# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License
# Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn


@dataclass
class ModelArgs:
    """模型参数."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # 后续由 tokenizer 定义.
    multiple_of: int = (
        256  # 使 SwiGLU 隐藏层的尺寸为较大的2的幂的整数倍.
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    """RMSNorm, 参考：docs/CN/RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """初始化 RMSNorm 归一化层.

        Args:
            dim (int): 输入张量的维度.
            eps (float, optional): 为保证数值稳定性, 在分母上添加的极小数值. 默认设为 1e-6.

        Attributes:
            eps (float): 为保证数值稳定性, 在分母上添加的极小数值.
            weight (nn.Parameter): 可学习的缩放系数, 对应docs/CN/RMSNorm 中的增益系数.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """对输入张量应用 RMSNorm 归一化.

        Args:
            x (torch.Tensor): 输入张量.

        Returns:
            torch.Tensor: 归一化后的张量.

        """
        # 对张量的每个元素除以其均方根, 加入 eps 保证其开方为正数.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """RMSNorm 层的前向过程.

        Args:
            x (torch.Tensor): 输入张量.

        Returns:
            torch.Tensor: 应用 RMSNorm 归一化层后的张量.

        """
        # TODO [NAN] https://github.com/keli-wen/meta-llama2-explain/issues/7
        # 执行 RMSNorm 归一化.
        output = self._norm(x.float()).type_as(x)
        # 对结果乘上学习的增益系数完成整个 RMSNorm.
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension
    'dim' and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the
    provided frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and
    the frequency tensor is reshaped for broadcasting compatibility. The resulting tensors contain
    rotary embeddings and are returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with
            rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """对于 key 或 query 在 n_kv_heads 维度进行重复扩充."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-Head Self-Attention (多头自注意力机制)."""

    def __init__(self, args: ModelArgs):
        """初始化 Attention 模块.

        Args:
            args (ModelArgs): 模型配置参数.

        Attributes:
            n_kv_heads (int): key 和 value head 的个数.
            n_local_heads (int): Local query heads 的个数.
            n_local_kv_heads (int): Local key heads 和 local value heads 的个数.
            n_rep (int): Local heads 中特征需要重复的次数.
            head_dim (int): 每个 attention head 的维度.
            wq (ColumnParallelLinear): 用于 queries 的线性变换.
            wk (ColumnParallelLinear): 用于 keys 的线性变换.
            wv (ColumnParallelLinear): 用于 values 的线性变换.
            wo (RowParallelLinear): 用于输出的线性变换.
            cache_k (torch.Tensor): Attention 中的 cached keys.
            cache_v (torch.Tensor): Attention 中的 cached values.

        """
        super().__init__()
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # Attention 中 key 的缓存.
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        # Attention 中 value 的缓存.
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """Attention 模块的前向过程.

        Args:
            x (torch.Tensor): 输入张量.
            start_pos (int): Attention caching 的起始位置.
            freqs_cis (torch.Tensor): 预计算的复指数频率张量.
            mask (torch.Tensor, optional): Attention mask 张量.

        Returns:
            torch.Tensor: 经过 attention 操作后的输出张量.

        """
        bsz, seqlen, _ = x.shape
        # 对输入 x 分别进行 query, key, value 变换.
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 调整 query, key, value 的尺寸, 完成特征在多头上的分配.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 对 query 和 key 执行 RoPE 操作.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 取出 KV-Cache 中 key 和 value 的缓存.
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # 根据起始位置和新序列的长度将新计算得到的 key 和 query 存入对应的 cache 中.
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 取出需要参与 attention 计算的 key 和 value.
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 若 n_kv_heads < n_heads, 则在 n_kv_heads 维度上进行重复扩充
        # 使 query, key, value 的尺寸匹配.
        # [Shape] keys: (batch_size, cache_len + seqlen, n_local_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # [Shape] values: (batch_size, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)

        # [Shape] xq: (batch_size, n_local_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        # [Shape] keys: (batch_size, n_local_heads, cache_len + seqlen, head_dim)
        keys = keys.transpose(1, 2)
        # [Shape] values: (batch_size, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)
        # 计算 attention scores.
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        # 若存在 mask, 则对 attention scores 执行加法操作将 masked 部分置为 -inf.
        # [Shape] scores: (batch_size, n_local_heads, seqlen, cache_len + seqlen)
        if mask is not None:
            scores = scores + mask
        # 使用 softmax 归一化 attention scores.
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 使用 attention scores 聚合 values 向量.
        # [Shape] output: (batch_size, n_local_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)
        # [Shape] output: (batch_size, seqlen, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 对输出执行一次线性变换.
        return self.wo(output)


class FeedForward(nn.Module):
    """FeedForward (前馈神经网络)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """初始化 FeedForward 模块.

        Args:
            dim (int): 输入维度.
            hidden_dim (int): feedforward layer 隐藏层维度.
            multiple_of (int): 保证隐藏层维度是该值的整数倍.
            ffn_dim_multiplier (float, optional): 自定义的隐藏层维度缩放参数.
                默认设置为 None.

        Attributes:
            w1 (ColumnParallelLinear): 第一层的线性变换.
            w2 (RowParallelLinear): 第二层的线性变换.
            w3 (ColumnParallelLinear): 第三层的线性变换.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # 自定义的隐藏层维度缩放参数.
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.w3 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer基础模块."""

    def __init__(self, layer_id: int, args: ModelArgs):
        """初始化一个 TransformerBlock.

        Args:
            layer_id (int): 网络层的标记符.
            args (ModelArgs): 模型配置参数.

        Attributes:
            n_heads (int): Attention head 的数量.
            dim (int): 模块的特征维度.
            head_dim (int): 每一个 attention head 的特征维度.
            attention (Attention): Attention 模块.
            feed_forward (FeedForward): FeedForward 模块.
            layer_id (int): 网络层的标记符.
            attention_norm (RMSNorm): Attention 层前使用的 normalization.
            ffn_norm (RMSNorm): FeedForward 层前使用的 normalization.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """执行 TransformerBlock 的前向过程.

        Args:
            x (torch.Tensor): 输入张量.
            start_pos (int): Attention caching 的起始位置.
            freqs_cis (torch.Tensor): 预计算的复指数频率张量.
            mask (torch.Tensor, optional): 计算 attention 时的遮罩. 默认设为 None.

        Returns:
            torch.Tensor: 经过 attention 和 FeedForward 层后输出的张量.

        """
        # 残差计算 attention.
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        # 残差计算 FeedForward.
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Transformer模型."""

    def __init__(self, params: ModelArgs):
        """初始化 Transformer 模型.

        Args:
            params (ModelArgs): 模型配置参数.

        Attributes:
            params (ModelArgs): 模型配置参数.
            vocab_size (int): 词表大小.
            n_layers (int): 模型中的层数.
            tok_embeddings (ParallelEmbedding): Token embeddings, 将token 索引
                转化为对应的 embedding.
            layers (torch.nn.ModuleList): Transformer blocks 的列表.
            norm (RMSNorm): 模型输出层前使用的 normalization.
            output (ColumnParallelLinear): 模型输出层, 用于最终输出的线性层.
            freqs_cis (torch.Tensor): 预计算的复指数频率张量.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        # 计算用于 RoPE 的复指数频率张量.
        self.freqs_cis = precompute_freqs_cis(
            # ? 暂时搁置
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the
            # Llama 2 generation of models is 4096. Adding this multiplier instead of using 4096
            # directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """执行 Transformer 模型的前向过程.

        Args:
            tokens (torch.Tensor): 输入的 token 索引.
            start_pos (int): Attention caching 的起始位置.

        Returns:
            torch.Tensor: Transformer 模型输出的 logits.

        """
        _bsz, seqlen = tokens.shape
        # 根据输入的 token 索引计算对应的 embedding
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        # 根据起始位置和序列长度截取对应的复指数频率张量.
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        # 若序列长度大于 1, 则需要计算用于 Attention 的 mask.
        if seqlen > 1:
            # 首先根据序列长度构建一个 [seqlen, seqlen] 的矩阵, 每一个元素的值均为 -inf.
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            # 保留该矩阵的上三角部分, 不保留主对角线的值, 即主对角线右上部分的元素保持为 -inf,
            # 其余元素均为0.
            # 表示每个 token 只能使用当前 token 以及已经处理过的 token 的信息.
            mask = torch.triu(mask, diagonal=1)

            # 由于使用了 key-value caching, 我们只需要为新序列计算 attention scores.
            # 因此, 我们需要计算的 attention scores 的尺寸为 (seqlen, cache_len + seqlen).
            # 那么我们需要 mask 的元素 (i, j) 当 j > cache_len + i. 实际操作时, 我们只需要
            # 在先前计算的尺寸为 [seqlen, seqlen] 的 mask 矩阵前补上 cached 部分, 即
            # 起始位置前的部分, 这一部分的所有元素都不需要被 mask, 所以只需要拼接上一个
            # 尺寸为 (seqlen, start_pos) 的全零矩阵即可.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # 遍历每一层网络.
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 在进入输出层前执行一次 normalization.
        h = self.norm(h)
        # 通过最终输出层得到 logits.
        output = self.output(h).float()
        return output
