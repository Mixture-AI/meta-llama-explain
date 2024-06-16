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
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
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
    """预计算复指数 (cis) 频率张量, 具有给定的维度.

    此函数使用给定的维度 'dim' 和终止索引 'end' 计算具有复指数的频率张量.
    'theta' 参数用于缩放频率.
    返回的张量包含 complex64 数据类型的复数值.

    Args:
        dim (int): 频率张量的维度.
        end (int): 用于预计算频率的终止索引.
        theta (float, 可选): 频率计算的缩放因子。默认值为 10000.0.

    Returns:
        torch.Tensor: 预计算的复指数频率张量.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整频率张量的形状以便与另一个张量进行广播.

    此函数将频率张量调整为与目标张量 'x' 具有相同的形状，
    以便在元素级操作期间广播频率张量.

    Args:
        freqs_cis (torch.Tensor): 要调整形状的频率张量.
        x (torch.Tensor): 目标张量以实现广播兼容性.

    Returns:
        torch.Tensor: 调整形状后的频率张量.

    Raises:
        AssertionError: 如果频率张量的形状不符合预期.
        AssertionError: 如果目标张量 'x' 的维度数不符合预期.
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
    """使用给定的频率张量对输入张量应用旋转嵌入.

    此函数使用提供的频率张量 'freqs_cis' 对给定的查询 'xq' 和键 'xk' 张量应用旋转嵌入.
    输入张量被重塑为复数, 并且频率张量被重塑以实现广播兼容性. 生成的张量包含旋转嵌入,
    并作为实数张量返回.

    Args:
        xq (torch.Tensor): 要应用旋转嵌入的查询张量.
        xk (torch.Tensor): 要应用旋转嵌入的键张量.
        freqs_cis (torch.Tensor): 预计算的复指数频率张量.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 包含旋转嵌入的修改后的查询张量和键张量的元组.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""
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
        """初始化注意力模块.

        Args:
            args (ModelArgs): 模型配置参数.

        Attributes:
            n_kv_heads (int): 键和值头的数量.
            n_local_heads (int): 本地查询头的数量.
            n_local_kv_heads (int): 本地键和值头的数量.
            n_rep (int): 本地头的重复次数.
            head_dim (int): 每个注意力头的维度大小.
            wq (ColumnParallelLinear): 查询的线性变换.
            wk (ColumnParallelLinear): 键的线性变换.
            wv (ColumnParallelLinear): 值的线性变换.
            wo (RowParallelLinear): 输出的线性变换.
            cache_k (torch.Tensor): 注意力的缓存键.
            cache_v (torch.Tensor): 注意力的缓存值.
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

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
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
        """注意力模块的前向传递.

        Args:
            x (torch.Tensor): 输入张量.
            start_pos (int): 缓存的起始位置.
            freqs_cis (torch.Tensor): 预计算的频率张量.
            mask (torch.Tensor, optional): 注意力掩码张量.

        Returns:
            torch.Tensor: 注意力后的输出张量.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        if mask is not None:
            scores = (
                scores + mask
            )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(
            scores, values
        )  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
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
        """初始化前馈模块.

        Args:
            dim (int): 输入维度.
            hidden_dim (int): 前馈层的隐藏维度.
            multiple_of (int): 确保隐藏维度是此值的倍数.
            ffn_dim_multiplier (float, optional): 隐藏维度的自定义倍增器.
                默认为 None.

        Attributes:
            w1 (ColumnParallelLinear): 第一层的线性变换.
            w2 (RowParallelLinear): 第二层的线性变换.
            w3 (ColumnParallelLinear): 第三层的线性变换.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
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
            layer_id (int): 层的标识符.
            args (ModelArgs): 模型配置参数.

        Attributes:
            n_heads (int): 注意力头的数量.
            dim (int): 模型的维度大小.
            head_dim (int): 每个注意力头的维度大小.
            attention (Attention): 注意力模块.
            feed_forward (FeedForward): 前馈模块.
            layer_id (int): 层的标识符.
            attention_norm (RMSNorm): 注意力输出的层归一化.
            ffn_norm (RMSNorm): 前馈输出的层归一化.
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
        """通过 TransformerBlock 执行前向传递.

        Args:
            x (torch.Tensor): 输入张量.
            start_pos (int): 注意力缓存的起始位置.
            freqs_cis (torch.Tensor): 预计算的余弦和正弦频率.
            mask (torch.Tensor, optional): 注意力的掩码张量. 默认为 None.

        Returns:
            torch.Tensor: 应用注意力和前馈层后的输出张量.
        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Transformer模型."""

    def __init__(self, params: ModelArgs):
        """初始化一个 Transformer 模型.

        Args:
            params (ModelArgs): 模型配置参数.

        Attributes:
            params (ModelArgs): 模型配置参数.
            vocab_size (int): 词汇表大小.
            n_layers (int): 模型中的层数.
            tok_embeddings (ParallelEmbedding): 令牌嵌入.
            layers (torch.nn.ModuleList): Transformer 块的列表.
            norm (RMSNorm): 模型输出的层归一化.
            output (ColumnParallelLinear): 最终输出的线性层.
            freqs_cis (torch.Tensor): 预计算的余弦和正弦频率.
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

        self.freqs_cis = precompute_freqs_cis(
            # 注意 self.params.max_seq_len 被乘以 2，因为 Llama 2 生成的模型的令牌限制为 4096。
            # 添加此乘数而不是直接使用 4096，允许在训练或微调期间令牌长度的动态性。
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """通过 Transformer 模型执行前向传递.

        Args:
            tokens (torch.Tensor): 输入令牌索引.
            start_pos (int): 注意力缓存的起始位置.

        Returns:
            torch.Tensor: 应用 Transformer 模型后的输出对数.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # 在执行键值缓存时，我们仅计算新序列的注意力分数。
            # 因此，分数矩阵的大小为 (seqlen, cache_len + seqlen)，并且唯一的掩码条目是 (i, j)，
            # 其中 j > cache_len + i，
            # 因为第 i 行对应于令牌 cache_len + i。

            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
