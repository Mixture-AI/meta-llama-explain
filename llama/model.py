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
    """Model Arguments."""

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
    """RMSNorm, refer to docs/EN/RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability.
                Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # Divide each element of the tensor by its root mean square (RMS),
        # adding eps to ensure the square root is positive.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # Perform RMSNorm normalization.
        output = self._norm(x.float()).type_as(x)
        # Multiply the result by the learned scaling factor to complete the RMSNorm.
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
    # Calculate the basis for rotation angles used in RoPE for different dimensions
    # based on theta and dim parameters.
    # Specifically compute θ_i = theta^{-2i / dim} for i ∈ [0, dim / 2).
    # [Shape] freqs: (dim / 2, )
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )

    # Generate t based on the end index to sample rotation angles for all positions.
    # [Shape] t: (end, )
    t = torch.arange(end, device=freqs.device)

    # Sample rotation angles for all positions and corresponding dimensions
    # using outer product operation.
    # For each position pos_i in t and each set of dimension-specific angle bases θ_j in freqs,
    # sample the rotation angle corresponding to position and dimension pos_i * θ_j.
    # [Shape] freqs: (end, dim / 2)
    freqs = torch.outer(t, freqs).float()

    # Convert all rotation angles corresponding to dimensions for all positions
    # into complex exponential frequencies.
    # That is, for each rotation angle θ, calculate the corresponding frequency e^{iθ}.
    # Refer to docs/CN/RoPE for description based on Euler's formula.
    # [Shape] freqs_cis: (end, dim / 2)
    # [Type] freqs_cis: complex64
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
    # Number of dimensions in the target tensor x
    ndim = x.ndim
    # Meaningless exception check?
    assert 0 <= 1 < ndim
    # Check if the precomputed frequency tensor matches the length and feature dimensions of
    # the target tensor x.
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Calculate the target shape for reshaping the frequency tensor (1, seq_len, 1, dim / 2).
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
    # Reshape the query tensor xq into (batch_size, seq_len, n_heads, dim / 2, 2) and then
    # represent it in complex form.
    # This groups the elements of the tensor into pairs and represents them as complex numbers,
    # each pair as a 2D vector.
    # [Shape] xq_: (batch_size, seq_len, n_heads, dim / 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

    # Perform the same operation on the key tensor xk.
    # [Shape] xk_: (batch_size, seq_len, n_heads, dim / 2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape the frequency tensor freqs_cis for broadcasting.
    # [Shape] freqs_cis: (seq_len, dim / 2) -> (1, seq_len, 1, dim / 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Perform rotation on the tensor using complex multiplication and convert the
    # corresponding results back to real numbers.
    # Flatten the last two dimensions to restore the tensor to its original shape.
    # This converts each grouped 2D tensor d = [d1, d2]^T into a complex number d1 + i d2.
    # Use complex multiplication with the corresponding complex frequency e^{iθ}, then convert
    # back to real numbers,
    # achieving R(θ)d, which rotates each grouped 2D tensor by the corresponding angle,
    # thereby implementing RoPE (Rotation Position Encoding).
    # [Shape] xq_out: (batch_size, seq_len, n_heads, dim)
    # [Shape] xk_out: (batch_size, seq_len, n_heads, dim)
    # flatten operation instance (Shape changes):
    # (batch_size, seq_len, n_heads, dim // 2, 2) -flatten(3)-> (batch_size, seq_len, n_heads, dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """在 n_kv_heads 维度上重复扩展 key 或 query 张量.

    Args:
        x (torch.Tensor): 输入张量, Shape: (batch_size, sequence_length, n_kv_heads, head_dim).
        n_rep (int): 重复次数.

    Returns:
        torch.Tensor: 扩展后的张量.
    """
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
        """Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

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
        # key_cache_in_Attention
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        # value_cache_in_Attention
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
        """Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        # Perform query, key, value transformations on input x.
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Adjust the sizes of query, key, and value to distribute features across multiple heads.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE operation on query and key.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # Store the newly computed key and query into the corresponding cache based on the
        # starting position and length of the new sequence.
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # Retrieve keys and values needed for attention computation.
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # If n_kv_heads < n_heads, expand in the n_kv_heads dimension to match the sizes of
        # query, key, and value.
        # Q: Why would n_kv_heads be less than n_heads?
        # A: This occurs when the transformer layer uses Grouped-Query Attention.
        # Multiple query heads share a pair of key and value heads to reduce the KV-Cache,
        # hence n_heads must be divisible by n_kv_heads.
        # [Shape] keys: (batch_size, cache_len + seq_len, n_local_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # [Shape] values: (batch_size, cache_len + seq_len, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)

        # [Shape] xq: (batch_size, n_local_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # [Shape] keys: (batch_size, n_local_heads, cache_len + seq_len, head_dim)
        keys = keys.transpose(1, 2)
        # [Shape] values: (batch_size, n_local_heads, cache_len + seq_len, head_dim)
        values = values.transpose(1, 2)
        # Calculate attention scores.
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        # If there is a mask, add it to attention scores to mask out parts of the scores.
        if mask is not None:
            scores = scores + mask
        # Normalize attention scores using softmax.
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # Aggregate the values vector using attention scores.
        # [Shape] output: (batch_size, n_local_heads, seq_len, head_dim)
        output = torch.matmul(scores, values)
        # [Shape] output: (batch_size, seq_len, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # Apply a linear transformation to the output.
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
        """Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension.
                Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Custom scaling parameter for hidden layer dimensions.
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # Ensure hidden_dim is a multiple of multiple_of.
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
        """Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

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
        """Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # Residual calculation for attention.
        # [Shape] h: (batch_size, seq_len, dim)
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )

        # Residual calculation for FeedForward.
        # [Shape] out: (batch_size, seq_len, dim)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Transformer模型."""

    def __init__(self, params: ModelArgs):
        """Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

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
        # Compute complex exponential frequency tensor for RoPE.
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the
            # Llama 2 generation of models is 4096. Adding this multiplier instead of using 4096
            # directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        # Compute embeddings corresponding to the input token indices.
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        # Slice the complex exponential frequency tensor based on the starting position
        # and sequence length.
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        # If the sequence length is greater than 1, calculate the mask for Attention.
        if seqlen > 1:
            # First, create a [seq_len, seq_len] matrix where each element is set to -inf.
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
            # Retain the upper triangular part of the matrix, excluding the main diagonal values,
            # which are set to -inf, while all other elements are set to 0.
            # Indicates each token can only attend to itself and previously processed tokens.

            mask = torch.triu(mask, diagonal=1)

            # Due to key-value caching, we only need to compute attention scores for the
            # new sequence.
            # Therefore, the desired size of attention scores is (seq_len, cache_len + seq_len).
            # For the mask, elements (i, j) where j > cache_len + i need to be masked out.
            # In practice, we prepend the previously computed [seq_len, seq_len] mask matrix
            # with a zero-filled matrix
            # of size (seq_len, start_pos). This part does not require any masking, so we
            # concatenate a zero matrix
            # of size (seq_len, start_pos) to the beginning of the mask.
            # ┌─────────────────────────────────────────────────────────┐
            # │ > mask maxtirx visualization                            │
            # │                                                         │
            # │                                     hstack              │
            # │                                       ↓                 │
            # │            ↙ [0][0][0][0][0][0][0][0] | [0][x][x][x][x] │
            # │           ↙  [0][0][0][0][0][0][0][0] | [0][0][x][x][x] │
            # │ seq_len(5) ← [0][0][0][0][0][0][0][0] | [0][0][0][x][x] │
            # │           ↖  [0][0][0][0][0][0][0][0] | [0][0][0][0][x] │
            # │            ↖ [0][0][0][0][0][0][0][0] | [0][0][0][0][0] │
            # │                                    ↑     ↘  seq_len  ↙  │
            # │                               cache_len(8)              │
            # │                                                         │
            # │ x: denote the -inf value (masked value)                 │
            # └─────────────────────────────────────────────────────────┘
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # Iterate through each layer in the network.
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # Perform normalization before entering the output layer.
        h = self.norm(h)
        # Obtain logits through the final output layer.
        output = self.output(h).float()
        return output
