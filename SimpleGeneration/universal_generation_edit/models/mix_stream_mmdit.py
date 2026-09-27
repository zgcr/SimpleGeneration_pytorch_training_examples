import math
from einops import rearrange

# flash-attn==2.8.3
# flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
from flash_attn import flash_attn_func, flash_attn_varlen_func

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

__all__ = [
    'MixStreamMMDiT_1B',
    'MixStreamMMDiT_2B',
    'MixStreamMMDiT_4B',
    'MixStreamMMDiT_6B',
    'MixStreamMMDiT_8B',
]


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    assert dim % 2 == 0

    t = time_factor * t.float()
    half_dim = dim // 2

    freqs = torch.exp(-math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=t.device) /
                      half_dim)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


def rope(pos, dim, theta):
    assert dim % 2 == 0

    pos = pos.float()
    scale = torch.arange(0, dim, 2, dtype=torch.float32,
                         device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    out = pos[..., None] * omega
    out = torch.complex(torch.cos(out), torch.sin(out))

    return out


def apply_rope(xq, xk, freqs_cis):
    freqs_cis = torch.view_as_real(freqs_cis)
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]

    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_out = torch.stack([
        cos * xq_[..., 0] +
        (-sin) * xq_[..., 1], sin * xq_[..., 0] + cos * xq_[..., 1]
    ],
                         dim=-1)
    xk_out = torch.stack([
        cos * xk_[..., 0] +
        (-sin) * xk_[..., 1], sin * xk_[..., 0] + cos * xk_[..., 1]
    ],
                         dim=-1)

    xq_out = xq_out.flatten(-2).type_as(xq)
    xk_out = xk_out.flatten(-2).type_as(xk)

    return xq_out, xk_out


def swiglu_hidden_dim(hidden_size, multiple_of=64):
    hidden_dim = int(round(hidden_size / 3 * 8 / multiple_of) * multiple_of)

    return hidden_dim


def center_image_ids(ids, index=None, num_index=1):
    """
    Re-centre the h / w axes of one image segment's position ids on 0.

    Out of place on purpose: `img_ids` / `ref_ids` are built ONCE by the wrapper
    and then reused for every sampler step and for both CFG branches, so an
    in-place shift would re-centre the same tensor again on every forward.

    Args:
        ids: [B, L, 4] position ids of ONE segment; only the h / w axes are
            touched, the t / l axes pass through untouched.
        index: [B, L] long per-token block index (`ref_index`), or None when the
            whole segment is a single image block (the noise target).
        num_index: number of distinct block indices (`max_ref_images`).

    Returns:
        centered_ids: [B, L, 4], same dtype / device as `ids`.
    """
    hw = ids[..., 1:3]

    if index is None:
        # A single block, whose ids must be a full 0-based grid for `amax` to be
        # its extent; that is exactly what the wrapper's `encode_image_latent`
        # builds.
        assert float(hw.amin()) == 0.0
        extent = hw.amax(dim=1, keepdim=True)
    else:
        # Per-block extent, gathered back onto every token of its own block.
        # Padded reference positions carry index 0 and all-zero ids, so they can
        # never raise a maximum.
        index = index[..., None].expand(-1, -1, 2)
        extent = hw.new_zeros(hw.shape[0], num_index, 2).scatter_reduce(
            1, index, hw, reduce='amax').gather(1, index)

    centered_ids = ids.clone()
    centered_ids[..., 1:3] = hw - (extent + 2).div(2, rounding_mode='floor')

    return centered_ids


def attention(q, k, v, pe, num_heads, num_kv_heads):
    """
    No-padding fast path of the 'sdpa' backend (dense SDPA).

    `attn_mask` is deliberately NOT a parameter: passing any mask disqualifies
    SDPA from its fused flash kernel and drops it onto the math kernel, which
    materializes the whole [B, H, L, L] score matrix. The padded case therefore
    goes through `attention_packed` instead, which reaches flash by packing.

    `enable_gqa` is only turned on when num_kv_heads != num_heads. The softmax
    scale is SDPA's default (1/sqrt(head_dim)), i.e. standard scaled
    dot-product attention.

    Segment isolation (noise / ctx / ref never attending to each other) is NOT
    done here: it comes from the caller handing each segment its own q/k/v and
    its own `AttentionArgs` (see the per-segment args built in the forward).

    Args:
        q: [B, Hq, L, D] (pre-RoPE).
        k/v: [B, Hkv, L, D].
        pe: complex64 RoPE table broadcastable to [B, 1, L, D//2].

    Returns:
        x: [B, L, Hq*D], cast back to the dtype of the incoming q. The kernel
            runs in bf16, except under an fp16 autocast, which re-casts it to
            fp16 (see the cast below).
    """
    q, k = apply_rope(q, k, pe)
    enable_gqa = num_kv_heads != num_heads

    # SDPA only reaches its fused flash kernel in fp16/bf16, so q/k/v are cast
    # to bf16 and the caller's dtype is restored afterwards. Without it a
    # pure-fp32 forward silently falls back to the math kernel and
    # materializes the [B, H, L, L] scores. Note that an fp16 autocast
    # OVERRIDES this cast (SDPA is on autocast's cast list, so it re-casts its
    # inputs to fp16), which is the one case where this backend does not agree
    # with 'flash_varlen': flash-attn is not registered with autocast and so
    # really does run in bf16 under every amp mode.
    compute_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    x = F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)
    x = rearrange(x, "B H L D -> B L (H D)").to(compute_dtype)

    return x


def attention_packed(q, k, v, pe, attn_args, num_heads, num_kv_heads):
    """
    Padded path of the 'sdpa' backend, kept on the flash kernel by packing.

    SDPA only reaches its fused flash kernel when no `attn_mask` is given, so a
    `[B, 1, 1, L]` key-padding mask would silently downgrade every padded
    forward to the math kernel. This function removes the mask instead of the
    padding it describes: it GATHERS the valid tokens of every sample into one
    flat sequence -- exactly like `attention_varlen` -- and hands SDPA a jagged
    nested tensor, whose `cu_seqlens` offsets carry the per-sample boundaries the
    mask used to carry.

    Mathematical equivalence to the masked SDPA call it replaces: each query
    attends, within its own sample, over exactly the same set of valid keys and
    never across sample boundaries; same RoPE (applied here, before packing),
    same softmax scale. GQA is expressed by repeating the K/V heads instead of
    `enable_gqa` (the nested layout does not accept that flag); the two are
    bit-identical, since `enable_gqa` is itself defined as that repeat.
    Padded query positions are simply dropped (they stay zero in the scattered
    output; those rows only ever act as masked-out keys downstream, and the
    supervised slice `img[:, :num_img_tokens]` is always fully valid).

    The packing args (flat_index / cu_seqlens) are NOT built here: they are
    precomputed ONCE per forward in `AttentionArgs` and reused by every block,
    so this function performs no GPU->CPU sync at all.

    Args:
        q: [B, Hq, L, D] (pre-RoPE, same layout as `attention()`).
        k/v: [B, Hkv, L, D].
        pe: complex64 RoPE table broadcastable to [B, 1, L, D//2].
        attn_args: `AttentionArgs` built from this segment's key-padding mask.

    Returns:
        x: [B, L, Hq*D] with padded positions left as zeros, cast back to the
            dtype of the incoming q.
    """
    q, k = apply_rope(q, k, pe)

    B, Hq, L, D = q.shape
    Hkv = k.shape[1]

    # [B, Hh, L, D] -> [B, L, Hh, D] then flatten the batch/sequence axes so the precomputed row-major flat_index can gather with index_select.
    q = q.transpose(1, 2).reshape(B * L, Hq, D)
    k = k.transpose(1, 2).reshape(B * L, Hkv, D)
    v = v.transpose(1, 2).reshape(B * L, Hkv, D)

    flat_index = attn_args.flat_index
    # [total, Hq, D]
    q_packed = q.index_select(0, flat_index)
    # [total, Hkv, D]
    k_packed = k.index_select(0, flat_index)
    # [total, Hkv, D]
    v_packed = v.index_select(0, flat_index)

    # Same cast as in `attention()`, and with the same fp16 caveat: an fp16
    # autocast overrides it and runs this kernel in fp16, while the
    # 'flash_varlen' backend stays in bf16 under every amp mode. Done here,
    # before the GQA repeat below, so only the packed (not the padded) tokens
    # are cast and the repeat itself runs on the narrower dtype.
    compute_dtype = q_packed.dtype
    q_packed = q_packed.to(torch.bfloat16)
    k_packed = k_packed.to(torch.bfloat16)
    v_packed = v_packed.to(torch.bfloat16)

    # The nested layout has no `enable_gqa`, so the K/V heads are repeated to
    # Hq explicitly. Bit-identical to `enable_gqa=True`, which is defined as
    # exactly this repeat, and only the packed (not the padded) tokens pay it.
    if num_kv_heads != num_heads:
        k_packed = k_packed.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_packed = v_packed.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # [B, Hq, j_total, D] jagged views over the packed buffers; cu_seqlens is
    # the same per-sample boundary tensor the varlen backend consumes.
    cu_seqlens = attn_args.cu_seqlens
    q_packed = torch.nested.nested_tensor_from_jagged(q_packed,
                                                      cu_seqlens).transpose(
                                                          1, 2)
    k_packed = torch.nested.nested_tensor_from_jagged(k_packed,
                                                      cu_seqlens).transpose(
                                                          1, 2)
    v_packed = torch.nested.nested_tensor_from_jagged(v_packed,
                                                      cu_seqlens).transpose(
                                                          1, 2)

    out = F.scaled_dot_product_attention(q_packed, k_packed, v_packed)
    # [total, Hq*D]
    out = out.transpose(1, 2).values().reshape(-1, Hq * D).to(compute_dtype)

    # scatter back to the padded layout (padded rows stay zero). index_copy is
    # the int-index counterpart of `x[valid_mask] = out` and writes exactly the
    # same rows, but its backward is a plain index_select (no nonzero / sync).
    x = q.new_zeros(B * L, Hq * D)
    x = x.index_copy(0, flat_index, out)
    x = x.view(B, L, Hq * D)

    return x


def attention_flash_dense(q, k, v, pe):
    """
    No-padding fast path of the 'flash_varlen' backend (dense flash-attn).

    When a segment carries no padding at all, packing the valid tokens is an
    identity permutation: `attention_varlen` would still pay 3 gathers, 1
    scatter and several GPU->CPU syncs to produce exactly the same numbers the
    dense kernel gives directly. So we skip the packing entirely here.

    Mathematically identical to `attention_varlen` with an all-True mask (a
    varlen call whose every segment length equals L attends over exactly the
    same keys), and identical to `attention()` with attn_mask=None up to the
    usual kernel-level floating-point reduction order -- except under an fp16
    autocast, where SDPA is re-cast to fp16 while the bf16 pinning below
    survives (flash-attn is not registered with autocast).

    Args:
        q: [B, Hq, L, D] (pre-RoPE, same layout as `attention()`).
        k/v: [B, Hkv, L, D]. Hkv <= Hq; Hkv < Hq (GQA) is handled
            natively by flash-attn, same as the varlen func.
        pe: complex64 RoPE table broadcastable to [B, 1, L, D//2].

    Returns:
        x: [B, L, Hq*D], cast back to the dtype of the incoming q. The kernel
            itself always runs in bf16, whatever the amp mode.
    """
    q, k = apply_rope(q, k, pe)

    B, Hq, L, D = q.shape

    # [B, Hh, L, D] -> [B, L, Hh, D] for flash-attn's (batch, seqlen, nheads, headdim) layout.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # flash-attn only supports fp16/bf16, and it is NOT registered with
    # autocast, so this cast really does pin the kernel to bf16 in every amp
    # mode (fp32 / fp16 / bf16); the caller's dtype is restored afterwards.
    # This is intentional, NOT a missing guard -- `attention_varlen` does the
    # same. No-op under the bf16 training configs; costs 3 mantissa bits under
    # fp16, where the 'sdpa' backend instead follows the autocast down to fp16.
    compute_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    # [B, L, Hq, D]
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    out = out.reshape(B, L, Hq * D).to(compute_dtype)

    return out


def attention_varlen(q, k, v, pe, attn_args):
    """
    flash-attn varlen counterpart of `attention()`.

    This is the "packing" backend: instead of masking the padded positions with
    an attn_mask over the full [B, L, L] score matrix (what `attention()` does
    via SDPA), it physically GATHERS the valid tokens of every sample into one
    flat sequence and hands flash-attn the per-sample boundaries via
    `cu_seqlens`, so padding tokens never enter the attention kernel at all.

    Mathematical equivalence to the SDPA path (on the valid tokens):
      * Both attend, within each sample, over exactly the same set of valid
        keys and never across sample boundaries (SDPA: key-padding mask +
        batch-separated; varlen: cu_seqlens segments). Padded query positions
        are simply dropped here (they stay zero in the scattered output; those
        rows only ever act as masked-out keys downstream, and the supervised
        slice `img[:, :num_img_tokens]` is always fully valid).
      * Same RoPE (applied here, identically, BEFORE packing), same GQA head
        layout (flash-attn consumes fewer K/V heads natively), same softmax
        scale (flash default 1/sqrt(head_dim) == SDPA default).
    So the two backends are mathematically identical on the valid tokens; only
    the floating-point reduction order of the two CUDA kernels differs, so the
    numerical outputs are close but not bit-identical. Under an fp16 autocast
    they additionally differ in precision: this path stays bf16 (see the cast
    below) while SDPA is re-cast to fp16.

    The packing args (flat_index / cu_seqlens / max_seqlen) are NOT built
    here: they are precomputed ONCE per forward in `AttentionArgs` and reused
    by every block, so this function performs no GPU->CPU sync at all. The
    gather / scatter go through `index_select` / `index_copy_` on the
    precomputed int64 index instead of boolean indexing, which would re-run
    `nonzero()` (and sync) on every single call.

    Args:
        q: [B, Hq, L, D]  (pre-RoPE, same layout as `attention()`).
        k/v: [B, Hkv, L, D].
        pe: complex64 RoPE table broadcastable to [B, 1, L, D//2].
        attn_args: `AttentionArgs` built from this segment's key-padding mask.

    Returns:
        x: [B, L, Hq*D] with padded positions left as zeros, cast back to the
            dtype of the incoming q.
    """
    q, k = apply_rope(q, k, pe)

    B, Hq, L, D = q.shape
    Hkv = k.shape[1]

    # [B, Hh, L, D] -> [B, L, Hh, D] then flatten the batch/sequence axes so the precomputed row-major flat_index can gather with index_select.
    q = q.transpose(1, 2).reshape(B * L, Hq, D)
    k = k.transpose(1, 2).reshape(B * L, Hkv, D)
    v = v.transpose(1, 2).reshape(B * L, Hkv, D)

    flat_index = attn_args.flat_index
    # [total, Hq, D]
    q_packed = q.index_select(0, flat_index)
    # [total, Hkv, D]
    k_packed = k.index_select(0, flat_index)
    # [total, Hkv, D]
    v_packed = v.index_select(0, flat_index)

    # flash-attn only supports fp16/bf16, and it is NOT registered with
    # autocast, so this cast really does pin the kernel to bf16 in every amp
    # mode (fp32 / fp16 / bf16); the caller's dtype is restored afterwards.
    # This is intentional, NOT a missing guard -- `attention_flash_dense` does
    # the same, and the two paths of this backend must agree since
    # `run_attention` picks between them per segment on `all_valid` alone.
    # No-op under the bf16 training configs; costs 3 mantissa bits under fp16,
    # where the 'sdpa' backend instead follows the autocast down to fp16.
    compute_dtype = q_packed.dtype
    q_packed = q_packed.to(torch.bfloat16)
    k_packed = k_packed.to(torch.bfloat16)
    v_packed = v_packed.to(torch.bfloat16)

    out = flash_attn_varlen_func(q_packed,
                                 k_packed,
                                 v_packed,
                                 cu_seqlens_q=attn_args.cu_seqlens,
                                 cu_seqlens_k=attn_args.cu_seqlens,
                                 max_seqlen_q=attn_args.max_seqlen,
                                 max_seqlen_k=attn_args.max_seqlen,
                                 dropout_p=0.0,
                                 causal=False)  # [total, Hq, D]
    out = out.reshape(out.shape[0], Hq * D).to(compute_dtype)

    # scatter back to the padded layout (padded rows stay zero). index_copy_ is
    # the int-index counterpart of `x[valid_mask] = out` and writes exactly the
    # same rows, but its backward is a plain index_select (no nonzero / sync).
    x = q.new_zeros(B * L, Hq * D, dtype=compute_dtype)
    x = x.index_copy(0, flat_index, out)
    x = x.view(B, L, Hq * D)

    return x


class AttentionArgs:
    """
    Precomputed attention args for ONE key-padding mask, shared by both
    backends.

    Built once per forward per distinct mask (noise / ctx / ref / joint) and
    reused by every block, instead of being rebuilt on every attention call.
    That is what collapses the GPU->CPU syncs (`mask.all()`, `.max().item()`
    and the `nonzero()` hidden inside boolean indexing) from ~170/step down to
    ~4; each sync stalls the CPU from enqueueing the next block.

    Both backends pack the padded batch through the SAME `flat_index` /
    `cu_seqlens` pair, so only `max_seqlen` (which flash-attn takes as a plain
    int, and SDPA's nested layout derives itself) is backend-specific; that
    keeps its `.item()` sync off the sdpa path.

    Numerics are unchanged: `all_valid` sends both backends down their dense
    no-mask path, which is equivalent to an all-True mask, and keeping the
    padded case mask-FREE (packed instead) is what leaves SDPA eligible for its
    fused flash kernel in that case too. The two backends do run at different
    precision under an fp16 autocast, though: sdpa follows it down to fp16
    while flash_varlen stays bf16 (see the casts in the attention functions).

    Args:
        valid_mask: [B, L] bool, True = valid token. None means the caller
            guarantees full validity (e.g. the noise refiner), so not even the
            `all()` sync is paid.
        attention_backend: 'sdpa' or 'flash_varlen'; only the active backend's
            extra args are built.

    Attributes (the three packing args below are all None when all_valid):
        all_valid: bool, True when there is no padding at all.
        flat_index: [total] int64 indices of the valid tokens in the flattened
            [B*L] axis; row-major, so sample 0's tokens come first and the
            packing lines up with cu_seqlens (both backends).
        cu_seqlens: [B+1] int32 per-sample segment boundaries of the packed
            sequence (both backends).
        max_seqlen: int, longest valid segment (flash_varlen).
    """

    def __init__(self, valid_mask, attention_backend='sdpa'):
        self.flat_index = None
        self.cu_seqlens = None
        self.max_seqlen = None

        # Caller guarantees every token is valid (e.g. the noise refiner), so we do not even run `all()`.
        if valid_mask is None:
            self.all_valid = True
        # The ONE `all()` sync per args object, previously paid on EVERY attention call.
        else:
            self.all_valid = bool(valid_mask.all())

        # The three fields above stay None: `run_attention` takes the dense
        # no-mask path and reads none of them. Returning here is what skips
        # the `nonzero()` / `.max().item()` syncs below.
        if self.all_valid:

            return

        self.flat_index = valid_mask.reshape(-1).nonzero(as_tuple=True)[0]
        seqlens = valid_mask.sum(dim=1).to(torch.int32)
        self.cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0),
                                (1, 0)).to(torch.int32)

        # Only flash-attn needs the longest segment as a python int, so its `.item()` sync is not paid on the sdpa path.
        if attention_backend == 'flash_varlen':
            self.max_seqlen = int(seqlens.max().item())


def run_attention(q, k, v, pe, attn_args, attention_backend, num_heads,
                  num_kv_heads):
    """
    Single dispatch point shared by every block so the two backends stay in
    lock-step (same q/k/v/pe/attn_args -> mathematically identical result on
    the valid tokens). Performs no GPU->CPU sync: everything mask-related was
    already resolved when `attn_args` was built at the top of the forward.

    Dispatch is a 2x2 on (backend, padding), and all four paths run a flash
    kernel -- the padded ones by PACKING the valid tokens rather than masking
    the padded ones, since a mask is exactly what would disqualify flash:
      * sdpa + all_valid: dense SDPA, no mask at all.
      * sdpa + padded: SDPA over a jagged nested tensor built from the
        precomputed cu_seqlens, so the padding never enters the kernel.
      * flash_varlen + all_valid: dense flash_attn_func, no packing at all.
      * flash_varlen + padded: flash varlen over the same precomputed
        cu_seqlens.
    """
    if attention_backend == 'flash_varlen':
        # No padding to pack, so skip the gather / scatter entirely.
        if attn_args.all_valid:
            return attention_flash_dense(q, k, v, pe)
        else:
            return attention_varlen(q, k, v, pe, attn_args)
    else:
        # No padding to pack, so skip the gather / scatter entirely.
        if attn_args.all_valid:
            return attention(q, k, v, pe, num_heads, num_kv_heads)
        else:
            return attention_packed(q, k, v, pe, attn_args, num_heads,
                                    num_kv_heads)


class EmbedND(nn.Module):

    def __init__(self, dim, theta, axes_dim):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([
            rope(ids[..., i], self.axes_dim[i], self.theta)
            for i in range(n_axes)
        ],
                        dim=-1)
        emb = emb.unsqueeze(1)

        return emb


class MLPEmbedder(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLPEmbedder, self).__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = x.to(self.in_layer.weight.dtype)
        x = self.out_layer(self.silu(self.in_layer(x)))

        return x


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        x = (x * rrms).to(dtype=x_dtype) * self.scale

        return x


class QKNorm(nn.Module):

    def __init__(self, dim):
        super(QKNorm, self).__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)

        q = q.to(v)
        k = k.to(v)

        return q, k


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))

        return x


class Modulation(nn.Module):

    def __init__(self, dim, double, adaln_dim=256):
        super(Modulation, self).__init__()

        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(adaln_dim, self.multiplier * dim, bias=False)

        # AdaLN-Zero: zero-init so that shift/scale/gate are all 0 at start, making each block an identity mapping for stable training from scratch.
        nn.init.zeros_(self.lin.weight)

    def forward(self, vec):
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        mod1 = (out[0], out[1], out[2])

        if self.is_double:
            mod2 = (out[3], out[4], out[5])
        else:
            mod2 = None

        return mod1, mod2


class RefinerBlock(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_hidden_dim,
                 num_kv_heads,
                 modulation=True,
                 adaln_dim=256):
        super(RefinerBlock, self).__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.modulation = modulation

        self.gate_act = nn.Tanh()

        if modulation:
            self.mod = Modulation(hidden_size,
                                  double=True,
                                  adaln_dim=adaln_dim)

        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(dim=hidden_size,
                                  num_heads=num_heads,
                                  num_kv_heads=num_kv_heads)
        self.attn_post_norm = RMSNorm(hidden_size)

        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_post_norm = RMSNorm(hidden_size)

    def forward(self, x, vec, pe, attn_args, attention_backend='sdpa'):
        if self.modulation:
            mod1, mod2 = self.mod(vec)
            shift1, scale1, gate1 = mod1
            shift2, scale2, gate2 = mod2

        h = self.norm1(x)

        if self.modulation:
            h = (1 + scale1) * h + shift1

        q = rearrange(self.attn.to_q(h),
                      "B L (H D) -> B H L D",
                      H=self.num_heads)
        k = rearrange(self.attn.to_k(h),
                      "B L (H D) -> B H L D",
                      H=self.num_kv_heads)
        v = rearrange(self.attn.to_v(h),
                      "B L (H D) -> B H L D",
                      H=self.num_kv_heads)
        q, k = self.attn.norm(q, k, v)
        a = self.attn.proj(
            run_attention(q, k, v, pe, attn_args, attention_backend,
                          self.num_heads, self.num_kv_heads))

        if self.modulation:
            gate_a = self.gate_act(gate1)
        else:
            gate_a = 1.0

        x = x + gate_a * self.attn_post_norm(a)

        h = self.norm2(x)

        if self.modulation:
            h = (1 + scale2) * h + shift2

        m = self.mlp(h)

        if self.modulation:
            gate_m = self.gate_act(gate2)
        else:
            gate_m = 1.0

        x = x + gate_m * self.mlp_post_norm(m)

        return x


class DoubleStreamBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, adaln_dim=256):
        super(DoubleStreamBlock, self).__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        mlp_hidden_dim = swiglu_hidden_dim(hidden_size)

        self.gate_act = nn.Tanh()

        # ---- image stream ----
        self.img_mod = Modulation(hidden_size,
                                  double=True,
                                  adaln_dim=adaln_dim)
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_attn = SelfAttention(dim=hidden_size,
                                      num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)

        self.img_norm2 = RMSNorm(hidden_size)
        self.img_mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.img_attn_post_norm = RMSNorm(hidden_size)
        self.img_mlp_post_norm = RMSNorm(hidden_size)

        # ---- text stream ----
        self.txt_mod = Modulation(hidden_size,
                                  double=True,
                                  adaln_dim=adaln_dim)
        self.txt_norm1 = RMSNorm(hidden_size)
        self.txt_attn = SelfAttention(dim=hidden_size,
                                      num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)
        self.txt_norm2 = RMSNorm(hidden_size)
        self.txt_mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.txt_attn_post_norm = RMSNorm(hidden_size)
        self.txt_mlp_post_norm = RMSNorm(hidden_size)

    @staticmethod
    def select_mod(mod_real, mod_zero, is_source, seq_len):
        m = is_source[..., None]
        out = []
        for a, b in zip(mod_real, mod_zero):
            out.append(
                torch.where(m, b.expand(-1, seq_len, -1),
                            a.expand(-1, seq_len, -1)))
        out = tuple(out)

        return out

    def forward(self,
                img,
                txt,
                vec,
                vec_zero,
                pe,
                is_source,
                attn_args,
                attention_backend='sdpa'):
        img_mod1, img_mod2 = self.img_mod(vec)

        # source (reference) image tokens use a zero-timestep modulation while the noise-target tokens keep the real timestep.
        if vec_zero is not None and is_source is not None:
            img_mod1_z, img_mod2_z = self.img_mod(vec_zero)
            L_img = img.shape[1]
            img_mod1 = self.select_mod(img_mod1, img_mod1_z, is_source, L_img)
            img_mod2 = self.select_mod(img_mod2, img_mod2_z, is_source, L_img)

        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        img_q = rearrange(self.img_attn.to_q(img_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_heads)
        img_k = rearrange(self.img_attn.to_k(img_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_kv_heads)
        img_v = rearrange(self.img_attn.to_v(img_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_kv_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare text for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        txt_q = rearrange(self.txt_attn.to_q(txt_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_heads)
        txt_k = rearrange(self.txt_attn.to_k(txt_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_kv_heads)
        txt_v = rearrange(self.txt_attn.to_v(txt_modulated),
                          "B L (H D) -> B H L D",
                          H=self.num_kv_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # joint attention (text tokens first, then image tokens)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = run_attention(q, k, v, pe, attn_args, attention_backend,
                             self.num_heads, self.num_kv_heads)
        num_txt_tokens = txt.shape[1]

        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]

        # calculate the img blocks (tanh gate + residual-side post-norm)
        img = img + self.gate_act(img_mod1_gate) * self.img_attn_post_norm(
            self.img_attn.proj(img_attn))
        img = img + self.gate_act(img_mod2_gate) * self.img_mlp_post_norm(
            self.img_mlp(
                (1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift))

        # calculate the txt blocks
        txt = txt + self.gate_act(txt_mod1_gate) * self.txt_attn_post_norm(
            self.txt_attn.proj(txt_attn))
        txt = txt + self.gate_act(txt_mod2_gate) * self.txt_mlp_post_norm(
            self.txt_mlp(
                (1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift))

        return img, txt


class SingleStreamBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, adaln_dim=256):
        super(SingleStreamBlock, self).__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        mlp_hidden_dim = swiglu_hidden_dim(hidden_size)

        self.gate_act = nn.Tanh()

        self.mod = Modulation(hidden_size, double=True, adaln_dim=adaln_dim)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(dim=hidden_size,
                                  num_heads=num_heads,
                                  num_kv_heads=num_kv_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.attn_post_norm = RMSNorm(hidden_size)
        self.mlp_post_norm = RMSNorm(hidden_size)

    @staticmethod
    def select_mod(mod_real, mod_zero, is_source, seq_len):
        m = is_source[..., None]
        out = []
        for a, b in zip(mod_real, mod_zero):
            out.append(
                torch.where(m, b.expand(-1, seq_len, -1),
                            a.expand(-1, seq_len, -1)))
        out = tuple(out)

        return out

    def forward(self,
                x,
                vec,
                vec_zero,
                pe,
                is_source,
                attn_args,
                attention_backend='sdpa'):
        mod1, mod2 = self.mod(vec)

        # source (reference) image tokens use a zero-timestep modulation while the text / noise-target tokens keep the real timestep.
        if vec_zero is not None and is_source is not None:
            mod1_z, mod2_z = self.mod(vec_zero)
            L = x.shape[1]
            mod1 = self.select_mod(mod1, mod1_z, is_source, L)
            mod2 = self.select_mod(mod2, mod2_z, is_source, L)

        mod1_shift, mod1_scale, mod1_gate = mod1
        mod2_shift, mod2_scale, mod2_gate = mod2

        # attention sub-layer
        h = self.norm1(x)
        h = (1 + mod1_scale) * h + mod1_shift
        q = rearrange(self.attn.to_q(h),
                      "B L (H D) -> B H L D",
                      H=self.num_heads)
        k = rearrange(self.attn.to_k(h),
                      "B L (H D) -> B H L D",
                      H=self.num_kv_heads)
        v = rearrange(self.attn.to_v(h),
                      "B L (H D) -> B H L D",
                      H=self.num_kv_heads)
        q, k = self.attn.norm(q, k, v)
        a = self.attn.proj(
            run_attention(q, k, v, pe, attn_args, attention_backend,
                          self.num_heads, self.num_kv_heads))
        x = x + self.gate_act(mod1_gate) * self.attn_post_norm(a)

        # mlp sub-layer
        h = self.norm2(x)
        h = (1 + mod2_scale) * h + mod2_shift
        m = self.mlp(h)
        x = x + self.gate_act(mod2_gate) * self.mlp_post_norm(m)

        return x


class LastLayer(nn.Module):

    def __init__(self, hidden_size, out_channels, adaln_dim=256):
        super(LastLayer, self).__init__()

        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(adaln_dim, 2 * hidden_size, bias=False))

        # AdaLN-Zero: zero-init the final modulation and output projection so that the model starts as an identity/zero-output mapping.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)

        return x


class MixStreamMMDiT(nn.Module):
    """
    Mix-stream MMDiT DENOISE model (pure DiT, no AE / no VLM).

    This is the denoise transformer ONLY. The AE (VAE encode/decode), the VLM
    text/image condition encoder and the rectified-flow interpolation live in
    the wrapper UniversalGenerationEditModel, which feeds this module already-
    encoded latent tokens + condition features.

    The tokens first run through a double-stream stage (image and text keep
    SEPARATE weights, mixing only inside the joint attention) and then through a
    single-stream stage (one joint [txt; img] sequence sharing a SINGLE weight
    set). This stream supports both the T2I and the TI2I task.

    forward() consumes the UNIFIED denoise-DiT signature shared with
    DoubleStreamMMDiT / SingleStreamMMDiT:
        forward(x_t, img_ids, timesteps, ctx, ctx_ids, ctx_mask,
                ref_tokens, ref_ids, ref_valid, ref_index)
    and returns the predicted velocity `model_pred` for the noise-target tokens.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_kv_heads,
                 depth_double_blocks,
                 depth_single_blocks,
                 in_channels=128,
                 context_in_dim=7680,
                 axes_dim=[8, 40, 40, 40],
                 theta=10000,
                 num_refiner_layers=2,
                 adaln_embed_dim=256,
                 max_ref_images=5,
                 use_gradient_checkpoint=False,
                 attention_backend='sdpa'):
        """
        Mix-stream MMDiT denoise transformer.

        Args
            hidden_size / num_heads: transformer width / heads.
            num_kv_heads: GQA K/V head count. Must be explicitly
                given by the model factory and must divide num_heads (SDPA
                `enable_gqa` requirement).
            depth_double_blocks: number of double-stream layers (each keeps a
                separate img / txt weight set, so ~2x the params of a single
                block).
            depth_single_blocks: number of single-stream layers (one shared
                weight set for the whole joint [txt; img] sequence).
            in_channels: VAE latent channels after 2x2 patchify (flux2:
                z_planes=32 x 2 x 2 = 128). out_channels is tied to it, so
                forward() predicts the velocity in this same space.
            context_in_dim: condition-feature width from the wrapper
                (len(deepstack_layers) * vlm_hidden_size, e.g. 3 * 2560 = 7680
                for Qwen3-VL-4B-Instruct with deepstack_layers=(9, 18, 27)).
            axes_dim: per-axis RoPE dims for the 4D (t, h, w, l) ids; must sum to
                head_dim. The split is deliberately asymmetric: the t axis only
                ever takes max_ref_images+1 discrete values (0 for the noise
                target, ref_time_coord_scale*(j+1) for reference image j), so 8
                dims already separate them, while h / w carry the latent grid
                and l the prompt positions and get 40 dims each.
            theta: RoPE base frequency.
            num_refiner_layers: layers per input-side refiner stack.
            adaln_embed_dim: low-dim AdaLN bottleneck width.
            max_ref_images: max reference images that get a distinct learnable
                identity index embedding; forward() requires
                ref_index.max() < max_ref_images.
            use_gradient_checkpoint: gradient checkpointing for the blocks.
            attention_backend: 'sdpa' (default) or 'flash_varlen'.
        """
        super(MixStreamMMDiT, self).__init__()
        assert hidden_size % num_heads == 0
        assert num_kv_heads >= 1 and num_heads % num_kv_heads == 0
        assert attention_backend in ['sdpa', 'flash_varlen']
        pe_dim = hidden_size // num_heads
        assert sum(axes_dim) == pe_dim

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_refiner_layers = num_refiner_layers
        self.adaln_embed_dim = adaln_embed_dim
        self.max_ref_images = max_ref_images
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.attention_backend = attention_backend

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.ref_image_patch_embedder = nn.Linear(in_channels,
                                                  hidden_size,
                                                  bias=False)
        self.image_index_embedding = nn.Parameter(
            torch.empty(max_ref_images, hidden_size))

        self.txt_in = nn.Sequential(
            RMSNorm(context_in_dim),
            nn.Linear(context_in_dim, hidden_size, bias=False))

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=adaln_embed_dim)

        self.x_pad_token = nn.Parameter(torch.empty(1, hidden_size))
        self.txt_pad_token = nn.Parameter(torch.empty(1, hidden_size))

        refiner_mlp_hidden_dim = swiglu_hidden_dim(hidden_size)
        self.noise_refiner = nn.ModuleList([
            RefinerBlock(hidden_size,
                         num_heads,
                         refiner_mlp_hidden_dim,
                         num_kv_heads=num_kv_heads,
                         modulation=True,
                         adaln_dim=adaln_embed_dim)
            for _ in range(num_refiner_layers)
        ])
        self.ref_image_refiner = nn.ModuleList([
            RefinerBlock(hidden_size,
                         num_heads,
                         refiner_mlp_hidden_dim,
                         num_kv_heads=num_kv_heads,
                         modulation=True,
                         adaln_dim=adaln_embed_dim)
            for _ in range(num_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            RefinerBlock(hidden_size,
                         num_heads,
                         refiner_mlp_hidden_dim,
                         num_kv_heads=num_kv_heads,
                         modulation=False,
                         adaln_dim=adaln_embed_dim)
            for _ in range(num_refiner_layers)
        ])

        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size,
                              num_heads,
                              num_kv_heads=num_kv_heads,
                              adaln_dim=adaln_embed_dim)
            for _ in range(depth_double_blocks)
        ])

        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size,
                              num_heads,
                              num_kv_heads=num_kv_heads,
                              adaln_dim=adaln_embed_dim)
            for _ in range(depth_single_blocks)
        ])

        self.final_layer = LastLayer(hidden_size,
                                     self.out_channels,
                                     adaln_dim=adaln_embed_dim)

        nn.init.normal_(self.image_index_embedding, std=0.02)
        nn.init.normal_(self.x_pad_token, std=0.02)
        nn.init.normal_(self.txt_pad_token, std=0.02)

    def forward(self,
                x_t,
                img_ids,
                timesteps,
                ctx,
                ctx_ids,
                ctx_mask,
                ref_tokens=None,
                ref_ids=None,
                ref_valid=None,
                ref_index=None):
        """
        Denoise one flow-matching step.

        Preconditions: the noise-target segment is assumed FULLY valid, so
        L_img must be identical across the batch (one aspect-ratio bucket per
        batch, noise tokens are never padded); padding is only allowed in the
        ctx / ref segments. The *_ids may come in as float or long (they are
        cast to float32 internally, see the RoPE note below).

        Args (all already-encoded by the wrapper):
            x_t: [B, L_img, in_channels] noised target latent tokens.
            img_ids: [B, L_img, 4] noise-target 4D position ids (t=0, h, w, l=0);
                h / w are the 0-based latent grid and get re-centred on 0
                internally (see `center_image_ids`).
            timesteps: [B] flow-matching timesteps.
            ctx: [B, L_ctx, context_in_dim] VLM condition features.
            ctx_ids: [B, L_ctx, 4] condition position ids.
            ctx_mask: [B, L_ctx] bool, True for valid (non-padding) text tokens.
            ref_tokens: [B, L_ref, in_channels] padded reference latents, or None
                (T2I). ref_ids / ref_valid / ref_index must be passed together
                with it and share L_ref.
            ref_ids: [B, L_ref, 4] reference 4D position ids; h / w are re-centred
                on 0 per reference image the same way as `img_ids`.
            ref_valid: [B, L_ref] bool, True for valid reference tokens.
            ref_index: [B, L_ref] long, per-token reference-image identity index;
                must stay < max_ref_images.

        Returns:
            model_pred: [B, L_img, in_channels] predicted velocity for the
                noise-target tokens only (the txt / ref / padded positions are
                sliced off before the final layer, so nothing is returned for
                them).
        """
        # `ctx` / `x_t` are FEATURE values that feed an nn.Linear, so they are
        # aligned to the RUNTIME WEIGHT dtype (fp32 / bf16 / fp16, whatever the
        # config built the DiT with) rather than to a hardcoded one. The VLM
        # always emits bf16 (the wrapper pins it with a nested bf16 autocast),
        # so without this cast an fp32 DiT would hit a dtype mismatch on the
        # paths that do NOT run under an outer autocast (pure fp32, DeepSpeed
        # bf16). Keeping them in the low precision is also deliberate: these are
        # the big [B, L, C] tensors, and they are normalized (RMSNorm) before the
        # projection, so bf16 range is plenty.
        ctx = ctx.to(self.img_in.weight.dtype)
        x_t = x_t.to(self.img_in.weight.dtype)

        # The *_ids are position INDICES, not features: they never touch a
        # Linear, only `rope()`. So they are pinned to float32
        # UNCONDITIONALLY -- explicitly NOT to the weight dtype above. bf16 has
        # only an 8-bit mantissa, so integers are exact only up to 256: on long
        # prompts / high resolutions token 257 would collapse onto token 256 and
        # the RoPE phase gets corrupted. The `pos.float()` inside `rope()` can
        # NOT repair that, the rounding already happened upstream. Cheap to do
        # here ([B, L, 4] only), and FLUX.2 keeps its ids in float32 as well.
        ctx_ids = ctx_ids.float()
        img_ids = img_ids.float()
        # The h / w axes are consumed CENTRED on 0, so callers keep handing in
        # the plain 0-based latent grid and every resolution shares one
        # coordinate system.
        img_ids = center_image_ids(img_ids)

        B = x_t.shape[0]
        device = x_t.device
        num_img_tokens = x_t.shape[1]
        num_txt_tokens = ctx.shape[1]

        has_ref = ref_tokens is not None
        max_ref_len = ref_tokens.shape[1] if has_ref else 0
        if has_ref:
            # ref_tokens is a value (weight dtype), ref_ids is an index (always float32).
            ref_tokens = ref_tokens.to(self.img_in.weight.dtype)
            ref_ids = ref_ids.float()
            # Every reference image is its own block, so `ref_index` selects the
            # extent each token is centred against.
            ref_ids = center_image_ids(ref_ids, ref_index, self.max_ref_images)

        # ---- embed streams ----
        # noise-target tokens use img_in; reference tokens use the independent ref embedder plus per-image identity index.
        noise_img = self.img_in(x_t)
        txt = self.txt_in(ctx)
        # learnable pad token for padded text positions.
        txt = torch.where(ctx_mask[..., None], txt,
                          self.txt_pad_token.to(txt.dtype))

        if has_ref:
            ref_img = self.ref_image_patch_embedder(ref_tokens)
            ref_img = ref_img + self.image_index_embedding[ref_index]
            ref_img = torch.where(ref_valid[..., None], ref_img,
                                  self.x_pad_token.to(ref_img.dtype))

        # ---- conditioning vectors (low-dim AdaLN) ----
        vec = self.time_in(timestep_embedding(timesteps, 256))

        # zero-timestep conditioning vector for source/reference tokens; only needed when reference images are present.
        # Two per-token source flags are needed because the two stages modulate
        # different sequences: in the double stage the modulation acts on the
        # image stream only, so `is_source_img` spans just the image tokens; in
        # the single stage it acts on the whole [txt; img] sequence, so
        # `is_source_full` spans that instead. Both mark only the ref tokens.
        vec_zero = None
        is_source_img = None
        is_source_full = None
        if has_ref:
            vec_zero = self.time_in(
                timestep_embedding(torch.zeros_like(timesteps), 256))
            is_source_img = torch.zeros(B,
                                        num_img_tokens + max_ref_len,
                                        dtype=torch.bool,
                                        device=device)
            is_source_img[:, num_img_tokens:] = True
            is_source_full = torch.zeros(B,
                                         num_txt_tokens + num_img_tokens +
                                         max_ref_len,
                                         dtype=torch.bool,
                                         device=device)
            is_source_full[:, num_txt_tokens + num_img_tokens:] = True

        # ---- position embeddings (per-segment for refiners, joint for main) ----
        pe_noise = self.pe_embedder(img_ids)
        pe_txt = self.pe_embedder(ctx_ids)

        # ---- attention args ----
        # One `AttentionArgs` per distinct key-padding mask, built HERE and
        # reused by every block below. All the mask bookkeeping (and the few
        # GPU->CPU syncs it needs) therefore happens once per forward instead of
        # once per attention call; `run_attention` then just dispatches.
        # noise segment: every noise-target token is valid -> no mask at all.
        noise_args = AttentionArgs(None, self.attention_backend)
        ctx_args = AttentionArgs(ctx_mask, self.attention_backend)
        ref_args = AttentionArgs(ref_valid,
                                 self.attention_backend) if has_ref else None

        # ---- input-side refiners ----
        # noise refiner: all noise tokens valid -> dense (no-mask) attention.
        for layer in self.noise_refiner:
            noise_img = layer(noise_img,
                              vec,
                              pe_noise,
                              noise_args,
                              attention_backend=self.attention_backend)

        # context refiner: modulation=False, mask out padded text keys.
        for layer in self.context_refiner:
            txt = layer(txt,
                        None,
                        pe_txt,
                        ctx_args,
                        attention_backend=self.attention_backend)

        if has_ref:
            pe_ref = self.pe_embedder(ref_ids)

            # reference/source tokens are modulated with the zero-timestep vector
            # `vec_zero` here (same as in the main blocks), so the refiner and the
            # backbone assume the SAME timestep condition for reference tokens
            # (avoids a refiner/backbone modulation mismatch that would hurt reference fidelity / convergence).
            for layer in self.ref_image_refiner:
                ref_img = layer(ref_img,
                                vec_zero,
                                pe_ref,
                                ref_args,
                                attention_backend=self.attention_backend)

        # ---- assemble image stream and joint sequence ----
        if has_ref:
            img = torch.cat((noise_img, ref_img), dim=1)
            x_input_ids = torch.cat((img_ids, ref_ids), dim=1)
        else:
            img = noise_img
            x_input_ids = img_ids

        ids = torch.cat((ctx_ids, x_input_ids), dim=1)
        pe = self.pe_embedder(ids)

        # joint key-padding mask (text valid + noise all valid + ref valid).
        img_valid = torch.ones(B,
                               img.shape[1],
                               dtype=torch.bool,
                               device=device)
        if has_ref:
            img_valid[:, num_img_tokens:] = ref_valid
        joint_valid = torch.cat((ctx_mask, img_valid), dim=1)
        # One args for every main block: the double and the single stage both
        # attend over this same joint [txt; img] sequence.
        joint_args = AttentionArgs(joint_valid, self.attention_backend)

        # ---- denoise model: double stream stage ----
        for block in self.double_blocks:
            if self.use_gradient_checkpoint:
                img, txt = checkpoint(block,
                                      img,
                                      txt,
                                      vec,
                                      vec_zero,
                                      pe,
                                      is_source_img,
                                      joint_args,
                                      self.attention_backend,
                                      use_reentrant=False)
            else:
                img, txt = block(img, txt, vec, vec_zero, pe, is_source_img,
                                 joint_args, self.attention_backend)

        # ---- denoise model: single stream stage ----
        # joint sequence: text tokens first, then image tokens.
        seq = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            if self.use_gradient_checkpoint:
                seq = checkpoint(block,
                                 seq,
                                 vec,
                                 vec_zero,
                                 pe,
                                 is_source_full,
                                 joint_args,
                                 self.attention_backend,
                                 use_reentrant=False)
            else:
                seq = block(seq, vec, vec_zero, pe, is_source_full, joint_args,
                            self.attention_backend)

        # Only the noise-target tokens are supervised, so slice them out BEFORE
        # the final RMSNorm + linear projection. This avoids running the final
        # layer over the text / reference / padded image tokens (whose outputs
        # are discarded anyway). Correct for both batch=1 (no padding) and
        # batch>1: noise-target tokens always sit right after the text tokens
        # and are always modulated with the real-timestep vector `vec`.
        model_pred = self.final_layer(
            seq[:, num_txt_tokens:num_txt_tokens + num_img_tokens], vec)

        return model_pred


def _mix_stream_mmdit(hidden_size, num_heads, num_kv_heads,
                      depth_double_blocks, depth_single_blocks, **kwargs):
    model = MixStreamMMDiT(hidden_size=hidden_size,
                           num_heads=num_heads,
                           num_kv_heads=num_kv_heads,
                           depth_double_blocks=depth_double_blocks,
                           depth_single_blocks=depth_single_blocks,
                           **kwargs)

    return model


def MixStreamMMDiT_1B(**kwargs):
    return _mix_stream_mmdit(hidden_size=1536,
                             num_heads=12,
                             num_kv_heads=6,
                             depth_double_blocks=7,
                             depth_single_blocks=14,
                             **kwargs)


def MixStreamMMDiT_2B(**kwargs):
    return _mix_stream_mmdit(hidden_size=2048,
                             num_heads=16,
                             num_kv_heads=8,
                             depth_double_blocks=9,
                             depth_single_blocks=18,
                             **kwargs)


def MixStreamMMDiT_4B(**kwargs):
    return _mix_stream_mmdit(hidden_size=2560,
                             num_heads=20,
                             num_kv_heads=10,
                             depth_double_blocks=11,
                             depth_single_blocks=24,
                             **kwargs)


def MixStreamMMDiT_6B(**kwargs):
    return _mix_stream_mmdit(hidden_size=3072,
                             num_heads=24,
                             num_kv_heads=12,
                             depth_double_blocks=12,
                             depth_single_blocks=26,
                             **kwargs)


def MixStreamMMDiT_8B(**kwargs):
    return _mix_stream_mmdit(hidden_size=3072,
                             num_heads=24,
                             num_kv_heads=12,
                             depth_double_blocks=17,
                             depth_single_blocks=34,
                             **kwargs)


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def build_image_ids(batch_size, latent_height, latent_width, t_coord=0.0):
        """Build the 4D (t, h, w, l) position ids of one image latent grid.

        Mirrors `encode_image_latent` in the wrapper: the noise target uses
        t_coord=0, reference image j uses t_coord=ref_time_coord_scale*(j+1),
        and the l axis stays 0 for every image token.
        """
        image_ids = torch.zeros(latent_height, latent_width, 4)
        image_ids[..., 0] = t_coord
        image_ids[..., 1] = torch.arange(latent_height)[:, None]
        image_ids[..., 2] = torch.arange(latent_width)[None, :]
        image_ids = rearrange(image_ids, "h w c -> (h w) c")

        return image_ids[None].repeat(batch_size, 1, 1)

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    # flux2 AE geometry (SimpleGeneration/flux_autoencoder/models/flux2_autoencoder.py):
    #   encoder conv downsample 8x  (planes_mult=[1, 2, 4, 4] -> 2^3 stages)
    #   x 2x2 patchify inside encode()
    #   = 16x total, and z_planes(32) * 2 * 2 = 128 latent channels.
    batch_size = 1
    ae_z_planes = 32
    ae_patch_size = 2
    ae_conv_downsample_ratio = 8
    ae_downsample_ratio = ae_conv_downsample_ratio * ae_patch_size
    in_channels = ae_z_planes * ae_patch_size * ae_patch_size
    context_in_dim = 7680
    # reference image j gets the temporal RoPE coordinate scale * (j + 1)
    ref_time_coord_scale = 10
    text_seq_len = 32

    print(f"batch_size: {batch_size}")
    print(f"ae_z_planes: {ae_z_planes}")
    print(f"ae_patch_size: {ae_patch_size}")
    print(f"ae_conv_downsample_ratio: {ae_conv_downsample_ratio}")
    print(f"ae_downsample_ratio: {ae_downsample_ratio}")
    print(f"in_channels: {in_channels}")
    print(f"context_in_dim: {context_in_dim}")
    print(f"ref_time_coord_scale: {ref_time_coord_scale}")
    print(f"text_seq_len: {text_seq_len}")

    model = MixStreamMMDiT_1B(in_channels=in_channels,
                              context_in_dim=context_in_dim,
                              use_gradient_checkpoint=False)
    model = model.cuda()
    model.eval()
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    input_resolution_list = [[256, 256], [512, 512], [1024, 1024]]
    input_reference_image_flag_list = [False, True]
    for image_height, image_width in input_resolution_list:
        for use_reference_image in input_reference_image_flag_list:
            task_name = 'TI2I' if use_reference_image else 'T2I'
            print(
                f"{task_name}, image_height: {image_height}, image_width: {image_width}"
            )

            # ---- image resolution -> AE latent grid -> noise-target tokens ----
            assert image_height % ae_downsample_ratio == 0
            assert image_width % ae_downsample_ratio == 0
            latent_height = image_height // ae_downsample_ratio
            latent_width = image_width // ae_downsample_ratio
            img_seq_len = latent_height * latent_width
            print(
                f"{task_name}, latent_height: {latent_height}, latent_width: {latent_width}, img_seq_len: {img_seq_len}"
            )

            # x0 stands in for AutoEncoder.encode(target_image) flattened to [B, latent_height*latent_width, in_channels].
            x0 = torch.randn(batch_size, img_seq_len, in_channels).cuda()
            img_ids = build_image_ids(batch_size,
                                      latent_height,
                                      latent_width,
                                      t_coord=0.0).cuda()
            print(f"{task_name}, x0 shape: {tuple(x0.shape)}")
            print(f"{task_name}, img_ids shape: {tuple(img_ids.shape)}")

            # ---- rectified-flow interpolation----
            noise = torch.randn_like(x0)
            timesteps = torch.rand(batch_size).cuda()
            t = timesteps[:, None, None]
            x_t = (1.0 - t) * x0 + t * noise
            target = noise - x0
            print(f"timesteps shape: {tuple(timesteps.shape)}")
            print(f"x_t shape: {tuple(x_t.shape)}")
            print(f"target shape: {tuple(target.shape)}")

            # ---- condition feature: stands in for the VLM deepstack concat ----
            # ctx_ids only advance along the l axis (t = h = w = 0).
            ctx = torch.randn(batch_size, text_seq_len, context_in_dim).cuda()
            ctx_ids = torch.zeros(batch_size, text_seq_len, 4).cuda()
            ctx_ids[..., 3] = torch.arange(text_seq_len).cuda()[None, :]
            ctx_mask = torch.ones(batch_size, text_seq_len,
                                  dtype=torch.bool).cuda()
            print(f"ctx shape: {tuple(ctx.shape)}")
            print(f"ctx_ids shape: {tuple(ctx_ids.shape)}")
            print(f"ctx_mask shape: {tuple(ctx_mask.shape)}")

            # ---- reference image latents ----
            if use_reference_image:
                # One reference image at the same resolution as the target.
                reference_image_height, reference_image_width = image_height, image_width
                ref_latent_height = reference_image_height // ae_downsample_ratio
                ref_latent_width = reference_image_width // ae_downsample_ratio
                ref_seq_len = ref_latent_height * ref_latent_width
                # reference image index j = 0
                ref_time_coord = float(ref_time_coord_scale * (0 + 1))

                ref_tokens = torch.randn(batch_size, ref_seq_len,
                                         in_channels).cuda()
                ref_ids = build_image_ids(batch_size,
                                          ref_latent_height,
                                          ref_latent_width,
                                          t_coord=ref_time_coord).cuda()
                ref_valid = torch.ones(batch_size,
                                       ref_seq_len,
                                       dtype=torch.bool).cuda()
                ref_index = torch.zeros(batch_size,
                                        ref_seq_len,
                                        dtype=torch.long).cuda()
                print(
                    f"ref_latent_height: {ref_latent_height}, ref_latent_width: {ref_latent_width}"
                )
                print(f"ref_seq_len: {ref_seq_len}")
                print(f"ref_time_coord: {ref_time_coord}")
                print(f"ref_tokens shape: {tuple(ref_tokens.shape)}")
                print(f"ref_ids shape: {tuple(ref_ids.shape)}")
                print(f"ref_valid shape: {tuple(ref_valid.shape)}")
                print(f"ref_index shape: {tuple(ref_index.shape)}")

                assert ref_ids[0, 0, 0].item() == ref_time_coord
            else:
                ref_tokens = None
                ref_ids = None
                ref_valid = None
                ref_index = None

            with torch.no_grad():
                model_pred = model(x_t,
                                   img_ids,
                                   timesteps,
                                   ctx,
                                   ctx_ids,
                                   ctx_mask,
                                   ref_tokens=ref_tokens,
                                   ref_ids=ref_ids,
                                   ref_valid=ref_valid,
                                   ref_index=ref_index)

            print(f"model_pred shape: {tuple(model_pred.shape)}")
            assert model_pred.shape == (batch_size, img_seq_len, in_channels)
            assert model_pred.shape == target.shape
            ########################################################################################################
            ########################################################################################################
            ########################################################################################################

    del model
    torch.cuda.empty_cache()
