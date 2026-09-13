"""Inference-only FlashAttention interfaces for PyTorch MPS tensors.

The public tensor layouts match FlashAttention 2.  Scheduling, cache ownership,
and rotary position selection remain explicit; all attention math runs in Metal.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskSpec:
    """A supported vLLM mask semantic, not a Python callback to execute."""

    kind: int
    sliding_window_left: int = -1
    clamp_prefix_to_window: bool = False


def _reshape(x, *shape):
    # PyTorch MPS can store FP8 but cannot copy/reorder it as a numeric dtype.
    # Reshaping its bytes preserves exact FP8 payloads even when a copy is needed.
    if x.dtype == torch.float8_e4m3fn:
        return x.view(torch.uint8).reshape(*shape).view(x.dtype)
    return x.reshape(*shape)


def _inference_only(dropout_p, return_attn_probs, *tensors):
    if dropout_p != 0:
        raise NotImplementedError("flash-attn-mps implements inference; dropout_p must be 0")
    if return_attn_probs:
        raise NotImplementedError(
            "return_attn_probs is a training/debug interface; use return_softmax_lse=True"
        )
    if any(t is not None and t.requires_grad for t in tensors):
        raise NotImplementedError("flash-attn-mps implements inference without backward")


def _validate(q, k, v, cu_q, cu_k, used_k, block_table, out=None):
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.device.type != "mps":
            raise ValueError(f"{name} must be an MPS tensor")
    if q.ndim != 3 or k.ndim not in (3, 4) or v.ndim != k.ndim:
        raise ValueError("expected q [T,H,D] and matching packed or paged K/V")
    if k.shape[:-1] != v.shape[:-1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K head dimensions and K/V token/head dimensions must match")
    if k.shape[-2] == 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("query heads must be divisible by KV heads")
    if not (1 <= q.shape[-1] <= 512 and 1 <= v.shape[-1] <= 512):
        raise ValueError("head dimensions must be between 1 and 512")
    if cu_q.ndim != 1 or cu_q.numel() < 1 or cu_q.dtype != torch.int32:
        raise ValueError("cu_seqlens_q must be int32 [batch+1]")
    batch = cu_q.numel() - 1
    if (cu_k is None) == (used_k is None):
        raise ValueError("provide exactly one of cu_seqlens_k and seqused_k")
    for name, tensor, shape in (
        ("cu_seqlens_q", cu_q, (batch + 1,)),
        ("cu_seqlens_k", cu_k, (batch + 1,)),
        ("seqused_k", used_k, (batch,)),
    ):
        if tensor is not None and (
            tensor.device != q.device or tensor.dtype != torch.int32
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{name} must be an MPS int32 tensor of shape {shape}")
    if block_table is not None:
        if (k.ndim != 4 or block_table.ndim != 2
                or block_table.shape[0] != batch or block_table.dtype != torch.int32
                or block_table.device != q.device):
            raise ValueError("paged KV requires an MPS int32 block_table [B,P]")
    elif k.ndim != 3:
        raise ValueError("4D KV requires block_table")
    if out is not None and (
        tuple(out.shape) != (q.shape[0], q.shape[1], v.shape[-1])
        or out.device != q.device
    ):
        raise ValueError("out must have shape [total_q, query_heads, value_head_dim] on MPS")


def _run(q, k, v, *, cu_seqlens_q, cu_seqlens_k=None, seqused_k=None,
         max_seqlen_q, max_seqlen_k, block_table=None, softmax_scale=None,
         causal=False, window_size=(-1, -1), alibi_slopes=None, softcap=0.0,
         s_aux=None, q_descale=None, k_descale=None, v_descale=None, out=None,
         num_splits=0, mask_mod=None, aux_tensors=None, leftpad_k=None,
         deterministic=False):
    _inference_only(0, False, q, k, v, alibi_slopes, s_aux,
                    q_descale, k_descale, v_descale, out)
    _validate(q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k, block_table, out)
    if len(window_size) != 2 or min(window_size) < -1:
        raise ValueError("window_size must contain two bounds >= -1")
    if softcap < 0 or num_splits < 0 or max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("softcap, lengths, and num_splits must be nonnegative")
    if mask_mod is not None and not isinstance(mask_mod, MaskSpec):
        raise TypeError("mask_mod must be an mm_prefix_mask() or rswa_mask() descriptor")
    batch, heads, kvheads = cu_seqlens_q.numel()-1, q.shape[1], k.shape[-2]
    if torch.is_tensor(causal) and (
        causal.shape != (batch,) or causal.dtype != torch.bool or causal.device != q.device
    ):
        raise ValueError("dynamic causal must be an MPS bool [B] tensor")
    for name, tensor in (("q_descale", q_descale), ("k_descale", k_descale),
                         ("v_descale", v_descale)):
        if tensor is not None and (
            tensor.shape != (batch, kvheads) or tensor.dtype != torch.float32
            or tensor.device != q.device
        ):
            raise ValueError(f"{name} must be an MPS float32 [B,Hkv] tensor")
    if alibi_slopes is not None and (
        tuple(alibi_slopes.shape) not in ((heads,), (batch, heads))
        or alibi_slopes.device != q.device or alibi_slopes.dtype != torch.float32
    ):
        raise ValueError("alibi_slopes must be MPS float32 [Hq] or [B,Hq]")
    if s_aux is not None and (s_aux.shape != (heads,) or s_aux.device != q.device):
        raise ValueError("s_aux must be an MPS [Hq] tensor")
    if mask_mod is not None:
        if mask_mod.kind not in (1, 2) or aux_tensors is None or len(aux_tensors) != 2:
            raise ValueError("a supported mask requires its two auxiliary tensors")
        shapes = ((q.shape[0], 2), (batch+1,)) if mask_mod.kind == 1 else ((batch,), (1,))
        for tensor, shape in zip(aux_tensors, shapes):
            if tensor.shape != shape or tensor.dtype != torch.int32 or tensor.device != q.device:
                raise ValueError(f"mask auxiliary tensor must be MPS int32 {shape}")
    elif aux_tensors is not None:
        raise ValueError("aux_tensors require a mask descriptor")
    from ._attention import attention

    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k, max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k, block_table=block_table,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        alibi_slopes=alibi_slopes, softcap=softcap, s_aux=s_aux,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        out=out, num_splits=num_splits, mask_mod=mask_mod, aux_tensors=aux_tensors,
    )
    if leftpad_k is not None:
        kwargs["leftpad_k"] = leftpad_k
    if deterministic:
        kwargs["deterministic"] = True
    return attention(q, k, v, **kwargs)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False,
    block_table=None, *, return_softmax_lse=False, out=None,
):
    """Packed or paged variable-length attention with lower-right causality."""
    _inference_only(dropout_p, return_attn_probs, q, k, v)
    if block_table is not None:
        if k.ndim != 4 or k.shape[1] % 16:
            raise ValueError("paged KV requires a page size divisible by 16")
    result, lse = _run(
        q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        block_table=block_table, softmax_scale=softmax_scale, causal=causal,
        window_size=window_size, softcap=softcap, alibi_slopes=alibi_slopes,
        num_splits=1 if deterministic else 0, out=out, deterministic=deterministic,
    )
    return (result, lse) if return_softmax_lse else result


def flash_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
    deterministic=False, return_attn_probs=False, *, return_softmax_lse=False,
):
    """Dense attention: Q [B,Sq,Hq,D], K/V [B,Sk,Hkv,D]."""
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[0] != k.shape[0]
            or k.shape[:3] != v.shape[:3]):
        raise ValueError("dense attention requires matching 4D batches")
    batch, sq = q.shape[:2]
    sk = k.shape[1]
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sq
    cu_k = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sk
    out, lse = flash_attn_varlen_func(
        _reshape(q, -1, *q.shape[2:]), _reshape(k, -1, *k.shape[2:]),
        _reshape(v, -1, *v.shape[2:]), cu_q, cu_k, sq, sk,
        dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
        deterministic, return_attn_probs, return_softmax_lse=True,
    )
    out = out.reshape(batch, sq, q.shape[2], v.shape[-1])
    lse = lse.reshape(q.shape[2], batch, sq).transpose(0, 1)
    return (out, lse) if return_softmax_lse else out


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    """Dense QKV [B,S,3,H,D]."""
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError("qkv must have shape [B,S,3,H,D]")
    return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], *args, **kwargs)


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    """Dense Q [B,Sq,Hq,D], KV [B,Sk,2,Hkv,D]."""
    if kv.ndim != 5 or kv.shape[2] != 2:
        raise ValueError("kv must have shape [B,S,2,H,D]")
    return flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], *args, **kwargs)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *args, **kwargs):
    """Packed QKV [T,3,H,D]."""
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError("qkv must have shape [T,3,H,D]")
    return flash_attn_varlen_func(
        qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen, *args, **kwargs,
    )


def flash_attn_varlen_kvpacked_func(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs,
):
    """Packed KV [T,2,Hkv,D]."""
    if kv.ndim != 4 or kv.shape[1] != 2:
        raise ValueError("kv must have shape [T,2,H,D]")
    return flash_attn_varlen_func(
        q, kv[:, 0], kv[:, 1], cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, *args, **kwargs,
    )


def flash_attn_with_kvcache(
    q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None,
    cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None, block_table=None,
    softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
    rotary_interleaved=True, alibi_slopes=None, num_splits=0,
    return_softmax_lse=False,
):
    """Attention over a cache, optionally appending KV at the supplied old lengths.

    Cache lengths are not mutated. They specify the physical exclusive end,
    before append when k/v are provided. cache_leftpad excludes an initial
    portion of a dense cache from attention without changing append positions.
    """
    _inference_only(0, False, q, k, v, k_cache, v_cache)
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q and KV caches must be 4D")
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together")
    if block_table is not None and cache_leftpad is not None:
        raise ValueError("FlashAttention does not combine paged cache with cache_leftpad")
    if block_table is not None and k_cache.shape[1] % 16:
        raise ValueError("paged KV requires a page size divisible by 16")
    batch, sq = q.shape[:2]
    if cache_seqlens is None:
        if k is not None or block_table is not None:
            raise ValueError("cache_seqlens is required for append or paged cache")
        cache_seqlens = k_cache.shape[1]
    if isinstance(cache_seqlens, int):
        lengths = torch.full((batch,), cache_seqlens, device=q.device, dtype=torch.int32)
    else:
        lengths = cache_seqlens
    if (tuple(lengths.shape) != (batch,) or lengths.dtype != torch.int32
            or lengths.device != q.device):
        raise ValueError("cache_seqlens must be an int or MPS int32 [B]")
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be provided together")
    if rotary_cos is not None:
        if k is None:
            raise ValueError("rotary embedding requires new k and v")
        from ._cache import rotary
        q, k = rotary(
            q, k, rotary_cos, rotary_sin, lengths,
            rotary_interleaved=rotary_interleaved, causal=causal, window_size=window_size,
        )
    if k is not None:
        if k.ndim != 4 or v.ndim != 4 or k.shape[0] != batch or v.shape[:3] != k.shape[:3]:
            raise ValueError("new k/v must have matching [B,Snew,Hkv,D] dimensions")
        from ._cache import append_kvcache
        append_kvcache(
            k, v, k_cache, v_cache, lengths, cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad, block_table=block_table,
        )
        lengths = lengths + k.shape[1]
    if block_table is None:
        indices = cache_batch_idx
        if indices is None:
            indices = torch.arange(batch, device=q.device, dtype=torch.int32)
        table = indices.to(torch.int32).reshape(batch, 1)
        max_k = k_cache.shape[1]
    else:
        # A page table already selects physical storage; its rows correspond
        # to the query batch even when cache_batch_idx was also supplied.
        table = block_table
        max_k = table.shape[1] * k_cache.shape[1]
    if cache_leftpad is not None:
        lengths = lengths - cache_leftpad
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sq
    out, lse = _run(
        _reshape(q, -1, *q.shape[2:]), k_cache, v_cache,
        cu_seqlens_q=cu_q, seqused_k=lengths, max_seqlen_q=sq,
        max_seqlen_k=max_k, block_table=table, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, alibi_slopes=alibi_slopes,
        softcap=softcap, num_splits=num_splits, leftpad_k=cache_leftpad,
    )
    out = out.reshape(batch, sq, q.shape[2], v_cache.shape[-1])
    lse = lse.reshape(q.shape[2], batch, sq).transpose(0, 1)
    return (out, lse) if return_softmax_lse else out
