"""vLLM 0.29.0 FLASH_ATTN / FLASH_ATTN_DIFFKV inference compatibility.

This module covers tensor contracts; it does not register a vLLM platform or
execute CUDA/CuTE callbacks. The two masks used by these backends have explicit
Metal descriptors below. Distributed transport remains the engine's job.
"""

from .interface import MaskSpec, _inference_only, _run


def mm_prefix_mask(sliding_window_left=None, clamp_prefix_to_window=False):
    """Use aux_tensors=[absolute_query_ranges[T,2], cu_seqlens_q[B+1]]."""
    window = -1 if sliding_window_left is None else sliding_window_left
    if window == 0 or window < -1:
        raise ValueError("sliding_window_left must be positive or None")
    if clamp_prefix_to_window and window == -1:
        raise ValueError("clamping the prefix requires a sliding window")
    return MaskSpec(1, window, clamp_prefix_to_window)


def rswa_mask():
    """Use aux_tensors=[prefix_lens[B] int32, window[1] int32]."""
    return MaskSpec(2)


def get_scheduler_metadata(*args, **kwargs):
    """CUDA scheduling metadata has no mathematical meaning on Metal."""
    return None


def flash_attn_varlen_func(
    q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
    cu_seqlens_k=None, seqused_k=None, q_v=None, dropout_p=0.0,
    softmax_scale=None, causal=False, window_size=None, softcap=0.0,
    alibi_slopes=None, deterministic=False, return_attn_probs=False,
    block_table=None, return_softmax_lse=False, out=None, scheduler_metadata=None,
    q_descale=None, k_descale=None, v_descale=None, num_splits=0,
    output_scale=None, fa_version=2, s_aux=None, cp_world_size=1,
    cp_rank=0, cp_tot_seqused_k=None, mask_mod=None, block_sparse_tensors=None,
    aux_tensors=None, aux_tensor_leading_dims=None, dynamic_causal=None,
):
    """Run the stable vLLM dense/paged inference contract on Metal.

    fa_version and scheduler_metadata are accepted compatibility hints. Every
    version selects Metal kernels. num_splits=1 selects a fixed per-query
    reduction schedule, including when decode requests are mixed with prefill.
    """
    _inference_only(dropout_p, return_attn_probs, q, k, v)
    if fa_version not in (2, 3, 4):
        raise ValueError("fa_version must be 2, 3, or 4")
    if (q_v is not None or cp_world_size != 1 or cp_rank != 0
            or cp_tot_seqused_k is not None or block_sparse_tensors is not None):
        raise NotImplementedError("MLA and separate sparse backends are outside the FLASH_ATTN contract")
    if output_scale is not None:
        raise NotImplementedError("vLLM FLASH_ATTN does not support fused output quantization")
    if aux_tensor_leading_dims is not None:
        raise NotImplementedError("custom CuTE auxiliary layouts are not supported")
    if block_table is not None and (k.ndim != 4 or k.shape[1] % 16):
        raise ValueError("paged KV requires a page size divisible by 16")
    if block_table is not None and seqused_k is None:
        raise ValueError("vLLM paged KV requires seqused_k")
    result, lse = _run(
        q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        block_table=block_table, softmax_scale=softmax_scale,
        causal=causal if dynamic_causal is None else dynamic_causal,
        window_size=(-1, -1) if window_size is None else tuple(window_size),
        alibi_slopes=alibi_slopes, softcap=softcap, s_aux=s_aux,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        out=out, num_splits=1 if deterministic else num_splits,
        mask_mod=mask_mod, aux_tensors=aux_tensors,
        deterministic=deterministic or num_splits == 1,
    )
    return (result, lse) if return_softmax_lse else result


def merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                      output_lse=None):
    """Merge two KV partitions without requiring a distributed runtime."""
    from ._merge import merge_attn_states as merge
    return merge(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                 out_lse=output_lse)
