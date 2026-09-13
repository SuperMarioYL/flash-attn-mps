# Inference contracts

The public interface follows FlashAttention 2's inference layouts. The separate
`flash_attn_mps.vllm` module follows vLLM **v0.29.0**, commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`, for `FLASH_ATTN` and
`FLASH_ATTN_DIFFKV`. Contract compatibility does not register an MPS platform
with vLLM or make its CUDA runtime executable on a Mac.

## Tensor layouts

| Entry point | Q | K/V | Output |
| --- | --- | --- | --- |
| `flash_attn_func` | `[B,Sq,Hq,Dqk]` | `[B,Sk,Hkv,Dqk/Dv]` | `[B,Sq,Hq,Dv]` |
| `flash_attn_varlen_func` | `[Tq,Hq,Dqk]` | packed `[Tk,Hkv,Dqk/Dv]` or paged `[Npages,P,Hkv,Dqk/Dv]` | `[Tq,Hq,Dv]` |
| `flash_attn_with_kvcache` | `[B,Sq,Hq,Dqk]` | continuous `[Bcache,capacity,Hkv,Dqk/Dv]` or paged cache | `[B,Sq,Hq,Dv]` |
| `store_kvcache` | — | new `[T,Hkv,Dqk/Dv]`, cache `[Npages,P,Hkv,Dqk/Dv]` | in-place writes |

`Hq` is divisible by `Hkv`; each query head maps directly to its KV head.
QK and V dimensions may differ and lie in 1–512. FP16, BF16 and FP32 compute
uses FP32 softmax/LSE accumulation. FP8 E4M3FN is storage interpreted by native
Metal code; vLLM descales have shape `[B,Hkv]` and may have zero strides.
FP8 Q defaults to FP16 output; vLLM's explicit `out` can select FP16/BF16/FP32.

The packed helpers accept `[B,S,3,H,D]` / `[T,3,H,D]` QKV and
`[B,S,2,Hkv,D]` / `[T,2,Hkv,D]` KV, respectively.

All inputs and sequence metadata are MPS tensors. Cumulative sequence lengths
and block tables are int32; cache scatter accepts int32 or int64 slots.
Strided views and nonzero storage offsets are supported without collecting the
historical cache into another tensor. Slot `-1` does not write a cache entry.

## Causality and cache ownership

With `Sq` new queries and `Sk` total keys, query row `i` has position
`i + Sk - Sq`; causal attention permits key `j <= i + Sk - Sq`. Fully masked
rows return zero output and negative-infinity LSE. Sliding-window bounds use
the same absolute positions.

The library owns kernel execution, not page allocation or prefix hashes.
`store_kvcache` writes slots before attention. When new K/V are supplied to
`flash_attn_with_kvcache`, `cache_seqlens` instead describes the exclusive end
**before** appending. The function does not modify the lengths tensor.
For continuous cache, `cache_leftpad` excludes leading entries; lengths still
include that physical padding, and appends occur at the old exclusive end.

RoPE is optional on the append interface and supports interleaved and split-half
layouts. An engine that already rotates Q/K must not enable it again. In
nano-vLLM, Q/K are already rotated and `context_lens` includes the newly stored
token, so the integration performs scatter then attention without append.

## vLLM semantics

The vLLM entry point preserves the stable backend's argument order. Packed KV
uses `cu_seqlens_k`; paged KV uses `seqused_k` and `block_table`. It supports
mixed query lengths, dynamic per-sequence causal flags, windows, ALiBi,
softcap, sinks, FP8 descales, and caller-owned strided output buffers.

`return_softmax_lse=True` returns natural-log FP32 LSE in `[Hq,Tq]` layout.
The dense/cache public entry points return `[B,Hq,Sq]` LSE. Split-KV and
`merge_attn_states` combine attention outputs using stable log-sum-exp; an empty
partition contributes zero probability. A sink must be counted exactly once
when the caller divides one logical attention operation into partitions.

The two mask patterns used by the pinned backend have explicit descriptors:

- `mm_prefix_mask(...)` with `aux_tensors=[query_ranges,cu_seqlens_q]`: absolute,
  inclusive query ranges `[Tq,2]`; `[-1,-1]` means no bidirectional range.
- `rswa_mask()` with `aux_tensors=[prefix_lens,window]`: an always-visible prefix
  plus a causal rolling window, with int32 shapes `[B]` and `[1]`.

These replace the corresponding CuTE factories at the compatibility boundary;
arbitrary CUDA/CuTE callbacks are not interpreted. The same tensor metadata and
mask behavior are exercised by the MPS contract tests.

`fa_version=2/3/4` is an interface compatibility hint; all versions dispatch to
Metal. `get_scheduler_metadata` returns `None`, since CUDA schedules have no
mathematical meaning here. Explicit vLLM `num_splits=1` selects the fixed
batch-invariant calculation path. Larger values perform real split-KV work;
the default allows the native dispatcher to choose its split count.

## Explicit exclusions

Backward, nonzero dropout, and `return_attn_probs=True` are rejected; inference
LSE is available without requesting a full probability matrix. Standalone MLA,
block-sparse attention, CUDA Graph/AOT machinery, arbitrary CuTE callbacks,
distributed transport, and fused output quantization are outside this contract.
The pinned vLLM main/DiffKV backends themselves reject fused output quantization.
