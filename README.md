# flash-attn-mps

Native Metal FlashAttention inference for PyTorch on Apple Silicon.

The library implements the inference tensor contracts used by vLLM v0.29.0's
`FLASH_ATTN` and `FLASH_ATTN_DIFFKV` backends, with nano-vLLM as the model-level
integration target. It runs native Metal kernels through PyTorch's MPS stream.
There is no CPU/SDPA attention fallback, Python loop over requests, or full
attention-score matrix. Paged kernels read cache pages directly.

On the tested M1 Max, 104 native tests and the nano-vLLM model scenarios pass.
The complete-call release matrix measures 1.19–4.33x speedups over the previous
MPS SDPA path; Qwen3-0.6B generation measures about 2.66–2.68x. These figures
apply to the recorded shapes and environment, not all Apple GPUs or workloads.
See [validation and raw results](docs/validation.md).

The unreleased `main` branch adds specialized paged decode and
improves DiffKV, windowed attention and softcap. See the
[same-machine comparison and optimization report](docs/performance-optimization.md)
for comparisons with SDPA, HF Metal and mtlflashattn, including the remaining
FP32 decode gap and complete Qwen3-0.6B measurements. The published v0.1.0
artifacts retain their original implementation.

## Runtime

- Apple Silicon with an available PyTorch MPS device
- Python 3.10 or newer and PyTorch 2.14 or newer
- Metal shaders compile through `torch.mps.compile_shader` on first use;
  installing the Python package does not require a C++ compiler or full Xcode.

The package name is `flash-attn-mps`; the import is `flash_attn_mps`.
It does not replace the CUDA `flash_attn` package or patch PyTorch globally.

## Use

Install the fixed GitHub release (the package is not published to PyPI):

```bash
python -m pip install 'flash-attn-mps @ https://github.com/SuperMarioYL/flash-attn-mps/releases/download/v0.1.0/flash_attn_mps-0.1.0-py3-none-any.whl'
```

```python
import torch
from flash_attn_mps import flash_attn_func

q = torch.randn(2, 128, 16, 128, device="mps", dtype=torch.float16)
k = torch.randn(2, 128, 8, 128, device="mps", dtype=torch.float16)
v = torch.randn_like(k)
output = flash_attn_func(q, k, v, causal=True)  # [2, 128, 16, 128]
```

The public interface includes dense, QKV/KV-packed, and variable-length
attention; `flash_attn_with_kvcache`; `store_kvcache`; and stable attention/LSE
merging. It supports GQA/MQA, different QK/V dimensions, causal and local
attention, ALiBi, softcap, KV append, and RoPE.

The `flash_attn_mps.vllm` entry point additionally exposes the pinned vLLM
parameter layout, FP8 E4M3 descales, sinks, dynamic causal batches, multimodal
prefix and rolling-window masks, caller-owned output, and split-KV/LSE results.
See [the exact contracts](docs/contracts.md) for layouts and cache ownership.

This is an inference library. Backward, nonzero dropout, full probability-matrix
returns, arbitrary CuTE callbacks, and a complete vLLM MPS platform are outside
its contract and are not silently emulated.

## Validate and benchmark

From an Apple Silicon checkout with an available MPS device:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python benchmarks/bench_attention.py --output .artifacts/attention.json
```

GPU tests are skipped on machines without MPS, so a hosted packaging CI pass is
not a GPU correctness result. The release validation must run on real MPS
hardware with CPU fallback disabled. The benchmark includes cache writes,
metadata, and synchronization for both implementations, reports all rounds,
and does not count a diagnostic `--quick` run as the release gate.
The wheel also installs these tests, benchmarks, and validation data under
`sys.prefix/share/flash-attn-mps`, so the published artifact can be verified
without importing code from a source checkout.

To validate the integrated nano-vLLM checkout:

```bash
python benchmarks/validate_nanovllm.py --model /path/to/Qwen3-0.6B \
  --output .artifacts/nano-correctness.json
python benchmarks/validate_nanovllm.py --model /path/to/Qwen3-0.6B \
  --output .artifacts/nano-performance.json --performance
```

The correctness harness checks teacher-forced logits and greedy choices,
including prefix reuse, chunking, cross-page decode, and preemption. The
performance mode omits logit comparison and reports real generation timings.
Kernel tests keep their fixed pointwise dtype tolerances. Model acceptance
requires finite logits, identical greedy choices, and per-request/per-step
`KL(P_SDPA || P_native) <= 1e-3` nats at temperature 1.0. Raw logit errors are
also reported; they are not expected to be bitwise identical through 28 FP16
layers. `--fp64-reference` optionally records an independent FP64-attention
trajectory without claiming that the entire model runs in FP64.

## License and sources

Apache-2.0. Portions of the tiled attention implementation derive from
Hugging Face's `metal-flash-sdpa` and Apple's MLX/STEEL kernels. The exact
source revisions and retained notices are recorded in `NOTICE` and `LICENSES/`.
