# Validation — v0.1.0

Measured on 2026-09-13 using Apple M1 Max (64 GB), macOS 26.6.2, Python 3.12.13, PyTorch 2.14.0 and Transformers 4.57.6. CPU fallback was disabled for native validation.

Native implementation commit: `6869079432b637508cc1eace5273a5f19ce15264`. vLLM tensor-contract baseline: `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. nano-vLLM base: `bb823b3e06983d71485a8e1f23715ebd87d98ef8` plus the local MPS integration; no nano-vLLM changes were pushed.

## Native correctness

**104 tests passed, zero skips, failures or errors.** This includes 20 core attention tests, 45 public API/vLLM contract tests and 39 cache/merge tests. [JUnit results](results/native-tests.xml)

The suite covers FP16/BF16/FP32/FP8 E4M3, odd and large dimensions through 512, DiffKV, strided buffers, ragged/paged attention, masks, scale broadcasting, sinks, split-KV/LSE merging, cache writes, append, RoPE and batch invariance. CPU FP64 references use fixed dtype tolerances. It verifies the vLLM main/DiffKV tensor contract, not a complete vLLM MPS engine.

## Complete-call performance

Both paths include slot writes and metadata handling. Each case ran 10 warmups and 50 measured calls per implementation in each of three rounds; implementation order alternates between rounds. Speedup below is the median of the three per-round p50 ratios. The gate requires at least 1.1x median speedup and a speedup greater than 1.0x in every round. [All timings, errors and resident-memory counters](results/attention.json)

| Case | Speedup | Three per-round speedups |
| --- | ---: | --- |
| prefill-b1-s2048 | 1.95x | 1.94x, 1.95x, 1.99x |
| prefill-b1-s4096 | 1.25x | 1.25x, 1.28x, 1.22x |
| prefill-b4-s2048 | 1.63x | 1.62x, 1.63x, 1.64x |
| prefill-b4-s4096 | 1.19x | 1.06x, 1.21x, 1.19x |
| decode-b1-k4096 | 3.25x | 3.25x, 2.75x, 3.84x |
| decode-b1-k8192 | 2.52x | 2.52x, 1.98x, 3.12x |
| decode-b4-k4096 | 4.33x | 4.02x, 4.38x, 4.33x |
| decode-b4-k8192 | 3.41x | 3.69x, 3.41x, 3.38x |
| decode-b16-k4096 | 4.01x | 3.90x, 4.12x, 4.01x |
| decode-b16-k8192 | 3.59x | 3.51x, 3.59x, 3.62x |
| short-prefill | 5.92x | 4.98x, 6.39x, 5.92x |
| short-decode | 5.29x | 5.17x, 5.29x, 5.30x |
| cached-prefill | 7.45x | 7.45x, 7.39x, 7.63x |

**All 10 release-gate cases passed.** The other three cases disclose short-sequence and cached-prefill results. The first-call column includes process-local pipeline setup; it is not a forced OS-wide cold shader compilation. Allocated/driver memory fields are resident counters, not measurements of peak allocation.

## nano-vLLM model correctness

Qwen3-0.6B runs the real Scheduler, BlockManager, ModelRunner and model. The baseline records greedy choices; the native run consumes those same tokens for teacher-forced comparison and independently checks that its own greedy choices agree. [Full results](results/nano-correctness.json)

| Scenario | Max KL (nats) | Max raw logit difference | Greedy | Exercised state |
| --- | ---: | ---: | --- | --- |
| normal | 4.2861e-05 | 0.046875 | identical | 16 sample calls; 0 prefix hits; 0 preemptions |
| cross-page | 2.30164e-05 | 0.103516 | identical | 16 sample calls; 0 prefix hits; 0 preemptions |
| shared-prefix | 2.20471e-05 | 0.299683 | identical | 24 sample calls; 2 prefix hits; 0 preemptions |
| chunked-prefill | 2.52827e-05 | 0.164062 | identical | 22 sample calls; 0 prefix hits; 0 preemptions |
| preemption | 8.47758e-05 | 0.103516 | identical | 30 sample calls; 0 prefix hits; 1 preemptions |

The frozen model criterion is finite logits, identical greedy choices, and per-request/per-step `KL(P_SDPA || P_native) <= 1e-3` nats at **temperature 1.0**. This is not a claim about every stochastic sampling temperature. The unmodified nano-vLLM `example.py` additionally completed both prompts with its original temperature 0.6 and maximum 256-token setting.

### Why model logits are not checked with the kernel tolerance

The initial near-zero logit budget (`atol=0.03, rtol=0.01`) failed despite matching greedy choices. We investigated rather than changing native arithmetic or weakening the kernel tests. On 280 real-model, same-input attention comparisons, native/SDPA RMS-error ratios relative to CPU FP64 ranged from 0.999907 to 1.000102. Small FP16 differences accumulated through 28 layers. A trial whole-model “twice the FP64-attention trajectory error” rule also failed: a local kernel error bound does not imply the same bound after nonlinear layers. This reference trajectory still uses FP16 weights and other operators, so it is not a complete FP64 model.

The model criterion therefore compares prediction distributions, which are invariant to a common logit shift, while retaining raw-logit diagnostics. Small KL does not by itself prove every logit difference is a common shift. The kernel pointwise tolerances were unchanged. The KL threshold was fixed before the formal model acceptance run. [Independent FP64 attention investigation](results/real-model-fp64-attention.json)

## Real generation performance

Fixed 1024-token prompts and 128 generated tokens per request, with greedy sampling, identical synchronization and no logit-copy/comparison work inside the timed runs. These are end-to-end observations from one paired run per batch size; the three-round hard gate above applies to the attention-call matrix. [Full generation data](results/nano-performance.json)

| Batch | SDPA tokens/s | Native tokens/s | Throughput speedup | SDPA/native TTFT (s) | SDPA/native median decode step (ms) |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 10.23 | 27.21 | 2.66x | 0.393 / 0.270 | 80.75 / 34.55 |
| 4 | 30.95 | 83.03 | 2.68x | 0.952 / 0.886 | 115.55 / 39.93 |

## Distribution and publication evidence

[GitHub Actions packaging/import run](https://github.com/SuperMarioYL/flash-attn-mps/actions/runs/34759036568) passed on Linux/CPU. It builds wheel and sdist, checks Metal/license resources, and imports the installed wheel outside the checkout. It does not claim GPU verification.

The release workflow is: publish the locally verified version, download its actual GitHub wheel into a clean environment, install nano-vLLM non-editably, and rerun native/model correctness. The resulting install report is attached to the [v0.1.0 release](https://github.com/SuperMarioYL/flash-attn-mps/releases/tag/v0.1.0). Hardware validation here is limited to the listed M1 Max; CUDA and other Apple GPU models were not hardware-tested.
