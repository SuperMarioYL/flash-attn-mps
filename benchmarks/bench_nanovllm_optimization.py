"""Real nano-vLLM release/development generation comparison.

Run with nano-vLLM's existing environment; no HF kernel/MTL dependency is needed:
  MTLFLASHATTN_SHIM=off PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    ../nano-vllm/.venv/bin/python benchmarks/bench_nanovllm_optimization.py

Only three imported attention functions are patched inside this test process.
Performance calls retain the original stochastic sampler. Independent greedy
teacher-forced correctness uses validate_nanovllm's existing helper and is never
used as a timing result. Every measured generation gets a fresh engine, explicit
warmup, and prompts that do not reuse the warmup's cached prefix.
"""

import argparse
import atexit
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


os.environ["MTLFLASHATTN_SHIM"] = "off"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch

from bench_optimization import ROOT, load_libraries, load_module, manifest


def find_model(explicit, nano_root):
    if explicit is not None:
        candidates = [explicit.expanduser()]
    else:
        # The first path is the local nano-vLLM example.py default, verified
        # during preparation. Do not download or guess a different model.
        candidates = [Path.home() / "huggingface/Qwen3-0.6B",
                      Path.home() / "models/Qwen3-0.6B", nano_root / "models/Qwen3-0.6B"]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (
            (candidate / "model.safetensors").is_file()
            or (candidate / "model.safetensors.index.json").is_file()
        ):
            return candidate.resolve()
    raise FileNotFoundError("No complete local Qwen3-0.6B model found; provide --model")


@contextmanager
def attention_backend(attention_module, library):
    names = ("store_kvcache", "flash_attn_varlen_func", "flash_attn_with_kvcache")
    with patch.multiple(attention_module, **{name: getattr(library, name) for name in names}):
        yield {name: getattr(attention_module, name).__module__ for name in names}


def prompt_tokens(tokenizer, length, batch):
    pattern = tokenizer.encode("Explain how a computer stores and retrieves information. ")
    prompts = [[pattern[(index + offset) % len(pattern)] for index in range(length)]
               for offset in range(batch)]
    # Prefix hashes include the preceding block hash; changing the first token
    # prevents every subsequent warmup block from matching measured prefixes.
    first_tokens = {tokens[0] for tokens in prompts}
    markers = [token for token in range(len(first_tokens) + batch) if token not in first_tokens][:batch]
    warmup = [[markers[index], *tokens[1:]] for index, tokens in enumerate(prompts)]
    return prompts, warmup


def performance_run(model, settings, prompts, warmup_prompts, generated, warmup_generated,
                    seed, library, attention_module, temperature):
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(seed)
    load_started = time.perf_counter()
    with attention_backend(attention_module, library) as routing:
        llm = LLM(str(model), device="mps", **settings)
        try:
            torch.mps.synchronize()
            load_seconds = time.perf_counter() - load_started
            counts = {"prefix_hit_requests": 0, "reused_prefix_blocks": 0, "preemptions": 0}
            allocate = llm.scheduler.block_manager.allocate

            def observed_allocate(sequence, cached_blocks):
                counts["prefix_hit_requests"] += int(cached_blocks > 0)
                counts["reused_prefix_blocks"] += cached_blocks
                return allocate(sequence, cached_blocks)

            preempt = llm.scheduler.preempt

            def observed_preempt(sequence):
                counts["preemptions"] += 1
                return preempt(sequence)

            llm.scheduler.block_manager.allocate = observed_allocate
            llm.scheduler.preempt = observed_preempt
            stamps = []
            original_sample = llm.model_runner.sampler.forward

            def sample_ready(logits, temperatures):
                # Delegate unchanged numerical sampling, with no logits/tokens
                # readback or correctness comparison in this instrumentation.
                tokens = original_sample(logits, temperatures)
                torch.mps.synchronize()
                stamps.append(time.perf_counter())
                return tokens

            with patch.object(llm.model_runner.sampler, "forward", sample_ready):
                torch.manual_seed(seed + 100_000)
                warmup_started = time.perf_counter()
                warmup_output = llm.generate(
                    warmup_prompts, SamplingParams(temperature=temperature,
                                                   max_tokens=warmup_generated, ignore_eos=True),
                    use_tqdm=False)
                torch.mps.synchronize()
                warmup_seconds = time.perf_counter() - warmup_started
                assert all(len(item["token_ids"]) == warmup_generated for item in warmup_output)
                del warmup_output
                stamps.clear()
                for key in counts:
                    counts[key] = 0
                before_memory = {"allocated_bytes": torch.mps.current_allocated_memory(),
                                 "driver_bytes": torch.mps.driver_allocated_memory()}
                torch.manual_seed(seed)
                torch.mps.synchronize()
                started = time.perf_counter()
                outputs = llm.generate(
                    prompts, SamplingParams(temperature=temperature, max_tokens=generated, ignore_eos=True),
                    use_tqdm=False)
                torch.mps.synchronize()
                elapsed = time.perf_counter() - started
            # All validation and serialization below occur after the timer.
            assert len(stamps) == generated, "Expected one unchunked prefill sample and subsequent decode samples"
            assert len(outputs) == len(prompts)
            assert all(len(item["token_ids"]) == generated for item in outputs)
            assert counts["prefix_hit_requests"] == 0, "Normal prefill was contaminated by a cached warmup prefix"
            assert counts["preemptions"] == 0, "Capacity must cover this normal generation case"
            intervals = [end - begin for begin, end in zip(stamps, stamps[1:])]
            emitted = sum(len(item["token_ids"]) for item in outputs)
            token_ids = [item["token_ids"] for item in outputs]
            return {
                "attention_routing": routing,
                "sampler": "Original production Sampler.forward, temperature sampling; wrapper only timestamps completion",
                "seed": seed, "batch": len(prompts), "prompt_tokens_per_request": len(prompts[0]),
                "generated_tokens_per_request": generated, "emitted_tokens": emitted,
                "elapsed_s": elapsed, "tokens_per_s": emitted / elapsed,
                "ttft_sample_ready_s": stamps[0] - started,
                "median_decode_inter_sample_ms": statistics.median(intervals) * 1000 if intervals else None,
                "decode_inter_sample_ms": [value * 1000 for value in intervals],
                "load_and_constructor_warmup_s": load_seconds,
                "explicit_generation_warmup_s": warmup_seconds,
                "sample_calls": len(stamps), "lifecycle_counts": counts,
                "resident_before_generation": before_memory,
                "resident_after_generation": {"allocated_bytes": torch.mps.current_allocated_memory(),
                                               "driver_bytes": torch.mps.driver_allocated_memory()},
                "output_tokens_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            }
        finally:
            # Same lifecycle cleanup as validate_nanovllm.run_case; each round
            # gets an independent engine/cache/process group, not persistent KV.
            atexit.unregister(llm.exit)
            llm.exit()
            del llm
            gc.collect()
            torch.mps.empty_cache()


def correctness_run(model, settings, prompts, generated, release, dev, attention_module, validation):
    batches = [(prompts, generated)]
    # The existing helper normally chooses SDPA for its reference. Point that
    # reference at the real nano forward method with the release functions
    # installed, keeping all its independent greedy/KL checks intact.
    with attention_backend(attention_module, release):
        with patch.object(validation, "reference_forward", attention_module.Attention.forward):
            reference, traces = validation.run_case(model, settings, batches)
    with attention_backend(attention_module, dev):
        actual, _ = validation.run_case(model, settings, batches, expected=traces)
    assert reference["output_tokens"] == actual["output_tokens"]
    errors = actual["logit_errors"]
    result = {
        "passed": True,
        "criterion": f"Finite logits; per-request/step KL(P_release || P_development) <= {validation.MODEL_KL_LIMIT}; identical greedy choices",
        "timing_eligible": False,
        "note": "This separate helper run replaces the test sampler for teacher forcing and reads logits; none of its durations are performance results",
        "sample_calls": actual["counters"]["sample_calls"],
        "max_kl_divergence": max(item["kl_divergence_max"] for item in errors),
        "max_logit_abs_error": max(item["max_abs"] for item in errors),
        "all_greedy_choices_match": all(item["greedy_matches"] for item in errors),
        "reference_lifecycle_counts": reference["counters"],
        "development_lifecycle_counts": actual["counters"],
    }
    del reference, actual, traces
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--nano-root", type=Path, default=ROOT.parent / "nano-vllm")
    parser.add_argument("--output", type=Path, default=ROOT / ".artifacts/optimization/nanovllm.json")
    parser.add_argument("--mode", choices=("all", "performance", "correctness"), default="all")
    parser.add_argument("--batch", type=int, action="append", choices=(1, 4))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--quick", action="store_true")
    opts = parser.parse_args()
    if opts.quick:
        opts.rounds, opts.generate_tokens, opts.warmup_tokens = 1, 8, 8
    nano_root = opts.nano_root.expanduser().resolve()
    model = find_model(opts.model, nano_root)
    sys.path.insert(0, str(nano_root))
    import nanovllm
    import nanovllm.layers.attention as attention_module
    from transformers import AutoTokenizer
    if Path(nanovllm.__file__).resolve().parent.parent != nano_root:
        raise RuntimeError("nano-vLLM imported from a different checkout")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS hardware is required")
    dev, release, _, _ = load_libraries()
    validation = load_module("_optimization_nano_validation", ROOT / "benchmarks/validate_nanovllm.py")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dev_dir = Path(dev.__file__).parent
    settings = {"max_num_seqs": 4, "max_model_len": max(1536, opts.prompt_length + opts.generate_tokens),
                "max_num_batched_tokens": max(4096, opts.prompt_length * 4), "num_kvcache_blocks": 32}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(model), "nano_root": str(nano_root),
        "nano_head": subprocess.check_output(["git", "-C", str(nano_root), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "python": platform.python_version(), "macos": platform.mac_ver()[0],
        "device": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "development_module": str(dev.__file__), "release_module": str(release.__file__),
        "source_manifest_before": manifest(dev_dir), "nano_manifest_before": manifest(nano_root / "nanovllm"),
        "settings": settings, "mode": opts.mode, "rounds": opts.rounds, "temperature": opts.temperature,
        "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "timing_note": "Original sampler preserved. TTFT is internal first sampled-token readiness, not HTTP/client streaming. Complete generate elapsed time includes scheduler, sampling, and final tokenizer decode. No correctness/logit readback in timed runs.",
        "warmup_note": "Fresh engine per backend/round; explicit generation warmup uses distinct first tokens and measured runs assert zero cached-prefix hits",
        "memory_note": "Allocated/driver counters are resident snapshots while the engine is loaded, not peak usage or a standalone allocation benchmark",
        "correctness_note": "Separate existing greedy teacher-forced helper; performance is not inferred from correctness durations",
        "cases": [],
    }

    def save():
        opts.output.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        for batch in opts.batch or (1, 4):
            prompts, warmup = prompt_tokens(tokenizer, opts.prompt_length, batch)
            row = {"name": f"normal-b{batch}", "batch": batch,
                   "prompt_tokens": opts.prompt_length, "generated_tokens": opts.generate_tokens,
                   "correctness": {"status": "not-run"}, "performance_rounds": []}
            report["cases"].append(row)
            if opts.mode in ("all", "correctness"):
                try:
                    row["correctness"] = correctness_run(
                        model, settings, prompts, opts.generate_tokens, release, dev, attention_module, validation)
                    print(row["name"], "CORRECT", row["correctness"]["max_kl_divergence"], flush=True)
                except Exception:
                    row["correctness"] = {"passed": False, "error": traceback.format_exc()}
                    save()
                    raise
                save()
            if opts.mode in ("all", "performance"):
                for round_id in range(opts.rounds):
                    order = [("release-0.1.0", release), ("development", dev)]
                    if round_id % 2:
                        order.reverse()
                    measurements = {}
                    for label, library in order:
                        measurements[label] = performance_run(
                            model, settings, prompts, warmup, opts.generate_tokens, opts.warmup_tokens,
                            202_609 + round_id, library, attention_module, opts.temperature)
                        item = measurements[label]
                        print(row["name"], round_id + 1, label,
                              {key: item[key] for key in ("tokens_per_s", "ttft_sample_ready_s", "median_decode_inter_sample_ms")},
                              flush=True)
                    row["performance_rounds"].append({"order": [label for label, _ in order], "measurements": measurements})
                    save()
                summary = {}
                for label in ("release-0.1.0", "development"):
                    summary[label] = {key: statistics.median(rnd["measurements"][label][key]
                                                           for rnd in row["performance_rounds"])
                                      for key in ("elapsed_s", "tokens_per_s", "ttft_sample_ready_s", "median_decode_inter_sample_ms")}
                row["summary"] = summary
                row["throughput_speedup_per_round"] = [
                    rnd["measurements"]["development"]["tokens_per_s"] /
                    rnd["measurements"]["release-0.1.0"]["tokens_per_s"] for rnd in row["performance_rounds"]]
                save()
    report["source_manifest_after"] = manifest(dev_dir)
    report["nano_manifest_after"] = manifest(nano_root / "nanovllm")
    report["source_unchanged_during_run"] = report["source_manifest_before"] == report["source_manifest_after"]
    report["nano_source_unchanged_during_run"] = report["nano_manifest_before"] == report["nano_manifest_after"]
    report["complete_formal_performance"] = (opts.mode in ("all", "performance") and not opts.quick
        and opts.rounds >= 3 and opts.prompt_length == 1024 and opts.generate_tokens == 128
        and opts.warmup_tokens >= 128 and {row["batch"] for row in report["cases"]} == {1, 4}
        and report["source_unchanged_during_run"] and report["nano_source_unchanged_during_run"])
    report["all_correctness_passed"] = all(row["correctness"].get("passed", False) for row in report["cases"])
    save()


if __name__ == "__main__":
    main()
