"""Compare development kernels with frozen v0.1.0 and the strongest known APIs.

Example (the library paths are also independently asserted and fingerprinted):
  MTLFLASHATTN_SHIM=off PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
    .artifacts/operator-comparison/env/bin/python benchmarks/bench_optimization.py --quick

The default suite contains six compute-only cases, two complete paged calls,
and small feature observations. Flex is deliberately excluded: previous native
MPS measurements established that it is not the best comparator on this host.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path


os.environ["MTLFLASHATTN_SHIM"] = "off"
os.environ["MTLFLASHATTN_KERNEL"] = "auto"
os.environ["MTLFLASHATTN_TRACE"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "761199956ba9baffbc93e0a3e08933668f06cf7a"
OPERATOR_CASES = [
    ("short-prefill", 1, 128, 128),
    ("prefill-b1-2048", 1, 2048, 2048),
    ("prefill-b4-4096", 4, 4096, 4096),
    ("decode-b1-8192", 1, 1, 8192),
    ("decode-b16-8192", 16, 1, 8192),
    ("cached-chunk-b4", 4, 64, 576),
]
FEATURE_CASES = [
    ("bf16-prefill", torch.bfloat16, 2, 128, 512, 128, 128),
    ("fp32-decode", torch.float32, 4, 1, 2048, 128, 128),
    ("diffkv-prefill", torch.float16, 2, 64, 512, 128, 64),
    ("fp8-descale-decode", torch.float8_e4m3fn, 4, 1, 2048, 128, 128),
    ("window-prefill", torch.float16, 2, 128, 1024, 128, 128),
    ("sink-decode", torch.float16, 4, 1, 2048, 128, 128),
    ("softcap-prefill", torch.float16, 2, 64, 512, 128, 128),
]


def load_module(name, path, package=False):
    options = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_libraries():
    # PYTHONPATH=. may expose the source tree's egg-info first. Restrict this
    # lookup to the interpreter's installed packages before loading the wheel.
    installed = Path(sysconfig.get_paths()["purelib"]).resolve()
    distribution = next((item for item in importlib.metadata.distributions(path=[str(installed)])
                         if item.metadata["Name"].lower().replace("_", "-") == "flash-attn-mps"), None)
    if distribution is None:
        raise RuntimeError("The interpreter does not contain an installed flash-attn-mps release")
    release_dir = Path(distribution.locate_file("flash_attn_mps")).resolve()
    dev_dir = (ROOT / "flash_attn_mps").resolve()
    if distribution.version != "0.1.0" or release_dir.parent.name != "site-packages":
        raise RuntimeError("Use the isolated environment containing the non-editable v0.1.0 wheel")
    if release_dir == dev_dir:
        raise RuntimeError("Development and released packages must be different directories")
    # Aliases give every relative import and Python/shader cache its own owner.
    release = load_module("_optimization_release", release_dir / "__init__.py", package=True)
    release.__benchmark_distribution_version__ = distribution.version
    dev = load_module("_optimization_development", dev_dir / "__init__.py", package=True)
    paths = load_module("_optimization_paths", ROOT / "benchmarks/compare_paths.py")
    paths.ours = release  # A frozen common writer in every complete-call adapter.
    oracle = load_module("_optimization_reference", ROOT / "tests/reference.py")
    return dev, release, paths, oracle


def manifest(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".metal")}


def measure(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    ordered = sorted(samples)
    return {"p50_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "samples_ms": samples}


def cpu_samples(q, k, v):
    qlen, klen, dim = q.shape[1], k.shape[1], q.shape[-1]
    rows = list(range(qlen)) if qlen <= 128 else sorted({
        0, 1, 7, 15, 31, 32, 63, 64, 127, 128, 255, 256,
        qlen // 2, qlen - 2, qlen - 1})
    query = q[:, rows].cpu().double().transpose(1, 2)
    key, value = (x.cpu().double().transpose(1, 2) for x in (k, v))
    mask = torch.arange(klen)[None, :] <= (torch.tensor(rows) + klen - qlen)[:, None]
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=dim ** -.5, enable_gqa=True)
    return rows, expected.transpose(1, 2)


def operator_case(spec, dev, release, hf):
    name, batch, qlen, klen = spec
    torch.manual_seed(123)
    q = torch.randn(batch, qlen, 16, 128, device="mps", dtype=torch.float16)
    k = torch.randn(batch, klen, 8, 128, device="mps", dtype=torch.float16)
    v = torch.randn_like(k)
    qs, ks, vs = (x.transpose(1, 2) for x in (q, k, v))
    qp, kp, vp = (x.flatten(0, 1) for x in (q, k, v))
    cuq = torch.arange(batch + 1, device="mps", dtype=torch.int32) * qlen
    cuk = torch.arange(batch + 1, device="mps", dtype=torch.int32) * klen
    mask = None if qlen in (1, klen) else (
        torch.arange(klen, device="mps")[None, :] <=
        torch.arange(klen - qlen, klen, device="mps")[:, None])

    def native(module):
        return lambda: module.flash_attn_varlen_func(
            qp, kp, vp, cuq, cuk, qlen, klen,
            causal=qlen > 1, softmax_scale=128 ** -.5).reshape_as(q)

    calls = {
        "development": native(dev), "release-0.1.0": native(release),
        "sdpa": lambda: F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=mask, is_causal=qlen == klen and qlen > 1,
            scale=128 ** -.5, enable_gqa=True).transpose(1, 2),
        "hf-metal": native(hf),
    }
    rows, expected = cpu_samples(q, k, v)
    row = {"name": name, "suite": "operator", "representative_gate": True,
           "batch": batch, "q_len": qlen, "kv_len": klen, "heads": [16, 8],
           "qk_dim": 128, "v_dim": 128, "dtype": "float16",
           "reference_query_rows": rows,
           "scope": "Public compute API; fixed input views/mask/cu lengths prepared outside timing; no cache update or gather"}
    return row, calls, lambda out: out[:, rows], expected, q.shape, torch.float16


def paged_case(name, lengths, qlen, dev, release, paths, hf, mtl):
    case = paths.make_case(name, lengths, qlen)

    def native(module):
        def call():
            release.store_kvcache(case.k, case.v, case.kcache, case.vcache, case.slots)
            return module.flash_attn_with_kvcache(
                case.q4, case.kcache, case.vcache, cache_seqlens=case.lengths,
                block_table=case.table, softmax_scale=128 ** -.5,
                causal=True).reshape(-1, 16, 128)
        return call

    grouped = paths.build_grouped_sdpa(paths.selected_pages)
    hf_call = paths.build_hf(hf, paths.selected_pages)
    mtl_call = paths.build_mtl(mtl, paths.selected_pages)
    calls = {"development": native(dev), "release-0.1.0": native(release),
             "sdpa-grouped-index-select": lambda: grouped(case),
             "sdpa-batched-index-select": lambda: paths.sdpa_batched(case, paths.selected_pages),
             "hf-index-select-pack": lambda: hf_call(case),
             "mtl-auto-index-select-grouped": lambda: mtl_call(case)}
    row = {"name": name, "suite": "paged", "representative_gate": True,
           "batch": len(lengths), "q_len": qlen, "kv_lengths": lengths,
           "heads": [16, 8], "qk_dim": 128, "v_dim": 128, "dtype": "float16",
           "scope": "Complete call including frozen common KV writer, live GPU metadata, index_select gather, dynamic masks/grouping, necessary copies and output merging",
           "padding": "Randomized physical pages; NaN invalid tails and positive sentinel pages",
           "hf_note": "Boolean packing and required MPS synchronization are inside the timed adapter"}
    return row, calls, lambda out: out, case.expected, case.q.shape, torch.float16


def feature_case(spec, dev, release, oracle, mtl, hf):
    name, dtype, batch, qlen, klen, dq, dv = spec
    torch.manual_seed(227)

    def tensor(shape):
        if dtype == torch.float8_e4m3fn:
            value = torch.randn(shape).to(dtype)
            return value.view(torch.uint8).to("mps").view(dtype)
        return torch.randn(shape, device="mps", dtype=dtype)

    q, k, v = tensor((batch, qlen, 16, dq)), tensor((batch, klen, 8, dq)), tensor((batch, klen, 8, dv))
    qp, kp, vp = (x.flatten(0, 1) for x in (q, k, v))
    cuq = torch.arange(batch + 1, device="mps", dtype=torch.int32) * qlen
    cuk = torch.arange(batch + 1, device="mps", dtype=torch.int32) * klen
    window = (127, 0) if name == "window-prefill" else (-1, -1)
    sink = torch.linspace(-1, 1, 16, device="mps") if name == "sink-decode" else None
    softcap = 3.0 if name == "softcap-prefill" else 0.0
    descales = {}
    if dtype == torch.float8_e4m3fn:
        for label, begin, end in (("q_descale", .75, 1.25), ("k_descale", .5, 1.0), ("v_descale", .7, 1.1)):
            descales[label] = torch.linspace(begin, end, 8, device="mps").expand(batch, 8)
    kwargs = dict(max_seqlen_q=qlen, cu_seqlens_q=cuq, max_seqlen_k=klen,
                  cu_seqlens_k=cuk, causal=True, softmax_scale=dq ** -.5,
                  window_size=window, s_aux=sink, softcap=softcap, **descales)

    def native(module):
        api = sys.modules[module.__name__ + ".vllm"]
        return lambda: api.flash_attn_varlen_func(qp, kp, vp, **kwargs).reshape(batch, qlen, 16, dv)

    calls = {"development": native(dev), "release-0.1.0": native(release)}
    exclusions = {}
    # These are the installed public API contracts, not emulations. Leave all
    # layout copies, metadata allocations and dispatch inside the API timing.
    if dtype == torch.float8_e4m3fn:
        exclusions["mtl-auto"] = "Only matching FP16/BF16/FP32 inputs are accepted; no FP8/descale contract"
    elif dq != dv:
        exclusions["mtl-auto"] = "Public dense API requires matching Q/K/V head dimensions"
    elif sink is not None:
        exclusions["mtl-auto"] = "Public dense API has no attention-sink parameter"
    else:
        def mtl_dense():
            return mtl.flash_attn_func(
                q, k, v, causal=qlen > 1, softmax_scale=dq ** -.5,
                window_size=window, softcap=softcap)
        calls["mtl-auto"] = mtl_dense
        if dtype == torch.float16:
            # The documented M1 tier remains a candidate; numerical failures
            # are excluded by exactly the same oracle/tolerance as every API.
            calls["mtl-v1"] = mtl_dense
        else:
            exclusions["mtl-v1"] = "The explicit v1 tier accepts FP16 only; auto selects the runtime dtype implementation"

    if dtype == torch.float8_e4m3fn:
        exclusions["hf-metal"] = "Pinned build has FP16/BF16/FP32 variants and no encoded FP8/descale API"
    elif dq != dv:
        exclusions["hf-metal"] = "Pinned wrapper allocates output like Q and exposes a shared head dimension"
    elif sink is not None:
        exclusions["hf-metal"] = "Pinned public API has no attention-sink parameter"
    elif window != (-1, -1):
        exclusions["hf-metal"] = "Pinned public varlen wrapper rejects nontrivial window_size"
    elif softcap:
        exclusions["hf-metal"] = "Pinned varlen wrapper has no softcap argument; its low-level softcapping is documented as fixed at 1.0"
    else:
        calls["hf-metal"] = lambda: hf.flash_attn_varlen_func(
            qp, kp, vp, cuq, cuk, qlen, klen, causal=qlen > 1,
            softmax_scale=dq ** -.5).reshape(batch, qlen, 16, dv)

    external_note = None
    if dtype != torch.float8_e4m3fn and softcap == 0:
        qs, ks, vs = (x.transpose(1, 2) for x in (q, k, v))
        qpos = torch.arange(klen - qlen, klen, device="mps")[:, None]
        kpos = torch.arange(klen, device="mps")[None, :]
        allowed = kpos <= qpos
        if window[0] >= 0:
            allowed &= kpos >= qpos - window[0]
        is_causal = qlen == klen and window == (-1, -1) and sink is None
        mask = None if (qlen == 1 or is_causal) and window == (-1, -1) else allowed
        if sink is None:
            calls["sdpa"] = lambda: F.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=mask, is_causal=is_causal,
                scale=dq ** -.5, enable_gqa=True).transpose(1, 2)
        else:
            # SDPA has no sink argument. Append the zero-valued sink inside the
            # timed adapter; the shape-dependent additive bias is fixed input.
            ordinary = torch.where(allowed, 0.0, -float("inf"))[None, None]
            additive = torch.cat((ordinary.expand(batch, 16, qlen, klen),
                                  sink[None, :, None, None].expand(batch, 16, qlen, 1)), -1)
            zk = torch.zeros(batch, 8, 1, dq, device="mps", dtype=dtype)
            zv = torch.zeros(batch, 8, 1, dv, device="mps", dtype=dtype)
            calls["sdpa-sink-adapter"] = lambda: F.scaled_dot_product_attention(
                qs, torch.cat((ks, zk), 2), torch.cat((vs, zv), 2),
                attn_mask=additive, scale=dq ** -.5, enable_gqa=True).transpose(1, 2)
    else:
        if dtype == torch.float8_e4m3fn:
            external_note = "No equivalent direct encoded FP8+descale API among these comparators; release regression only, not a best-in-class claim"
        else:
            external_note = "Stock SDPA has no softcap parameter; compare the direct MTL API without constructing a score-matrix adapter"
    expected, _ = oracle.attention_reference(
        qp, kp, vp, cuq, cuk, causal=True, window=window, softcap=softcap,
        sinks=sink, **descales)
    expected = expected.reshape(batch, qlen, 16, dv)
    output_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    row = {"name": name, "suite": "feature", "representative_gate": False,
           "batch": batch, "q_len": qlen, "kv_len": klen, "heads": [16, 8],
           "qk_dim": dq, "v_dim": dv, "dtype": str(dtype).removeprefix("torch."),
           "window": window, "sink": sink is not None, "softcap": softcap,
           "direct_api_exclusions": exclusions,
           "external_note": external_note,
           "scope": "Public inference compute APIs against full CPU FP64 feature oracle; fixed input views/masks excluded, API-internal copies/metadata and SDPA sink append copies included"}
    return row, calls, lambda out: out, expected, (batch, qlen, 16, dv), output_dtype


def select_backend(label):
    # Public library setting, changed once per measured block, never per call.
    if label.startswith("mtl-"):
        os.environ["MTLFLASHATTN_KERNEL"] = "v1" if label == "mtl-v1" else "auto"


def run_case(built, opts, report, save):
    row, calls, sample, expected, shape, output_dtype = built
    # BF16 uses the existing tests/reference.py precision contract; all other
    # rows retain the previous comparison's fixed .003/.01 tolerances.
    atol, rtol = (.02, .05) if output_dtype == torch.bfloat16 else (.003, .01)
    row["tolerance"] = {"atol": atol, "rtol": rtol}
    row["implementations"], row["rounds"] = {}, []
    report["cases"].append(row)
    if opts.backends:
        selected = set(opts.backends.split(","))
        calls = {label: call for label, call in calls.items() if label in selected}
    active = {}
    for label, call in calls.items():
        try:
            select_backend(label)
            if label.startswith("mtl-"):
                from metal_flash_attn import _kernel as mtl_kernel
                mtl_kernel._trace.clear()
                os.environ["MTLFLASHATTN_TRACE"] = "1"
            torch.mps.synchronize()
            started = time.perf_counter()
            output = call()
            torch.mps.synchronize()
            first_ms = (time.perf_counter() - started) * 1000
            assert tuple(output.shape) == tuple(shape) and output.dtype == output_dtype
            assert output.device.type == "mps" and torch.isfinite(output).all()
            actual = sample(output).cpu().double()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            error = (actual - expected).abs()
            row["implementations"][label] = {"valid": True, "first_call_ms": first_ms,
                "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
            if label.startswith("mtl-"):
                item = row["implementations"][label]
                item["requested_mtl_mode"] = os.environ["MTLFLASHATTN_KERNEL"]
                item["backend_trace"] = mtl_kernel._trace_summary_lines()
                tiers = sorted({key[-1] for key in mtl_kernel._trace})
                item["selected_mtl_tiers"] = tiers
                item["execution_kind"] = ("pytorch_mps_composite" if "torch" in tiers
                                           else "native_metal" if tiers else "unverified")
            active[label] = call
        except Exception:
            row["implementations"][label] = {"valid": False, "error": traceback.format_exc()}
            print(row["name"], label, "NOT RANKED", row["implementations"][label]["error"], flush=True)
        finally:
            if label.startswith("mtl-") and not row["implementations"].get(label, {}).get("valid"):
                from metal_flash_attn import _kernel as mtl_kernel
                row["implementations"][label].update(
                    requested_mtl_mode=os.environ["MTLFLASHATTN_KERNEL"],
                    backend_trace=mtl_kernel._trace_summary_lines(),
                    selected_mtl_tiers=sorted({key[-1] for key in mtl_kernel._trace}))
            os.environ["MTLFLASHATTN_TRACE"] = "0"
        save()
    labels = list(active)
    if not labels:
        row["measurement_status"] = "No selected backend passed correctness"
        save()
        return
    for round_id in range(opts.rounds):
        order = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
        timing = {}
        for label in order:
            select_backend(label)
            timing[label] = measure(active[label], opts.warmup, opts.iterations)
        row["rounds"].append({"order": order, "timings": timing})
        print(row["name"], round_id + 1,
              {label: value["p50_ms"] for label, value in timing.items()}, flush=True)
        save()
    for label in labels:
        item = row["implementations"][label]
        for metric in ("p50_ms", "p95_ms"):
            item[metric] = statistics.median(rnd["timings"][label][metric] for rnd in row["rounds"])
    dev_result = row["implementations"].get("development", {})
    if dev_result.get("valid"):
        comparators = {label: item["p50_ms"] for label, item in row["implementations"].items()
                       if label != "development" and item.get("valid")}
        external = {label: latency for label, latency in comparators.items() if label != "release-0.1.0"}
        if comparators:
            best = min(comparators, key=comparators.get)
            row["best_measured_comparator"] = best
            row["development_latency_ratio_to_best"] = dev_result["p50_ms"] / comparators[best]
        if external:
            best_external = min(external, key=external.get)
            row["best_external_comparator"] = best_external
            row["development_latency_ratio_to_best_external"] = dev_result["p50_ms"] / external[best_external]
        if "release-0.1.0" in comparators:
            row["development_speedup_vs_release"] = comparators["release-0.1.0"] / dev_result["p50_ms"]
        row["near_best"] = (bool(comparators)
                            and (bool(external) or not row["representative_gate"])
                            and row["development_latency_ratio_to_best"] <= opts.near_best_factor)
    else:
        row["near_best"] = False
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".artifacts/optimization/results.json")
    parser.add_argument("--suite", choices=("all", "operator", "paged", "feature"), default="all")
    parser.add_argument("--case", action="append", help="Repeat to select several case names")
    parser.add_argument("--backends", help="Optional comma-separated backend selection")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--near-best-factor", type=float, default=1.10)
    opts = parser.parse_args()
    if opts.quick:
        opts.warmup, opts.iterations, opts.rounds = 2, 5, 1
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS hardware is required")
    if F.scaled_dot_product_attention.__module__ != "torch._C._nn":
        raise RuntimeError("Launch the interpreter with MTLFLASHATTN_SHIM=off")
    dev, release, paths, oracle = load_libraries()
    dev_dir, release_dir = Path(dev.__file__).parent, Path(release.__file__).parent
    from kernels import get_kernel
    import metal_flash_attn as mtl
    hf = get_kernel("kernels-community/metal-flash-sdpa", revision=HF_REVISION)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": platform.mac_ver()[0], "python": platform.python_version(), "torch": torch.__version__,
        "development_module": str(dev.__file__), "release_module": str(release.__file__),
        "release_version": release.__benchmark_distribution_version__,
        "development_alias": dev.__name__, "release_alias": release.__name__,
        "source_manifest_before": manifest(dev_dir), "release_manifest": manifest(release_dir),
        "helper_manifest": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (Path(__file__), ROOT / "benchmarks/compare_paths.py", ROOT / "tests/reference.py")},
        "hf_revision": HF_REVISION, "mtl_version": importlib.metadata.version("mtlflashattn"),
        "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "shim": os.environ["MTLFLASHATTN_SHIM"],
        "warmup": opts.warmup, "iterations": opts.iterations, "rounds": opts.rounds,
        "near_best_factor": opts.near_best_factor,
        "ratio_definition": "Development p50 / fastest valid measured comparator p50; below 1 is faster. Improvement over release alone does not satisfy near-best.",
        "feature_policy": "Observe major feature paths and correctness; no universal fastest claim where an equivalent external API is absent",
        "cold_note": "Input work synchronized before first call; first-call numbers include process-local pipeline setup, not forced OS-cold compilation",
        "memory_note": "No peak memory or resident-memory ranking is claimed",
        "cases": [],
    }

    def save():
        opts.output.write_text(json.dumps(report, indent=2) + "\n")

    tasks = [("operator", spec[0], lambda spec=spec: operator_case(spec, dev, release, hf))
             for spec in OPERATOR_CASES]
    tasks += [("paged", name, lambda name=name, lengths=lengths, qlen=qlen:
               paged_case(name, lengths, qlen, dev, release, paths, hf, mtl))
              for name, lengths, qlen in (("cached-prefill", [576] * 4, 64),
                                         ("mixed-paged-decode", [4096, 8192] * 8, 1))]
    tasks += [("feature", spec[0], lambda spec=spec: feature_case(spec, dev, release, oracle, mtl, hf))
              for spec in FEATURE_CASES]
    if opts.case:
        unknown = set(opts.case) - {name for _, name, _ in tasks}
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
    with torch.inference_mode():
        for suite, name, build in tasks:
            if opts.suite not in ("all", suite) or (opts.case and name not in opts.case):
                continue
            run_case(build(), opts, report, save)
            gc.collect()
            torch.mps.empty_cache()
    report["source_manifest_after"] = manifest(dev_dir)
    report["source_unchanged_during_run"] = report["source_manifest_before"] == report["source_manifest_after"]
    report["complete_formal_suite"] = (not opts.quick and not opts.case and not opts.backends
                                       and opts.suite == "all" and opts.warmup >= 10
                                       and opts.iterations >= 50 and opts.rounds >= 3
                                       and report["source_unchanged_during_run"])
    representative = [row for row in report["cases"] if row["representative_gate"]]
    report["all_representative_near_best"] = (len(representative) == 8 and
                                             all(row.get("near_best", False) for row in representative))
    report["all_development_correct"] = bool(report["cases"]) and all(
        row["implementations"].get("development", {}).get("valid", False) for row in report["cases"])
    report["formal_performance_target_verified"] = (report["complete_formal_suite"]
        and report["all_representative_near_best"] and report["all_development_correct"])
    save()


if __name__ == "__main__":
    main()
