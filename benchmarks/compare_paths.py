"""Compare paged attention adapters from identical nano-vLLM-style inputs.

Every measured call starts with the same native KV writer, then consumes GPU
query/length/page-table metadata. The comparison isolates attention adaptation;
it does not compare the competing projects' KV-writing kernels. Gather, padding,
dynamic masks, necessary contiguous copies and output packing are timed. Only
shape-dependent constants and conservative Flex block lists are prepared early.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


# These must also be set before interpreter startup: mtlflashattn has a .pth hook.
os.environ["MTLFLASHATTN_SHIM"] = "off"
os.environ["MTLFLASHATTN_KERNEL"] = "auto"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn.functional as F

import flash_attn_mps as ours


HF_REVISION = "761199956ba9baffbc93e0a3e08933668f06cf7a"
PAGE, HQ, HK, DIM = 256, 16, 8, 128
SCALE = DIM ** -0.5


@dataclass
class Case:
    name: str
    batch: int
    qlen: int
    max_k: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kcache: torch.Tensor
    vcache: torch.Tensor
    slots: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    cuq: torch.Tensor
    key_positions: torch.Tensor
    query_positions: torch.Tensor
    zero: torch.Tensor
    expected: torch.Tensor
    initial_lengths: list

    @property
    def q4(self):
        return self.q.view(self.batch, self.qlen, HQ, DIM)


def cpu_reference(q, kc, vc, table, lengths, qlen):
    """Independent FP64 logical attention; never used in the timed adapters."""
    result = []
    for batch_index, length in enumerate(lengths):
        pages = table[batch_index, :math.ceil(length / PAGE)].long()
        key = kc[pages].flatten(0, 1)[:length].double()
        value = vc[pages].flatten(0, 1)[:length].double()
        query = q[batch_index * qlen:(batch_index + 1) * qlen].double()
        query = query.reshape(qlen, HK, HQ // HK, DIM).permute(1, 2, 0, 3)
        key = key.permute(1, 0, 2).unsqueeze(1)
        value = value.permute(1, 0, 2).unsqueeze(1)
        score = (query @ key.transpose(-1, -2)) * SCALE
        valid = torch.arange(length).unsqueeze(0) <= (
            torch.arange(qlen) + length - qlen).unsqueeze(1)
        score.masked_fill_(~valid, -float("inf"))
        output = score.softmax(-1) @ value
        result.append(output.permute(2, 0, 1, 3).reshape(qlen, HQ, DIM))
    return torch.cat(result)


def make_case(name, lengths, qlen):
    torch.manual_seed(314159)
    batch, max_k = len(lengths), max(lengths)
    counts = [math.ceil(length / PAGE) for length in lengths]
    physical = torch.randperm(sum(counts))
    sentinel = sum(counts)
    table = torch.full((batch, math.ceil(max_k / PAGE)), sentinel, dtype=torch.int32)
    offset = 0
    for index, count in enumerate(counts):
        table[index, :count] = physical[offset:offset + count].int()
        offset += count
    kc = torch.randn(sentinel + 1, PAGE, HK, DIM, dtype=torch.float16)
    vc = torch.randn_like(kc)
    kc[sentinel].fill_(float("nan"))
    vc[sentinel].fill_(float("nan"))
    for index, length in enumerate(lengths):
        if length % PAGE:
            last = int(table[index, counts[index] - 1])
            kc[last, length % PAGE:].fill_(float("nan"))
            vc[last, length % PAGE:].fill_(float("nan"))
    packed = torch.randn(batch * qlen, HQ + 2 * HK, DIM, dtype=torch.float16)
    q, k, v = packed.split((HQ, HK, HK), dim=1)
    slots = torch.tensor([
        int(table[index, position // PAGE]) * PAGE + position % PAGE
        for index, length in enumerate(lengths)
        for position in range(length - qlen, length)
    ], dtype=torch.int32)
    # All calls make this same idempotent update; reference sees those new tokens.
    kc.flatten(0, 1).index_copy_(0, slots.long(), k)
    vc.flatten(0, 1).index_copy_(0, slots.long(), v)
    expected = cpu_reference(q, kc, vc, table, lengths, qlen)
    packed = packed.to("mps")
    q, k, v = packed.split((HQ, HK, HK), dim=1)
    return Case(
        name, batch, qlen, max_k, q, k, v, kc.to("mps"), vc.to("mps"),
        slots.to("mps"), table.to("mps"),
        torch.tensor(lengths, device="mps", dtype=torch.int32),
        torch.arange(batch + 1, device="mps", dtype=torch.int32) * qlen,
        torch.arange(max_k, device="mps").unsqueeze(0),
        torch.arange(qlen, device="mps").view(1, qlen, 1),
        torch.zeros(1, device="mps", dtype=torch.int32), expected, lengths,
    )


def write_cache(case):
    ours.store_kvcache(case.k, case.v, case.kcache, case.vcache, case.slots)


def fancy_pages(kcache, vcache, table):
    pages = table.long()
    return kcache[pages].flatten(1, 2), vcache[pages].flatten(1, 2)


def selected_pages(kcache, vcache, table):
    # These benchmark tables use nonnegative physical IDs, including a positive
    # sentinel page. Clamp therefore preserves their contents exactly.
    pages = table.clamp_min(0).flatten()
    shape = (table.size(0), table.size(1) * kcache.size(1), HK, DIM)
    return (torch.index_select(kcache, 0, pages).view(shape),
            torch.index_select(vcache, 0, pages).view(shape))


def gather(case, clear_padding, page_reader=fancy_pages):
    """Read current pages and lengths on every call; never cache gathered KV."""
    k, v = page_reader(case.kcache, case.vcache, case.table)
    k, v = k[:, :case.max_k], v[:, :case.max_k]
    valid = case.key_positions < case.lengths.unsqueeze(1)
    if clear_padding:
        # A zero softmax probability times a NaN V is still NaN. SDPA/Flex read
        # dense storage, so mask alone cannot make poisoned padded KV harmless.
        k.masked_fill_(~valid[:, :, None, None], 0)
        v.masked_fill_(~valid[:, :, None, None], 0)
    return k, v, valid


def native_paged(case):
    write_cache(case)
    return ours.flash_attn_with_kvcache(
        case.q4, case.kcache, case.vcache, cache_seqlens=case.lengths,
        block_table=case.table, softmax_scale=SCALE, causal=True,
    ).reshape(-1, HQ, DIM)


def sdpa_batched(case, page_reader=fancy_pages):
    write_cache(case)
    k, v, valid = gather(case, clear_padding=True, page_reader=page_reader)
    # Handles changing GPU lengths without transferring values to the host.
    if case.qlen == 1:
        mask = valid[:, None, None, :]
    else:
        visible = case.key_positions[:, None, :] <= (
            case.query_positions + case.lengths[:, None, None] - case.qlen)
        mask = (visible & valid[:, None, :]).unsqueeze(1)
    output = F.scaled_dot_product_attention(
        case.q4.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=False, enable_gqa=True, scale=SCALE,
    )
    return output.transpose(1, 2).contiguous().view(-1, HQ, DIM)


def build_hf(hf, page_reader=fancy_pages):
    def call(case):
        write_cache(case)
        k, v, valid = gather(case, clear_padding=False, page_reader=page_reader)
        # Boolean packing and cumulative lengths are dynamic GPU operations.
        # Their nonzero/shape synchronization is part of this complete call.
        k, v = k[valid], v[valid]
        cuk = torch.cat((case.zero, case.lengths.cumsum(0, dtype=torch.int32)))
        q = case.q.contiguous()
        # This pinned HF extension aborts if entered immediately after these
        # PyTorch MPS preparation operations with an encoder still active.
        # Keep the required boundary inside the complete-call measurement.
        torch.mps.synchronize()
        return hf.flash_attn_varlen_func(
            q, k, v, case.cuq, cuk, case.qlen, case.max_k,
            causal=case.qlen > 1, softmax_scale=SCALE,
        ).contiguous()
    return call


def build_mtl(mtl, page_reader=fancy_pages):
    def call(case):
        write_cache(case)
        groups = {}
        # This dense interface has no effective-length mask. Group by live
        # lengths, paying the metadata read and grouping inside the timer.
        for index, length in enumerate(case.lengths.tolist()):
            groups.setdefault(length, []).append(index)
        output = torch.empty((case.batch, case.qlen, HQ, DIM),
                             device=case.q.device, dtype=case.q.dtype)
        for length, members in groups.items():
            indices = torch.tensor(members, device=case.q.device, dtype=torch.int64)
            if page_reader is selected_pages:
                table = case.table.index_select(0, indices)[:, :math.ceil(length / PAGE)]
                q = case.q4.index_select(0, indices)
            else:
                table = case.table[indices, :math.ceil(length / PAGE)]
                q = case.q4[indices]
            k, v = page_reader(case.kcache, case.vcache, table)
            k, v = k[:, :length], v[:, :length]
            result = mtl.flash_attn_func(
                q, k, v, causal=case.qlen > 1, softmax_scale=SCALE)
            output.index_copy_(0, indices, result)
        return output.view(-1, HQ, DIM)
    return call


def build_grouped_sdpa(page_reader):
    from torch.nn.attention.bias import causal_lower_right

    def call(case):
        write_cache(case)
        groups = {}
        for index, length in enumerate(case.lengths.tolist()):
            groups.setdefault(length, []).append(index)
        output = torch.empty((case.batch, case.qlen, HQ, DIM),
                             device=case.q.device, dtype=case.q.dtype)
        for length, members in groups.items():
            indices = torch.tensor(members, device=case.q.device, dtype=torch.int64)
            table = case.table.index_select(0, indices)[:, :math.ceil(length / PAGE)]
            k, v = page_reader(case.kcache, case.vcache, table)
            k, v = k[:, :length], v[:, :length]
            q = case.q4.index_select(0, indices)
            is_causal = case.qlen == length and case.qlen > 1
            mask = (causal_lower_right(case.qlen, length)
                    if case.qlen > 1 and case.qlen != length else None)
            result = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=mask, is_causal=is_causal, enable_gqa=True, scale=SCALE)
            output.index_copy_(0, indices, result.transpose(1, 2))
        return output.view(-1, HQ, DIM)
    return call


def build_flex(case, artifact_dir):
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    import torch._inductor.codegen.metal_flex_attention_template as metal_template

    torch.compiler.set_stance("default")
    torch._dynamo.config.suppress_errors = False
    # Recreate source once per process so native lowering can be evidenced;
    # this does not disable torch.compile's in-process graph reuse in timing.
    torch.compiler.config.force_disable_caches = True
    lens, qlen = case.lengths, case.qlen

    def dynamic_mask(batch, head, query, key):
        return (key < lens[batch]) & (key <= query + lens[batch] - qlen)

    # Every possible KV block is partial. Thus lengths may change without
    # invalidating this shape-only list or incorrectly marking a block as full.
    block = 128
    nq, nk = math.ceil(case.qlen / block), math.ceil(case.max_k / block)
    indices = torch.arange(nk, device="mps", dtype=torch.int32).view(1, 1, 1, nk)
    indices = indices.expand(case.batch, 1, nq, nk).contiguous()
    counts = torch.full((case.batch, 1, nq), nk, device="mps", dtype=torch.int32)
    mask = BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=block, mask_mod=dynamic_mask,
        seq_lengths=(case.qlen, case.max_k), compute_q_blocks=False,
    )
    evidence = {"backend": "inductor", "fullgraph": True,
                "mask": "all-partial block list, dynamically updated GPU counts and length predicate",
                "generated_metal_shader_sha256": []}
    source_path = artifact_dir / f"paths-flex-{case.name}.metal"
    evidence["generated_metal_source"] = str(source_path.resolve())
    generator = getattr(metal_template._generate_metal_shader,
                        "_comparison_original", metal_template._generate_metal_shader)

    def record_shader(*args, **kwargs):
        source = generator(*args, **kwargs)
        evidence["generated_metal_shader_sha256"].append(
            hashlib.sha256(source.encode()).hexdigest())
        source_path.write_text(source)
        return source

    record_shader._comparison_original = generator
    metal_template._generate_metal_shader = record_shader
    compiled = torch.compile(flex_attention, backend="inductor", fullgraph=True,
                             dynamic=False)

    def call(current):
        write_cache(current)
        k, v, _ = gather(current, clear_padding=True)
        # Updating counts is timed. No complete invalid KV block needs scanning,
        # while partial boundary blocks still use the live length predicate.
        counts.copy_(((current.lengths + block - 1) // block)
                     .view(current.batch, 1, 1).expand(current.batch, 1, nq))
        output = compiled(
            current.q4.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            block_mask=mask, enable_gqa=True, scale=SCALE)
        return output.transpose(1, 2).contiguous().view(-1, HQ, DIM)

    return call, evidence


def measure(fn, case, warmup, iterations):
    for _ in range(warmup):
        fn(case)
    torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(case)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return {"p50_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "samples_ms": samples}


def select_backend(label):
    """Select MTL outside the timed segment, using its public mode setting."""
    if label.startswith("mtl-"):
        os.environ["MTLFLASHATTN_KERNEL"] = "v1" if label.startswith("mtl-v1-") else "auto"


def run_supplement(opts):
    """Bounded gather microbenchmark, then grouped SDPA versus the native path."""
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__, "ours_module": str(Path(ours.__file__).resolve()),
        "warmup": opts.warmup, "iterations": opts.iterations, "rounds": opts.rounds,
        "scope": "Supplement; original six-backend paths.json remains unchanged",
        "shared_writer": "flash_attn_mps.store_kvcache in every timed complete call",
        "gather_micro": "One cache, same live nonnegative table, equal_nan=True, 10 warmups/20 calls",
        "reference": "Independent CPU FP64 attention, atol=0.003 rtol=0.01",
        "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "shim": os.environ["MTLFLASHATTN_SHIM"], "cases": [],
    }

    def save():
        opts.output.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        for name, lengths, qlen in [("cached-prefill", [576] * 4, 64),
                                    ("mixed-paged-decode", [4096, 8192] * 8, 1)]:
            case = make_case(name, lengths, qlen)

            def fancy_one(current):
                return current.kcache[current.table.long()]

            def selected_one(current):
                pages = current.table.clamp_min(0).flatten()
                shape = (*current.table.shape, *current.kcache.shape[1:])
                return torch.index_select(current.kcache, 0, pages).view(shape)

            fancy, selected = fancy_one(case), selected_one(case)
            torch.testing.assert_close(fancy, selected, rtol=0, atol=0, equal_nan=True)
            del fancy, selected
            micro = {"exact_equal_including_nan": True,
                     "fancy": measure(fancy_one, case, 10, 20),
                     "index_select": measure(selected_one, case, 10, 20)}
            ratio = micro["fancy"]["p50_ms"] / micro["index_select"]["p50_ms"]
            micro["index_select_speedup"] = ratio
            reader = selected_pages if ratio > 1.05 else fancy_pages
            reader_name = "index-select" if reader is selected_pages else "fancy"
            row = {"name": name, "batch": len(lengths), "query_length": qlen,
                   "key_lengths": lengths, "gather_micro": micro,
                   "chosen_gather": reader_name,
                   "implementations": {}, "rounds": []}
            report["cases"].append(row)
            print(name, "gather-micro", ratio,
                  micro["fancy"]["p50_ms"], micro["index_select"]["p50_ms"], flush=True)
            candidates = {"ours-paged": native_paged,
                          f"sdpa-grouped-{reader_name}": build_grouped_sdpa(reader)}
            if reader is selected_pages:
                candidates["sdpa-batched-index-select"] = (
                    lambda current: sdpa_batched(current, selected_pages))
            for label, fn in candidates.items():
                torch.mps.synchronize()
                output = fn(case).cpu().double()
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output, case.expected, atol=3e-3, rtol=1e-2)
                error = (output - case.expected).abs()
                row["implementations"][label] = {"status": "correct",
                    "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
            save()
            labels = list(candidates)
            for round_id in range(opts.rounds):
                order = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
                times = {}
                for label in order:
                    times[label] = measure(candidates[label], case, opts.warmup, opts.iterations)
                    print(name, round_id + 1, label, times[label]["p50_ms"], flush=True)
                row["rounds"].append({"order": order, "timings": times})
                save()
            for label in labels:
                result = row["implementations"][label]
                result["median_p50_ms"] = statistics.median(
                    rnd["timings"][label]["p50_ms"] for rnd in row["rounds"])
                result["median_p95_ms"] = statistics.median(
                    rnd["timings"][label]["p95_ms"] for rnd in row["rounds"])
                result["speedup_vs_ours_per_round"] = [
                    rnd["timings"]["ours-paged"]["p50_ms"] / rnd["timings"][label]["p50_ms"]
                    for rnd in row["rounds"]]
            save()
            del case, candidates
            gc.collect()
            torch.mps.empty_cache()
    report["complete_measurement"] = (not opts.quick and opts.iterations >= 50
                                      and opts.rounds >= 3 and opts.warmup >= 10)
    save()


def run_third_party_supplement(opts):
    """Apply the same measured gather improvement to third-party adapters."""
    from kernels import get_kernel
    import metal_flash_attn as mtl
    from metal_flash_attn._kernel import _trace, _trace_summary_lines
    hf = get_kernel("kernels-community/metal-flash-sdpa", revision=HF_REVISION)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__, "ours_module": str(Path(ours.__file__).resolve()),
        "hf_revision": HF_REVISION, "mtl_version": importlib.metadata.version("mtlflashattn"),
        "warmup": opts.warmup, "iterations": opts.iterations, "rounds": opts.rounds,
        "scope": "Third-party index_select supplement; previous raw measurements are preserved",
        "shared_writer": "flash_attn_mps.store_kvcache in every timed complete call",
        "reference": "Independent CPU FP64 attention, atol=0.003 rtol=0.01",
        "hf_note": "Timed index_select gather, boolean packing, cuK preparation, contiguous Q, and required MPS synchronization",
        "mtl_note": "Timed live GPU lengths to host, grouping, index_select gather, public dense API, and output merging",
        "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "shim": os.environ["MTLFLASHATTN_SHIM"], "cases": [],
    }

    def save():
        opts.output.write_text(json.dumps(report, indent=2) + "\n")

    candidates = {
        "ours-paged": native_paged,
        "hf-index-select-pack": build_hf(hf, selected_pages),
        "mtl-auto-index-select-grouped": build_mtl(mtl, selected_pages),
        "mtl-v1-index-select-grouped": build_mtl(mtl, selected_pages),
    }
    with torch.inference_mode():
        for name, lengths, qlen in [("cached-prefill", [576] * 4, 64),
                                    ("mixed-paged-decode", [4096, 8192] * 8, 1)]:
            case = make_case(name, lengths, qlen)
            row = {"name": name, "batch": len(lengths), "query_length": qlen,
                   "key_lengths": lengths, "implementations": {}, "rounds": []}
            report["cases"].append(row)
            for label, fn in candidates.items():
                select_backend(label)
                if label.startswith("mtl-"):
                    _trace.clear()
                    os.environ["MTLFLASHATTN_TRACE"] = "1"
                torch.mps.synchronize()
                output = fn(case).cpu().double()
                os.environ["MTLFLASHATTN_TRACE"] = "0"
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output, case.expected, atol=3e-3, rtol=1e-2)
                error = (output - case.expected).abs()
                row["implementations"][label] = {"status": "correct",
                    "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
                if label.startswith("mtl-"):
                    row["implementations"][label]["backend_trace"] = _trace_summary_lines()
                    row["implementations"][label]["kernel_selection"] = os.environ["MTLFLASHATTN_KERNEL"]
            save()
            labels = list(candidates)
            for round_id in range(opts.rounds):
                order = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
                times = {}
                for label in order:
                    select_backend(label)
                    times[label] = measure(candidates[label], case, opts.warmup, opts.iterations)
                    print(name, round_id + 1, label, times[label]["p50_ms"], flush=True)
                row["rounds"].append({"order": order, "timings": times})
                save()
            for label in labels:
                result = row["implementations"][label]
                result["median_p50_ms"] = statistics.median(
                    rnd["timings"][label]["p50_ms"] for rnd in row["rounds"])
                result["median_p95_ms"] = statistics.median(
                    rnd["timings"][label]["p95_ms"] for rnd in row["rounds"])
                result["speedup_vs_ours_per_round"] = [
                    rnd["timings"]["ours-paged"]["p50_ms"] / rnd["timings"][label]["p50_ms"]
                    for rnd in row["rounds"]]
            save()
            del case
            gc.collect()
            torch.mps.empty_cache()
    report["complete_measurement"] = (not opts.quick and opts.iterations >= 50
                                      and opts.rounds >= 3 and opts.warmup >= 10)
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(".artifacts/operator-comparison/paths.json"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", choices=("cached-prefill", "mixed-paged-decode"))
    parser.add_argument("--no-flex", action="store_true")
    parser.add_argument("--supplement", action="store_true",
                        help="Measure gather APIs and append optimized grouped SDPA comparisons")
    parser.add_argument("--third-party-supplement", action="store_true",
                        help="Apply index_select to the HF and MTL adapters, preserving earlier results")
    opts = parser.parse_args()
    if opts.quick:
        opts.iterations, opts.rounds, opts.warmup = 3, 1, 1
    assert torch.backends.mps.is_available(), "Real MPS hardware is required"
    assert F.scaled_dot_product_attention.__module__ == "torch._C._nn", (
        "SDPA was replaced by an auto shim: launch with MTLFLASHATTN_SHIM=off")
    if opts.third_party_supplement:
        if opts.output == Path(".artifacts/operator-comparison/paths.json"):
            opts.output = opts.output.with_name("paths-third-party-supplement.json")
        run_third_party_supplement(opts)
        return
    if opts.supplement:
        if opts.output == Path(".artifacts/operator-comparison/paths.json"):
            opts.output = opts.output.with_name("paths-supplement.json")
        run_supplement(opts)
        return
    from kernels import get_kernel
    import metal_flash_attn as mtl
    hf = get_kernel("kernels-community/metal-flash-sdpa", revision=HF_REVISION)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "torch": torch.__version__, "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "device": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("flash-attn-mps", "mtlflashattn", "kernels")},
        "ours_module": str(Path(ours.__file__).resolve()), "hf_revision": HF_REVISION,
        "dtype": "float16", "heads": [HQ, HK], "dim": DIM, "page_size": PAGE,
        "iterations": opts.iterations, "rounds": opts.rounds, "warmup": opts.warmup,
        "shim": os.environ["MTLFLASHATTN_SHIM"],
        "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "scope": "Complete paged attention adapter; all competitors share ours.store_kvcache",
        "timed": ["KV write", "GPU page metadata", "gather/pad/pack",
                  "dynamic length mask", "required copies", "attention", "output packing"],
        "untimed": ["input generation", "shape-only arange/zero/cuQ",
                    "conservative Flex block-list construction", "warmup/compilation"],
        "cold_note": "First call starts after input synchronization; process-local setup, not OS-cold compilation",
        "memory_note": "Resident counters after the complete case; no peak-memory comparison",
        "reference": "Independent CPU FP64 attention, atol=0.003 rtol=0.01",
        "hf_interop_note": "MPS synchronization after live KV packing is timed; without it this pinned extension aborts with an active command-encoder assertion",
        "cases": [],
    }

    def save():
        opts.output.write_text(json.dumps(report, indent=2) + "\n")

    cases = [("cached-prefill", [576] * 4, 64),
             ("mixed-paged-decode", [4096, 8192] * 8, 1)]
    if opts.case:
        cases = [case for case in cases if case[0] == opts.case]
    with torch.inference_mode():
        for name, lengths, qlen in cases:
            case = make_case(name, lengths, qlen)
            row = {"name": name, "batch": len(lengths), "query_length": qlen,
                   "key_lengths": lengths, "page_order": "randomized",
                   "padding": "NaN in unused tail tokens and sentinel pages",
                   "implementations": {}, "rounds": []}
            report["cases"].append(row)
            candidates = {"ours-paged": native_paged, "sdpa-batched": sdpa_batched,
                          "hf-varlen-gather": build_hf(hf),
                          "mtl-auto-grouped-gather": build_mtl(mtl),
                          "mtl-v1-grouped-gather": build_mtl(mtl)}
            if not opts.no_flex:
                try:
                    fn, evidence = build_flex(case, opts.output.parent)
                    candidates["flex-compiled-gather"] = fn
                    row["flex_evidence"] = evidence
                except Exception:
                    row["implementations"]["flex-compiled-gather"] = {
                        "status": "setup-failed", "error": traceback.format_exc()}
            valid_candidates = {}
            for label, fn in candidates.items():
                try:
                    select_backend(label)
                    if label.startswith("mtl-"):
                        from metal_flash_attn._kernel import _trace
                        _trace.clear()
                        os.environ["MTLFLASHATTN_TRACE"] = "1"
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    actual = fn(case)
                    torch.mps.synchronize()
                    first_ms = (time.perf_counter() - start) * 1000
                    actual = actual.cpu().double()
                    assert torch.isfinite(actual).all(), "nonfinite output with poisoned padding"
                    torch.testing.assert_close(actual, case.expected, atol=3e-3, rtol=1e-2)
                    error = (actual - case.expected).abs()
                    row["implementations"][label] = {
                        "status": "correct", "first_call_ms": first_ms,
                        "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
                    if label.startswith("mtl-"):
                        from metal_flash_attn._kernel import _trace_summary_lines
                        row["implementations"][label]["backend_trace"] = _trace_summary_lines()
                        row["implementations"][label]["kernel_selection"] = os.environ["MTLFLASHATTN_KERNEL"]
                    valid_candidates[label] = fn
                except Exception:
                    row["implementations"][label] = {
                        "status": "failed", "error": traceback.format_exc()}
                finally:
                    os.environ["MTLFLASHATTN_TRACE"] = "0"
                save()
                print(name, label, json.dumps(row["implementations"][label]), flush=True)
            labels = list(valid_candidates)
            if not labels:
                row["timing_status"] = "No implementation passed correctness"
                save()
                continue
            for round_id in range(opts.rounds):
                order = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
                times = {}
                for label in order:
                    select_backend(label)
                    times[label] = measure(valid_candidates[label], case,
                                           opts.warmup, opts.iterations)
                    print(name, round_id + 1, label,
                          json.dumps({k: v for k, v in times[label].items() if k != "samples_ms"}),
                          flush=True)
                row["rounds"].append({"order": order, "timings": times})
                save()
            for label in labels:
                results = [r["timings"][label] for r in row["rounds"]]
                summary = row["implementations"][label]
                summary["median_p50_ms"] = statistics.median(r["p50_ms"] for r in results)
                summary["median_p95_ms"] = statistics.median(r["p95_ms"] for r in results)
                if "sdpa-batched" in labels:
                    ratios = [r["timings"]["sdpa-batched"]["p50_ms"] /
                              r["timings"][label]["p50_ms"] for r in row["rounds"]]
                    summary["per_round_speedup_vs_sdpa"] = ratios
                    summary["median_speedup_vs_sdpa"] = statistics.median(ratios)
            row["allocated_bytes_after_case"] = torch.mps.current_allocated_memory()
            row["driver_bytes_after_case"] = torch.mps.driver_allocated_memory()
            save()
            del case, candidates, valid_candidates
            gc.collect()
            torch.mps.empty_cache()
    report["complete_measurement"] = (not opts.quick and not opts.case and
                                      opts.iterations >= 50 and opts.rounds >= 3 and opts.warmup >= 10)
    save()


if __name__ == "__main__":
    main()
