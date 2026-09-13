"""End-to-end attention-call comparison with nano-vLLM's frozen MPS SDPA path.

Both paths include slot writes, metadata handling, and attention. This deliberately
does not time an isolated shader against the entire previous implementation.
"""
import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from flash_attn_mps import flash_attn_varlen_func, flash_attn_with_kvcache, store_kvcache


def sdpa_baseline(q, k, v, kc, vc, context):
    """Frozen behavior of nano-vLLM macos forward_mps before native integration."""
    if kc.numel():
        slots = context.slot_mapping.long()
        valid = slots >= 0
        kc.flatten(0, 1).index_copy_(0, slots[valid], k[valid])
        vc.flatten(0, 1).index_copy_(0, slots[valid], v[valid])
    if context.is_prefill:
        offsets = context.cu_seqlens_q.tolist()
        key_offsets = context.cu_seqlens_k.tolist()
        lengths = [b - a for a, b in zip(key_offsets, key_offsets[1:])]
    else:
        offsets = list(range(q.size(0) + 1))
        lengths = context.context_lens.tolist()
    outputs = []
    for i, length in enumerate(lengths):
        start, end = offsets[i:i + 2]
        query = q[start:end].transpose(0, 1).unsqueeze(0)
        if context.block_tables is not None:
            block_size = kc.size(1)
            blocks = context.block_tables[i, :math.ceil(length / block_size)].long()
            key = kc[blocks].flatten(0, 1)[:length]
            value = vc[blocks].flatten(0, 1)[:length]
        else:
            key, value = k[start:end], v[start:end]
        output = F.scaled_dot_product_attention(
            query, key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(end - start, length), scale=0.08838834764831845,
            enable_gqa=True,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def native(q, k, v, kc, vc, context):
    store_kvcache(k, v, kc, vc, context.slot_mapping)
    if context.is_prefill:
        key, value = (kc, vc) if context.block_tables is not None else (k, v)
        return flash_attn_varlen_func(
            q, key, value, context.cu_seqlens_q, context.cu_seqlens_k,
            context.max_q, context.max_k, softmax_scale=128**-0.5, causal=True,
            block_table=context.block_tables,
        )
    return flash_attn_with_kvcache(
        q.unsqueeze(1), kc, vc, cache_seqlens=context.context_lens,
        block_table=context.block_tables, softmax_scale=128**-0.5, causal=True,
    ).reshape_as(q)


def make_case(batch, qlen, klen, prefill):
    torch.manual_seed(123)
    page = 256
    pages_per_seq = math.ceil(klen / page)
    # Physical page order intentionally differs from request/logical order.
    table_cpu = torch.randperm(batch * pages_per_seq).reshape(batch, pages_per_seq)
    table = table_cpu.to(device='mps', dtype=torch.int32)
    packed = torch.randn(batch * qlen, 32, 128, device='mps', dtype=torch.float16)
    q, k, v = packed.split([16, 8, 8], dim=1)
    kc = torch.randn(batch * pages_per_seq, page, 8, 128, device='mps', dtype=torch.float16)
    vc = torch.randn_like(kc)
    slots = [int(table_cpu[b, pos // page]) * page + pos % page
             for b in range(batch) for pos in range(klen - qlen, klen)]
    ctx = SimpleNamespace(
        is_prefill=prefill, max_q=qlen, max_k=klen,
        slot_mapping=torch.tensor(slots, device='mps', dtype=torch.int32),
        cu_seqlens_q=torch.arange(batch + 1, device='mps', dtype=torch.int32) * qlen,
        cu_seqlens_k=torch.arange(batch + 1, device='mps', dtype=torch.int32) * klen,
        context_lens=torch.full((batch,), klen, device='mps', dtype=torch.int32),
        block_tables=table if (not prefill or klen > qlen) else None,
    )
    return q, k, v, kc, vc, ctx


def measure(fn, args, warmup, iterations):
    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return {'p50_ms': statistics.median(samples), 'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--case', help='Run only a named case')
    parser.add_argument('--quick', action='store_true', help='Diagnostic run; does not constitute the release gate')
    opts = parser.parse_args()
    assert torch.backends.mps.is_available(), 'Real MPS hardware is required'
    cases = [(f'prefill-b{b}-s{s}', b, s, s, True, True) for b in (1, 4) for s in (2048, 4096)]
    cases += [(f'decode-b{b}-k{k}', b, 1, k, False, True) for b in (1, 4, 16) for k in (4096, 8192)]
    cases += [('short-prefill', 1, 128, 128, True, False), ('short-decode', 1, 1, 256, False, False),
              ('cached-prefill', 4, 64, 576, True, False)]
    if opts.case:
        cases = [case for case in cases if case[0] == opts.case]
        assert cases, 'Unknown case'
    if opts.quick:
        opts.iterations, opts.rounds, opts.warmup = 5, 1, 2
    report = {
        'torch': torch.__version__, 'python': platform.python_version(), 'macos': platform.mac_ver()[0],
        'device': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'dtype': 'float16', 'heads': [16, 8], 'head_dim': 128, 'page_size': 256,
        'iterations': opts.iterations, 'rounds': opts.rounds, 'warmup': opts.warmup,
        'baseline': 'nano-vllm macos SDPA, frozen before native library integration',
        'memory_note': 'Allocated and driver memory are resident counters after execution, not peak allocation.',
        'cases': [],
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, batch, qlen, klen, prefill, gate in cases:
            args = make_case(batch, qlen, klen, prefill)
            start = time.perf_counter()
            actual = native(*args)
            torch.mps.synchronize()
            first_ms = (time.perf_counter() - start) * 1000
            expected = sdpa_baseline(*args)
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=1e-2)
            row = {'name': name, 'release_gate': gate, 'first_call_ms': first_ms,
                   'max_abs_error': float((actual - expected).abs().max()), 'rounds': []}
            del actual, expected
            for round_id in range(opts.rounds):
                # Alternate order to reduce thermal/order bias.
                order = [('sdpa', sdpa_baseline), ('native', native)]
                if round_id % 2:
                    order.reverse()
                times = {label: measure(fn, args, opts.warmup, opts.iterations) for label, fn in order}
                times['speedup'] = times['sdpa']['p50_ms'] / times['native']['p50_ms']
                row['rounds'].append(times)
                print(name, round_id + 1, json.dumps(times), flush=True)
            row['median_speedup'] = statistics.median(r['speedup'] for r in row['rounds'])
            row['passes_speed_gate'] = row['median_speedup'] >= 1.1 and all(r['speedup'] > 1 for r in row['rounds'])
            row['allocated_bytes'] = torch.mps.current_allocated_memory()
            row['driver_bytes'] = torch.mps.driver_allocated_memory()
            report['cases'].append(row)
            opts.output.write_text(json.dumps(report, indent=2) + '\n')
            del args
            gc.collect()
            torch.mps.empty_cache()
    report['complete_release_measurement'] = not opts.quick and not opts.case and opts.rounds >= 3 and opts.iterations >= 50 and opts.warmup >= 10
    report['release_gate_passed'] = report['complete_release_measurement'] and all(r['passes_speed_gate'] for r in report['cases'] if r['release_gate'])
    opts.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
