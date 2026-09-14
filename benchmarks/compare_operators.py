"""Same-device operator API comparison, separate from paged-cache adapters.

Run in an isolated environment with MTLFLASHATTN_SHIM=off. Each implementation
receives the same logical values in token-major storage. Views, fixed masks and
sequence metadata are prepared before timing; copies performed by a public API
itself remain in its latency. No cache writes or page gather are timed here.
"""
import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import flash_attn_mps
import metal_flash_attn
import metal_flash_attn._kernel as mtl_kernel
from kernels import get_kernel

HF_REVISION = '761199956ba9baffbc93e0a3e08933668f06cf7a'


def measure(call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return {'p50_ms': statistics.median(samples),
            'p95_ms': ordered[math.ceil(.95 * len(ordered))-1]}


def make_flex(q, k, v, qlen, klen):
    # No debug/eager backend: fullgraph prevents a silently uncompiled baseline.
    torch._dynamo.reset()
    shift = klen-qlen
    def mask(batch, head, qi, ki):
        return ki <= qi+shift
    block = None if qlen == 1 else create_block_mask(
        mask, B=q.shape[0], H=None, Q_LEN=qlen, KV_LEN=klen, device='mps')
    compiled = torch.compile(flex_attention, backend='inductor', fullgraph=True, dynamic=False)
    return lambda: compiled(q, k, v, block_mask=block, scale=128**-.5,
                            enable_gqa=True).transpose(1, 2)


def reference_samples(q, k, v):
    qlen, klen = q.shape[1], k.shape[1]
    rows = list(range(qlen)) if qlen <= 128 else sorted({
        0, 1, 7, 15, 31, 32, 63, 64, 127, 128, 255, 256, qlen//2, qlen-2, qlen-1})
    query = q[:, rows].cpu().double().transpose(1, 2)
    key, value = k.cpu().double().transpose(1, 2), v.cpu().double().transpose(1, 2)
    mask = torch.arange(klen)[None, :] <= (torch.tensor(rows) + klen-qlen)[:, None]
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                             scale=128**-.5, enable_gqa=True).transpose(1, 2)
    return rows, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--backends', help='Comma-separated subset; omitted runs every backend')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('MTLFLASHATTN_SHIM') == 'off'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1'
    assert torch.__version__ == '2.14.0' and torch.backends.mps.is_available()
    assert version('flash-attn-mps') == '0.1.0'
    os.environ['MTLFLASHATTN_TRACE'] = '0'
    os.environ['MTLFLASHATTN_KERNEL'] = 'auto'
    selected = set(args.backends.split(',')) if args.backends else None
    torch.compiler.set_stance('default')
    torch._dynamo.config.suppress_errors = False
    hf = get_kernel('kernels-community/metal-flash-sdpa', revision=HF_REVISION)
    cases = [('short-prefill', 1, 128, 128), ('prefill-b1-2048', 1, 2048, 2048),
             ('prefill-b4-4096', 4, 4096, 4096), ('decode-b1-8192', 1, 1, 8192),
             ('decode-b16-8192', 16, 1, 8192), ('cached-chunk-b4', 4, 64, 576)]
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
        assert cases, 'Unknown case'
    warmup, iterations, rounds = (2, 5, 1) if args.quick else (10, 50, 3)
    report = {'device': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
              'macos': platform.mac_ver()[0], 'python': platform.python_version(), 'torch': torch.__version__,
              'versions': {p: version(p) for p in ('flash-attn-mps','mtlflashattn','kernels')},
              'hf_revision': HF_REVISION, 'hf_module': str(hf.__file__),
              'input_layout': 'contiguous B,S,H,D token-major; heads-second/packed views prepared outside timing',
              'dtype': 'float16', 'heads': [16,8], 'head_dim': 128,
              'timing': {'warmup':warmup,'iterations':iterations,'rounds':rounds,
                         'scope':'public compute API including internal allocations/copies; fixed input/mask/meta prep excluded; no KV writes or page gather',
                         'cold_note':'first call occurs after input initialization sync; process-local caches may already be warm for later cases; Flex compilation and mask setup are reported separately'},
              'validation': 'Full output finite; CPU FP64 for all queries when Q<=128, otherwise 15 deterministic boundary/interior query rows per batch; atol=.003,rtol=.01',
              'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, batch, qlen, klen in cases:
            torch.manual_seed(123)
            q = torch.randn(batch,qlen,16,128,device='mps',dtype=torch.float16)
            k = torch.randn(batch,klen,8,128,device='mps',dtype=torch.float16)
            v = torch.randn_like(k)
            qs, ks, vs = (x.transpose(1,2) for x in (q,k,v))
            qp, kp, vp = (x.flatten(0,1) for x in (q,k,v))
            cuq = torch.arange(batch+1,device='mps',dtype=torch.int32)*qlen
            cuk = torch.arange(batch+1,device='mps',dtype=torch.int32)*klen
            # Q=1 causal decode sees every supplied key, so avoid an unnecessary mask.
            mask = None if qlen in (1,klen) else (
                torch.arange(klen,device='mps')[None,:] <= torch.arange(klen-qlen,klen,device='mps')[:,None])
            sdpa = lambda: F.scaled_dot_product_attention(qs,ks,vs,attn_mask=mask,
                is_causal=(qlen==klen and qlen>1),scale=128**-.5,enable_gqa=True).transpose(1,2)
            calls = {
                'sdpa': sdpa,
                'ours': lambda: flash_attn_mps.flash_attn_varlen_func(qp,kp,vp,cuq,cuk,qlen,klen,
                    causal=(qlen>1),softmax_scale=128**-.5).reshape_as(q),
                'hf_metal': lambda: hf.flash_attn_varlen_func(qp,kp,vp,cuq,cuk,qlen,klen,
                    causal=(qlen>1),softmax_scale=128**-.5).reshape_as(q),
                'mtl_auto': lambda: metal_flash_attn.flash_attn_func(q,k,v,causal=(qlen>1),softmax_scale=128**-.5),
                'mtl_v1': lambda: metal_flash_attn.flash_attn_func(q,k,v,causal=(qlen>1),softmax_scale=128**-.5),
            }
            rows, ref = reference_samples(q,k,v)
            torch.mps.synchronize()
            flex_setup_ms = None
            if selected is None or 'flex_compiled' in selected:
                before = time.perf_counter()
                calls['flex_compiled'] = make_flex(qs,ks,vs,qlen,klen)
                torch.mps.synchronize()
                flex_setup_ms = (time.perf_counter()-before)*1000
            if selected is not None:
                assert selected <= calls.keys(), 'Unknown backend'
                calls = {label: call for label, call in calls.items() if label in selected}
            row = {'name':name,'batch':batch,'q_len':qlen,'kv_len':klen,
                   'reference_query_rows':rows,'flex_mask_setup_ms':flex_setup_ms,
                   'mtl_effective_kernel':mtl_kernel._effective_kernel_label(qs,ks,vs),
                   'implementations':{}}
            for label, call in list(calls.items()):
                try:
                    os.environ['MTLFLASHATTN_KERNEL'] = 'v1' if label == 'mtl_v1' else 'auto'
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    out = call()
                    torch.mps.synchronize()
                    first = (time.perf_counter()-started)*1000
                    assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
                    assert torch.isfinite(out).all()
                    sampled = out[:, rows].cpu().double()
                    torch.testing.assert_close(sampled,ref,atol=3e-3,rtol=1e-2)
                    row['implementations'][label]={'valid':True,'first_call_ms':first,
                        'max_abs_error':float((sampled-ref).abs().max()),'rounds':[]}
                    if label.startswith('mtl_'):
                        row['implementations'][label]['selected_mtl_kernel'] = mtl_kernel._effective_kernel_label(qs,ks,vs)
                    del out, sampled
                except Exception as exc:
                    row['implementations'][label]={'valid':False,'error':f'{type(exc).__name__}: {str(exc)[:1200]}'}
                    del calls[label]
                    print(name,label,'NOT RANKED',str(exc)[:200],flush=True)
            labels = list(calls)
            for r in range(rounds):
                order = labels[r % len(labels):]+labels[:r % len(labels)]
                for label in order:
                    os.environ['MTLFLASHATTN_KERNEL'] = 'v1' if label == 'mtl_v1' else 'auto'
                    result = measure(calls[label],warmup,iterations)
                    row['implementations'][label]['rounds'].append(result)
                print(name,'round',r+1,{label:row['implementations'][label]['rounds'][-1]['p50_ms'] for label in labels},flush=True)
            for item in row['implementations'].values():
                if item['valid']:
                    item['p50_ms'] = statistics.median(r['p50_ms'] for r in item['rounds'])
                    item['p95_ms'] = statistics.median(r['p95_ms'] for r in item['rounds'])
            report['cases'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            del calls, q,k,v,qs,ks,vs,qp,kp,vp,ref,cuq,cuk,mask
            gc.collect()
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
