"""Native Metal attention dispatch; no host request loop or dense score tensor."""

from functools import lru_cache
from pathlib import Path

import torch


# This list generates the Metal enum as well as the host parameter buffer.
_META = (
    'T HQ HK DQ DV PAGE PAGED CUK USED CAUSAL DYNAMIC WIN_L WIN_R '
    'ALIBI SINK QDS KDS VDS MASK MASK_WIN CLAMP SPLITS MAX_Q OSPLIT '
    'QS0 QS1 QS2 KS0 KS1 KS2 KS3 VS0 VS1 VS2 VS3 OS0 OS1 OS2 '
    'BT0 BT1 CQ0 CK0 U0 CA0 A0 A1 SI0 QD0 QD1 KD0 KD1 VD0 VD1 '
    'MR0 MR1 MP0 LEFTPAD LP0'
).split()
_TYPES = {
    torch.float16: 'half', torch.bfloat16: 'bfloat', torch.float32: 'float',
    torch.float8_e4m3fn: 'uchar',
}
# Kernel variants depend on features, never sequence lengths or tensor values.
_STATIC_META = 'PAGED CUK USED CAUSAL DYNAMIC ALIBI SINK QDS KDS VDS MASK CLAMP LEFTPAD'.split()


@lru_cache(maxsize=96)
def _library(qdtype, kdtype, vdtype, odtype, dq, dv, specialization, softcap):
    root = Path(__file__).parent / 'kernels'
    # Keep common FP16/BF16 tiles in their native type to preserve occupancy.
    # Mixed precision uses FP32 staging so values retain their exponent range.
    bq, bk, wm = (32, 16, 4) if max(dq, dv) <= 128 else (16, 8, 2)
    bd, bv = (dq + 7) // 8 * 8, (dv + 7) // 8 * 8
    shared = _TYPES[qdtype] if qdtype == kdtype == vdtype else 'float'
    if shared == 'uchar':
        shared = 'half'
    header = '\n'.join((
        'enum { ' + ', '.join(_META) + ' };',
        f'using QType = {_TYPES[qdtype]}; using KType = {_TYPES[kdtype]};',
        f'using VType = {_TYPES[vdtype]}; using OType = {_TYPES[odtype]};',
        f'using SharedType = {shared};',
        f'constant constexpr int BQ={bq}, BK={bk}, WM={wm}, BD={bd}, BV={bv};',
    ))
    source = (root / 'vendor' / 'steel.metal').read_text() + header
    body = (root / 'attention.metal').read_text()
    if max(dq, dv) > 256:
        start = body.index('kernel void attention_tiled(')
        end = body.index('// SIMD-vector online softmax')
        body = body[:start] + body[end:]
    for name, value in (('DQ', dq), ('DV', dv), *specialization):
        body = body.replace(f'p[{name}]', str(value))
    if not softcap:
        body = body.replace('f[1]', '0.0f')
    return torch.mps.compile_shader(source + body), bq, wm


@lru_cache(maxsize=32)
def _fast_library(dtype, dim, causal, softcap=False):
    root = Path(__file__).parent / 'kernels'
    bq, bk, wm = (32, 16, 4) if dim <= 128 else (16, 8, 2)
    body = (root / 'attention_fast.metal').read_text()
    for key, value in dict(TYPE=_TYPES[dtype], D=dim, BQ=bq, BK=bk, WM=wm,
                           CAUSAL='true' if causal else 'false',
                           SOFTCAP='true' if softcap else 'false').items():
        body = body.replace(f'__{key}__', str(value))
    source = (root / 'vendor' / 'steel.metal').read_text()
    source += 'enum { ' + ', '.join(_META) + ' };\n' + body
    return torch.mps.compile_shader(source), bq, wm


@lru_cache(maxsize=1)
def _dummy():
    # Unused arguments remain bound to a valid buffer; shader flags erase reads.
    return torch.zeros(2, dtype=torch.int32, device='mps')


def _float_arg(tensor, device):
    return _dummy() if tensor is None else tensor.to(device=device, dtype=torch.float32)


def _int_arg(tensor, device):
    return _dummy() if tensor is None else tensor.to(device=device, dtype=torch.int32)


def _strides(values, tensor, names):
    if tensor is not None:
        values.update(zip(names, tensor.stride()))


def attention(
    q, k, v, *, cu_seqlens_q, cu_seqlens_k=None, seqused_k=None,
    max_seqlen_q, max_seqlen_k, block_table=None, softmax_scale=None,
    causal=False, window_size=(-1, -1), alibi_slopes=None, softcap=0.,
    s_aux=None, q_descale=None, k_descale=None, v_descale=None, out=None,
    num_splits=0, mask_mod=None, aux_tensors=None, leftpad_k=None, deterministic=False,
):
    """Return packed output and natural-log LSE in ``[heads, total_q]`` order.

    All sequence metadata is consumed by the GPU. Python dispatch depends only
    on shapes and the caller's existing maximum lengths, never tensor values.
    """
    if q.device.type != 'mps' or k.device != q.device or v.device != q.device:
        raise ValueError('q, k and v must be on the same MPS device')
    if q.ndim != 3 or k.ndim not in (3, 4) or v.ndim != k.ndim:
        raise ValueError('q must be packed 3-D; k/v must be packed 3-D or paged 4-D')
    if any(t.dtype not in _TYPES for t in (q, k, v)):
        raise TypeError('attention supports FP16, BF16, FP32 and FP8 E4M3FN')
    tokens, heads, dq = q.shape
    kh, dv = k.shape[-2], v.shape[-1]
    if not (1 <= dq <= 512 and 1 <= dv <= 512):
        raise ValueError('query/key and value head dimensions must be in 1..512')
    if k.shape[-1] != dq or v.shape[:-1] != k.shape[:-1]:
        raise ValueError('key/value shapes must agree except for their last dimension')
    if heads < 1 or kh < 1 or heads % kh:
        raise ValueError('query heads must be a positive multiple of KV heads')
    if (k.ndim == 4) != (block_table is not None):
        raise ValueError('paged 4-D KV requires block_table; packed 3-D KV does not')
    if cu_seqlens_k is None and seqused_k is None:
        raise ValueError('cu_seqlens_k or seqused_k is required')
    if num_splits < 0:
        raise ValueError('num_splits must be nonnegative')
    default_dtype = torch.float16 if q.dtype == torch.float8_e4m3fn else q.dtype
    if out is None:
        out = torch.empty((tokens, heads, dv), dtype=default_dtype, device=q.device)
    elif out.shape != (tokens, heads, dv) or out.device != q.device:
        raise ValueError('out must have shape [total_q, query_heads, value_dim] on MPS')
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('out must have FP16, BF16 or FP32 dtype')
    if tokens == 0:
        return out, torch.empty((heads, 0), device=q.device, dtype=torch.float32)
    if max_seqlen_q <= 0 or max_seqlen_k < 0:
        raise ValueError('nonempty q requires positive max_seqlen_q and nonnegative max_seqlen_k')
    batch = cu_seqlens_q.numel() - 1
    if batch < 1:
        raise ValueError('cu_seqlens_q must contain at least two offsets')
    if (max_seqlen_q == 1 and not deterministic
            and alibi_slopes is None and mask_mod is None
            and (isinstance(causal, bool) or isinstance(causal, torch.Tensor))):
        # A single right-aligned query has no future keys, for either causal
        # value. Partition its visible KV instead of padding 31 query rows.
        from ._decode import decode_attention
        return decode_attention(
            q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
            block_table=block_table, softmax_scale=softmax_scale, out=out,
            leftpad_k=leftpad_k, num_splits=num_splits,
            s_aux=s_aux, q_descale=q_descale, k_descale=k_descale,
            v_descale=v_descale, window_size=window_size, softcap=softcap,
        )
    lse = torch.empty((heads, tokens), device=q.device, dtype=torch.float32)
    fast = (not deterministic and num_splits in (0, 1)
            and q.dtype == k.dtype == v.dtype == out.dtype
            and q.dtype != torch.float8_e4m3fn and dq == dv
            and dq in (32,64,72,80,96,128,256) and not isinstance(causal, torch.Tensor)
            and window_size[0] == window_size[1] == -1
            and alibi_slopes is None and s_aux is None
            and q_descale is None and k_descale is None and v_descale is None
            and mask_mod is None and leftpad_k is None
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
            and (k.ndim == 3 or k.shape[1] % (8 if dq == 256 else 16) == 0))
    vector = not fast and (deterministic or max_seqlen_q == 1 or max(dq, dv) > 256)
    splits = 1 if deterministic else num_splits or (
        min(32, max(1, (max_seqlen_k + 255)//256)) if vector else 1)
    if splits > 1:
        target = torch.empty((splits, tokens, heads, dv), device=q.device, dtype=torch.float32)
        target_lse = torch.empty((splits, heads, tokens), device=q.device, dtype=torch.float32)
        output_view = target[0]
    else:
        target, target_lse, output_view = out, lse, out
    cq = _int_arg(cu_seqlens_q, q.device)
    ck = _int_arg(cu_seqlens_k, q.device)
    used = _int_arg(seqused_k, q.device)
    table = _int_arg(block_table, q.device)
    lp = _int_arg(leftpad_k, q.device)
    dynamic = isinstance(causal, torch.Tensor)
    ca = causal.to(device=q.device, dtype=torch.bool) if dynamic else _dummy()
    alibi = _float_arg(alibi_slopes, q.device)
    sink = _float_arg(s_aux, q.device)
    qds, kds, vds = (_float_arg(x, q.device) for x in (q_descale, k_descale, v_descale))
    kind = getattr(mask_mod, 'kind', 0)
    aux_tensors = aux_tensors or ()
    ranges = _int_arg(aux_tensors[0], q.device) if kind == 1 else _dummy()
    prefix = _int_arg(aux_tensors[0], q.device) if kind == 2 else _dummy()
    window = _int_arg(aux_tensors[1], q.device) if kind == 2 else _dummy()
    params = dict.fromkeys(_META, 0)
    params.update(T=tokens, HQ=heads, HK=kh, DQ=dq, DV=dv,
                  PAGE=k.shape[1] if k.ndim == 4 else 0, PAGED=int(k.ndim == 4),
                  CUK=int(cu_seqlens_k is not None), USED=int(seqused_k is not None),
                  CAUSAL=0 if dynamic else int(causal), DYNAMIC=int(dynamic),
                  WIN_L=window_size[0], WIN_R=window_size[1], ALIBI=int(alibi_slopes is not None),
                  SINK=int(s_aux is not None), QDS=int(q_descale is not None),
                  KDS=int(k_descale is not None), VDS=int(v_descale is not None),
                  MASK=kind, MASK_WIN=getattr(mask_mod, 'sliding_window_left', -1),
                  CLAMP=int(getattr(mask_mod, 'clamp_prefix_to_window', False)),
                  SPLITS=splits, MAX_Q=max_seqlen_q, OSPLIT=target.stride(0) if splits > 1 else 0,
                  LEFTPAD=int(leftpad_k is not None))
    _strides(params, q, ('QS0', 'QS1', 'QS2'))
    _strides(params, k, ('KS0','KS1','KS2','KS3') if k.ndim == 4 else ('KS1','KS2','KS3'))
    _strides(params, v, ('VS0','VS1','VS2','VS3') if v.ndim == 4 else ('VS1','VS2','VS3'))
    _strides(params, output_view, ('OS0','OS1','OS2'))
    _strides(params, table, ('BT0','BT1'))
    for tensor, name in ((cq,'CQ0'),(ck,'CK0'),(used,'U0'),(ca,'CA0'),(sink,'SI0'),(lp,'LP0')):
        _strides(params, tensor, (name,))
    if alibi_slopes is not None:
        if alibi.ndim == 1:
            params['A1'] = alibi.stride(0)
        else:
            _strides(params, alibi, ('A0','A1'))
    for original, tensor, names in ((q_descale,qds,('QD0','QD1')),
                                    (k_descale,kds,('KD0','KD1')),
                                    (v_descale,vds,('VD0','VD1'))):
        if original is not None and tensor.ndim:
            _strides(params, tensor, names if tensor.ndim == 2 else (names[1],))
    _strides(params, ranges, ('MR0','MR1'))
    _strides(params, prefix, ('MP0',))
    if fast:
        lib,bq,wm = _fast_library(q.dtype,dq,bool(causal),softcap > 0)
        lib.fast_attention(q,k,v,out,tuple(params[name] for name in _META),
                           (float(softmax_scale if softmax_scale is not None else dq**-.5),float(softcap)),
                           _dummy(),cq,ck,lse,table,used,
                           threads=(((max_seqlen_q+bq-1)//bq)*32,heads*wm,batch),
                           group_size=(32,wm,1))
        return out,lse
    specialization = tuple((name, params[name]) for name in _STATIC_META)
    specialization += tuple((name, -1) for name in ('WIN_L', 'WIN_R') if params[name] < 0)
    lib, bq, wm = _library(q.dtype, k.dtype, v.dtype, target.dtype, dq, dv,
                           specialization, softcap > 0)
    args = (q,k,v,target,target_lse,tuple(params[name] for name in _META),
            (float(softmax_scale if softmax_scale is not None else dq**-0.5),float(softcap)),
            cq,ck,used,table,ca,alibi,sink,qds,kds,vds,ranges,prefix,window,lp)
    if vector:
        lib.attention_vector(*args, threads=(max_seqlen_q*32,heads,batch*splits), group_size=(32,1,1))
    else:
        lib.attention_tiled(*args, threads=(((max_seqlen_q+bq-1)//bq)*32,heads*wm,batch*splits),
                            group_size=(32,wm,1))
    if splits > 1:
        from ._merge import _merge_partials
        merged, lse = _merge_partials(target, target_lse)
        out.copy_(merged)
    return out, lse
