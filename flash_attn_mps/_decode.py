"""Single-token attention with grouped KV reads and native split reduction."""
from functools import lru_cache
from pathlib import Path

import torch

_META = ('T HQ GQA SPAN Q0 Q1 Q2 K0 K1 K2 K3 V0 V1 V2 V3 O0 O1 O2 '
         'CQ CK USED BT0 BT1 LEFT QSC0 QSC1 KSC0 KSC1 VSC0 VSC1 SINK0 WINDOW_LEFT PARTS').split()
_DTYPES = {torch.float16:'half',torch.bfloat16:'bfloat',torch.float32:'float',
           torch.float8_e4m3fn:'uchar'}
_FEATURES = ('Q_SCALE','K_SCALE','V_SCALE','SINK','WINDOW','CAP')
_SIMD_GROUPS = 8


@lru_cache(maxsize=96)
def _shader(dtypes, out_dtype, dq, dv, page, group, partitioned, unit_d, has_cuk, has_used, has_left, features):
    source=Path(__file__).with_name('kernels').joinpath('attention_decode.metal').read_text()
    definitions={
        'DQ':dq,'DV':dv,'PAGE_SIZE':max(1,page),'PAGED':int(page>0),
        'Q_GROUP':group,'SG_COUNT':_SIMD_GROUPS,'PARTITIONED':int(partitioned),
        'UNIT_D':int(unit_d),'HAS_CUK':int(has_cuk),'HAS_USED':int(has_used),
        'HAS_LEFT':int(has_left),
    }
    definitions.update((f'HAS_{name}',int(enabled)) for name,enabled in zip(_FEATURES,features))
    prefix='#include <metal_stdlib>\nusing namespace metal;\n'
    prefix+='enum { '+', '.join(_META)+' };\n'
    prefix+=''.join(f'using {name}={_DTYPES[dtype]};\n'
                    for name,dtype in zip(('Query','Key','Value'),dtypes))
    prefix+=f'using Out={_DTYPES[out_dtype]};\n'
    prefix+=f'using Partial={"float" if partitioned else _DTYPES[out_dtype]};\n'
    prefix+='\n'.join(f'constant constexpr int {name}={value};' for name,value in definitions.items())+'\n'
    return torch.mps.compile_shader(prefix+source)


@lru_cache(maxsize=1)
def _dummy():
    return torch.empty(1,dtype=torch.int32,device='mps')


def decode_attention(q,k,v,*,cu_seqlens_q,cu_seqlens_k=None,seqused_k=None,
                     max_seqlen_k,block_table=None,softmax_scale=None,out=None,
                     leftpad_k=None,num_splits=0,s_aux=None,q_descale=None,
                     k_descale=None,v_descale=None,window_size=(-1,-1),softcap=0.):
    """Attend one query per nonempty request, returning output and natural LSE.

    The caller supplies valid GPU sequence metadata, with cuQ differences in
    {0,1}. K/V are packed or paged, and head dimensions lie in 1..512. FP8
    descales, sinks, Q1 windows and score softcapping execute inside the kernel.
    Public validation belongs to interface._run and the attention dispatcher.
    """
    tokens,heads,dq=q.shape
    kvheads,dv=k.shape[-2],v.shape[-1]
    batch=cu_seqlens_q.numel()-1
    if tokens>batch:
        raise ValueError('decode accepts at most one query per request')
    out_dtype=torch.float16 if q.dtype==torch.float8_e4m3fn else q.dtype
    if out is None:out=torch.empty((tokens,heads,dv),dtype=out_dtype,device=q.device)
    lse=torch.empty((heads,tokens),dtype=torch.float32,device=q.device)
    if not tokens:return out,lse
    # Pair grouped queries without exceeding the threadgroup memory budget.
    ratio=heads//kvheads
    group=2 if ratio%2==0 and dv<=256 else 1
    effective_max=min(max_seqlen_k,window_size[0]+1) if window_size[0]>=0 else max_seqlen_k
    splits=num_splits or max(1,min(32,(effective_max+255)//256))
    span=((effective_max+splits-1)//splits+7)//8*8
    page=k.shape[1] if block_table is not None else 0
    unit_d=q.stride(-1)==k.stride(-1)==v.stride(-1)==1
    features=(q_descale is not None,k_descale is not None,v_descale is not None,
              s_aux is not None,window_size[0]>=0,softcap>0)
    lib=_shader((q.dtype,k.dtype,v.dtype),out.dtype,dq,dv,page,group,splits>1,unit_d,
                cu_seqlens_k is not None,seqused_k is not None,leftpad_k is not None,features)
    def integer(tensor):
        return _dummy() if tensor is None else tensor
    cq,ck,used,table=map(integer,(cu_seqlens_q,cu_seqlens_k,seqused_k,block_table))
    left=_dummy() if leftpad_k is None else leftpad_k.to(device=q.device,dtype=torch.int32)
    qs,ks,vs=(_dummy() if tensor is None else tensor for tensor in (q_descale,k_descale,v_descale))
    sinks=_dummy() if s_aux is None else s_aux.float()
    params=dict.fromkeys(_META,0)
    params.update(T=tokens,HQ=heads,GQA=ratio,SPAN=span,WINDOW_LEFT=window_size[0],PARTS=splits)
    for tensor,names in ((q,('Q0','Q1','Q2')),
        (k,('K0','K1','K2','K3') if page else ('K1','K2','K3')),
        (v,('V0','V1','V2','V3') if page else ('V1','V2','V3')),
        (out,('O0','O1','O2')),(cq,('CQ',)),(ck,('CK',)),(used,('USED',)),
        (table,('BT0','BT1')),(left,('LEFT',)),(qs,('QSC0','QSC1')),
        (ks,('KSC0','KSC1')),(vs,('VSC0','VSC1')),(sinks,('SINK0',))):
        params.update(zip(names,tensor.stride()))
    parameters=tuple(params[name] for name in _META)
    if splits>1:
        destination=torch.empty((tokens,heads,splits,dv),dtype=torch.float32,device=q.device)
        partial_lse=torch.empty((tokens,heads,splits),dtype=torch.float32,device=q.device)
    else:destination,partial_lse=out,lse
    lib.decode_partitions(q,k,v,destination,partial_lse,parameters,
        (float(softmax_scale if softmax_scale is not None else dq**-.5),float(softcap)),
        cq,ck,used,table,left,qs,ks,vs,sinks,
        threads=(heads//group*32,batch*_SIMD_GROUPS,splits),group_size=(32,_SIMD_GROUPS,1))
    if splits>1:
        lib.decode_merge(destination,partial_lse,out,lse,parameters,
                         threads=tokens*heads*32,group_size=32)
    return out,lse
