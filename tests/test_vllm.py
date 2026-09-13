import pytest
import torch

from flash_attn_mps.vllm import flash_attn_varlen_func, mm_prefix_mask, rswa_mask, merge_attn_states
from reference import attention_reference, assert_attention_close


def inputs(device, qlens=(1,4,17), klens=(65,100,17), d=64, dv=32):
    q=torch.randn(sum(qlens),4,d,device=device,dtype=torch.float16)
    k=torch.randn(sum(klens),2,d,device=device,dtype=torch.float16)
    v=torch.randn(sum(klens),2,dv,device=device,dtype=torch.float16)
    cuq=torch.tensor([0,*torch.tensor(qlens).cumsum(0).tolist()],device=device,dtype=torch.int32)
    cuk=torch.tensor([0,*torch.tensor(klens).cumsum(0).tolist()],device=device,dtype=torch.int32)
    kw=dict(cu_seqlens_q=cuq,cu_seqlens_k=cuk,max_seqlen_q=max(qlens),max_seqlen_k=max(klens))
    return q,k,v,kw


@pytest.mark.parametrize("causal", [True,False,"mixed"])
def test_scores_sinks_out_and_dynamic_causal(device,causal):
    q,k,v,kw=inputs(device)
    dynamic=torch.tensor([True,False,True],device=device) if causal=="mixed" else None
    slopes=torch.tensor([.1,.2,.3,.4],device=device)
    sinks=torch.tensor([-1.,0.,1.,2.],device=device)
    storage=torch.full((q.shape[0]+5,4,32),123.,device=device,dtype=q.dtype)
    out=storage[:q.shape[0]]
    actual,lse=flash_attn_varlen_func(q,k,v,**kw,causal=True if dynamic is not None else causal,
        dynamic_causal=dynamic,window_size=[15,15],softcap=2.,alibi_slopes=slopes,
        s_aux=sinks,out=out,return_softmax_lse=True,fa_version=4)
    ref,refl=attention_reference(q,k,v,kw['cu_seqlens_q'],kw['cu_seqlens_k'],
        causal=dynamic if dynamic is not None else causal,window=(15,15),softcap=2.,alibi=slopes,sinks=sinks)
    assert actual.data_ptr()==out.data_ptr()
    assert_attention_close(actual,ref)
    assert_attention_close(lse,refl)
    assert torch.all(storage[q.shape[0]:]==123)


@pytest.mark.parametrize("kind", ["mm","mm_window","mm_clamp","rswa"])
def test_vllm_masks(device,kind):
    q,k,v,kw=inputs(device,(96,),(257,))
    table=torch.tensor([[5,1,7,3,0]],device=device,dtype=torch.int32)
    paged_k=torch.full((8,64,2,64),float('nan'),device=device,dtype=k.dtype)
    paged_v=torch.full((8,64,2,32),float('nan'),device=device,dtype=v.dtype)
    pos=torch.arange(257,device=device)
    pages=table[0,pos//64].long()
    paged_k[pages,pos%64]=k
    paged_v[pages,pos%64]=v
    k,v=paged_k,paged_v
    del kw['cu_seqlens_k']
    kw['seqused_k']=torch.tensor([257],device=device,dtype=torch.int32)
    kw['block_table']=table
    ranges=torch.full((96,2),-1,device=device,dtype=torch.int32)
    ranges[31:61]=torch.tensor([192,221],device=device,dtype=torch.int32)
    if kind=="rswa":
        mask=rswa_mask()
        aux=[torch.tensor([64],device=device,dtype=torch.int32),torch.tensor([16],device=device,dtype=torch.int32)]
    else:
        mask=mm_prefix_mask(None if kind=="mm" else 16,kind=="mm_clamp")
        aux=[ranges,kw['cu_seqlens_q']]
    actual,lse=flash_attn_varlen_func(q,k,v,**kw,mask_mod=mask,aux_tensors=aux,
                                    return_softmax_lse=True,fa_version=4)
    ref,refl=attention_reference(q,k,v,kw['cu_seqlens_q'],used_k=kw['seqused_k'],
                                block_table=table,mask=mask,aux=aux)
    assert_attention_close(actual,ref)
    assert_attention_close(lse,refl)


def test_fp8_descales_with_broadcast_strides(device):
    q,k,v,kw=inputs(device,(3,7),(19,23),d=32,dv=64)
    q,k,v=(t.cpu().to(torch.float8_e4m3fn).to(device) for t in (q,k,v))
    scales=[torch.tensor([[.8,1.2]],device=device).expand(2,2),
            torch.tensor([[.3],[.7]],device=device).expand(2,2),
            torch.tensor([[.9,1.1]],device=device).expand(2,2)]
    actual,lse=flash_attn_varlen_func(q,k,v,**kw,q_descale=scales[0],k_descale=scales[1],
        v_descale=scales[2],return_softmax_lse=True,causal=True,fa_version=3)
    ref,refl=attention_reference(q,k,v,kw['cu_seqlens_q'],kw['cu_seqlens_k'],causal=True,
        q_descale=scales[0],k_descale=scales[1],v_descale=scales[2])
    assert actual.dtype==torch.float16
    assert_attention_close(actual,ref)
    assert_attention_close(lse,refl)


def test_cascade_and_dcp_lse_merge(device):
    q,k,v,kw=inputs(device,(4,),(129,),d=32,dv=64)
    full,full_lse=flash_attn_varlen_func(q,k,v,**kw,return_softmax_lse=True)
    parts=[]
    for start,end in [(0,64),(64,64),(64,129)]:
        cu=torch.tensor([0,end-start],device=device,dtype=torch.int32)
        part=flash_attn_varlen_func(q,k[start:end],v[start:end],
            max_seqlen_q=4,cu_seqlens_q=kw['cu_seqlens_q'],max_seqlen_k=end-start,
            cu_seqlens_k=cu,return_softmax_lse=True)
        parts.append(part)
    output=torch.empty_like(full)
    output_lse=torch.empty_like(full_lse)
    merge_attn_states(output,*parts[0],*parts[1],output_lse=output_lse)
    first,first_lse=output.clone(),output_lse.clone()
    merge_attn_states(output,first,first_lse,*parts[2],output_lse=output_lse)
    torch.testing.assert_close(output,full,atol=2e-3,rtol=1e-2)
    torch.testing.assert_close(output_lse,full_lse,atol=2e-3,rtol=1e-2)


@pytest.mark.parametrize("splits", [1,2,4])
def test_splitkv_equivalence(device,splits):
    q,k,v,kw=inputs(device,(1,4),(1025,513),d=64,dv=64)
    actual,lse=flash_attn_varlen_func(q,k,v,**kw,causal=True,num_splits=splits,return_softmax_lse=True)
    ref,refl=attention_reference(q,k,v,kw['cu_seqlens_q'],kw['cu_seqlens_k'],causal=True)
    assert_attention_close(actual,ref)
    assert_attention_close(lse,refl)


@pytest.mark.parametrize("page", [16,32,64,128,256])
def test_fp8_paged_mixed_dtype_and_block_sizes(device,page):
    # The split K/V cache has nontrivial token/head strides and V offset.
    raw=torch.randn(7,2,page,128).to(torch.float8_e4m3fn).to(device)
    k,v=raw.transpose(1,2).split(64,-1)
    q=torch.randn(5,4,64,device=device,dtype=torch.bfloat16)
    cu=torch.tensor([0,1,5],device=device,dtype=torch.int32)
    used=torch.tensor([page+1,2*page-1],device=device,dtype=torch.int32)
    table=torch.tensor([[4,1],[5,2]],device=device,dtype=torch.int32)
    kd=torch.tensor([[.7,1.2]],device=device).expand(2,2)
    vd=torch.tensor([[.9],[1.1]],device=device).expand(2,2)
    result,lse=flash_attn_varlen_func(q,k,v,max_seqlen_q=4,cu_seqlens_q=cu,
        max_seqlen_k=2*page,seqused_k=used,block_table=table,causal=True,
        k_descale=kd,v_descale=vd,return_softmax_lse=True,fa_version=4)
    ref,refl=attention_reference(q,k,v,cu,used_k=used,block_table=table,
                                 causal=True,k_descale=kd,v_descale=vd)
    assert_attention_close(result,ref,torch.bfloat16)
    assert_attention_close(lse,refl)


def test_sink_counted_once_when_splitting_or_merging(device):
    q,k,v,kw=inputs(device,(1,),(513,),d=32,dv=32)
    sink=torch.tensor([1.,2.,3.,4.],device=device)
    actual,lse=flash_attn_varlen_func(q,k,v,**kw,s_aux=sink,num_splits=4,return_softmax_lse=True)
    ref,refl=attention_reference(q,k,v,kw['cu_seqlens_q'],kw['cu_seqlens_k'],sinks=sink)
    assert_attention_close(actual,ref)
    assert_attention_close(lse,refl)
    prefix=flash_attn_varlen_func(q,k[:256],v[:256],max_seqlen_q=1,
        cu_seqlens_q=kw['cu_seqlens_q'],max_seqlen_k=256,
        cu_seqlens_k=torch.tensor([0,256],device=device,dtype=torch.int32),
        s_aux=sink,return_softmax_lse=True)
    suffix=flash_attn_varlen_func(q,k[256:],v[256:],max_seqlen_q=1,
        cu_seqlens_q=kw['cu_seqlens_q'],max_seqlen_k=257,
        cu_seqlens_k=torch.tensor([0,257],device=device,dtype=torch.int32),
        return_softmax_lse=True)
    out=torch.empty_like(actual)
    out_lse=torch.empty_like(lse)
    merge_attn_states(out,*prefix,*suffix,output_lse=out_lse)
    assert_attention_close(out,ref)
    assert_attention_close(out_lse,refl)


def test_num_splits_one_is_batch_invariant(device):
    q,k,v,kw=inputs(device,(1,17),(513,529),d=64,dv=64)
    batched=flash_attn_varlen_func(q,k,v,**kw,causal=True,num_splits=1)
    single=flash_attn_varlen_func(q[:1],k[:513],v[:513],max_seqlen_q=1,
        cu_seqlens_q=torch.tensor([0,1],device=device,dtype=torch.int32),
        max_seqlen_k=513,cu_seqlens_k=torch.tensor([0,513],device=device,dtype=torch.int32),
        causal=True,num_splits=1)
    torch.testing.assert_close(single,batched[:1],atol=0,rtol=0)
