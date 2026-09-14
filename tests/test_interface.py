import pytest
import torch

from flash_attn_mps import (
    flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache,
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func,
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func,
)
from reference import attention_reference, assert_attention_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", [(40,40), (128,128), (192,128), (256,256), (512,512)])
def test_ragged_gqa_and_diffkv(device, dtype, dims):
    d, dv = dims
    q = torch.randn(21, 4, d, device=device, dtype=dtype)
    k = torch.randn(25, 2, d, device=device, dtype=dtype)
    v = torch.randn(25, 2, dv, device=device, dtype=dtype)
    cu_q = torch.tensor([0, 1, 4, 21], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 8, 25], device=device, dtype=torch.int32)
    out, lse = flash_attn_varlen_func(q,k,v,cu_q,cu_k,17,17,causal=True,return_softmax_lse=True)
    ref, ref_lse = attention_reference(q,k,v,cu_q,cu_k,causal=True)
    assert_attention_close(out, ref, dtype)
    assert_attention_close(lse, ref_lse, torch.float16)


@pytest.mark.parametrize("sq,sk", [(2,5), (5,2), (0,3)])
def test_dense_lower_right_and_empty(device, sq, sk):
    q = torch.randn(2,sq,4,32,device=device,dtype=torch.float16)
    k = torch.randn(2,sk,2,32,device=device,dtype=torch.float16)
    v = torch.randn_like(k)
    out, lse = flash_attn_func(q,k,v,causal=True,return_softmax_lse=True)
    cuq = torch.arange(3,device=device,dtype=torch.int32)*sq
    cuk = torch.arange(3,device=device,dtype=torch.int32)*sk
    ref, refl = attention_reference(q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),cuq,cuk,causal=True)
    assert_attention_close(out.flatten(0,1),ref)
    assert_attention_close(lse.transpose(0,1).reshape(4,-1),refl)


def test_dense_interleaved_shapes_and_fresh_values(device):
    # Reusing immutable offsets must not retain inputs or a previous batch's
    # boundaries when layers alternate between shapes.
    for batch, sq, sk in ((2,3,5), (1,7,4), (2,3,5), (3,1,17)):
        q = torch.randn(batch,sq,4,32,device=device,dtype=torch.float16)
        k = torch.randn(batch,sk,2,32,device=device,dtype=torch.float16)
        v = torch.randn_like(k)
        out = flash_attn_func(q,k,v,causal=True)
        cuq = torch.arange(batch+1,dtype=torch.int32)*sq
        cuk = torch.arange(batch+1,dtype=torch.int32)*sk
        ref,_ = attention_reference(q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),
                                    cuq,cuk,causal=True)
        assert_attention_close(out.flatten(0,1),ref)


def test_decode_public_validation_precedes_shader_dispatch(device,monkeypatch):
    from flash_attn_mps import _decode
    from flash_attn_mps.vllm import flash_attn_varlen_func as vllm_attention
    q=torch.randn(1,4,32,device=device,dtype=torch.float16)
    k=torch.randn(17,2,32,device=device,dtype=torch.float16)
    cq=torch.tensor([0,1],device=device,dtype=torch.int32)
    ck=torch.tensor([0,17],device=device,dtype=torch.int32)
    base=dict(q=q,k=k,v=k,cu_seqlens_q=cq,cu_seqlens_k=ck,
              max_seqlen_q=1,max_seqlen_k=17)
    def forbidden(*args,**kwargs):
        pytest.fail('invalid public input reached shader preparation')
    monkeypatch.setattr(_decode,'_shader',forbidden)
    for change in (
        dict(q=q.int()), dict(cu_seqlens_q=cq.long()), dict(k=k[...,:16]),
        dict(out=torch.empty_like(q,dtype=torch.int32)),
        dict(k_descale=torch.ones(1,1,device=device)),
        dict(s_aux=torch.ones(2,device=device)), dict(window_size=(-2,0)),
        dict(q=q.expand(2,-1,-1)),
    ):
        with pytest.raises((ValueError,TypeError)):
            vllm_attention(**(base|change))


def test_packed_entrypoints(device):
    qkv = torch.randn(2,7,3,4,32,device=device,dtype=torch.float16)
    expected = flash_attn_func(qkv[:,:,0],qkv[:,:,1],qkv[:,:,2],causal=True)
    torch.testing.assert_close(flash_attn_qkvpacked_func(qkv,causal=True),expected)
    torch.testing.assert_close(flash_attn_kvpacked_func(qkv[:,:,0],qkv[:,:,1:],causal=True),expected)
    packed = qkv.flatten(0,1)
    cu = torch.tensor([0,7,14],device=device,dtype=torch.int32)
    torch.testing.assert_close(flash_attn_varlen_qkvpacked_func(packed,cu,7,causal=True),expected.flatten(0,1))
    torch.testing.assert_close(flash_attn_varlen_kvpacked_func(packed[:,0],packed[:,1:],cu,cu,7,7,causal=True),expected.flatten(0,1))


def test_dense_fp8_noncontiguous_reshape(device):
    qb=torch.randn(2,4,9,32).to(torch.float8_e4m3fn)
    kb=torch.randn(2,2,13,32).to(torch.float8_e4m3fn)
    vb=torch.randn(2,2,13,32).to(torch.float8_e4m3fn)
    q,k,v=(x.to(device).transpose(1,2) for x in (qb,kb,vb))
    out=flash_attn_func(q,k,v,causal=True)
    cuq=torch.tensor([0,9,18],dtype=torch.int32)
    cuk=torch.tensor([0,13,26],dtype=torch.int32)
    ref,_=attention_reference(qb.float().transpose(1,2).flatten(0,1),
        kb.float().transpose(1,2).flatten(0,1),vb.float().transpose(1,2).flatten(0,1),
        cuq,cuk,causal=True)
    assert_attention_close(out.flatten(0,1),ref)


def test_paged_prefix_and_storage_views(device):
    parent = torch.randn(3,9,2,32,128,device=device,dtype=torch.float16)
    k, v = parent[1].transpose(1,2).split(64,-1)
    q = torch.randn(11,4,64,device=device,dtype=torch.float16)
    cuq = torch.tensor([0,1,11],device=device,dtype=torch.int32)
    cuk = torch.tensor([0,65,114],device=device,dtype=torch.int32)
    table = torch.tensor([[4,1,7],[2,6,-1]],device=device,dtype=torch.int32)
    out = flash_attn_varlen_func(q,k,v,cuq,cuk,10,65,causal=True,block_table=table)
    ref,_ = attention_reference(q,k,v,cuq,used_k=cuk[1:]-cuk[:-1],block_table=table,causal=True)
    assert_attention_close(out,ref)


def test_with_cache_append_leftpad_and_batch_index(device):
    kc = torch.randn(4,23,2,32,device=device,dtype=torch.float16)
    vc = torch.randn_like(kc)
    beforek,beforev=kc.clone(),vc.clone()
    q = torch.randn(2,2,4,32,device=device,dtype=torch.float16)
    k = torch.randn(2,2,2,32,device=device,dtype=torch.float16)
    v = torch.randn_like(k)
    lengths = torch.tensor([9,11],device=device,dtype=torch.int32)
    leftpad = torch.tensor([3,5],device=device,dtype=torch.int32)
    indices = torch.tensor([3,1],device=device,dtype=torch.int32)
    out = flash_attn_with_kvcache(q,kc,vc,k,v,cache_seqlens=lengths,
                                cache_batch_idx=indices,cache_leftpad=leftpad,causal=True)
    for b,row in enumerate([3,1]):
        end=[9,11][b]
        beforek[row,end:end+2]=k[b]
        beforev[row,end:end+2]=v[b]
    torch.testing.assert_close(kc,beforek,rtol=0,atol=0)
    torch.testing.assert_close(vc,beforev,rtol=0,atol=0)
    cu=torch.tensor([0,2,4],device=device,dtype=torch.int32)
    ref,_=attention_reference(q.flatten(0,1),kc,vc,cu,used_k=lengths+2-leftpad,
                              block_table=indices[:,None],causal=True,leftpad=leftpad.cpu())
    assert_attention_close(out.flatten(0,1),ref)
    torch.testing.assert_close(lengths,torch.tensor([9,11],device=device,dtype=torch.int32))


@pytest.mark.parametrize("interleaved", [True,False])
def test_rope_append_matches_explicit_rotation(device,interleaved):
    q=torch.randn(1,2,4,32,device=device,dtype=torch.float16)
    k=torch.randn(1,2,2,32,device=device,dtype=torch.float16)
    v=torch.randn_like(k)
    kc=torch.randn(1,16,2,32,device=device,dtype=torch.float16)
    vc=torch.randn_like(kc)
    theta=torch.randn(16,8,device=device,dtype=torch.float16)
    cos,sin=theta.cos(),theta.sin()
    lens=torch.tensor([5],device=device,dtype=torch.int32)
    out=flash_attn_with_kvcache(q,kc,vc,k,v,cos,sin,lens,causal=True,
                               rotary_interleaved=interleaved)
    def rotate(x):
        x=x.cpu().float()
        c,s=cos[5:7].cpu().float()[None,:,None,:],sin[5:7].cpu().float()[None,:,None,:]
        result=x.clone()
        if interleaved:
            a,b=x[...,:16:2],x[...,1:16:2]
            result[...,:16:2]=a*c-b*s
            result[...,1:16:2]=a*s+b*c
        else:
            a,b=x[...,:8],x[...,8:16]
            result[...,:8]=a*c-b*s
            result[...,8:16]=a*s+b*c
        return result.half().to(device)
    torch.testing.assert_close(kc[:,5:7],rotate(k),atol=2e-3,rtol=1e-3)
    cu=torch.tensor([0,2],device=device,dtype=torch.int32)
    table=torch.tensor([[0]],device=device,dtype=torch.int32)
    ref,_=attention_reference(rotate(q).flatten(0,1),kc,vc,cu,used_k=lens+2,block_table=table,causal=True)
    assert_attention_close(out.flatten(0,1),ref)


def test_paged_decode_crosses_page_boundary(device):
    kc=torch.randn(8,16,2,32,device=device,dtype=torch.float16)
    vc=torch.randn_like(kc)
    table=torch.tensor([[5,1,7]],device=device,dtype=torch.int32)
    for old_len in (15,16,17):
        q=torch.randn(1,1,4,32,device=device,dtype=torch.float16)
        k=torch.randn(1,1,2,32,device=device,dtype=torch.float16)
        v=torch.randn_like(k)
        lengths=torch.tensor([old_len],device=device,dtype=torch.int32)
        out=flash_attn_with_kvcache(q,kc,vc,k,v,cache_seqlens=lengths,block_table=table,causal=True)
        page=int(table[0,old_len//16])
        torch.testing.assert_close(kc[page,old_len%16],k[0,0],rtol=0,atol=0)
        ref,_=attention_reference(q.flatten(0,1),kc,vc,
            torch.tensor([0,1],device=device,dtype=torch.int32),
            used_k=lengths+1,block_table=table,causal=True)
        assert_attention_close(out.flatten(0,1),ref)


def test_reject_training_and_cpu():
    q=torch.randn(1,2,2,16)
    with pytest.raises(NotImplementedError,match="dropout"):
        flash_attn_func(q,q,q,dropout_p=.1)
    with pytest.raises(ValueError,match="MPS"):
        flash_attn_func(q,q,q)
