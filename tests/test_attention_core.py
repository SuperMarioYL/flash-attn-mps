"""Core contracts that cannot be inferred from a dense wrapper smoke test."""

import pytest
import torch

from flash_attn_mps._attention import attention
from reference import attention_reference, assert_attention_close


@pytest.mark.parametrize('dims', [(1,1), (7,19), (57,99), (255,129), (257,33), (1,511)])
def test_arbitrary_head_dimensions_and_nonunit_strides(device, dims):
    dq, dv = dims
    qbase = torch.randn(15, 4, dq*2, device=device)
    kbase = torch.randn(19, 2, dq*2, device=device)
    vbase = torch.randn(19, 2, dv*2, device=device)
    q, k, v = qbase[2:13,:,::2], kbase[1:18,:,::2], vbase[1:18,:,::2]
    cuq = torch.tensor([0,0,4,11], dtype=torch.int32, device=device)
    cuk = torch.tensor([0,3,8,17], dtype=torch.int32, device=device)
    storage = torch.full((15,4,dv*2),123.,device=device)
    out = storage[2:13,:,::2]
    actual, lse = attention(q,k,v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                           max_seqlen_q=7,max_seqlen_k=9,causal=True,out=out)
    ref, refl = attention_reference(q,k,v,cuq,cuk,causal=True)
    assert_attention_close(actual,ref,torch.float32)
    assert_attention_close(lse,refl,torch.float32)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.all(storage[:2] == 123) and torch.all(storage[13:] == 123)
    assert torch.all(storage[:,:,1::2] == 123)


@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('splits', [0,1,3])
def test_paged_direct_cumulative_lengths_and_leftpad(device,dtype,splits):
    # This calls the core with cuK directly: physical cache offsets must never
    # include the cumulative length of a preceding request.
    parent = torch.randn(12,2,32,160,device=device,dtype=dtype)
    k,v = parent.transpose(1,2).split((96,64),dim=-1)
    q = torch.randn(39,4,96,device=device,dtype=dtype)
    table = torch.tensor([[7,2,9,1],[4,6,3,10],[5,8,0,11]],device=device,dtype=torch.int32)
    cuq = torch.tensor([0,1,6,39],device=device,dtype=torch.int32)
    cuk = torch.tensor([0,69,150,183],device=device,dtype=torch.int32)
    left = torch.tensor([5,9,1],device=device,dtype=torch.int32)
    actual,lse = attention(q,k,v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                          max_seqlen_q=33,max_seqlen_k=81,block_table=table,
                          leftpad_k=left,causal=True,num_splits=splits)
    ref,refl = attention_reference(q,k,v,cuq,used_k=cuk[1:]-cuk[:-1],
                                  block_table=table,leftpad=left.cpu(),causal=True)
    assert_attention_close(actual,ref,dtype)
    assert_attention_close(lse,refl,torch.float32)


@pytest.mark.parametrize('sink', [False,True])
def test_empty_kv_and_all_masked_rows_across_splits(device,sink):
    q=torch.randn(13,4,32,device=device,dtype=torch.float16)
    k=torch.empty(0,2,32,device=device,dtype=torch.float16)
    cuq=torch.tensor([0,7,13],device=device,dtype=torch.int32)
    cuk=torch.tensor([0,0,0],device=device,dtype=torch.int32)
    sinks=torch.tensor([-10.,0.,1.,10.],device=device) if sink else None
    out,lse=attention(q,k,k,cu_seqlens_q=cuq,cu_seqlens_k=cuk,max_seqlen_q=7,
                      max_seqlen_k=0,s_aux=sinks,num_splits=4)
    assert torch.count_nonzero(out)==0
    expected=sinks[:,None].expand(4,13) if sink else torch.full_like(lse,-float('inf'))
    torch.testing.assert_close(lse,expected,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('qlen,klen', [(257,1025),(1,4097)])
def test_long_softmax_stability(device,qlen,klen):
    torch.manual_seed(191)
    q=torch.randn(qlen,4,128,device=device,dtype=torch.float16)*3
    k=torch.randn(klen,2,128,device=device,dtype=torch.float16)*3
    v=torch.randn(klen,2,128,device=device,dtype=torch.float16)
    cuq=torch.tensor([0,qlen],device=device,dtype=torch.int32)
    cuk=torch.tensor([0,klen],device=device,dtype=torch.int32)
    out,lse=attention(q,k,v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,
                      max_seqlen_q=qlen,max_seqlen_k=klen,causal=True)
    ref,refl=attention_reference(q,k,v,cuq,cuk,causal=True)
    assert_attention_close(out,ref)
    assert_attention_close(lse,refl,torch.float32)


def test_fp8_paged_decode_matches_dequantized_reference(device):
    # Byte representations cover subnormals and the largest finite E4M3 value.
    finite=torch.arange(256,dtype=torch.uint8)
    finite=finite[(finite!=127)&(finite!=255)].view(torch.float8_e4m3fn)
    raw=finite.repeat((8*16*2*32+253)//254)[:8*16*2*32]
    k=raw.reshape(8,16,2,32).to(device)
    v=raw.view(torch.uint8).flip(0).view(torch.float8_e4m3fn).reshape(8,16,2,32).to(device)
    q=torch.randn(2,4,32,device=device,dtype=torch.float16)*.01
    cuq=torch.tensor([0,1,2],device=device,dtype=torch.int32)
    used=torch.tensor([97,63],device=device,dtype=torch.int32)
    table=torch.tensor([[5,0,7,2,6,4,1],[2,1,4,3,0,0,0]],device=device,dtype=torch.int32)
    scale=torch.tensor([[.125,.0625]],device=device).expand(2,2)
    out,lse=attention(q,k,v,cu_seqlens_q=cuq,seqused_k=used,max_seqlen_q=1,
                      max_seqlen_k=97,block_table=table,k_descale=scale,v_descale=scale)
    ref,refl=attention_reference(q,k,v,cuq,used_k=used,block_table=table,
                                 k_descale=scale,v_descale=scale)
    assert_attention_close(out,ref)
    assert_attention_close(lse,refl,torch.float32)


@pytest.mark.parametrize('window,causal', [((0,0),True),((15,2),False),((-1,7),False),((127,-1),True)])
def test_window_tile_bounds_with_empty_splits_and_sink(device,window,causal):
    q=torch.randn(71,4,32,device=device,dtype=torch.float16)
    k=torch.randn(307,2,32,device=device,dtype=torch.float16)
    v=torch.randn_like(k)
    cq=torch.tensor([0,3,38,71],device=device,dtype=torch.int32)
    ck=torch.tensor([0,300,300,307],device=device,dtype=torch.int32)
    sinks=torch.tensor([-2.,0.,1.,3.],device=device)
    out,lse=attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_q=35,
        max_seqlen_k=300,window_size=window,causal=causal,s_aux=sinks,num_splits=3)
    ref,refl=attention_reference(q,k,v,cq,ck,window=window,causal=causal,sinks=sinks)
    assert_attention_close(out,ref)
    assert_attention_close(lse,refl,torch.float32)


def test_generic_shader_reused_across_lengths_and_window_widths(device):
    from flash_attn_mps._attention import _library
    q=torch.randn(3,4,64,device=device,dtype=torch.float16)
    k=torch.randn(1025,2,64,device=device,dtype=torch.float16)
    v=torch.randn(1025,2,32,device=device,dtype=torch.float16)
    cq=torch.tensor([0,3],device=device,dtype=torch.int32)
    misses=None
    for length,width in ((513,15),(769,63),(1025,127)):
        ck=torch.tensor([0,length],device=device,dtype=torch.int32)
        out,lse=attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_q=3,
            max_seqlen_k=length,window_size=(width,0),causal=True)
        ref,refl=attention_reference(q,k,v,cq,ck,window=(width,0),causal=True)
        assert_attention_close(out,ref)
        assert_attention_close(lse,refl,torch.float32)
        if misses is not None:
            assert _library.cache_info().misses==misses
        misses=_library.cache_info().misses


@pytest.mark.parametrize('dtype,dim', [(torch.float16,128),(torch.bfloat16,64),(torch.float32,256)])
@pytest.mark.parametrize('causal', [True,False])
def test_fast_softcap_preserves_masked_rows_and_lse(device,dtype,dim,causal):
    q=torch.randn(73,4,dim,device=device,dtype=dtype)*3
    k=torch.randn(100,2,dim,device=device,dtype=dtype)*3
    v=torch.randn_like(k)
    cq=torch.tensor([0,3,38,73],device=device,dtype=torch.int32)
    ck=torch.tensor([0,0,97,100],device=device,dtype=torch.int32)
    # cap=1 must remain a real cap; it was an unused upstream disable sentinel.
    for cap in (.25,1.,3.):
        out,lse=attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_q=35,
            max_seqlen_k=97,causal=causal,softcap=cap)
        ref,refl=attention_reference(q,k,v,cq,ck,causal=causal,softcap=cap)
        assert_attention_close(out,ref,dtype)
        assert_attention_close(lse,refl,torch.float32)
