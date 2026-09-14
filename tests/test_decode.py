import pytest
import torch

from flash_attn_mps._decode import decode_attention
from reference import attention_reference, assert_attention_close


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('dims',[(64,64),(128,128),(256,256),(96,128)])
def test_decode_packed_against_fp64(device,dtype,dims):
    dq,dv=dims
    q=torch.randn(4,6,dq,device=device,dtype=dtype)
    k=torch.randn(1027,2,dq,device=device,dtype=dtype)
    v=torch.randn(1027,2,dv,device=device,dtype=dtype)
    cq=torch.tensor([0,1,2,2,3,4],device=device,dtype=torch.int32)
    ck=torch.tensor([0,0,1,257,770,1027],device=device,dtype=torch.int32)
    actual,lse=decode_attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_k=513)
    expected,el=attention_reference(q,k,v,cq,ck)
    assert_attention_close(actual,expected,dtype)
    torch.testing.assert_close(lse.cpu().double(),el,atol=8e-6,rtol=2e-5)


@pytest.mark.parametrize('splits',[1,3,16,32])
@pytest.mark.parametrize('page',[16,31,256])
def test_decode_paged_leftpad_strides_and_partial_tail(device,splits,page):
    lengths=[0,255,513]
    pads=[3,7,1]
    columns=(max(lengths)+max(pads)+page-1)//page
    table=torch.randperm(3*columns,device=device).int().reshape(3,columns)
    # Shared K/V storage views have unequal token and head strides, plus a
    # nonunit head-dimension stride and a nonzero storage offset.
    raw=torch.randn(3*columns+2,2,page,512,device=device,dtype=torch.float16)
    k,v=raw[1:-1].transpose(1,2)[...,::2].split(128,dim=-1)
    for b,(length,pad) in enumerate(zip(lengths,pads)):
        for pos in range(pad+length,columns*page):
            physical=int(table[b,pos//page])
            k[physical,pos%page]=float('nan')
            v[physical,pos%page]=float('nan')
    q=torch.randn(5,8,256,device=device,dtype=torch.float16)[1:4,:,::2]
    cq=torch.tensor([0,1,2,3],device=device,dtype=torch.int32)
    used=torch.tensor(lengths,device=device,dtype=torch.int32)
    left=torch.tensor(pads,device=device,dtype=torch.int32)
    out_storage=torch.full((5,8,256),123.,device=device,dtype=torch.float16)
    out=out_storage[1:4,:,::2]
    actual,lse=decode_attention(q,k,v,cu_seqlens_q=cq,seqused_k=used,max_seqlen_k=513,
        block_table=table,leftpad_k=left,num_splits=splits,out=out)
    expected,el=attention_reference(q,k,v,cq,used_k=used,block_table=table,leftpad=pads)
    assert actual.data_ptr()==out.data_ptr()
    assert_attention_close(actual,expected)
    torch.testing.assert_close(lse.cpu().double(),el,atol=8e-6,rtol=2e-5)
    assert torch.all(out_storage[0]==123) and torch.all(out_storage[4]==123)
    assert torch.all(out_storage[:,:,1::2]==123)


@pytest.mark.parametrize('heads,kvheads',[(1,1),(16,1),(16,8),(8,8)])
def test_decode_gqa_mqa_and_scaling(device,heads,kvheads):
    q=torch.randn(2,heads,128,device=device,dtype=torch.float32)
    k=torch.randn(260,kvheads,128,device=device,dtype=torch.float32)
    v=torch.randn_like(k)
    cq=torch.tensor([0,1,2],device=device,dtype=torch.int32)
    ck=torch.tensor([0,3,260],device=device,dtype=torch.int32)
    actual,lse=decode_attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,
                               max_seqlen_k=257,softmax_scale=.125,num_splits=8)
    expected,el=attention_reference(q,k,v,cq,ck,scale=.125)
    torch.testing.assert_close(actual.cpu().double(),expected,atol=8e-6,rtol=2e-4)
    torch.testing.assert_close(lse.cpu().double(),el,atol=8e-6,rtol=2e-5)


def test_decode_empty_queries_and_kv(device):
    q=torch.empty(0,4,128,device=device,dtype=torch.float16)
    k=torch.empty(0,2,128,device=device,dtype=torch.float16)
    cq=torch.tensor([0,0,0],device=device,dtype=torch.int32)
    result,lse=decode_attention(q,k,k,cu_seqlens_q=cq,cu_seqlens_k=cq,max_seqlen_k=0)
    assert result.shape==(0,4,128) and lse.shape==(4,0)
    q=torch.randn(2,4,128,device=device,dtype=torch.float16)
    result,lse=decode_attention(q,k,k,cu_seqlens_q=torch.tensor([0,1,2],device=device,dtype=torch.int32),
                               cu_seqlens_k=cq,max_seqlen_k=0,num_splits=4)
    assert torch.count_nonzero(result)==0
    assert torch.isneginf(lse).all()


def _typed_random(shape,dtype,device):
    # MPS stores FP8 bytes but does not implement the numeric float->FP8 cast.
    return torch.randn(shape).to(dtype).to(device)


@pytest.mark.parametrize('dtypes',[
    (torch.float16,torch.float8_e4m3fn,torch.float8_e4m3fn),
    (torch.float8_e4m3fn,torch.float8_e4m3fn,torch.float8_e4m3fn),
    (torch.float32,torch.bfloat16,torch.float16),
    (torch.bfloat16,torch.float8_e4m3fn,torch.float32),
    (torch.float16,torch.bfloat16,torch.float32),
])
def test_decode_independent_dtypes_and_broadcast_descales(device,dtypes):
    q=_typed_random((3,8,128),dtypes[0],device)
    k=_typed_random((261,2,128),dtypes[1],device)
    v=_typed_random((261,2,128),dtypes[2],device)
    cq=torch.tensor([0,1,2,3],device=device,dtype=torch.int32)
    ck=torch.tensor([0,1,4,261],device=device,dtype=torch.int32)
    qs=torch.tensor([[.5],[.75],[1.]],device=device).expand(3,2)
    ks=torch.tensor([[.4,.6]],device=device).expand(3,2)
    vs=torch.tensor([[.2,.9]],device=device).expand(3,2)
    out,lse=decode_attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_k=257,
                            q_descale=qs,k_descale=ks,v_descale=vs)
    expected,el=attention_reference(q,k,v,cq,ck,q_descale=qs,k_descale=ks,v_descale=vs)
    assert out.dtype==(torch.float16 if dtypes[0]==torch.float8_e4m3fn else dtypes[0])
    assert_attention_close(out,expected,out.dtype)
    torch.testing.assert_close(lse.cpu().double(),el,atol=1e-5,rtol=2e-5)


@pytest.mark.parametrize('window',[(-1,-1),(0,0),(15,0),(63,-1)])
@pytest.mark.parametrize('splits',[0,1,4])
def test_decode_paged_fp8_sink_window_and_softcap(device,window,splits):
    raw=torch.randn(25,16,2,256)
    # Poison the last physical page; unused page-table entries point there.
    raw[24]=float('nan')
    packed=raw.to(torch.float8_e4m3fn).to(device)
    k,v=packed.split(128,dim=-1)
    q=_typed_random((3,8,128),torch.float16,device)
    cq=torch.tensor([0,1,2,3],device=device,dtype=torch.int32)
    used=torch.tensor([0,33,97],device=device,dtype=torch.int32)
    left=torch.tensor([0,3,1],device=device,dtype=torch.int32)
    table=torch.tensor([[24]*8,[3,8,11,24,24,24,24,24],
                        [4,10,7,6,2,13,21,24]],device=device,dtype=torch.int32)
    qs=torch.tensor([[1.,.75]],device=device).expand(3,2)
    ks=torch.tensor([[.5]],device=device).expand(3,2)
    vs=torch.tensor([[.25,1.]],device=device).expand(3,2)
    sinks=torch.linspace(-3,3,16,device=device,dtype=torch.float16)[::2]
    output=torch.empty((3,8,128),device=device,dtype=torch.float32)
    out,lse=decode_attention(q,k,v,cu_seqlens_q=cq,seqused_k=used,max_seqlen_k=97,
        block_table=table,leftpad_k=left,num_splits=splits,window_size=window,softcap=.7,
        q_descale=qs,k_descale=ks,v_descale=vs,s_aux=sinks,out=output)
    expected,el=attention_reference(q,k,v,cq,used_k=used,block_table=table,leftpad=[0,3,1],
        q_descale=qs,k_descale=ks,v_descale=vs,sinks=sinks,softcap=.7,window=window)
    assert out.data_ptr()==output.data_ptr()
    torch.testing.assert_close(out.cpu().double(),expected,atol=8e-6,rtol=2e-4)
    torch.testing.assert_close(lse.cpu().double(),el,atol=1e-5,rtol=2e-5)


@pytest.mark.parametrize('splits',[1,8])
def test_decode_sinks_seed_empty_kv_only_once(device,splits):
    q=torch.randn(2,16,128,device=device,dtype=torch.float16)
    k=torch.empty(0,8,128,device=device,dtype=torch.float16)
    cq=torch.tensor([0,1,2],device=device,dtype=torch.int32)
    ck=torch.tensor([0,0,0],device=device,dtype=torch.int32)
    sinks=torch.linspace(-5,5,16,device=device)
    out,lse=decode_attention(q,k,k,cu_seqlens_q=cq,cu_seqlens_k=ck,max_seqlen_k=0,
                            num_splits=splits,s_aux=sinks,softcap=.5)
    assert torch.count_nonzero(out)==0
    torch.testing.assert_close(lse,sinks[:,None].expand(16,2),atol=1e-6,rtol=1e-6)


def test_decode_kv_growth_and_large_split_count_reuse_shader(device):
    from flash_attn_mps._decode import _shader

    q=torch.randn(1,4,128,device=device,dtype=torch.float16)
    k=torch.randn(1025,2,128,device=device,dtype=torch.float16)
    v=torch.randn_like(k)
    cq=torch.tensor([0,1],device=device,dtype=torch.int32)
    _shader.cache_clear()
    for length in (513,769,1025):
        ck=torch.tensor([0,length],device=device,dtype=torch.int32)
        actual,lse=decode_attention(q,k[:length],v[:length],cu_seqlens_q=cq,
                                    cu_seqlens_k=ck,max_seqlen_k=length)
        expected,el=attention_reference(q,k[:length],v[:length],cq,ck)
        assert_attention_close(actual,expected)
        torch.testing.assert_close(lse.cpu().double(),el,atol=1e-5,rtol=2e-5)
    assert _shader.cache_info().misses==1
    for splits in (33,65,129):
        actual,lse=decode_attention(q,k,v,cu_seqlens_q=cq,cu_seqlens_k=ck,
                                    max_seqlen_k=1025,num_splits=splits)
        assert_attention_close(actual,expected)
        torch.testing.assert_close(lse.cpu().double(),el,atol=1e-5,rtol=2e-5)
    assert _shader.cache_info().misses==1
