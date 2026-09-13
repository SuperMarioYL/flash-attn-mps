import pytest
import torch

from flash_attn_mps._cache import _unit_scale, append_kvcache, rotary, store_kvcache


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_strides_offsets_and_skipped_slots(dtype, index_dtype):
    k = torch.arange(5*4*12, device="mps", dtype=torch.float32).to(dtype).reshape(5, 4, 12)[1:, ::2, 1::2]
    v = (k[..., :4] + 100).transpose(0, 1).contiguous().transpose(0, 1)
    kb = torch.full((4, 8, 4, 14), -3, device="mps", dtype=dtype)
    vb = torch.full((4, 8, 4, 10), -5, device="mps", dtype=dtype)
    kc, vc = kb[1:3, ::2, ::2, 1:13:2], vb[1:3, ::2, ::2, 1:9:2]
    ke, ve = kb.cpu(), vb.cpu()
    kev, vev = ke[1:3, ::2, ::2, 1:13:2], ve[1:3, ::2, ::2, 1:9:2]
    slots = torch.tensor([99, 6, 99, -1, 99, 0, 99, 3], dtype=index_dtype, device="mps")[1::2]
    for t, slot in enumerate([6, -1, 0, 3]):
        if slot >= 0:
            kev[slot//4, slot%4] = k[t].cpu()
            vev[slot//4, slot%4] = v[t].cpu()
    store_kvcache(k, v, kc, vc, slots)
    assert torch.equal(kb.cpu(), ke)
    assert torch.equal(vb.cpu(), ve)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_preserves_float_bits(dtype):
    special = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan"), 1e-30, -1e-30], dtype=dtype)
    k = special.reshape(1, 1, -1).to("mps")
    cache = torch.empty((1, 1, 1, special.numel()), dtype=dtype, device="mps")
    store_kvcache(k, k, cache, cache, torch.tensor([0], device="mps"))
    assert torch.equal(cache.cpu().view(torch.uint8).flatten(), special.view(torch.uint8))


def test_fp8_rne_saturation_and_scales():
    representable = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    ties = (representable[:-1] + representable[1:]) / 2
    values = torch.cat((representable, ties, -representable, -ties, torch.tensor([1000., -1000., float("inf"), -float("inf")])))
    values = values[:, None, None].expand(-1, 2, 3).contiguous()
    scale = torch.tensor([.5, 2.])
    k = (values * scale[None, :, None]).to("mps")
    v = (values * 4).to("mps")
    # Each cache is a view with a nonzero storage offset and a strided dimension.
    kb = torch.empty((2, len(values), 2, 6), dtype=torch.float8_e4m3fn, device="mps")
    vb = torch.empty_like(kb)
    kc, vc = kb[1:, :, :, ::2], vb[1:, :, :, ::2]
    store_kvcache(k, v, kc, vc, torch.arange(len(values), device="mps"), k_scale=scale.to("mps"), v_scale=4.)
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(kc.view(torch.uint8).cpu()[0], expected)
    assert torch.equal(vc.view(torch.uint8).cpu()[0], expected)


@pytest.mark.parametrize("paged", [False, True])
def test_append_batch_remapping_and_boundaries(paged):
    k = torch.arange(2*3*2*4, device="mps", dtype=torch.float32).reshape(2, 3, 2, 4)
    v = k[..., :2] + 70
    kc = torch.full((6 if paged else 3, 4 if paged else 12, 2, 4), -1., device="mps")
    vc = torch.full((*kc.shape[:-1], 2), -2., device="mps")
    table = torch.tensor([[5, 0, 2], [1, 4, 3], [2, 3, 1]], device="mps", dtype=torch.int32) if paged else None
    lengths = torch.tensor([3, 6], device="mps", dtype=torch.int32)
    rows = torch.tensor([1, 0], device="mps", dtype=torch.int32)
    ke, ve = kc.cpu(), vc.cpu()
    for b, row in enumerate([1, 0]):
        for j in range(3):
            pos = [3, 6][b] + j
            block, offset = (int(table.cpu()[b, pos//4]), pos%4) if paged else (row, pos)
            ke[block, offset], ve[block, offset] = k[b, j].cpu(), v[b, j].cpu()
    append_kvcache(k, v, kc, vc, lengths, cache_batch_idx=rows, cache_leftpad=torch.tensor([2, 1], device="mps"), block_table=table)
    assert torch.equal(kc.cpu(), ke)
    assert torch.equal(vc.cpu(), ve)
    assert torch.equal(lengths.cpu(), torch.tensor([3, 6], dtype=torch.int32))


def test_append_fp8_scales():
    k = torch.randn((2, 2, 3, 5), device="mps")
    v = torch.randn((2, 2, 3, 7), device="mps")
    kc = torch.empty((2, 4, 3, 5), dtype=torch.float8_e4m3fn, device="mps")
    vc = torch.empty((2, 4, 3, 7), dtype=torch.float8_e4m3fn, device="mps")
    scale = torch.tensor([.5, 1., 2.], device="mps")
    append_kvcache(k, v, kc, vc, 1, k_scale=scale, v_scale=2.)
    ke = (k.cpu()/scale.cpu()[None, None, :, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    ve = (v.cpu()/2).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(kc.cpu()[:, 1:3].contiguous().view(torch.uint8), ke)
    assert torch.equal(vc.cpu()[:, 1:3].contiguous().view(torch.uint8), ve)


def test_fp8_default_scale_with_float16_default_dtype():
    old_dtype = torch.get_default_dtype()
    _unit_scale.cache_clear()
    try:
        torch.set_default_dtype(torch.float16)
        k = torch.tensor([[[1., 2., 3., 4.]]], device="mps")
        cache = torch.empty((1, 1, 1, 4), device="mps", dtype=torch.float8_e4m3fn)
        store_kvcache(k, k, cache, cache, torch.tensor([0], device="mps"))
        expected = k.cpu().to(torch.float8_e4m3fn).view(torch.uint8)
        assert torch.equal(cache.view(torch.uint8).cpu()[0], expected)
    finally:
        torch.set_default_dtype(old_dtype)
        _unit_scale.cache_clear()


def _rotary_reference(x, cos, sin, lengths, interleaved, advance):
    result = x.clone()
    half = cos.shape[1]
    for b in range(x.shape[0]):
        for j in range(x.shape[1]):
            position = int(lengths[b]) + (j if advance else 0)
            c, s = cos[position], sin[position]
            a = x[b, j, :, :2*half:2] if interleaved else x[b, j, :, :half]
            z = x[b, j, :, 1:2*half:2] if interleaved else x[b, j, :, half:2*half]
            first, second = a*c-z*s, z*c+a*s
            if interleaved:
                result[b, j, :, :2*half:2] = first
                result[b, j, :, 1:2*half:2] = second
            else:
                result[b, j, :, :half] = first
                result[b, j, :, half:2*half] = second
    return result


@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("mode", ["global", "causal", "local"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rotary_modes_strides_and_partial_head(interleaved, mode, dtype):
    q = torch.randn((2, 3, 4, 16), device="mps").to(dtype)[..., 1::2]
    k = torch.randn((2, 2, 2, 16), device="mps").to(dtype)[..., 1::2]
    angles = torch.randn((20, 4), device="mps")[:, ::2]
    cos, sin = angles.cos(), angles.sin()
    lengths = torch.tensor([2, 7], device="mps")
    window = (3, -1) if mode=="local" else [-1, -1]
    qr, kr = rotary(q, k, cos, sin, lengths, rotary_interleaved=interleaved, causal=mode=="causal", window_size=window)
    qe = _rotary_reference(q.cpu().float(), cos.cpu(), sin.cpu(), lengths.cpu(), interleaved, mode!="global").to(dtype)
    ke = _rotary_reference(k.cpu().float(), cos.cpu(), sin.cpu(), lengths.cpu(), interleaved, True).to(dtype)
    torch.testing.assert_close(qr.cpu(), qe, rtol=1e-5 if dtype==torch.float32 else 8e-3, atol=5e-7 if dtype==torch.float32 else 8e-3)
    torch.testing.assert_close(kr.cpu(), ke, rtol=1e-5 if dtype==torch.float32 else 8e-3, atol=5e-7 if dtype==torch.float32 else 8e-3)
    assert torch.equal(qr.cpu()[..., 4:], q.cpu()[..., 4:])
