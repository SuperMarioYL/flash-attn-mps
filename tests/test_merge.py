import pytest
import torch

from flash_attn_mps._merge import _merge_partials, merge_attn_states


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires native MPS")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pair_merge_strided_and_extreme_lse(dtype):
    left = torch.randn((4, 3, 14), device="mps").to(dtype)[1:, :, 1::2]
    right = torch.randn((3, 6, 7), device="mps").to(dtype)[:, ::2]
    la = torch.tensor([[1000., -1000., -float("inf")], [-float("inf"), 0., 10.], [-float("inf"), -50., 50.]], device="mps")
    lb = torch.tensor([[1001., -1001., 0.], [-float("inf"), -float("inf"), 11.], [1., 50., -50.]], device="mps")
    lse_base = torch.empty((6, 6), device="mps")
    lse = lse_base[::2, 1::2]
    out_base = torch.full((3, 3, 14), 99., device="mps", dtype=dtype)
    out = out_base[..., 1::2]
    merge_attn_states(out, left, la, right, lb, lse)
    le = torch.logaddexp(la.cpu().double(), lb.cpu().double())
    wa = (la.cpu().double()-le).exp().nan_to_num(0).T.unsqueeze(-1)
    wb = (lb.cpu().double()-le).exp().nan_to_num(0).T.unsqueeze(-1)
    expected = (wa*left.cpu().double()+wb*right.cpu().double()).to(dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=8e-3 if dtype==torch.bfloat16 else 2e-3, atol=8e-3 if dtype==torch.bfloat16 else 2e-6)
    torch.testing.assert_close(lse.cpu(), le.float(), rtol=1e-6, atol=2e-6)
    assert torch.equal(out_base.cpu()[..., ::2], torch.full((3, 3, 7), 99., dtype=dtype))


def test_merge_inplace_and_empty_nan_payload():
    left = torch.randn((2, 3, 4), device="mps")
    right = torch.randn_like(left)
    la = torch.zeros((3, 2), device="mps")
    lb = torch.ones_like(la)
    expected = (left.cpu()+right.cpu()*torch.e)/(1+torch.e)
    merge_attn_states(left, left, la, right, lb, la)
    torch.testing.assert_close(left.cpu(), expected)
    torch.testing.assert_close(la.cpu(), torch.full((3, 2), torch.log(torch.tensor(1+torch.e))))
    left.fill_(float("nan"))
    right.fill_(float("nan"))
    la.fill_(-float("inf"))
    lb.fill_(-float("inf"))
    merge_attn_states(left, left, la, right, lb, la)
    assert torch.equal(left.cpu(), torch.zeros_like(left.cpu()))
    assert torch.isneginf(la.cpu()).all()


@pytest.mark.parametrize("splits", [0, 1, 5])
def test_merge_partials_noncontiguous_and_empty(splits):
    p = torch.randn((splits, 4, 6, 14), device="mps")[:, 1:, ::2, 1::2]
    l = torch.randn((splits, 6, 6), device="mps")[:, ::2, 1::2]
    if splits:
        l[:, 0, 1] = -float("inf")
        p[:, 1, 0] = float("nan")
    out, lse = _merge_partials(p, l)
    expected_lse = torch.logsumexp(l.cpu().double(), dim=0)
    weights = (l.cpu().double()-expected_lse).exp().nan_to_num(0).transpose(1, 2).unsqueeze(-1)
    values = torch.where(weights>0, p.cpu().double(), 0.)
    expected = (weights*values).sum(0).float()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=5e-7)
    torch.testing.assert_close(lse.cpu(), expected_lse.float(), rtol=1e-5, atol=5e-7)
