"""Stable native Metal merging of independently computed attention states."""

from functools import lru_cache
from pathlib import Path

import torch

from ._cache import _dtype, _mps, _params


@lru_cache(None)
def _shader(dtype):
    value, read, write = _dtype(dtype)
    source = Path(__file__).with_name("kernels").joinpath("merge.metal").read_text()
    for old, new in (("READ_VALUE", read), ("WRITE_VALUE", write), ("VALUE", value)):
        source = source.replace(old, new)
    return torch.mps.compile_shader(source)


def merge_attn_states(out, left_out, left_lse, right_out, right_lse, out_lse=None):
    """Write the LSE-weighted merge to ``out``; LSE uses natural logarithms.

    Outputs are [tokens, heads, dim], LSE tensors are FP32 [heads, tokens].
    An empty state has LSE=-inf; merging two empty states yields zero and -inf.
    ``out`` and ``out_lse`` may alias the corresponding input states.
    """
    _mps(out, left_out, left_lse, right_out, right_lse, out_lse)
    if out.ndim != 3 or left_out.shape != out.shape or right_out.shape != out.shape:
        raise ValueError("output states must have matching [tokens, heads, dim] shapes")
    if left_out.dtype != out.dtype or right_out.dtype != out.dtype:
        raise TypeError("output state dtypes must match")
    expected = (out.shape[1], out.shape[0])
    for lse in (left_lse, right_lse, out_lse):
        if lse is not None and (lse.shape != expected or lse.dtype != torch.float32):
            raise ValueError("LSE tensors must be float32 [heads, tokens]")
    if not out.numel():
        return
    destination = out_lse if out_lse is not None else left_lse
    lse_stride = out_lse.stride() if out_lse is not None else (0, 0)
    p = _params([*out.shape, *out.stride(), *left_out.stride(), *right_out.stride(), *left_lse.stride(), *right_lse.stride(), *lse_stride, int(out_lse is not None)])
    _shader(out.dtype).merge_pair(out, left_out, left_lse, right_out, right_lse, destination, p, threads=out.shape[0]*out.shape[1])


def _merge_partials(partial_out, partial_lse):
    """Merge [splits,tokens,heads,dim] and [splits,heads,tokens] states."""
    _mps(partial_out, partial_lse)
    if partial_out.ndim != 4 or partial_lse.shape != (partial_out.shape[0], partial_out.shape[2], partial_out.shape[1]):
        raise ValueError("partial states require [splits,tokens,heads,dim] and [splits,heads,tokens]")
    if partial_lse.dtype != torch.float32:
        raise TypeError("partial LSE must be float32")
    splits, tokens, heads, dim = partial_out.shape
    output = torch.empty((tokens, heads, dim), dtype=partial_out.dtype, device=partial_out.device)
    lse = torch.empty((heads, tokens), dtype=torch.float32, device=partial_out.device)
    if tokens and heads:
        p = _params([splits, tokens, heads, dim, *partial_out.stride(), *partial_lse.stride()])
        _shader(partial_out.dtype).merge_partials(partial_out, partial_lse, output, lse, p, threads=tokens*heads)
    return output, lse
