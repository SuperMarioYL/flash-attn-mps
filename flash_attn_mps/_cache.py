"""Metal cache scatter, append and rotary position encoding."""

from functools import lru_cache
from pathlib import Path

import torch


def _mps(*tensors):
    if any(t.device.type != "mps" for t in tensors if t is not None):
        raise ValueError("operations require MPS tensors")


def _dtype(dtype):
    if dtype == torch.float32:
        return "float", "float(value)", "float(value)"
    if dtype == torch.float16:
        return "half", "float(value)", "half(value)"
    if dtype == torch.bfloat16:
        return "ushort", "as_type<float>(uint(value) << 16)", "ushort((as_type<uint>(value) + 0x7fff + ((as_type<uint>(value) >> 16) & 1)) >> 16)"
    raise TypeError("input dtype must be float16, bfloat16 or float32")


@lru_cache(None)
def _shader(dtype, cache_dtype, index_dtype):
    src, read, write = _dtype(dtype)
    fp8 = cache_dtype == torch.float8_e4m3fn
    if not fp8 and cache_dtype != dtype:
        raise TypeError("cache dtype must match input or be float8_e4m3fn")
    source = Path(__file__).with_name("kernels").joinpath("cache.metal").read_text()
    replacements = {
        "READ_VALUE": read,
        "WRITE_VALUE": write,
        "STORE_VALUE": "encode_e4m3(read_value(input[src]) / scale[h*p[12]])" if fp8 else "input[src]",
        "APPEND_VALUE": "encode_e4m3(read_value(input[src]) / scale[h*p[12]])" if fp8 else "input[src]",
        "SRC": src,
        "DST": "uchar" if fp8 else src,
        "INDEX": "long" if index_dtype == torch.int64 else "int",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=1)
def _unit_scale():
    return torch.ones(1, device="mps", dtype=torch.float32)


@lru_cache(maxsize=1)
def _index_dummy():
    return torch.zeros(1, device="mps", dtype=torch.int32)


def _scale(scale, heads, device):
    scale = _unit_scale() if scale is None else torch.as_tensor(scale, device=device, dtype=torch.float32)
    if scale.numel() not in (1, heads):
        raise ValueError("scale must be scalar or have one value per KV head")
    return scale.reshape(-1).contiguous(), 0 if scale.numel() == 1 else 1


def _params(values):
    # compile_shader passes Python integer tuples directly as int64 constants.
    return tuple(values)


def store_kvcache(k, v, k_cache, v_cache, slot_mapping, *, k_scale=None, v_scale=None):
    """Scatter [tokens, heads, dim] K/V into physical cache slots; -1 skips."""
    _mps(k, v, k_cache, v_cache, slot_mapping)
    if k.ndim != 3 or v.ndim != 3 or k.shape[:2] != v.shape[:2]:
        raise ValueError("K/V must have matching [tokens, heads, dim] shapes")
    if slot_mapping.ndim != 1 or slot_mapping.numel() != k.shape[0] or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must be int32/int64 with one slot per token")
    for x, cache, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        if cache.ndim != 4 or cache.shape[2:] != x.shape[1:]:
            raise ValueError("cache must have shape [blocks, block_size, heads, dim]")
        if not x.numel():
            continue
        scale, scale_stride = _scale(scale, x.shape[1], x.device)
        p = _params([cache.shape[1], *x.shape[1:], *x.stride(), *cache.stride(), slot_mapping.stride(0), cache.shape[0]*cache.shape[1], scale_stride])
        _shader(x.dtype, cache.dtype, slot_mapping.dtype).store(
            x, cache, slot_mapping, scale, p,
            threads=(x.shape[2], x.shape[1], x.shape[0]),
            group_size=(min(x.shape[2], 256), 1, 1),
        )


def _integers(value, count, device):
    if value is None:
        value = 0
    value = torch.as_tensor(value, dtype=torch.int32, device=device)
    if value.numel() == 1:
        value = value.expand(count)
    if value.numel() != count:
        raise ValueError(f"expected {count} integer values")
    return value.contiguous()


def append_kvcache(k, v, k_cache, v_cache, cache_seqlens, *, cache_batch_idx=None, cache_leftpad=None, block_table=None, k_scale=None, v_scale=None):
    """Append [batch, new_tokens, heads, dim] K/V without changing lengths."""
    _mps(k, v, k_cache, v_cache, block_table)
    if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
        raise ValueError("K/V must have matching [batch, new_tokens, heads, dim] shapes")
    batch = k.shape[0]
    rows = k_cache.shape[0] if block_table is None else block_table.shape[0]
    lengths = _integers(cache_seqlens, batch, k.device)
    remap = block_table is None and cache_batch_idx is not None
    indices = _integers(cache_batch_idx, batch, k.device) if remap else _index_dummy()
    # FA2 lengths are physical exclusive ends, already including left padding.
    # New tokens append at length+j, never at length+leftpad+j.
    if cache_leftpad is not None:
        _integers(cache_leftpad, batch, k.device)
    table = _index_dummy() if block_table is None else block_table.to(torch.int32).contiguous()
    if block_table is not None and block_table.ndim != 2:
        raise ValueError("block_table must have shape [cache_batch, blocks_per_sequence]")
    for x, cache, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        if cache.ndim != 4 or cache.shape[2:] != x.shape[2:]:
            raise ValueError("cache must have shape [rows, capacity, heads, dim]")
        if not x.numel():
            continue
        scale, scale_stride = _scale(scale, x.shape[2], x.device)
        p = _params([cache.shape[1], *x.shape[1:], *x.stride(), *cache.stride(), scale_stride, 0, int(block_table is not None), 0 if block_table is None else block_table.shape[1], cache.shape[0], rows, int(remap)])
        _shader(x.dtype, cache.dtype, torch.int32).append(
            x, cache, lengths, indices, table, scale, p,
            threads=(x.shape[3], x.shape[2], x.shape[0]*x.shape[1]),
            group_size=(min(x.shape[3], 256), 1, 1),
        )


def rotary(q, k, cos, sin, cache_seqlens, *, rotary_interleaved=True, causal=False, window_size=(-1, -1)):
    """Rotate new K at old_length+j and Q at its causal/local absolute position."""
    _mps(q, k, cos, sin)
    if q.ndim != 4 or (k is not None and (k.ndim != 4 or q.shape[0] != k.shape[0])):
        raise ValueError("Q/K must have shape [batch, tokens, heads, dim]")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1]*2 > q.shape[-1]:
        raise ValueError("rotary cosine/sine must have shape [positions, rotary_dim/2]")
    cosine, sine = cos.float().contiguous(), sin.float().contiguous()
    lengths = _integers(cache_seqlens, q.shape[0], q.device)
    outputs = []
    for x, advance in ((q, causal or tuple(window_size) != (-1, -1)), (k, True)):
        if x is None:
            outputs.append(None)
            continue
        if cos.shape[1]*2 > x.shape[-1]:
            raise ValueError("rotary dimension exceeds head dimension")
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.numel():
            p = _params([x.shape[0], *x.shape[1:], *x.stride(), cos.shape[1], int(rotary_interleaved), int(advance)])
            _shader(x.dtype, x.dtype, torch.int32).rotate(
                x, output, cosine, sine, lengths, p,
                threads=(x.shape[3], x.shape[2], x.shape[0]*x.shape[1]),
                group_size=(min(x.shape[3], 256), 1, 1),
            )
        outputs.append(output)
    return tuple(outputs)
