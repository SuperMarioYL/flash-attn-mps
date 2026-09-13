"""Independent CPU float64 oracle; never imported by the production package."""

import torch


def attention_reference(q, k, v, cu_q, cu_k=None, used_k=None, block_table=None,
                        scale=None, causal=False, window=(-1, -1), softcap=0.,
                        alibi=None, sinks=None, q_descale=None, k_descale=None,
                        v_descale=None, mask=None, aux=None, leftpad=None):
    def cpu_double(x):
        x = x.detach()
        if x.dtype == torch.float8_e4m3fn:
            # MPS does not implement strided float8 device copies. Copy the
            # storage bytes and decode only in this CPU test oracle.
            return x.view(torch.uint8).cpu().view(x.dtype).double()
        return x.cpu().double()
    q, k, v = (cpu_double(x) for x in (q, k, v))
    cu_q = cu_q.cpu().tolist()
    if cu_k is not None:
        cu_k = cu_k.cpu().tolist()
        lengths = [b - a for a, b in zip(cu_k, cu_k[1:])]
    else:
        lengths = used_k.cpu().tolist()
    table = None if block_table is None else block_table.cpu()
    if torch.is_tensor(causal):
        causal = causal.cpu().tolist()
    heads, hkv, dv = q.shape[1], k.shape[-2], v.shape[-1]
    groups = torch.arange(heads) // (heads // hkv)
    out = torch.zeros(q.shape[0], heads, dv, dtype=torch.float64)
    lse = torch.full((heads, q.shape[0]), -torch.inf, dtype=torch.float64)
    scale = q.shape[-1] ** -0.5 if scale is None else scale

    def descale(tensor, b):
        if tensor is None:
            return 1.
        return tensor.detach().cpu().double()[b][groups].reshape(heads, 1, 1)

    for b, length in enumerate(lengths):
        start, end = cu_q[b:b + 2]
        if start == end:
            continue
        query = q[start:end].transpose(0, 1) * descale(q_descale, b)
        if table is not None:
            offset = 0 if leftpad is None else int(leftpad[b])
            positions = torch.arange(offset, offset + length)
            physical = table[b, positions // k.shape[1]].long()
            key = k[physical, positions % k.shape[1]]
            value = v[physical, positions % v.shape[1]]
        else:
            ks = cu_k[b]
            key, value = k[ks:ks+length], v[ks:ks+length]
        key = key[:, groups].transpose(0, 1) * descale(k_descale, b)
        value = value[:, groups].transpose(0, 1) * descale(v_descale, b)
        scores = query @ key.transpose(-2, -1) * scale
        if softcap:
            scores = torch.tanh(scores / softcap) * softcap
        qpos = torch.arange(length - (end-start), length)[:, None]
        kpos = torch.arange(length)[None, :]
        if alibi is not None:
            slopes = alibi.detach().cpu().double()
            slopes = slopes if slopes.ndim == 1 else slopes[b]
            scores -= slopes[:, None, None] * (qpos-kpos).abs()
        causal_b = causal[b] if isinstance(causal, list) else causal
        keep = (kpos <= qpos) if causal_b else torch.ones(end-start, length, dtype=torch.bool)
        if window[0] >= 0:
            keep &= kpos >= qpos-window[0]
        if window[1] >= 0:
            keep &= kpos <= qpos+window[1]
        if mask is not None and mask.kind == 1:
            ranges = aux[0].cpu()[start:end]
            keep = kpos <= qpos
            if mask.sliding_window_left >= 0:
                keep &= qpos-kpos < mask.sliding_window_left
            prefix = (kpos >= ranges[:, :1]) & (kpos <= ranges[:, 1:2])
            if mask.clamp_prefix_to_window:
                prefix &= qpos-kpos < mask.sliding_window_left
            keep |= prefix
        elif mask is not None and mask.kind == 2:
            prefix_len, width = int(aux[0][b]), int(aux[1][0])
            keep = (kpos <= qpos) & ((kpos < prefix_len) | (qpos-kpos < width))
        scores = scores.masked_fill(~keep, -torch.inf)
        all_scores = scores
        if sinks is not None:
            sink = sinks.detach().cpu().double().reshape(heads, 1, 1).expand(-1,end-start,-1)
            all_scores = torch.cat((scores, sink), -1)
        l = torch.logsumexp(all_scores, -1)
        probabilities = torch.exp(scores-l.unsqueeze(-1)).nan_to_num(0.)
        out[start:end] = (probabilities @ value).transpose(0, 1)
        lse[:, start:end] = l
    return out, lse


def assert_attention_close(actual, expected, dtype=torch.float16):
    atol, rtol = (2e-2, 5e-2) if dtype == torch.bfloat16 else (2e-3, 1e-2)
    torch.testing.assert_close(actual.detach().cpu().double(), expected,
                               atol=atol, rtol=rtol, check_dtype=False)
