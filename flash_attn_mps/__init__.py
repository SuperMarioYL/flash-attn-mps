"""Native Metal FlashAttention inference for PyTorch MPS."""

from .interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from .vllm import merge_attn_states

__version__ = "0.1.0"


def store_kvcache(key, value, k_cache, v_cache, slot_mapping, *, k_scale=None, v_scale=None):
    """Scatter new K/V to physical cache slots; negative slots are skipped."""
    from ._cache import store_kvcache as store
    return store(key, value, k_cache, v_cache, slot_mapping,
                 k_scale=k_scale, v_scale=v_scale)


__all__ = [
    "flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func", "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func", "flash_attn_with_kvcache", "store_kvcache",
    "merge_attn_states",
]
