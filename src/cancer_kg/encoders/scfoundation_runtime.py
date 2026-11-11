"""Scoped native-MHA fastpath control; keeps all genes/weights/precision intact."""
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F


@contextmanager
def encoder_runtime(cfg, audit=None):
    audit = {} if audit is None else audit
    previous = torch.backends.mha.get_fastpath_enabled()
    original_sdpa = F.scaled_dot_product_attention
    disabled = cfg.get('disable_mha_fastpath', False)
    policy = cfg.get('sdpa_policy', 'native_auto')
    if policy not in ('native_auto', 'efficient_only'):
        raise ValueError('Unknown explicit SDPA policy')
    if policy == 'efficient_only' and (not disabled or cfg['device'] != 'cuda'):
        raise ValueError('efficient_only requires CUDA and disabled native MHA fastpath')
    audit.update(disable_mha_fastpath=disabled, previous_mha_fastpath=previous,
                 sdpa_policy=policy, sdpa_calls=0, sdpa_input_shapes=[], sdpa_query_dtypes=[])
    def counted_sdpa(query, key, value, *args, **kwargs):
        audit['sdpa_calls'] += 1
        shape=list(query.shape)
        if shape not in audit['sdpa_input_shapes']:audit['sdpa_input_shapes'].append(shape)
        dtype=str(query.dtype)
        if dtype not in audit['sdpa_query_dtypes']:audit['sdpa_query_dtypes'].append(dtype)
        return original_sdpa(query,key,value,*args,**kwargs)
    if policy == 'efficient_only':
        from torch.nn.attention import sdpa_kernel,SDPBackend
        kernels=sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
        audit['allowed_sdpa_backends']=['EFFICIENT_ATTENTION']
    else:
        kernels=nullcontext()
        audit['allowed_sdpa_backends']=['framework_default_not_claimed_memory_efficient']
    try:
        if disabled:torch.backends.mha.set_fastpath_enabled(False)
        F.scaled_dot_product_attention=counted_sdpa
        audit['active_mha_fastpath']=torch.backends.mha.get_fastpath_enabled()
        with kernels:
            yield audit
        if disabled and audit['sdpa_calls']==0:
            raise RuntimeError('Native fastpath disabled but no SDPA call observed; encoder route is not verified')
    finally:
        F.scaled_dot_product_attention=original_sdpa
        torch.backends.mha.set_fastpath_enabled(previous)
        audit['restored_mha_fastpath']=torch.backends.mha.get_fastpath_enabled()
