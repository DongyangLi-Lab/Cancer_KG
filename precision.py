"""Explicit AMP policy: never reuse detached cast weights across two-pass replay."""
import contextlib
import torch
AMP_CACHE_ENABLED=False
def forward_amp(device='cuda'):
    device=torch.device(device)
    if device.type!='cuda':return contextlib.nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=AMP_CACHE_ENABLED)
