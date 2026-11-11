"""Exact full-risk Cox gradient with deterministic per-microbatch RNG replay."""
import torch
from cancer_kg.runtime.precision import forward_amp
def cox(r,t,e):
    if t.dtype not in [torch.float32,torch.float64]:raise ValueError('Survival time must retain FP32/FP64')
    r=r.float()
    ix=torch.nonzero(e.bool(),as_tuple=True)[0]
    if not len(ix):raise ValueError('No events')
    return (torch.logsumexp(r[None,:].expand(len(ix),-1).masked_fill(t[None,:]<t[ix,None],-torch.inf),1)-r[ix]).mean()
def get_rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def set_rng(s):
    torch.set_rng_state(s['cpu'])
    if s['cuda'] is not None:torch.cuda.set_rng_state_all(s['cuda'])
def fullrisk_backward(model,tokens,valid,t,e,microbatch=1):
    """Caller steps optimizer only after this function; never microbatch Cox."""
    model.zero_grad(set_to_none=True);states=[];risks=[];n=len(tokens)
    initial_rng=get_rng();final_rng=None
    try:
        with torch.no_grad():
            for start in range(0,n,microbatch):
                states.append(get_rng())
                with forward_amp(tokens.device):risks.append(model(tokens[start:start+microbatch],valid[start:start+microbatch]).float())
        final_rng=get_rng();risk=torch.cat(risks).detach().requires_grad_(True);loss=cox(risk,t,e)
        if not torch.isfinite(loss):raise ValueError('Nonfinite full-risk loss')
        gradient=torch.autograd.grad(loss,risk)[0].detach()
        for k,start in enumerate(range(0,n,microbatch)):
            set_rng(states[k])
            with forward_amp(tokens.device):pred=model(tokens[start:start+microbatch],valid[start:start+microbatch]).float()
            if not torch.allclose(pred.detach(),risk.detach()[start:start+microbatch],atol=1e-5,rtol=1e-5):raise ValueError('RNG/model replay risk mismatch')
            pred.backward(gradient[start:start+microbatch])
    finally:set_rng(final_rng if final_rng is not None else initial_rng)
    return float(loss.detach()),risk.detach(),dict(full_risk_patients=n,microbatch=microbatch,passes=2,rng_replayed=True,amp_cache_enabled=False,forward_autocast_context_per_microbatch=True)
