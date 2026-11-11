"""Shared actual-model padding and gradient gates, using synthetic objectives."""
import math
import torch
from token_model import canonicalize_tokens,PACKING_POLICY
def error(reference,actual):
    maximum=max(float((reference[k]-actual[k]).abs().max()) for k in reference)
    numerator=sum(float((reference[k]-actual[k]).double().square().sum()) for k in reference);denominator=sum(float(reference[k].double().square().sum()) for k in reference)
    return dict(max_abs=maximum,relative_l2=math.sqrt(numerator/max(denominator,1e-30)))
def gradient_health(model):
    rows={};adapter_total=0.
    for name,p in model.named_parameters():
        if p.requires_grad:
            good=p.grad is not None and bool(torch.isfinite(p.grad).all());magnitude=float(p.grad.float().abs().sum()) if good else None
            rows[name]=dict(present=p.grad is not None,finite=good,absolute_sum=magnitude)
            if name.startswith('adapter.') and good:adapter_total+=magnitude
    base_clean=all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    return dict(passed=all(v['finite'] for v in rows.values()) and adapter_total>0 and base_clean,all_trainable_gradients_present_finite=all(v['finite'] for v in rows.values()),adapter_gradient_abs_sum=adapter_total,adapter_gradient_nonzero=adapter_total>0,base_frozen_and_no_gradient=base_clean,trainable_gradients=rows)
def padding_gate(model,tokens,valid,amp_context,routes=None,expected_layers=32):
    """BF16 gate thresholds fixed before GPU execution; no outcome labels."""
    was_training=model.training;model.eval();before=[r['calls'] for r in routes] if routes is not None else None
    try:
        model.zero_grad(set_to_none=True);base=tokens[:2].detach().clone().requires_grad_(True);mask=valid[:2]
        with amp_context():reference=model(base,mask).float();objective=reference.sum()
        objective.backward();reference_grad={k:p.grad.detach().clone() for k,p in model.adapter.named_parameters()}
        generator=torch.Generator(device=tokens.device).manual_seed(20261023);noise=torch.randn((len(base),7,base.shape[-1]),device=base.device,dtype=base.dtype,generator=generator)*3+1
        padded=torch.cat([base.detach(),noise],1).requires_grad_(True);padded_mask=torch.cat([mask,torch.zeros((len(mask),7),device=mask.device,dtype=torch.bool)],1);model.zero_grad(set_to_none=True)
        original_packed,original_mask=canonicalize_tokens(base.detach(),mask);candidate_packed,candidate_mask=canonicalize_tokens(padded.detach(),padded_mask)
        canonical_same=torch.equal(original_packed,candidate_packed) and torch.equal(original_mask,candidate_mask)
        if not canonical_same:raise RuntimeError('Canonical padding layout gate failed')
        with amp_context():candidate=model(padded,padded_mask).float();objective=candidate.sum()
        objective.backward();actual_grad={k:p.grad.detach().clone() for k,p in model.adapter.named_parameters()};comparison=error(reference_grad,actual_grad)
        zero=bool(torch.equal(padded.grad[~padded_mask],torch.zeros_like(padded.grad[~padded_mask])))
        finite=bool(torch.isfinite(padded.grad).all());risk_same=bool(torch.allclose(reference.detach(),candidate.detach(),atol=1e-3,rtol=2e-3))
        complete=routes is None or (len(routes)==expected_layers and all(r['calls']>n for r,n in zip(routes,before)))
        health=gradient_health(model)
        passed=risk_same and comparison['relative_l2']<=5e-3 and comparison['max_abs']<=5e-4 and zero and finite and complete and health['passed']
        return dict(passed=passed,packing_policy=PACKING_POLICY,canonical_inputs_and_masks_exact=canonical_same,canonical_shape=list(original_packed.shape),valid_token_count=int(original_mask.sum()),synthetic_target='sum of risks in eval; no outcomes',added_false_padding_tokens=7,added_padding_nonzero_noise=True,valid_tokens_unchanged=bool(torch.equal(padded[:,:base.shape[1]].detach(),base.detach())),risk_max_abs=float((reference.detach()-candidate.detach()).abs().max()),risk_atol=1e-3,risk_rtol=2e-3,adapter_gradient=comparison,gradient_relative_l2_limit=5e-3,gradient_max_abs_limit=5e-4,invalid_token_input_gradient_exact_zero=zero,all_input_gradients_finite=finite,all_expected_native_layers_executed=complete,gradient_health=health)
    finally:model.train(was_training)
