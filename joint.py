"""One complete-risk backward plus globally normalized KG-only auxiliary backward."""
import math
import torch
from calibrated import auxiliary_forward
from fullrisk_replay import fullrisk_backward
from gradient_gates import gradient_health
def memory_grads(model):return {n:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n,p in model.memory.named_parameters()}
def gradient_accounting(before,after,coefficient):
    if coefficient==0:return dict(reader_Cox_gradient_L2=math.sqrt(sum(float(x.double().square().sum()) for x in before.values())),reader_weighted_aux_gradient_L2=0.,reader_aux_to_Cox_gradient_ratio=None,reader_gradient_cosine=None)
    aa=sum(float(v.double().square().sum()) for v in before.values());bb=sum(float((after[k]-v).double().square().sum()) for k,v in before.items());dot=sum(float((v.double()*(after[k]-v).double()).sum()) for k,v in before.items())
    return dict(reader_Cox_gradient_L2=math.sqrt(aa),reader_weighted_aux_gradient_L2=math.sqrt(bb),reader_aux_to_Cox_gradient_ratio=math.sqrt(bb/aa) if aa>0 else None,reader_gradient_cosine=dot/math.sqrt(aa*bb) if aa>0 and bb>0 else None)
def auxiliary_backward(model,aux,tokens,valid,token_ids,target,available,micro,coefficient):
    denominator=int(available.sum())
    if denominator<=0:raise ValueError('Whole-fit available auxiliary denominator must be positive')
    total=0.
    for i in range(0,len(tokens),micro):
        pred,mask,_=auxiliary_forward(model,aux,tokens[i:i+micro],valid[i:i+micro],token_ids[i:i+micro])
        if not torch.equal(mask,available[i:i+micro]):raise ValueError('Static auxiliary availability differs from pooling')
        loss=((pred-target[i:i+micro]).square()*mask).sum()/denominator;total+=float(loss.detach());(coefficient*loss).backward()
    return total
def joint_backward(model,aux,tokens,valid,t,event,token_ids,target,available,micro,coefficient):
    aux.zero_grad(set_to_none=True);loss,risk,policy=fullrisk_backward(model,tokens,valid,t,event,micro);before=memory_grads(model)
    mse=auxiliary_backward(model,aux,tokens,valid,token_ids,target,available,micro,coefficient) if coefficient else None
    accounting=gradient_accounting(before,memory_grads(model),coefficient)
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad):raise ValueError('Main gradients missing or nonfinite')
    if coefficient and not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in aux.parameters()):raise ValueError('Joint auxiliary gradients missing/nonfinite')
    health=gradient_health(model)
    aux_health=dict(enabled=coefficient>0,passed=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in aux.parameters()) if coefficient else all(p.grad is None for p in aux.parameters()),parameters_with_gradient=sum(p.grad is not None for p in aux.parameters()))
    return loss,risk,dict(main_gradient_health=health,auxiliary_gradient_health=aux_health,cox_loss=loss,aux_MSE=mse,wholefit_available_items=int(available.sum()),weighted_aux_loss=coefficient*mse if mse is not None else 0.,coefficient=coefficient,fullrisk=policy,**accounting)
