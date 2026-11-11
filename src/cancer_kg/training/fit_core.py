"""Executed alignment fit; imports localized only."""
import time,hashlib
import numpy as np
import torch
from cancer_kg.models.calibrated import AlignmentModel,auxiliary_forward,canonicalize_tokens
from cancer_kg.training.auxiliary import make_aux,LAMBDA
from cancer_kg.training.joint import joint_backward
from cancer_kg.training.fullrisk_replay import cox
from cancer_kg.runtime.precision import forward_amp
from cancer_kg.training.state import trainable_state,load_trainable

def make_model(encoder,data):return AlignmentModel(encoder,*[torch.as_tensor(data['bank'][k]) for k in ['facts','anchors','local_index','local_role']])

def aux_state(aux):return {k:v.detach().cpu().clone() for k,v in aux.state_dict().items()}

def aux_digest(aux):return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in aux.state_dict().values())).hexdigest()

def predict(model,x,valid,micro):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),micro):
            with forward_amp(x.device):out.append(model(x[i:i+micro],valid[i:i+micro]).float())
    r=torch.cat(out)
    if not bool(torch.isfinite(r).all()):raise ValueError('Nonfinite risk')
    return r

def knowledge_description(model,aux,data,arm,micro,device):
    begin=time.perf_counter();model.eval();acc={};minimum={};maximum={};total=0;error=0.;den=0
    with torch.no_grad():
        for start in range(0,len(data['fit']),micro):
            positions=np.arange(start,min(start+micro,len(data['fit'])));ix=data['fit'][positions];packed=torch.as_tensor(data['tokens'][ix],device=device);valid=torch.as_tensor(data['valid'][ix],device=device);ids=torch.as_tensor(data['saved_token_ids'][ix],device=device);joined=torch.cat([packed,ids[:,:,None].float()],-1);joined,valid=canonicalize_tokens(joined,valid);states,k,valid,receipt=model.knowledge(joined[:,:,:769],valid);entity=joined[:,:,768].long();mapped=receipt['routes'][-1];weights=receipt['weights'];glob=receipt['routes'][3];local_valid=receipt['routes'][5];g=glob.shape[1];raw=joined[:,:,:768];ratio=(states-raw).square().mean(-1).sqrt()/raw.square().mean(-1).sqrt().clamp_min(1e-30);mass=weights[:,:,g:].sum(-1);entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)/(g+local_valid.sum(-1)).float().log();rows=torch.as_tensor(data['bank']['entity_rows'],device=device)[entity.clamp_min(0)];facts=torch.as_tensor(data['bank']['bank'],device=device)[glob];incident=(facts[:,None,:,0]==rows[:,:,None])|(facts[:,None,:,2]==rows[:,:,None]);incident_mass=(weights[:,:,:g]*incident).sum(-1)+mass
            for key,value in [('residual_to_token_RMS',ratio),('local_slot_mass',mass),('normalized_entropy',entropy),('query_anchor_incident_global_plus_local_mass',incident_mass)]:
                a=value[mapped];acc[key]=acc.get(key,0.)+float(a.double().sum());minimum[key]=min(minimum.get(key,float('inf')),float(a.min()));maximum[key]=max(maximum.get(key,-float('inf')),float(a.max()))
            total+=int(mapped.sum())
            if LAMBDA[arm]:
                pred,mask,_=aux(k,joined[:,:,-1].long(),valid);expected=torch.as_tensor(data['aux_available'][positions],device=device)
                if not torch.equal(mask,expected):raise ValueError('Final training auxiliary mask changed')
                target=torch.as_tensor(data['target']['targets_train'][positions],device=device);error+=float(((pred-target).square()*mask).double().sum());den+=int(mask.sum())
    result=dict(training_rows_only=True,extra_Tahoe_forwards=0,seconds=time.perf_counter()-begin,gate=float(model.memory.gate.detach()),mapped_tokens=total,metrics={key:dict(mean=acc[key]/total,min=minimum[key],max=maximum[key]) for key in acc},incident_scope='actual gene for real grounding; shuffled query anchor for shuffled grounding',auxiliary_head_trained=LAMBDA[arm]>0,training_aux_MSE=error/den if den else None,training_aux_available_items=den)
    return result

def fit(encoder,data,arm,epochs,micro,early,device='cuda'):
    if not 1<=epochs<=24:raise ValueError('Frozen horizon1..24')
    torch.manual_seed(42);model=make_model(encoder,data).to(device);aux=make_aux(data).to(device);initial_aux=aux_digest(aux);coefficient=LAMBDA[arm];params=[p for p in model.parameters() if p.requires_grad]+(list(aux.parameters()) if coefficient else []);optimizer=torch.optim.AdamW(params,lr=.0001,weight_decay=.01)
    x=torch.as_tensor(data['tokens'],device=device);valid=torch.as_tensor(data['valid'],device=device);ids=torch.as_tensor(data['saved_token_ids'],device=device);fitix=torch.as_tensor(data['fit'],device=device);held=torch.as_tensor(data['held'],device=device);t=torch.as_tensor(data['time'],device=device,dtype=torch.float32);event=torch.as_tensor(data['event'],device=device,dtype=torch.bool);target=torch.as_tensor(data['target']['targets_train'],device=device);available=torch.as_tensor(data['aux_available'],device=device)
    trace=[];best=float('inf');best_epoch=0;saved=None;saved_aux=None;stale=0
    for epoch in range(1,epochs+1):
        begin=time.perf_counter();model.train();aux.train();optimizer.zero_grad(set_to_none=True);_,_,row=joint_backward(model,aux,x[fitix],valid[fitix],t[fitix],event[fitix],ids[fitix],target,available,micro,coefficient);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();row.update(epoch=epoch,preclip_gradient_norm=float(norm),auxiliary_head_trained=coefficient>0,seconds=time.perf_counter()-begin)
        if early:
            risk=predict(model,x[held],valid[held],micro);validation=float(cox(risk,t[held],event[held]));row['validation_cox']=validation
            if not np.isfinite(validation):raise ValueError('Nonfinite inner Cox validation')
            if validation<best:best=validation;best_epoch=epoch;stale=0;saved=trainable_state(model);saved_aux=aux_state(aux)
            else:stale+=1
        trace.append(row)
        if early and stale>=6:break
    if early:
        if saved is None or saved_aux is None:raise ValueError('No finite selected inner state')
        load_trainable(model,saved);aux.load_state_dict(saved_aux,strict=True)
    risk=predict(model,x[held],valid[held],micro).cpu().numpy();main=trainable_state(model);auxiliary=aux_state(aux);load_trainable(model,main);aux.load_state_dict(auxiliary,strict=True)
    if not np.array_equal(risk,predict(model,x[held],valid[held],micro).cpu().numpy()):raise ValueError('Selected main/aux state prediction replay failed')
    description=knowledge_description(model,aux,data,arm,micro,device);final_aux=aux_digest(aux)
    if not coefficient and final_aux!=initial_aux:raise ValueError('Control auxiliary head was accidentally trained')
    info=dict(best_epoch=best_epoch if early else epochs,epochs_run=len(trace),best_inner_loss=best if early else None,trace=trace,knowledge_reader_summary=description,auxiliary_head_trained=coefficient>0,initial_auxiliary_weights_sha256=initial_aux,final_auxiliary_weights_sha256=final_aux)
    return risk,main,auxiliary,info,model,aux
