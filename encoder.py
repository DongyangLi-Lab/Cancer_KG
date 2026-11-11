"""Strict original-vocabulary frozen OmicsFM, final real-protein token max."""
import importlib.util
from pathlib import Path
import torch

def load(path,architecture,device='cpu'):
    # Checkpoint is public trusted, pinned source; never load arbitrary user pickle.
    z=torch.load(path,map_location='cpu',weights_only=False)
    c=z['config'];m=c['model'];weights={k.removeprefix('_orig_mod.'):v for k,v in z['model'].items()}
    if len(weights)!=len(z['model']):raise ValueError('Compiled-prefix collision')
    if m.get('use_esmc',False) or m.get('protein_groups',{}).get('enabled',False):raise ValueError('Only learned individual identity checkpoint registered')
    names=list(z['feature_names']);n=weights['feature_emb.weight'].shape[0]-1
    if len(names)!=n or len(set(names))!=n:raise ValueError('Checkpoint vocabulary mismatch')
    spec=importlib.util.spec_from_file_location('omicsfm_frozen_architecture',architecture);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg={k:m[k] for k in ('d_model','n_heads','n_layers','d_ff','dropout')};cfg.update(num_features=n,num_bins=c['data']['num_bins'],use_esmc=False)
    model=module.ProtGPT(cfg);model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False).to(device)
    return model,{str(v):i+1 for i,v in enumerate(names)},c

def batch(ids,bins,device='cpu'):
    if len(ids)!=len(bins) or not len(ids) or len(ids)>1024:raise ValueError('Require 1..1024 observed selected proteins')
    if any(int(v)<=0 for v in ids) or any(not 1<=int(v)<=10 for v in bins):raise ValueError('Invalid real protein ID/bin')
    return dict(role=torch.ones((1,len(ids)+1),dtype=torch.long,device=device),feature_ids=torch.tensor([[[0]]+[[int(v)] for v in ids]],device=device),bin_values=torch.tensor([[0]+list(map(int,bins))],device=device))

@torch.inference_mode()
def pooled(model,b):
    out=model(b)['transformer_out'];valid=(b['role']==1)&(b['feature_ids'][...,0]>0)
    if not valid.any(1).all():raise ValueError('No real observed selected protein')
    return out.masked_fill(~valid[...,None],-torch.inf).amax(1).float()
