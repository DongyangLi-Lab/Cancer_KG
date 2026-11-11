"""Prototype: patient-conditioned sparse read of a shared frozen fact bank."""
import math
import torch
from torch import nn

class KGReader(nn.Module):
    def __init__(self,facts,anchors,local_index,in_dim=768,width=64,block_size=64,blocks_read=8,local_role=None):
        super().__init__()
        if facts.ndim!=2 or len(facts)%block_size or local_index.ndim!=2 or local_index.shape[1]>16:raise ValueError('Shared fixed blocks/local budget')
        self.register_buffer('facts',facts.detach().clone(),persistent=False);self.register_buffer('anchors',anchors.detach().clone(),persistent=False);self.register_buffer('local_index',local_index.long().clone(),persistent=False)
        self.block_size=block_size;self.blocks_read=blocks_read;self.width=width
        role=torch.full_like(local_index,2) if local_role is None else local_role
        self.register_buffer('local_role',role.long().clone(),persistent=False)
        self.incidence_role=nn.Embedding(3,width)
        self.query=nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,width,bias=False));self.anchor_query=nn.Linear(anchors.shape[1],width,bias=False)
        self.key=nn.Linear(facts.shape[1],width,bias=False);self.value=nn.Linear(facts.shape[1],width,bias=False);self.out=nn.Linear(width,in_dim,bias=False);self.gate=nn.Parameter(torch.tensor(.01))
    def routes(self,tokens,valid,entity):
        mapped=valid&(entity>=0);safe=entity.clamp_min(0)
        q=self.query(tokens.float())+self.anchor_query(self.anchors[safe].float())
        q=q*mapped[...,None];patient=q.sum(1)/mapped.sum(1,keepdim=True).clamp_min(1)
        # Trainable projections recomputed each forward; only raw frozen bank persists.
        keys=self.key(self.facts.float());values=self.value(self.facts.float())
        block_keys=keys.reshape(-1,self.block_size,self.width).mean(1);scores=patient@block_keys.T/math.sqrt(self.width)
        blocks=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:self.blocks_read]
        global_index=(blocks[:,:,None]*self.block_size+torch.arange(self.block_size,device=blocks.device)).flatten(1)
        local=self.local_index[safe];local=torch.where(mapped[...,None],local,-1)
        # Global slots already present are masked in local slots, with constant compute shape.
        duplicate=(local[...,None]==global_index[:,None,None,:]).any(-1);local_valid=(local>=0)&~duplicate
        return q,keys,values,global_index,local,local_valid,self.local_role[safe],mapped
    def read_selected(self,q,keys,values,global_index,local,local_valid,role,mapped):
        global_keys=keys[global_index];global_values=values[global_index]
        global_logits=torch.einsum('bld,bkd->blk',q,global_keys)/math.sqrt(self.width)
        # Role refers only to canonical endpoint A/B, never biological direction.
        local_keys=keys[local.clamp_min(0)];local_values=values[local.clamp_min(0)]
        local_keys=local_keys+self.incidence_role(role);local_values=local_values+self.incidence_role(role)
        local_logits=(q[:,:,None,:]*local_keys).sum(-1)/math.sqrt(self.width)
        logits=torch.cat([global_logits,local_logits],-1);mask=torch.cat([mapped[:,:,None].expand_as(global_logits),local_valid&mapped[:,:,None]],-1)
        weights=torch.softmax(logits.masked_fill(~mask,-1e30),-1)*mask
        a,b=weights.split([global_logits.shape[-1],local_logits.shape[-1]],-1)
        result=torch.einsum('blk,bkd->bld',a,global_values)+(b[...,None]*local_values).sum(-2)
        return result,weights
    def forward(self,tokens,valid,entity,return_receipt=False):
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            routes=self.routes(tokens,valid,entity);result,weights=self.read_selected(*routes)
            residual=self.gate*self.out(result);output=tokens+residual.to(tokens.dtype)*routes[-1][...,None]
            if return_receipt:return output,dict(global_indices=routes[3].detach(),local_indices=routes[4].detach(),local_valid=routes[5].detach(),mapped=routes[7].detach(),fact_budget=routes[3].shape[1]+routes[4].shape[-1],weights=weights.detach())
            return output
