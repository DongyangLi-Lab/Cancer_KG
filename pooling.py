"""Shared pathway readout accepts KG-branch768 ONLY, with no raw-token/RNA skip."""
import torch
from torch import nn
class PathwayReadout(nn.Module):
    def __init__(self,membership,selected_token_ids,mapped_gene_mask,width=768,seed=200042):
        super().__init__();member=torch.as_tensor(membership,dtype=torch.bool);ids=torch.as_tensor(selected_token_ids,dtype=torch.long);mapped=torch.as_tensor(mapped_gene_mask,dtype=torch.bool)
        if member.ndim!=2 or member.shape[1]!=len(ids) or mapped.shape!=ids.shape or len(torch.unique(ids))!=len(ids) or bool(((ids<0)|(ids>=19264)).any()) or bool(member[:,~mapped].any()):raise ValueError('Static real pathway membership must contain only mapped biological selected genes')
        lookup=torch.full((19266,),-1,dtype=torch.long);lookup[ids]=torch.arange(len(ids));self.register_buffer('token_column',lookup,persistent=False);self.register_buffer('membership',member,persistent=False);self.register_buffer('mapped_gene_mask',mapped,persistent=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed);self.readout=nn.Linear(width,1,bias=True)
    def forward(self,kg_branch_out,token_ids,valid):
        if kg_branch_out.ndim!=3 or token_ids.shape!=kg_branch_out.shape[:2] or valid.shape!=token_ids.shape or valid.dtype!=torch.bool or token_ids.dtype!=torch.long:raise ValueError('KG-only states and exact saved token IDs/valid required')
        if bool((((token_ids<0)|(token_ids>=19266))&valid).any()):raise ValueError('Valid saved token ID outside biological/T/S vocabulary')
        safe=token_ids.clamp(0,19265);column=self.token_column[safe];biological=valid&(token_ids<19264)&(column>=0);mapped=self.mapped_gene_mask[column.clamp_min(0)];active=biological&mapped
        for i in range(len(token_ids)):
            selected=token_ids[i][active[i]]
            if len(torch.unique(selected))!=len(selected):raise ValueError('Duplicate valid gene token would double-count pathway membership')
        weights=self.membership[:,column.clamp_min(0)].permute(1,2,0)&active[:,:,None];counts=weights.sum(1);available=counts>0
        # FP32 pooling/loss coordinates; caller chooses the still-pending KG calibration.
        with torch.autocast(device_type=kg_branch_out.device.type,enabled=False):
            used=weights.any(-1)
            if not bool(torch.isfinite(kg_branch_out[used]).all()):raise ValueError('Nonfinite active KG-only pathway states')
            safe_states=torch.where(used[:,:,None],kg_branch_out.float(),torch.zeros_like(kg_branch_out,dtype=torch.float32))
            pooled=torch.einsum('blp,bld->bpd',weights.float(),safe_states)/counts.clamp_min(1)[:,:,None];prediction=self.readout(pooled).squeeze(-1);prediction=prediction*available
        return prediction,available,counts
