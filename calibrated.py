"""Executed calibrated KG reader; historical import paths removed."""
import math
import torch
from torch import nn
from kg_reader import KGReader
from token_model import TokenAdapterTahoe,canonicalize_tokens

def calibrate(tokens,u,mapped,gate):
    k=u/torch.sqrt(u.square().mean(-1,keepdim=True)+1e-8);k=k*mapped[:,:,None]
    token_rms=torch.sqrt(tokens.float().square().mean(-1,keepdim=True)).detach()
    return tokens+gate*token_rms*k,k

class CalibratedReader(KGReader):
    def read_selected(self,q,keys,values,global_index,local,local_valid,role,mapped):
        g=global_index.shape[1];global_logits=torch.einsum('bld,bkd->blk',q,keys[global_index])/math.sqrt(self.width)-math.log(g)
        add=self.incidence_role(role);local_keys=keys[local.clamp_min(0)]+add;local_values=values[local.clamp_min(0)]+add
        count=local_valid.sum(-1,keepdim=True);local_logits=(q[:,:,None]*local_keys).sum(-1)/math.sqrt(self.width)-count.clamp_min(1).float().log()
        mask=torch.cat([mapped[:,:,None].expand_as(global_logits),local_valid&mapped[:,:,None]],-1);weights=torch.softmax(torch.cat([global_logits,local_logits],-1).masked_fill(~mask,-1e30),-1)*mask
        a,b=weights.split([g,local.shape[-1]],-1);result=torch.einsum('blk,bkd->bld',a,values[global_index])+(b[:,:,:,None]*local_values).sum(2)
        return result,weights
    def knowledge(self,tokens,valid,entity):
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            routes=self.routes(tokens.float(),valid,entity);read,weights=self.read_selected(*routes);u=self.out(read);states,k=calibrate(tokens.float(),u,routes[-1],self.gate)
            return states,k,dict(routes=routes,weights=weights,ungated=u)
    def forward(self,tokens,valid,entity):return self.knowledge(tokens,valid,entity)[0]

class AlignmentModel(TokenAdapterTahoe):
    def __init__(self,encoder,facts,anchors,local_index,local_role):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42);super().__init__(encoder,checkpoint_blocks=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(100042);self.memory=CalibratedReader(facts,anchors,local_index,local_role=local_role)
    def knowledge(self,packed,valid):
        if packed.shape[-1]!=769 or packed.dtype!=torch.float32:raise ValueError('Exact original FP32 state/entity sidechannel required')
        packed,valid=canonicalize_tokens(packed,valid);entity=packed[:,:,-1].long()
        if not torch.equal(entity.float(),packed[:,:,-1]) or bool(((entity < -1)|(entity>=len(self.memory.anchors))).any()):raise ValueError('Invalid compact entity ID')
        states,k,receipt=self.memory.knowledge(packed[:,:,:768],valid,entity)
        return states,k,valid,receipt
    def forward(self,packed,valid):
        states,_,valid,_=self.knowledge(packed,valid);return super().forward(states,valid)

def auxiliary_forward(model,aux,packed,valid,token_ids):
    joined=torch.cat([packed,token_ids[:,:,None].float()],-1);joined,valid=canonicalize_tokens(joined,valid)
    _,k,valid,_=model.knowledge(joined[:,:,:769],valid)
    return aux(k,joined[:,:,-1].long(),valid)
