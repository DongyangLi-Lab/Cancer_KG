import numpy as np
from pooling import PathwayReadout
LAMBDA={'aligned_real':.1}

def availability(token_ids,valid,target):
    ids=target['selected_token_ids'];lookup=np.full(19266,-1,np.int64);lookup[ids]=np.arange(len(ids));safe=np.clip(token_ids,0,19265);column=lookup[safe];active=valid&(token_ids>=0)&(token_ids<19264)&(column>=0);active&=target['mapped_gene_mask'][np.maximum(column,0)]
    member=target['membership'][:,np.maximum(column,0)].transpose(1,2,0)&active[:,:,None]
    return member.sum(1)>0

def make_aux(data):
    t=data['target'];return PathwayReadout(t['membership'],t['selected_token_ids'],t['mapped_gene_mask'])
