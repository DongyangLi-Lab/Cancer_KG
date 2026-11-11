"""Outcome-free static context from this fit's saved train-selected gene IDs."""
import numpy as np

def hash_rows(a):
    a=a.astype(np.uint64)
    with np.errstate(over='ignore'):
        x=a[:,0]*np.uint64(0x9E3779B185EBCA87)^a[:,1]*np.uint64(0xC2B2AE3D27D4EB4F)^a[:,2]*np.uint64(0x165667B19E3779F9)
        x^=x>>np.uint64(30);x*=np.uint64(0xBF58476D1CE4E5B9);x^=x>>np.uint64(27)
    return x

def build_bank(triples,train_selected_entities,original_relation_ids,bank_size=16384,local_limit=16):
    anchors=np.unique(np.asarray(train_selected_entities,dtype=np.int64));anchors=anchors[anchors>=0]
    allowed=~np.isin(np.asarray(original_relation_ids)[triples[:,1]],[43,44,45])
    candidates=np.asarray(triples[allowed&(np.isin(triples[:,0],anchors)|np.isin(triples[:,2],anchors))])
    candidates=np.unique(candidates,axis=0)
    if len(candidates)<bank_size:raise ValueError('Insufficient unique public incident facts; refuse fake padding')
    hashes=hash_rows(candidates);order=np.argsort(hashes,kind='stable');chosen=set()
    # Per-gene coverage priority, no expression/heldout values or outcome inputs.
    lists=[]
    for g in anchors:
        incident=order[(candidates[order,0]==g)|(candidates[order,2]==g)]
        rels=np.unique(candidates[incident,1]);pools_g=[incident[candidates[incident,1]==r] for r in rels];row=[];rank=0
        while len(row)<min(local_limit,len(incident)):
            for pool in pools_g:
                if rank<len(pool) and len(row)<local_limit:row.append(pool[rank])
            rank+=1
        lists.append(np.asarray(row,dtype=np.int64))
    for rank in range(local_limit):
        for ix in lists:
            if rank<len(ix) and len(chosen)<bank_size:chosen.add(int(ix[rank]))
    # Relation-balanced fill; deterministic public-only sample hashes.
    pools=[order[candidates[order,1]==r] for r in np.unique(candidates[:,1])];at=[0]*len(pools)
    while len(chosen)<bank_size:
        for j,pool in enumerate(pools):
            while at[j]<len(pool) and int(pool[at[j]]) in chosen:at[j]+=1
            if at[j]<len(pool) and len(chosen)<bank_size:chosen.add(int(pool[at[j]]));at[j]+=1
    selected=candidates[sorted(chosen)]
    # Semantic blocks: relation then canonical incident anchor then opposite endpoint.
    incident_anchor=np.where(np.isin(selected[:,0],anchors),selected[:,0],selected[:,2])
    selected=selected[np.lexsort((selected[:,2],selected[:,0],incident_anchor,selected[:,1]))]
    local={};selected_order=np.argsort(hash_rows(selected),kind='stable')
    for g in anchors:
        incident=selected_order[(selected[selected_order,0]==g)|(selected[selected_order,2]==g)]
        pools_g=[incident[selected[incident,1]==r] for r in np.unique(selected[incident,1])];row=[];rank=0
        while len(row)<min(local_limit,len(incident)):
            for pool in pools_g:
                if rank<len(pool) and len(row)<local_limit:row.append(int(pool[rank]))
            rank+=1
        local[int(g)]=row
    relations,counts=np.unique(selected[:,1],return_counts=True)
    report=dict(candidate_unique_facts=len(candidates),bank_unique_facts=len(selected),anchor_genes=len(anchors),anchor_genes_with_bank_facts=sum(bool(v) for v in local.values()),relations={str(int(original_relation_ids[r])):int(n) for r,n in zip(relations,counts)},layout='relation/canonical incident anchor/endpoint; fixed64 blocks',direction='canonical endpoint A/relation/B, not biological direction',no_patient_state_or_outcomes=True)
    return selected,local,report
