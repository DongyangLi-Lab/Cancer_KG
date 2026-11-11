"""Outcome-free fit-only RNA-derived pathway proxies, preserving saved RNA columns."""
import numpy as np
GENE_SD_THRESHOLD=1e-6
PATHWAY_VARIANCE_THRESHOLD=1e-6
def bind_columns(raw,fit,selected_symbols,selected_indices,canonical_gene_ids,canonical_values,gene_axis_symbols,gene_axis_ids,raw_column_symbols=None):
    symbols=np.asarray(selected_symbols,dtype=str);indices=np.asarray(selected_indices)
    if len(symbols)!=len(set(symbols)) or indices.shape!=symbols.shape or not np.issubdtype(indices.dtype,np.integer) or len(set(indices.tolist()))!=len(indices):raise ValueError('Unique saved selection columns required')
    if np.any(indices<0) or np.any(indices>=len(canonical_gene_ids)) or not np.array_equal(np.asarray(canonical_gene_ids,dtype=str)[indices],symbols):raise ValueError('Selection indices/symbols/canonical axis disagree')
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=len(symbols):raise ValueError('Raw RNA column count differs')
    if raw_column_symbols is not None:
        labels=np.asarray(raw_column_symbols,dtype=str)
        if labels.shape!=symbols.shape or len(set(labels))!=len(labels) or set(labels)!=set(symbols):raise ValueError('Explicit raw labels differ from saved selection')
        order={s:j for j,s in enumerate(labels)};raw=raw[:,[order[s] for s in symbols]]
    fit=np.asarray(fit)
    if fit.ndim!=1 or not np.issubdtype(fit.dtype,np.integer) or len(fit)<2 or len(np.unique(fit))!=len(fit) or np.any(fit<0) or np.any(fit>=len(raw)):raise ValueError('Valid unique training rows required')
    # Only TRAIN rows participate, including this exact column-order check.
    reference=np.asarray(canonical_values)[np.ix_(fit,indices)].astype(np.float32)
    if not np.array_equal(raw[fit].astype(np.float32),reference):raise ValueError('Unlabeled raw column permutation or source mismatch; never infer order from KG IDs')
    axis_symbols=np.asarray(gene_axis_symbols,dtype=str);axis_ids=np.asarray(gene_axis_ids)
    if len(set(axis_symbols))!=len(axis_symbols) or len(set(axis_ids.tolist()))!=len(axis_ids):raise ValueError('Unique scF symbol/token axis required')
    lookup=dict(zip(axis_symbols,axis_ids));tokens=np.array([lookup[s] for s in symbols],np.int64)
    if np.any(tokens<0) or np.any(tokens>=19264):raise ValueError('Selected RNA must map to biological scF token IDs')
    return raw,tokens
def memberships(bank,selected_token_ids,true_entity,entity_rows,node_types,node_ids,original_relation_ids,min_genes=5,max_pathways=64):
    bank=np.asarray(bank);token=np.asarray(selected_token_ids);compact=np.asarray(true_entity)[token];rows=np.full(len(token),-1,np.int64);mapped=compact>=0;rows[mapped]=np.asarray(entity_rows)[compact[mapped]]
    if len(set(rows[mapped].tolist()))!=int(mapped.sum()):raise ValueError('Two saved RNA columns map to one gene entity')
    column={int(g):j for j,g in enumerate(rows) if g>=0};groups={};relation_ids=np.asarray(original_relation_ids)
    if relation_ids[29]!=30 or relation_ids[33]!=34:raise ValueError('Locked prepared/public relation index mapping differs')
    for a,r,b in bank:
        if int(r) not in [29,33]:continue
        ta,tb=node_types[a],node_types[b]
        if ta=='gene/protein' and tb=='pathway':gene,pathway=int(a),int(b)
        elif tb=='gene/protein' and ta=='pathway':gene,pathway=int(b),int(a)
        else:raise ValueError('Pathway relation is not unambiguous gene/protein to pathway')
        if gene in column:groups.setdefault(pathway,set()).add(column[gene])
    eligible=[(pathway,cols) for pathway,cols in groups.items() if len(cols)>=min_genes];eligible.sort(key=lambda row:(-len(row[1]),int(node_ids[row[0]])))
    selected=eligible[:max_pathways];matrix=np.zeros((len(selected),len(token)),bool)
    for i,(_,cols) in enumerate(selected):matrix[i,list(cols)]=True
    return matrix,np.array([r[0] for r in selected],np.int64),dict(eligible_before_cap=len(eligible),selected_before_variance_filter=len(selected),min_distinct_genes=min_genes,max_pathways=max_pathways,selection_rule='membercount descending then original raw node_id ascending; cap BEFORE variance filter; no backfill',duplicate_relation_membership_merged=True,mapped_selected_genes=int(mapped.sum()))
def fit_targets(raw,fit,membership,pathway_rows):
    fit=np.asarray(fit);raw=np.asarray(raw)
    if fit.ndim!=1 or not np.issubdtype(fit.dtype,np.integer) or len(fit)<2 or len(np.unique(fit))!=len(fit) or np.any(fit<0) or np.any(fit>=len(raw)):raise ValueError('Unique registered training rows required')
    x=raw[fit].astype(np.float64);member=np.asarray(membership,bool)
    if x.ndim!=2 or member.ndim!=2 or member.shape[1]!=x.shape[1] or len(member)==0 or np.any(member.sum(1)<5) or not np.isfinite(x).all():raise ValueError('Finite training RNA and >=5-gene pathways required')
    mean=x.mean(0);observed_sd=x.std(0,ddof=0);constant=observed_sd<=GENE_SD_THRESHOLD;scale=np.where(constant,1.,observed_sd);z=(x-mean)/scale
    activity=z@member.T/member.sum(1)[None,:];pathway_mean=activity.mean(0);variance=activity.var(0,ddof=0);keep=variance>PATHWAY_VARIANCE_THRESHOLD
    if not keep.any():raise ValueError('All selected pathway targets have low training variance; stop without changing threshold')
    pathway_scale=np.sqrt(variance[keep]);target=(activity[:,keep]-pathway_mean[keep])/pathway_scale
    stats=dict(gene_mean=mean,gene_observed_std=observed_sd,gene_scale=scale,constant_gene_mask=constant,pathway_mean=pathway_mean[keep],pathway_scale=pathway_scale,pathway_variance_before_filter=variance,pathway_keep_mask=keep,membership=member[keep],pathway_rows=np.asarray(pathway_rows)[keep],fit_indices=fit.copy(),targets_train=target.astype(np.float32))
    return stats
def transform_rows(raw_rows,stats):
    """Pure application of fitted statistics; preparation never calls this on held rows."""
    x=np.asarray(raw_rows,dtype=np.float64);z=(x-stats['gene_mean'])/stats['gene_scale'];member=stats['membership'];activity=z@member.T/member.sum(1)[None,:]
    return ((activity-stats['pathway_mean'])/stats['pathway_scale']).astype(np.float32)
