import numpy as np

def align_row(values, observed, source_ids, target_ids):
    """Missing gets numerical zero for native APIs, with separate observed mask."""
    index = {str(g): i for i, g in enumerate(source_ids)}
    aligned = np.zeros(len(target_ids), dtype=np.float32)
    mask = np.zeros(len(target_ids), dtype=bool)
    for j, gene in enumerate(target_ids):
        i = index.get(str(gene))
        if i is not None and observed[i]:
            aligned[j], mask[j] = values[i], True
    return aligned, mask



def egsp_bulk(values,observed,mode,target_sum=None):
    if mode!='T':raise ValueError('EGSP fit-specific cache requires T normalized_log1p')
    x=np.where(observed,values,0).astype(np.float64);depth=np.log10(x.sum()+1e-8)
    if depth<=0 or not np.isfinite(depth) or not (x>0).any():raise ValueError('Invalid selected-expression EGSP depth')
    return np.concatenate([x,[depth,depth]]).astype(np.float32)
