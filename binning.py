"""Fit-independent native ranks and explicit exact-unique public mapping."""
import hashlib,re
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fasta_mapping(path):
    out={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.startswith('>'):continue
            match=re.search(r'\bGN=([^ ]+)',line)
            if match:out.setdefault(match.group(1),set()).add(line.split('|')[1])
    return {k:next(iter(v)) for k,v in out.items() if len(v)==1}, {k:sorted(v) for k,v in out.items() if len(v)>1}

def native_bins(values,num_bins=10,seed=0):
    """Official lexsort random-tie ranks, finite signed/zero values observed.

    Source-axis order is fixed before this call. Each patient resets seed0;
    rank all mapped observed proteins BEFORE a fit-specific selection.
    """
    values=np.asarray(values,dtype=np.float64)
    observed=np.isfinite(values);v=values[observed];result=np.zeros(len(values),dtype=np.uint8)
    if len(v):
        tie=np.random.default_rng(seed).random(len(v));order=np.lexsort((tie,v))
        ranks=np.empty(len(v),dtype=np.int64);ranks[order]=np.arange(1,len(v)+1)
        result[observed]=np.ceil(ranks/len(v)*num_bins).astype(np.uint8)
    return result

def fit_candidates(x,fit):
    a=np.asarray(x,dtype=np.float64)[fit];n=np.isfinite(a).sum(0)
    eligible=(n>=10)&(n>=len(fit)*.5)
    median=np.nanmedian(a,axis=0);filled=np.where(np.isfinite(a),a,median)
    eligible &= np.isfinite(median)&(np.std(filled,axis=0)>0)
    return eligible,median,filled
