import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def rank_cox(x,t,e,folder=None,workers=1):
    from lifelines import CoxPHFitter
    p=np.full(x.shape[1],np.inf);failures=[]
    cache=Path(folder)/'cox_rank_partial.npz' if folder else None
    fingerprint=hashlib.sha256(x.tobytes()+np.asarray(t).tobytes()+np.asarray(e).tobytes()).hexdigest()
    done=np.zeros(x.shape[1],bool)
    if cache and cache.exists():
        z=np.load(cache,allow_pickle=False);p=z['p'];done=z['done']
        if p.shape!=(x.shape[1],) or str(z['fingerprint'].item())!=fingerprint:raise ValueError('Rank checkpoint fit-expression/time/event mismatch')
        failures=json.loads(str(z['failures_json'].item()))
    def one(j):
        if x[:,j].std()<1e-12:return j,np.inf,'constant'
        try:
            model=CoxPHFitter(penalizer=0).fit(pd.DataFrame(dict(value=x[:,j],time=t,event=e)),duration_col='time',event_col='event',show_progress=False)
            v=float(model.summary.loc['value','p']);return j,v if np.isfinite(v) else np.inf,''
        except Exception as exc:return j,np.inf,type(exc).__name__
    from joblib import Parallel,delayed
    pending=np.flatnonzero(~done)
    for begin in range(0,len(pending),128):
        result=Parallel(n_jobs=workers,prefer='threads')(delayed(one)(int(j)) for j in pending[begin:begin+128])
        for j,v,error in result:
            p[j]=v;done[j]=True
            if error:failures.append(dict(gene_index=j,error=error))
        if cache:
            temporary=cache.with_suffix('.partial')
            with temporary.open('wb') as f:np.savez(f,p=p,done=done,fingerprint=fingerprint,failures_json=json.dumps(failures))
            temporary.replace(cache)
        print(json.dumps(dict(fit=str(folder),Cox_genes_complete=int(done.sum()),total=x.shape[1])),flush=True)
    eligible=np.flatnonzero(np.isfinite(p));selected=eligible[np.argsort(p[eligible],kind='stable')[:1024]]
    if len(selected)!=1024:raise ValueError('Fewer than1024 finite train-only univariate Cox genes; do not silently variance-fill')
    return selected,p,failures
