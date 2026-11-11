"""Portable I/O around the executed C_pool training loop."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import cancer_kg.models.model as m
HEAD_SHA='b437930f10604840a82144401946f8c153f1a95ad9ae8455d14bcb57d2049ba7'
CANDIDATES=[dict(lr=lr,wd=wd) for lr in [1e-4,1e-3] for wd in [1e-4,.01,.1]]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked(path,digest):
    if sha(path)!=digest:raise ValueError('SHA mismatch: '+str(path))
    return Path(path)

def design(z,m):
    scf=np.asarray(z['scf']);omics=np.asarray(z['omics']);fit=np.asarray(z['fit'])
    if scf.shape!=(133,768) or omics.shape!=(133,256) or not np.isfinite(scf).all() or not np.isfinite(omics).all():raise ValueError('Finite frozen pooled matrices133x768/256 required')
    normalized,stats=m.standardize(omics,fit);x=np.zeros((133,1792),dtype=np.float64);x[:,1024:]=scf;x[:,:256]=normalized
    return x,dict(omics=stats,scf='already exact historical fit-standardized bundle',scf_mean=z['scf_mean'],scf_sd=z['scf_sd'])

def finite(r,n):
    if np.asarray(r).shape!=(n,) or not np.isfinite(r).all():raise ValueError('Risk must be finite vector matching held patients')

def counts(t,e,r):
    finite(r,len(t));comp=e[:,None].astype(bool)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:].astype(bool)));delta=r[:,None]-r[None,:]
    return float((((delta>1e-8)+.5*(abs(delta)<=1e-8))*comp).sum()),int(comp.sum())

def choose(groups):
    scores=[sum(r['credit'] for r in group)/sum(r['pairs'] for r in group) for group in groups];return int(np.argmax(scores)),scores

def validate_metadata(z):
    ids=np.asarray(z['patient_ids']);t=np.asarray(z['time']);e=np.asarray(z['event'])
    if ids.shape!=(133,) or ids.dtype.kind!='U' or any(not value.strip() for value in ids) or len(set(ids.tolist()))!=133:raise ValueError('Unique nonempty Unicode patient IDs required')
    if t.shape!=(133,) or t.dtype.kind not in 'iuf' or not np.isfinite(t).all() or np.any(t<=0):raise ValueError('Finite positive133-vector times required')
    if e.shape!=(133,) or e.dtype.kind not in 'biuf' or not np.isfinite(e).all() or not np.isin(e,[0,1]).all():raise ValueError('Binary133-vector events required')
    for key in ['fit','held']:
        a=np.asarray(z[key])
        if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or len(set(a.tolist()))!=len(a) or np.any(a<0) or np.any(a>=133):raise ValueError('Nonempty unique role indices required')
        if not e[a].any():raise ValueError('Each fit and held role requires an event')

def validate_fits(fits):
    for z in fits.values():validate_metadata(z)
    expected={f'r0_f{f}_{s}' for f in range(3) for s in ['outer','inner0','inner1','inner2']}
    if set(fits)!=expected:raise ValueError('Require exactly12 frozen fits')
    ref=fits['r0_f0_outer'];ids=ref['patient_ids'].astype(str)
    if len(ids)!=133 or len(set(ids))!=133 or ref['event'].sum()!=106:raise ValueError('133/106 cohort mismatch')
    outerheld=[]
    for name,z in fits.items():
        for k in ['patient_ids','time','event']:np.testing.assert_array_equal(z[k],ref[k])
        for k in ['fit','held']:
            a=np.asarray(z[k])
            if a.dtype.kind not in 'iu' or len(set(a))!=len(a) or np.any(a<0) or np.any(a>=133):raise ValueError('Invalid fit indices')
        if set(z['fit'])&set(z['held']):raise ValueError('Role overlap')
    for fold in range(3):
        outer=fits[f'r0_f{fold}_outer'];train=set(outer['fit']);held=set(outer['held']);outerheld.extend(held)
        if len(held)!=[45,44,44][fold] or train|held!=set(range(133)):raise ValueError('Outer partition mismatch')
        validation=[]
        for j in range(3):
            z=fits[f'r0_f{fold}_inner{j}'];a,b=set(z['fit']),set(z['held'])
            if a|b!=train or not a<=train or not b<=train:raise ValueError('Inner outside frozen outer train')
            validation.extend(b)
        if len(validation)!=len(set(validation)) or set(validation)!=train:raise ValueError('Inner held coverage mismatch')
    if len(outerheld)!=133 or len(set(outerheld))!=133:raise ValueError('Outer held coverage mismatch')

def run(c):
    m.torch.set_num_threads(4);doc=json.loads(Path(c['packaged_manifest']).read_text());fits={};bundle_sha={}
    code={p.relative_to(Path(__file__).resolve().parents[1]).as_posix():sha(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}
    for row in doc['fits']:
        path=Path(row['file'])
        if not path.is_absolute():path=Path(c['packaged_manifest']).parent/path
        if row['fit_id'] in fits:raise ValueError('Duplicate packaged fit')
        with np.load(checked(path,row['sha256']),allow_pickle=False) as z:fits[row['fit_id']]={k:z[k].copy() for k in z.files}
        bundle_sha[row['fit_id']]=row['sha256']
    validate_fits(fits);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();records=[];selected=[];oof=[];manifest=[]
    (out/'execution.json').write_text(json.dumps(dict(mode='pooled_control',config=c,source_sha256=HEAD_SHA,code_sha256=code,recipe=dict(head='small64/16/dropout.2',seed=42,candidates=CANDIDATES,max_epochs=150,patience=15,selection='sum inner credit / sum inner pairs; firsttie; outer integer median bestepoch'),historical_reference_replay_not_executed_by_portable_wrapper=True),indent=2))
    def train(name,ci,epochs,early):
        z=fits[name];x,norm=design(z,m);step=time.perf_counter();r,state,info=m.mlp(x,z['time'],z['event'],z['fit'],z['held'],'small',42,**CANDIDATES[ci],epochs=epochs,early=early,patience=15);finite(r,len(z['held']));replay=m.replay(state,x[z['held']]);finite(replay,len(r));np.testing.assert_allclose(replay,r,rtol=1e-6,atol=1e-6);credit,pairs=counts(z['time'][z['held']],z['event'][z['held']],r)
        record=dict(fit_id=name,arm='C_pool',candidate_index=ci,candidate=CANDIDATES[ci],credit=credit,pairs=pairs,info=info,seconds=time.perf_counter()-step,bundle_sha256=bundle_sha[name]);target=out/(name+('_candidate'+str(ci) if early else '')+'.pkl');target.write_bytes(pickle.dumps(dict(state=state,normalization=norm,record=record,source_sha256=HEAD_SHA)));manifest.append(dict(file=target.name,sha256=sha(target),fit_id=name,candidate_index=ci,inner=early));return r,record
    for fold in range(3):
        groups=[]
        for ci in range(6):
            group=[]
            for j in range(3):
                _,record=train(f'r0_f{fold}_inner{j}',ci,150,True);records.append(record);group.append(record);(out/'inner_candidates.json').write_text(json.dumps(records,indent=2))
            groups.append(group)
        best,scores=choose(groups);epochs=max(1,int(np.median([r['info']['best_epoch'] for r in groups[best]])));choice=dict(fold=fold,arm='C_pool',candidate_index=best,candidate=CANDIDATES[best],inner_c=scores,epochs=epochs);selected.append(choice)
        name=f'r0_f{fold}_outer';risk,record=train(name,best,epochs,False);choice['outer_credit']=record['credit'];choice['outer_pairs']=record['pairs'];z=fits[name]
        oof.extend(dict(arm='C_pool',fold=fold,case_id=str(z['patient_ids'][i]),time=float(z['time'][i]),event=int(z['event'][i]),risk=float(r)) for i,r in zip(z['held'],risk))
        import pandas as pd
        pd.DataFrame(oof).to_csv(out/'private_oof.tsv',sep='\t',index=False);(out/'selected.json').write_text(json.dumps(selected,indent=2));(out/'CHECKPOINT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    if len(records)!=54 or len(manifest)!=57:raise ValueError('Incomplete54inner+3outer training')
    credit=sum(s['outer_credit'] for s in selected);pairs=sum(s['outer_pairs'] for s in selected)
    if pairs!=2345:raise ValueError('Frozen comparable-pair count mismatch')
    runtime=dict(seconds=time.perf_counter()-started,torch=m.torch.__version__,numpy=np.__version__,threads=m.torch.get_num_threads(),fit_count=57)
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2));(out/'COMPLETE.json').write_text(json.dumps(dict(mode='pooled_control',complete=True,credit=credit,pairs=pairs,cindex=credit/pairs,inner_fits=54,outer_fits=3,private_oof_sha256=sha(out/'private_oof.tsv'),selected_sha256=sha(out/'selected.json'),inner_candidates_sha256=sha(out/'inner_candidates.json'),checkpoint_manifest_sha256=sha(out/'CHECKPOINT_MANIFEST.json'),source_sha256=HEAD_SHA,code_sha256=code,config=c,runtime=runtime),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    run(dict(packaged_manifest=str(Path(a.manifest).resolve()),output=str(Path(a.output).resolve())))
