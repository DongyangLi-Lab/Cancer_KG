"""Portable nested orchestration around the executed scientific fit engine."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cancer_kg.data.inputs import PreparedInputs,sha
from cancer_kg.encoders.upstream_loader import load_encoder
from cancer_kg.runtime.flash_policy import native_flash_policy
import cancer_kg.training.fit_core as fit_core

ARM='aligned_real'
MAX_EPOCHS=24

def counts(t,e,r):
    if r.shape!=t.shape or not np.isfinite(r).all():raise ValueError('Finite held risk vector required')
    comparable=e[:,None].astype(bool)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:].astype(bool)))
    delta=r[:,None]-r[None,:]
    return float((((delta>1e-8)+.5*(abs(delta)<=1e-8))*comparable).sum()),int(comparable.sum())

def run(manifest,assets,output):
    arm=ARM
    if not torch.cuda.is_available():raise ValueError('Native CUDA runtime required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    inputs=PreparedInputs(manifest);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(assets).read_text());config['device']='cuda';source={p.relative_to(Path(__file__).resolve().parents[1]).as_posix():sha(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}
    records=[];oof=[];inner={};began=time.perf_counter();completed=False;policy=native_flash_policy(True)
    execution=dict(arm=arm,seed=42,microbatch=4,max_epochs=MAX_EPOCHS,patience=6,lr=.0001,weight_decay=.01,clip=1.,manifest_sha256=sha(manifest),assets_config_sha256=sha(assets),code_sha256=source,selection='first minimum inner Cox; outer integer median of3 inner best epochs',portable_wrapper=True)
    (out/'execution.json').write_text(json.dumps(execution,indent=2))
    try:
        with policy:
            encoder,audit=load_encoder(config);(out/'native_audit.json').write_text(json.dumps(audit,indent=2))
            for fold in range(3):
                for j in [0,1,2,None]:
                    name=f'r0_f{fold}_'+('outer' if j is None else f'inner{j}');data=inputs.load_fit(name)
                    epochs=MAX_EPOCHS if j is not None else int(np.median(inner[fold]));start=time.perf_counter()
                    risk,state,aux,info,model,aux_model=fit_core.fit(encoder,data,arm,epochs,4,j is not None)
                    del model,aux_model
                    if set(state)!={'adapter','pool_norm','head','memory'}:raise ValueError('Only small trainable modules may be saved')
                    path=out/(name+'_trainable.pt')
                    torch.save(dict(trainable=state,auxiliary=aux,fit_id=name,arm=arm,seed=42,epochs=info['best_epoch'],bank_sha256=data['bank_sha256'],data_sha256=data['data_sha256'],target_sha256=data.get('target_sha256'),asset_sha256=config['asset_sha256'],code_sha256=source),path)
                    record=dict(fit_id=name,fold=fold,inner=j,arm=arm,seconds=time.perf_counter()-start,checkpoint_sha256=sha(path),**info);records.append(record)
                    if j is not None:inner.setdefault(fold,[]).append(info['best_epoch'])
                    else:
                        credit,pairs=counts(data['time'][data['held']],data['event'][data['held']],risk);record.update(credit=credit,pairs=pairs)
                        oof.extend(dict(arm=arm,fold=fold,case_id=str(data['patient_ids'][i]),time=float(data['time'][i]),event=int(data['event'][i]),risk=float(r)) for i,r in zip(data['held'],risk))
                        pd.DataFrame(oof).to_csv(out/'private_oof.tsv',sep='\t',index=False)
                    (out/'fit_execution.json').write_text(json.dumps(records,indent=2));del data,state,aux;gc.collect();torch.cuda.empty_cache()
            if len(records)!=12 or len(oof)!=133 or len({r['case_id'] for r in oof})!=133 or sum(r['event'] for r in oof)!=106:raise ValueError('Incomplete12-fit133/106 output')
            outer=[r for r in records if r['inner'] is None]
            if sum(r['pairs'] for r in outer)!=2345:raise ValueError('Original cohort comparable-pair count differs')
            completed=True
    finally:
        receipt=policy.summary();(out/'FLASH_POLICY.json').write_text(json.dumps(receipt,indent=2))
        restored=receipt['original_function_restored'] and receipt['source_file_unchanged'] and receipt['calls']>0 and receipt['effective_deterministic_true_calls']==receipt['calls'] and receipt['effective_deterministic_false_calls']==0
        (out/'status.json').write_text(json.dumps(dict(completed=completed and restored,fit_count=len(records),seconds=time.perf_counter()-began),indent=2))
    if not restored:raise RuntimeError('Native Flash integrity or restoration failed')
    outer=[r for r in records if r['inner'] is None];pairs=sum(r['pairs'] for r in outer);credit=sum(r['credit'] for r in outer)
    if pairs!=2345:raise ValueError('Original cohort comparable-pair count differs')
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,arm=arm,fits=12,patients=133,events=106,credit=credit,pairs=pairs,cindex=credit/pairs,execution_sha256=sha(out/'execution.json'),fits_sha256=sha(out/'fit_execution.json'),private_oof_sha256=sha(out/'private_oof.tsv')),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['manifest','assets','output']:parser.add_argument('--'+name,required=True)
    a=parser.parse_args();run(a.manifest,a.assets,a.output)
