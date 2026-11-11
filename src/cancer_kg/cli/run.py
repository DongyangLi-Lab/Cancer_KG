"""Portable entry point; executed kernel and Cox implementations are unchanged."""
import argparse
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import numpy as np
from cancer_kg.data.common import ROLES, sha, write_json
from cancer_kg.data.data import load_manifest, load_fit
from cancer_kg.models.kernels import build_kernels
ARMS = ('real_additive_selfquadratic',)
from cancer_kg.training.cox import ALPHAS, eigensystem, solve, risk_loss_gradient, select_alpha
from cancer_kg.evaluation.metrics import score_oof


def run(manifest, output):
    index, roles, digest = load_manifest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    cache = output/'kernels';cache.mkdir()
    build = {}
    # Complete all label-independent support gates before any survival solve.
    for name in ROLES:
        data = load_fit(index[name])
        built = build_kernels(data)
        if not built['support_gate']['passed'] or not built['effective_gate']['passed']:
            raise ValueError('Insufficient common nondegenerate pathway support: '+name)
        path = cache/(name+'.npz')
        np.savez(path, **{arm: built['arms'][arm] for arm in ARMS})
        build[name] = dict(sha256=sha(path),source_sha256=index[name]['sha256'],
            support=built['support_gate'],effective=built['effective_gate'])
        del data,built
        print('Built',name,flush=True)
    write_json(output/'BUILD.json',dict(input_manifest_sha256=digest,roles=build))
    reference=roles['r0_f0_outer']
    risks={arm:np.full(len(reference['time']),np.nan) for arm in ARMS}
    folds=np.full(len(reference['time']),-1,dtype=np.int64)
    completed=0;selection={};fit_receipts=[]
    for arm in ARMS:
        target=output/arm;target.mkdir()
        for outer in range(3):
            validation={alpha:[] for alpha in ALPHAS}
            for inner in range(3):
                name=f'r0_f{outer}_inner{inner}';role=roles[name]
                with np.load(cache/(name+'.npz'),allow_pickle=False) as z:
                    system=eigensystem(z[arm],role['fit'])
                features=system['features'];fit,held=role['fit'],role['held']
                for alpha in ALPHAS:
                    solved=solve(features[fit],role['time'][fit],role['event'][fit],alpha)
                    risk=features[held]@solved['beta']
                    loss,_=risk_loss_gradient(risk,role['time'][held],role['event'][held])
                    validation[alpha].append(dict(loss=loss,events=int(role['event'][held].sum())))
                    name_alpha=f'{name}_alpha{alpha:g}'
                    np.savez(target/(name_alpha+'.npz'),beta=solved['beta'],projection=system['projection'],risk=risk,held=held)
                    fit_receipts.append(dict(arm=arm,role=name,alpha=alpha,
                        gradient_maximum=solved['gradient_maximum'],validation_loss=loss))
                    completed+=1
            alpha,scores=select_alpha(validation)
            name=f'r0_f{outer}_outer';role=roles[name];fit,held=role['fit'],role['held']
            with np.load(cache/(name+'.npz'),allow_pickle=False) as z:
                system=eigensystem(z[arm],fit)
            solved=solve(system['features'][fit],role['time'][fit],role['event'][fit],alpha)
            risk=system['features'][held]@solved['beta']
            risks[arm][held]=risk;folds[held]=outer
            np.savez(target/(name+'.npz'),beta=solved['beta'],projection=system['projection'],risk=risk,held=held)
            selection[f'{arm}/{outer}']=dict(alpha=alpha,inner_scores=scores)
            fit_receipts.append(dict(arm=arm,role=name,alpha=alpha,gradient_maximum=solved['gradient_maximum']))
            completed+=1
        print('Fitted',arm,flush=True)
    assert completed==39
    # Patient-level products stay in the user's private output directory.
    np.savez(output/'private_oof.npz',patient_ids=reference['patient_ids'],time=reference['time'],
             event=reference['event'],fold=folds,**risks)
    report=score_oof(reference['patient_ids'],reference['time'],reference['event'],folds,risks)
    report.update(solves=completed,input_manifest_sha256=digest,selection=selection,
        validation_scope='Portable entry point; original independent full-checkpoint replay is not bundled')
    write_json(output/'SOLVES.json',fit_receipts)
    write_json(output/'SUMMARY.json',report)
    print(json.dumps(report['arms'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.manifest,args.output)
