"""Synthetic-only computation, isolation and checkpoint smoke."""
import json
import numpy as np
import torch
import cancer_kg.models.model as model
import cancer_kg.cli.train as train

torch.set_num_threads(1)
rng=np.random.default_rng(3)
z=dict(scf=rng.normal(size=(133,768)),omics=rng.normal(size=(133,256)),fit=np.arange(80),scf_mean=np.zeros(768),scf_sd=np.ones(768))
x,stats=train.design(z,model)
np.testing.assert_array_equal(x[:,1024:],z['scf'])
assert np.all(x[:,256:1024]==0)
before=x[z['fit']].copy();z['omics'][80:]=1e6
y,other=train.design(z,model)
np.testing.assert_array_equal(before,y[z['fit']])
np.testing.assert_array_equal(stats['omics']['mean'],other['omics']['mean'])
assert train.choose([[dict(credit=2,pairs=4)]*3,[dict(credit=1,pairs=2)]*3])[0]==0
t=np.array([1,1,2,3,4,5,6,7,8,9,10,11.],float);e=np.array([1,1,0,1,1,1,1,0,1,1,0,1],bool)
r,state,info=model.mlp(x[:12],t,e,np.arange(8),np.arange(8,12),'small',42,1e-4,.01,2,early=True,patience=15)
np.testing.assert_array_equal(model.replay(state,x[8:12]),r)
assert info['parameters']==115808 and np.isfinite(r).all()

# Complete synthetic133/106 roles and malformed-schema negative cases.
def synthetic_roles():
    ids=np.array(['synthetic_%03d'%i for i in range(133)])
    events=(np.arange(133)%5!=0).astype(np.int64)
    assert events.sum()==106
    rows={};allix=np.arange(133)
    for f,held in enumerate(np.array_split(allix,3)):
        fit=np.setdiff1d(allix,held)
        rows[f'r0_f{f}_outer']=dict(patient_ids=ids.copy(),time=allix.astype(float)+1,event=events.copy(),fit=fit,held=held)
        for j,val in enumerate(np.array_split(fit,3)):
            rows[f'r0_f{f}_inner{j}']=dict(patient_ids=ids.copy(),time=allix.astype(float)+1,event=events.copy(),fit=np.setdiff1d(fit,val),held=val)
    return rows
roles=synthetic_roles();train.validate_fits(roles)
bad_values=[('time',np.ones((133,1))),('time',np.full(133,np.nan)),('time',np.zeros(133)),('time',np.full(133,np.inf)),('event',np.full(133,.5)),('event',np.ones((133,1))),('event',np.full(133,np.inf)),('patient_ids',np.arange(133)),('patient_ids',np.array([' ']*133)),('patient_ids',np.array(['']+['id%d'%i for i in range(132)])),('fit',np.array([],dtype=int)),('held',np.array([],dtype=int)),('fit',np.array([[1,2]],dtype=int))]
rejected=0
for key,value in bad_values:
    bad=synthetic_roles();bad['r0_f0_outer'][key]=value
    try:train.validate_fits(bad)
    except (ValueError,AssertionError):rejected+=1
    else:raise AssertionError('Accepted malformed '+key)
for role in ['fit','held']:
    bad=synthetic_roles();bad['r0_f0_outer'][role]=np.array([0,5,10])
    try:train.validate_fits(bad)
    except (ValueError,AssertionError):rejected+=1
    else:raise AssertionError('Accepted event-free '+role)
assert rejected==15
print(json.dumps(dict(passed=True,synthetic_only=True,input_negative_cases=15,parameters=info['parameters'],numpy=np.__version__,torch=torch.__version__)))
