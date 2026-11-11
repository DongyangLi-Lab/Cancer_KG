"""Synthetic negative cases for endpoint axes and nested role isolation."""
from copy import deepcopy
import numpy as np

def run_checks(validator):
    roles={}
    for fold in range(3):
        outer_held=np.arange(6*fold,6*fold+6)
        outer_fit=np.setdiff1d(np.arange(18),outer_held)
        for role in ('inner0','inner1','inner2','outer'):
            held=outer_held if role=='outer' else np.array_split(outer_fit,3)[int(role[-1])]
            fit=outer_fit if role=='outer' else np.setdiff1d(outer_fit,held)
            roles[f'r0_f{fold}_{role}']=dict(patient_ids=np.array([f'synthetic{i}' for i in range(18)]),time=np.arange(1.,19.),event=np.ones(18),fit=fit,held=held)
    validator(roles)
    for kind in ('zero_time','bad_event','empty_id','bytes_id','float_index','duplicate_index','empty_held','outer_leak','time_shape'):
        bad=deepcopy(roles)
        for data in bad.values():
            if kind=='zero_time':data['time'][0]=0
            elif kind=='bad_event':data['event'][0]=2
            elif kind=='empty_id':data['patient_ids'][0]=''
            elif kind=='bytes_id':data['patient_ids']=data['patient_ids'].astype('S')
            elif kind=='float_index':data['fit']=data['fit'].astype(float)
            elif kind=='duplicate_index':data['fit'][0]=data['fit'][1]
            elif kind=='empty_held':data['held']=np.array([],dtype=np.int64)
            elif kind=='time_shape':data['time']=data['time'][:-1]
        if kind=='outer_leak':bad['r0_f0_inner0']['fit'][0]=bad['r0_f0_outer']['held'][0]
        try:validator(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted: '+kind)
