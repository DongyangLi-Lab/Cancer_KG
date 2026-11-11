"""Portable SHA-bound prepared inputs; no source patient matrix or fit selection here."""
import hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def checked(base,row):
    p=Path(row['file']);p=p if p.is_absolute() else base/p
    if sha(p)!=row['sha256']:raise ValueError('Input SHA mismatch: '+str(p))
    return p

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def validate_metadata(z):
    ids=np.asarray(z['patient_ids']);t=np.asarray(z['time']);e=np.asarray(z['event'])
    if ids.shape!=(133,) or ids.dtype.kind!='U' or any(not v.strip() for v in ids) or len(set(ids.tolist()))!=133:raise ValueError('Unique nonempty Unicode133 patient IDs required')
    if t.shape!=(133,) or t.dtype.kind not in 'iuf' or not np.isfinite(t).all() or np.any(t<=0):raise ValueError('Finite positive time vector required')
    if e.shape!=(133,) or e.dtype.kind not in 'biuf' or not np.isfinite(e).all() or not np.isin(e,[0,1]).all() or e.sum()!=106:raise ValueError('Binary106/133 event vector required')
    for key in ['fit','held']:
        a=np.asarray(z[key])
        if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or len(set(a.tolist()))!=len(a) or np.any(a<0) or np.any(a>=133):raise ValueError('Nonempty unique role indices required')
        if not e[a].any():raise ValueError('Every fit/held role requires events')
    if set(z['fit'])&set(z['held']):raise ValueError('Role overlap')

def validate_roles(fits):
    expected={f'r0_f{f}_{s}' for f in range(3) for s in ['outer','inner0','inner1','inner2']}
    if set(fits)!=expected:raise ValueError('Exactly12 original named roles required')
    ref=fits['r0_f0_outer'];outerheld=[]
    for z in fits.values():
        validate_metadata(z)
        for k in ['patient_ids','time','event']:np.testing.assert_array_equal(z[k],ref[k])
    for f in range(3):
        z=fits[f'r0_f{f}_outer'];train=set(z['fit']);held=set(z['held']);validation=[]
        if len(held)!=[45,44,44][f] or train|held!=set(range(133)):raise ValueError('Outer partition')
        outerheld.extend(held)
        for j in range(3):
            inner=fits[f'r0_f{f}_inner{j}']
            if set(inner['fit'])|set(inner['held'])!=train:raise ValueError('Inner touches outer held')
            validation.extend(inner['held'].tolist())
        if len(validation)!=len(set(validation)) or set(validation)!=train:raise ValueError('Inner held partition')
    if len(outerheld)!=133 or len(set(outerheld))!=133:raise ValueError('Outer held partition')

def validate_bank(b):
    facts=b['facts'];anchors=b['anchors'];local=b['local_index'];role=b['local_role'];bank=b['bank'];rows=b['entity_rows'];g=len(anchors)
    if facts.shape!=(16384,192) or facts.dtype!=np.float32 or not np.isfinite(facts).all():raise ValueError('Original16384x192 FP32 fact bank required')
    if anchors.shape!=(g,64) or not 0<g<=1024 or anchors.dtype!=np.float32 or not np.isfinite(anchors).all():raise ValueError('Finite64-width frozen anchors required')
    if bank.shape!=(16384,3) or bank.dtype.kind not in 'iu' or np.any(bank<0) or len(np.unique(bank,axis=0))!=16384:raise ValueError('Unique original bank triples required')
    if rows.shape!=(g,) or rows.dtype.kind not in 'iu' or np.any(rows<0) or not np.array_equal(rows,np.unique(rows)):raise ValueError('Original sorted unique entity rows required')
    if local.shape!=(g,16) or local.dtype.kind!='i' or np.any(local < -1) or np.any(local>=len(bank)) or role.shape!=local.shape or role.dtype.kind not in 'iu' or not np.isin(role,[0,1,2]).all():raise ValueError('Local bank indices/roles differ')
    for j in range(g):
        valid=local[j]>=0;ix=local[j,valid]
        if len(np.unique(ix))!=len(ix) or not np.all((bank[ix,0]==rows[j])|(bank[ix,2]==rows[j])):raise ValueError('Local slots must be unique incident facts')
        if not np.array_equal(role[j,valid],np.where(bank[ix,0]==rows[j],0,1)) or np.any(role[j,~valid]!=2):raise ValueError('Canonical endpoint roles differ')
        for k in ix:
            vec=facts[k,:64] if bank[k,0]==rows[j] else facts[k,128:]
            if not np.array_equal(vec,anchors[j]):raise ValueError('Anchor/fact endpoint embedding mismatch')
    true=b['true_entity']
    if true.shape!=(19266,) or true.dtype.kind!='i' or np.any(true < -1) or np.any(true>=g) or np.any(true[-2:]!=-1):raise ValueError('Exact biological/T/S entity map required')
    mapped=true>=0
    if not np.array_equal(np.unique(true[mapped]),np.arange(g)):raise ValueError('Compact anchor axis incomplete')

def validate_tokens(z):
    x=z['tokens'];valid=z['valid'];ids=z['token_ids'];selected=z['selected_token_ids']
    if x.ndim!=3 or x.shape[0]!=133 or x.shape[2]!=768 or x.dtype!=np.float32 or valid.dtype!=np.bool_ or valid.shape!=x.shape[:2] or ids.shape!=valid.shape or ids.dtype.kind not in 'iu':raise ValueError('FP32 native768 tokens, bool mask and exact integer IDs required')
    if not np.isfinite(x[valid]).all() or not valid.any(1).all() or np.any(valid.sum(1)>1026) or np.any(ids[valid]<0) or np.any(ids[valid]>=19266):raise ValueError('Invalid native token values/IDs')
    if selected.shape!=(1024,) or selected.dtype.kind not in 'iu' or len(np.unique(selected))!=1024 or np.any(selected<0) or np.any(selected>=19264):raise ValueError('Saved training-selected1024 biological IDs required')
    for token,mask in zip(ids,valid):
        active=token[mask]
        if len(np.unique(active))!=len(active) or np.sum(active==19264)!=1 or np.sum(active==19265)!=1 or not set(active[active<19264])<=set(selected):raise ValueError('Native identity/T/S or selection mismatch')

def validate_target(t,z,b):
    if not np.array_equal(t['fit_indices'],z['fit']) or not np.array_equal(t['train_patient_ids'],z['patient_ids'][z['fit']]):raise ValueError('Auxiliary target training axis differs')
    selected=t['selected_token_ids'];member=t['membership'];mapped=t['mapped_gene_mask'];target=t['targets_train']
    if not np.array_equal(selected,z['selected_token_ids']):raise ValueError('Target selected-gene axis differs')
    if member.dtype!=np.bool_ or member.ndim!=2 or member.shape[1]!=1024 or not 1<=len(member)<=64 or np.any(member.sum(1)<5):raise ValueError('Fixed5gene/64pathway target membership required')
    if mapped.dtype!=np.bool_ or mapped.shape!=(1024,) or not np.array_equal(mapped,b['true_entity'][selected]>=0) or member[:,~mapped].any():raise ValueError('Real target membership must use real mapped genes')
    if target.shape!=(len(z['fit']),len(member)) or target.dtype!=np.float32 or not np.isfinite(target).all():raise ValueError('Only finite training targets allowed')
    if not np.allclose(target.mean(0),0,atol=1e-5) or not np.allclose(target.std(0),1,atol=1e-5):raise ValueError('Targets must use original training-only standardization')
    if any(k.startswith(('held_target','outer_target','validation_target')) for k in t):raise ValueError('Held targets do not belong in training artifact')

class PreparedInputs:
    def __init__(self,manifest):
        self.path=Path(manifest);self.base=self.path.parent;doc=json.loads(self.path.read_text());self.records={};self.fits={}
        for row in doc['fits']:
            name=row['fit_id']
            if name in self.records:raise ValueError('Duplicate fit record')
            p=checked(self.base,row['data'])
            with np.load(p,allow_pickle=False) as z:self.fits[name]={k:z[k].copy() for k in ['patient_ids','time','event','fit','held']}
            self.records[name]=row
        validate_roles(self.fits)
    def load_fit(self,name):
        row=self.records[name];z=read_npz(checked(self.base,row['data']));b=read_npz(checked(self.base,row['bank']));validate_metadata(z);validate_tokens(z);validate_bank(b)
        for k in self.fits[name]:np.testing.assert_array_equal(z[k],self.fits[name][k])
        if np.any(b['true_entity'][np.setdiff1d(np.arange(19264),z['selected_token_ids'])]>=0):raise ValueError('Bank map includes genes outside frozen selected axis')
        entitymap=b['true_entity'];ids=z['token_ids'];entity=np.full(ids.shape,-1,np.int64);entity[z['valid']]=entitymap[ids[z['valid']]]
        data=dict(**z,fit_id=name,arm='aligned_real',bank=b,saved_token_ids=ids.copy(),bank_sha256=row['bank']['sha256'],data_sha256=row['data']['sha256'])
        data['tokens']=np.concatenate([z['tokens'],entity[:,:,None].astype(np.float32)],axis=-1)
        from cancer_kg.training.auxiliary import availability
        target=read_npz(checked(self.base,row['targets']));validate_target(target,z,b)
        mask=availability(ids[z['fit']],z['valid'][z['fit']],target)
        if not mask.any():raise ValueError('No available training pathway items')
        data.update(target=target,aux_available=mask,target_sha256=row['targets']['sha256'])
        return data
