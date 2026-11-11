"""Checks for encoded four-modal features and matched topology controls."""
import numpy as np


def validate_features(data, require_graph=True):
    n=len(data['patient_ids'])
    for key,width in (('rna',768),('protein',256)):
        pool=np.asarray(data[key+'_pool'])
        if pool.shape!=(n,width) or not np.isfinite(pool).all():
            raise ValueError('Invalid pooled feature: '+key)
        if require_graph:
            value,mask=data[key],data[key+'_mask']
            if value.ndim!=3 or value.shape[0]!=n or value.shape[2]!=width or value.shape[1]<1:
                raise ValueError('Invalid native feature: '+key)
            if mask.dtype.kind!='b' or mask.shape!=value.shape[:2] or not np.isfinite(value).all():
                raise ValueError('Invalid native mask/value: '+key)
    for key in ('ptm','cnv'):
        value,mask=data[key],data[key+'_mask']
        if value.ndim!=2 or value.shape[0]!=n or value.shape[1]<1 or mask.shape!=value.shape:
            raise ValueError('Invalid numerical shape: '+key)
        if mask.dtype.kind!='b' or not np.isfinite(value).all():
            raise ValueError('Prepared numerical values must be finite with a boolean mask')
    if not require_graph:return
    ids,types=data['node_ids'],data['node_type'];nodes=len(ids)
    if ids.ndim!=1 or ids.dtype.kind!='U' or len(set(ids))!=nodes or types.shape!=(nodes,):
        raise ValueError('Unique Unicode graph node identities required')
    if types.dtype.kind not in 'iu' or not np.isin(types,[0,1,2]).all() or not (types==0).any() or not (types==2).any():
        raise ValueError('Invalid node types or empty gene/pathway inventory')
    for key in ('rna','protein','ptm','cnv'):
        mapping=data[key+'_node']
        if mapping.shape!=(data[key].shape[1],) or mapping.dtype.kind not in 'iu' or np.any(mapping<0) or np.any(mapping>=nodes):
            raise ValueError('Invalid modality-to-node map: '+key)
        if np.any(types[mapping]==2):raise ValueError('An assay cannot map directly to a pathway node')
    sets={}
    for topology in ('real','shuffled'):
        for relation,source_type,target_type in (('member',0,2),('member_reverse',2,0),('parent',2,2),('parent_reverse',2,2)):
            edge=data[topology+'_'+relation]
            if edge.ndim!=2 or edge.shape[1]!=2 or edge.dtype.kind not in 'iu' or np.any(edge<0) or np.any(edge>=nodes):
                raise ValueError('Invalid edge indices')
            pairs=set(map(tuple,edge.tolist()))
            if len(pairs)!=len(edge) or np.any(edge[:,0]==edge[:,1]):raise ValueError('Duplicate or self edge')
            if np.any(types[edge[:,0]]!=source_type) or np.any(types[edge[:,1]]!=target_type):
                raise ValueError('Incorrect typed edge direction')
            sets[topology,relation]=pairs
        for relation in ('member','parent'):
            if {(b,a) for a,b in sets[topology,relation]} != sets[topology,relation+'_reverse']:
                raise ValueError('Reverse relation mismatch')
    for relation in ('member','parent'):
        if sets['real',relation]==sets['shuffled',relation]:raise ValueError('Shuffle did not change topology')
        for axis in (0,1):
            first=np.bincount(data['real_'+relation][:,axis],minlength=nodes)
            second=np.bincount(data['shuffled_'+relation][:,axis],minlength=nodes)
            if not np.array_equal(first,second):raise ValueError('Shuffled node degrees changed')
