"""Original small-module checkpoint functions."""

def trainable_state(model):return {name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()} for name in ['adapter','pool_norm','head','memory']}

def load_trainable(model,state):
    if set(state)!={'adapter','pool_norm','head','memory'}:raise ValueError('Only small trainable modules may be checkpointed')
    for name,values in state.items():getattr(model,name).load_state_dict(values,strict=True)
