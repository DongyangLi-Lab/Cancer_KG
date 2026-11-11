"""Executed historical small-head computations, extracted verbatim."""
import numpy as np
import torch
from torch import nn

def standardize(x,fit):
    mean=x[fit].astype(np.float64).mean(0);sd=x[fit].astype(np.float64).std(0);sd[sd<1e-8]=1
    return (x.astype(np.float64)-mean)/sd,dict(mean=mean,sd=sd)

def cox(r,t,e):
    ix=torch.nonzero(e,as_tuple=True)[0]
    return (torch.logsumexp(r[None,:].expand(len(ix),-1).masked_fill(t[None,:]<t[ix,None],-torch.inf),1)-r[ix]).mean()

class MLP(nn.Module):
    def __init__(self,d,head):
        super().__init__();layers=[]
        for width in ([64,16] if head=='small' else [1024,512,256,128]):layers.extend([nn.Linear(d,width),nn.ReLU(),nn.Dropout(.2)]);d=width
        layers.append(nn.Linear(d,1,bias=False));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).flatten()

def mlp(x,t,e,fit,held,head,seed,lr,wd,epochs,early=False,patience=15):
    torch.manual_seed(seed);m=MLP(x.shape[1],head);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    xx=torch.tensor(x,dtype=torch.float32);tt=torch.tensor(t,dtype=torch.float32);ee=torch.tensor(e,dtype=torch.bool)
    best=float('inf');bestepoch=0;stale=0;saved=None
    for epoch in range(epochs):
        m.train();opt.zero_grad();loss=cox(m(xx[fit]),tt[fit],ee[fit])
        if not torch.isfinite(loss):raise ValueError('Nonfinite Cox train loss')
        loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if early:
            m.eval()
            with torch.no_grad():val=float(cox(m(xx[held]),tt[held],ee[held]))
            if not np.isfinite(val):raise ValueError('Nonfinite Cox validation loss')
            if val<best:best=val;bestepoch=epoch+1;stale=0;saved={k:v.detach().clone() for k,v in m.state_dict().items()}
            else:stale+=1
            if stale>=patience:break
    if early:m.load_state_dict(saved)
    m.eval()
    with torch.no_grad():r=m(xx[held]).numpy()
    return r,dict(weights={k:v.detach().cpu() for k,v in m.state_dict().items()},head=head,dimension=x.shape[1]),dict(best_epoch=bestepoch if early else epochs,best_val_loss=best if early else None,epochs_run=epoch+1,parameters=sum(v.numel() for v in m.parameters()))

def replay(state,x):
    if 'head' not in state:return x@state['weights']
    m=MLP(state['dimension'],state['head']);m.load_state_dict(state['weights']);m.eval()
    with torch.no_grad():return m(torch.tensor(x,dtype=torch.float32)).numpy()
