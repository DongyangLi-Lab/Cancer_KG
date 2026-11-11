"""Interface candidate only: existing frozen Tahoe TXEncoder; no new transformer."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
PACKING_POLICY='stable_valid_gather_then_microbatch_zero_pad_v1'
def canonicalize_tokens(tokens,valid):
    """Preserve every valid vector in original order; remove only masked slots."""
    if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool or not valid.any(1).all():raise ValueError('Nonempty valid=True sequences required')
    rows=[tokens[i][valid[i]] for i in range(len(tokens))]
    packed=nn.utils.rnn.pad_sequence(rows,batch_first=True,padding_value=0.)
    lengths=valid.sum(1)
    mask=torch.arange(packed.shape[1],device=valid.device)[None,:]<lengths[:,None]
    return packed,mask

class CheckpointBlock(nn.Module):
    def __init__(self,block):super().__init__();self.block=block
    def forward(self,x,attn_bias=None,flash_attn_padding_info=None):
        def call(value):return self.block(value,attn_bias=attn_bias,flash_attn_padding_info=flash_attn_padding_info)
        if not torch.is_grad_enabled():return call(x)
        return checkpoint(call,x,use_reentrant=False,preserve_rng_state=True)
class TokenAdapterTahoe(nn.Module):
    def __init__(self,encoder,in_dim=768,model_dim=2560,checkpoint_blocks=False):
        super().__init__();self.encoder=encoder
        for p in encoder.parameters():p.requires_grad_(False)
        if checkpoint_blocks:
            if not hasattr(encoder,'layers'):raise ValueError('Expected native TXEncoder.layers')
            encoder.layers=nn.ModuleList([b if isinstance(b,CheckpointBlock) else CheckpointBlock(b) for b in encoder.layers])
        self.adapter=nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,model_dim))
        self.pool_norm=nn.LayerNorm(model_dim)
        self.head=nn.Sequential(nn.Linear(model_dim,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,16),nn.ReLU(),nn.Dropout(.2),nn.Linear(16,1,bias=False))
        self.train()
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,tokens,valid):
        if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool or not valid.any(1).all():raise ValueError('Expected [B,L,768] tokens and valid=True mask')
        tokens,valid=canonicalize_tokens(tokens,valid)
        x=self.adapter(tokens.float())
        # Native API handles Flash padding. No invented inputs_embeds kwarg or bias.
        x=self.encoder(total_embs=x,key_padding_mask=valid,gen_mask=torch.zeros_like(valid))
        pooled=x.masked_fill(~valid[:,:,None],-torch.inf).max(1).values
        return self.head(self.pool_norm(pooled.float())).flatten()
    def counts(self):return dict(pretrained_used=sum(p.numel() for p in self.encoder.parameters()),trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),adapter=sum(p.numel() for p in self.adapter.parameters()),pool_norm=sum(p.numel() for p in self.pool_norm.parameters()),head=sum(p.numel() for p in self.head.parameters()))
