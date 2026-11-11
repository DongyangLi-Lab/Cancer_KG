import inspect
from pathlib import Path
import torch

def audit_tahoe_state_keys(incompatible):
    """No-drug removal may discard only exact native chemical-encoder keys."""
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    forbidden = [k for k in unexpected if not k.startswith('model.chem_encoder.')]
    if missing or forbidden:
        raise RuntimeError(f'Unloaded Tahoe parameters: missing={missing}; nonchemical unexpected={forbidden}')
    return dict(missing_keys=missing, discarded_chemical_keys=unexpected,
                allowed_discard_prefix='model.chem_encoder.', nonchemical_parameters_complete=True)

def unwrap_tahoe(wrapper):
    """ComposerTX.forward(batch) differs from TXModel.forward(genes,values,...)."""
    native = wrapper.model
    required = {'genes', 'values', 'gen_masks', 'key_padding_mask'}
    if not required.issubset(inspect.signature(native.forward).parameters):
        raise TypeError('Expected ComposerTX.model to expose native TXModel forward signature')
    if native.use_chem_token or native.keep_first_n_tokens != 1:
        raise ValueError('No-drug native model must agree with one-CLS collator configuration')
    return native

def load_tahoe_local_audited(cfg):
    """Mirror official local loader while recording/validating all state-key gaps.

    Unlike the official loader's silent strict=False branch, the no-drug branch
    refuses every missing key and every unexpected key outside chem_encoder.
    """
    from omegaconf import OmegaConf
    from tahoe_x1.model.model import ComposerTX
    from tahoe_x1.tokenizer import GeneVocab
    directory = Path(cfg['model_dir'])
    mc = OmegaConf.load(directory/'model_config.yml')
    cc = OmegaConf.load(directory/'collator_config.yml')
    vocab = GeneVocab.from_file(directory/'vocab.json')
    if mc['attn_config']['attn_impl'] == 'triton':
        mc['attn_config']['attn_impl'] = 'flash'
        mc['attn_config']['use_attn_mask'] = False
    mc['do_mlm'] = False
    mc['return_gene_embeddings'] = True
    cc['do_mlm'] = False
    cc['use_chem_token'] = False
    cc['keep_first_n_tokens'] = 1
    cc.pop('drug_to_id_path', None)
    mc.pop('chemical_encoder', None)
    wrapper = ComposerTX(model_config=mc, collator_config=cc)
    checkpoint = torch.load(directory/'best-model.pt', map_location='cpu', weights_only=False)
    incompatible = wrapper.load_state_dict(checkpoint['state']['model'], strict=False)
    audit = audit_tahoe_state_keys(incompatible)
    del checkpoint
    native = unwrap_tahoe(wrapper)
    native.to(cfg['device']).eval()
    audit.update(wrapper_type=type(wrapper).__name__, forward_type=type(native).__name__,
                 model_keep_first_n_tokens=native.keep_first_n_tokens,
                 collator_keep_first_n_tokens=cc['keep_first_n_tokens'], use_chem_token=False)
    return native, vocab, mc, cc, audit
