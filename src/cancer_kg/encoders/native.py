import importlib
import sys
from pathlib import Path
import numpy as np
import torch
from cancer_kg.data.core import align_row
from cancer_kg.encoders.scfoundation_runtime import encoder_runtime
from cancer_kg.data.core import egsp_bulk as scfoundation_bulk

def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return sum(p.numel() for p in model.parameters())

def scfoundation_convert_audited(raw, native, key='cell'):
    """Accept only official MMF cell layout or its explicit native conversion.

    No generic key-prefix search, partial encoder load, key dropping or shape
    repair is allowed. GenBio's own backbone uses this official MMF pathway.
    """
    if key in raw:
        cell = raw[key]
        if not isinstance(cell, dict) or not {'config', 'state_dict'}.issubset(cell):
            raise ValueError('MMF cell entry requires config and full state_dict')
        keys = list(cell['state_dict'])
        if any(not k.startswith('model.') or 'model.' in k[len('model.'):] for k in keys):
            raise ValueError('Unexpected MMF prefix; refuse heuristic key renaming')
        expected = [k[len('model.'):] for k in keys]
        if len(expected) != len(set(expected)):
            raise ValueError('Duplicate keys after exact MMF model. prefix removal')
        converted = native.convertconfig(cell)
        if set(converted['model_state_dict']) != set(expected):
            raise ValueError('Official convertconfig lost or introduced checkpoint keys')
        for source_key, target_key in zip(keys, expected):
            if converted['model_state_dict'][target_key] is not cell['state_dict'][source_key]:
                raise ValueError('Official conversion changed tensor provenance')
        layout = 'official_MMF_cell_exact_model_dot_prefix'
    elif {'config', 'model_state_dict'}.issubset(raw):
        converted = raw
        keys = list(raw['model_state_dict'])
        layout = 'explicit_native_config_and_model_state_dict_no_rename'
    else:
        raise ValueError('Unsupported checkpoint layout; native config/full key audit required before conversion')
    return converted, dict(checkpoint_layout=layout, source_state_key_count=len(keys),
                           converted_state_key_count=len(converted['model_state_dict']),
                           conversion_dropped_keys=[], conversion_guessed_keys=[])

def load_scfoundation_audited(cfg, native):
    # Acquisition gate verifies trusted distribution hash before this loader.
    raw = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    converted, audit = scfoundation_convert_audited(raw, native, key='cell')
    config = dict(converted['config'])
    original_config_device = config.get('device')
    config['device'] = cfg['device']
    if 'qv_dim' not in config and config.get('model') != 'mae_autobin':
        config['qv_dim'] = config.get('dim_head', 64)
    config.setdefault('ppi_edge', None)
    model = native.select_model(config)
    incompatible = model.load_state_dict(converted['model_state_dict'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError('Strict scFoundation checkpoint load has key gaps')
    audit.update(missing_keys=[], unexpected_keys=[], strict_load_passed=True,
                 configured_model_type=config.get('model_type'), full_native_model_constructed=True,
                 original_config_device=original_config_device, execution_config_device=cfg['device'])
    model.to(cfg['device']).eval()
    return model, config, audit

class ScFoundation:
    def __init__(self, cfg):
        self.cfg = cfg
        sys.path.insert(0, str(Path(cfg['repo_dir']) / 'model'))
        self.native = importlib.import_module('load')
        self.model, self.model_cfg, self.load_audit = load_scfoundation_audited(cfg, self.native)
        self.model.to(cfg['device'])
        self.params = freeze(self.model)
        import csv
        with open(cfg['gene_index'], encoding='utf-8') as f:
            self.genes = [r['gene_name'] for r in csv.DictReader(f, delimiter='\t')]
        if len(self.genes) != 19264 or len(set(self.genes)) != 19264:
            raise ValueError('Expected unique 19264-gene official ordered symbol vocabulary')

    def encode(self, row, observed, source_ids, sample_id, include_tokens):
        x, mask = align_row(row, observed, source_ids, self.genes)
        z = scfoundation_bulk(x, mask, self.cfg['pre_normalized'], self.cfg.get('target_sum'))
        z = torch.from_numpy(z).unsqueeze(0).to(self.cfg['device'])
        gene_ids = torch.arange(19266, device=z.device).unsqueeze(0)
        selected = z > 0
        x, padding = self.native.gatherData(z, selected, self.model_cfg['pad_token_id'])
        ids, _ = self.native.gatherData(gene_ids, selected, self.model_cfg['pad_token_id'])
        x = self.model.token_emb(x.unsqueeze(2).float(), output_weight=0)
        self.runtime_audit = {}
        with encoder_runtime(self.cfg,self.runtime_audit):
            token = self.model.encoder(x + self.model.pos_emb(ids), padding)
        # Explicit IDs/padding protect T/S identity if upstream changes batch alignment.
        valid = ~padding[0]
        biological = valid & (ids[0] < 19264)
        t_position = valid & (ids[0] == 19264)
        s_position = valid & (ids[0] == 19265)
        if not biological.any() or t_position.sum() != 1 or s_position.sum() != 1:
            raise ValueError('Missing biological genes or unique T/S identity')
        gene = token[0, biological]
        pooled_all = torch.cat([token[0, s_position][0], token[0, t_position][0],
                                gene.max(0).values, gene.mean(0)], dim=0)
        pooled_max = token[0, valid].max(0).values  # official pool_type=max includes T/S
        self.pooled_variants = {'pooled_all': pooled_all.float().cpu().numpy(),
                                'pooled_max': pooled_max.float().cpu().numpy()}
        pooled = pooled_all if self.cfg.get('pool_type', 'all') == 'all' else pooled_max
        self.special_mask = ((ids[0] >= 19264) | padding[0]).cpu().numpy()
        # Encoder token representation is not the decoder gene output [19264,512].
        return (pooled.float().cpu().numpy(), ids[0].cpu().numpy(),
                (~padding[0]).cpu().numpy(), mask,
                token[0].float().cpu().numpy() if include_tokens else None)
