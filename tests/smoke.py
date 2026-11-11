"""Synthetic-only CPU smoke; fake native encoder, no pretrained assets/patients."""
import copy, json, tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cancer_kg.training.fullrisk_replay import cox, fullrisk_backward, get_rng, set_rng
from cancer_kg.training.gradient_gates import error
from cancer_kg.models.kg_reader import KGReader
from cancer_kg.models.token_model import canonicalize_tokens
from cancer_kg.data.inputs import validate_metadata, validate_roles, validate_bank, validate_tokens, validate_target, PreparedInputs, sha
import cancer_kg.training.fit_core as fit_core
ALIGNMENT = True

class Block(nn.Module):

    def forward(self, x, **kwargs):
        return x

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block()])

    def forward(self, total_embs, key_padding_mask, gen_mask):
        return total_embs

def reader_reference_check():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    facts = torch.randn(32, 12)
    anchors = torch.randn(8, 4)
    local = torch.arange(8 * 3).reshape(8, 3)
    model = KGReader(facts, anchors, local, in_dim=6, width=4, block_size=4, blocks_read=2)
    tokens = torch.randn(2, 5, 6, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    entity = torch.tensor([[0, 1, -1, 2, -1], [4, 5, 6, -1, -1]])
    result, receipt = model(tokens, valid, entity, True)
    routes = model.routes(tokens, valid, entity)
    q, k, v, glob, loc, lvalid, role, mapped = routes
    rows = []
    for b in range(2):
        values = []
        for j in range(5):
            if not mapped[b, j]:
                values.append(torch.zeros(4))
                continue
            ix = loc[b, j][lvalid[b, j]]
            roles = model.incidence_role(role[b, j][lvalid[b, j]])
            keys = torch.cat([k[glob[b]], k[ix] + roles])
            vals = torch.cat([v[glob[b]], v[ix] + roles])
            w = torch.softmax((keys * q[b, j]).sum(-1) / 2, 0)
            values.append((w[:, None] * vals).sum(0))
        rows.append(torch.stack(values))
    reference = tokens + model.gate * model.out(torch.stack(rows)) * mapped[..., None]
    assert torch.allclose(result, reference, atol=1e-07, rtol=1e-06)
    checks = dict(sparse_vs_same_selected_reference=True, all_original_tokens_retained=result.shape == tokens.shape, unmapped_special_and_padding_unchanged=torch.equal(result[~mapped], tokens[~mapped]), frozen_fact_and_entity_tables=not model.facts.requires_grad and (not model.anchors.requires_grad))
    result.sum().backward()
    checks['memory_gradients_present_finite'] = all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    repeat = model(tokens, valid, entity)
    checks['route_and_output_replay_exact'] = torch.equal(repeat, result)
    with torch.no_grad():
        model.key.weight.add_(0.5)
    changed = model(tokens, valid, entity)
    checks['trainable_kv_not_stale_cached'] = not torch.equal(changed, repeat)
    with torch.no_grad():
        model.gate.zero_()
    checks['zero_residual_gate_exact_identity'] = torch.equal(model(tokens, valid, entity), tokens)
    checks['fully_unmapped_finite_exact_identity'] = torch.equal(model(tokens, valid, torch.full_like(entity, -1)), tokens)
    checks['static_buffers_nonpersistent'] = not any((k in model.state_dict() for k in ['facts', 'anchors', 'local_index', 'local_role']))
    assert all(checks.values()), checks
    return dict(passed=True, checks={k: bool(v) for k, v in checks.items()}, patient_data_used=False, real_KG_training_executed=False)

def synthetic_roles():
    out = {}
    allix = np.arange(133)
    ids = np.array(['synthetic_%03d' % i for i in allix])
    e = (allix % 5 != 0).astype(int)
    for f, held in enumerate(np.array_split(allix, 3)):
        fit = np.setdiff1d(allix, held)
        out[f'r0_f{f}_outer'] = dict(patient_ids=ids, time=allix.astype(float) + 1, event=e, fit=fit, held=held)
        for j, val in enumerate(np.array_split(fit, 3)):
            out[f'r0_f{f}_inner{j}'] = dict(patient_ids=ids, time=allix.astype(float) + 1, event=e, fit=np.setdiff1d(fit, val), held=val)
    return out

def input_checks():
    rng = np.random.default_rng(8)
    roles = synthetic_roles()
    validate_roles(roles)
    rejected = 0
    for key, value in [('time', np.zeros(133)), ('time', np.full(133, np.nan)), ('time', np.ones((133, 1))), ('event', np.full(133, 0.5)), ('event', np.full(133, np.inf)), ('patient_ids', np.arange(133)), ('patient_ids', np.array([' '] * 133)), ('fit', np.array([], int)), ('held', np.array([0, 5, 10]))]:
        bad = copy.deepcopy(roles)
        bad['r0_f0_outer'][key] = value
        try:
            validate_roles(bad)
        except (ValueError, AssertionError):
            rejected += 1
        else:
            raise AssertionError(key)
    g = 6
    anchors = rng.normal(size=(g, 64)).astype('float32')
    ix = np.arange(16384)
    bank = np.stack([ix % g, np.zeros(len(ix), int), ix + g], axis=1)
    facts = rng.normal(size=(16384, 192)).astype('float32')
    facts[:, :64] = anchors[ix % g]
    local = np.stack([np.flatnonzero(ix % g == j)[:16] for j in range(g)])
    true = np.full(19266, -1, np.int64)
    true[10:16] = np.arange(g)
    b = dict(facts=facts, anchors=anchors, bank=bank, entity_rows=np.arange(g), local_index=local, local_role=np.zeros_like(local), true_entity=true)
    validate_bank(b)
    ids = np.tile(np.array([10, 11, 12, 13, 14, 15, 19264, 19265]), (133, 1))
    tokens = rng.normal(size=(133, 8, 768)).astype('float32')
    valid = np.ones((133, 8), bool)
    selected = np.arange(10, 1034)
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        np.savez(p / 'bank.npz', **b)
        records = []
        for name, meta in roles.items():
            z = dict(**meta, tokens=tokens, valid=valid, token_ids=ids, selected_token_ids=selected)
            validate_tokens(z)
            np.savez(p / (name + '.npz'), **z)
            row = dict(fit_id=name, data=dict(file=name + '.npz', sha256=sha(p / (name + '.npz'))), bank=dict(file='bank.npz', sha256=sha(p / 'bank.npz')))
            target = rng.normal(size=(len(z['fit']), 2))
            target = ((target - target.mean(0)) / target.std(0)).astype('float32')
            member = np.zeros((2, 1024), bool)
            member[0, :5] = True
            member[1, 1:6] = True
            t = dict(fit_indices=z['fit'], train_patient_ids=z['patient_ids'][z['fit']], selected_token_ids=selected, membership=member, mapped_gene_mask=true[selected] >= 0, targets_train=target)
            validate_target(t, z, b)
            np.savez(p / (name + '_target.npz'), **t)
            row['targets'] = dict(file=name + '_target.npz', sha256=sha(p / (name + '_target.npz')))
            records.append(row)
        (p / 'input.json').write_text(json.dumps(dict(fits=records)))
        real = PreparedInputs(p / 'input.json').load_fit('r0_f0_inner0')
        np.testing.assert_array_equal(real['tokens'][:, :, :768], tokens)
        np.testing.assert_array_equal(real['tokens'][:, :, -1][:, -2:], -np.ones((133, 2)))
        data = {k: v for k, v in real.items()}
        take = np.arange(6)
        data.update(tokens=real['tokens'][take], valid=real['valid'][take], saved_token_ids=real['saved_token_ids'][take], fit=np.arange(4), held=np.array([4, 5]), time=np.arange(1, 7, dtype=float), event=np.array([1, 1, 0, 1, 1, 0]))
        smallbank = {k: v.copy() for k, v in b.items()}
        smallbank['facts'] = b['facts'][:1024].copy()
        smallbank['bank'] = b['bank'][:1024].copy()
        data['bank'] = smallbank
        member = np.zeros((2, 1024), bool)
        member[0, :5] = True
        member[1, 1:6] = True
        data['target'] = dict(membership=member, selected_token_ids=selected, mapped_gene_mask=true[selected] >= 0, targets_train=rng.normal(size=(4, 2)).astype('float32'))
        data['aux_available'] = np.ones((4, 2), bool)
    return (data, rejected)
torch.set_num_threads(1)
torch.manual_seed(123)
reader_check = reader_reference_check()
data, rejected = input_checks()
model = fit_core.make_model(Encoder(), data)
other = fit_core.make_model(Encoder(), data)
assert all((torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()))
x = torch.tensor(data['tokens'][:4])
valid = torch.tensor(data['valid'][:4])
t = torch.tensor([1.0, 1.0, 2.0, 3.0])
e = torch.tensor([1, 1, 0, 1])
model.train()
torch.manual_seed(42)
rngstate = get_rng()
risk = torch.cat([model(x[i:i + 2], valid[i:i + 2]) for i in range(0, 4, 2)])
loss = cox(risk, t, e)
loss.backward()
expected = {k: p.grad.clone() for k, p in model.named_parameters() if p.requires_grad}
set_rng(rngstate)
value, replayed, receipt = fullrisk_backward(model, x, valid, t, e, 2)
difference = error(expected, {k: p.grad for k, p in model.named_parameters() if p.requires_grad})
assert difference['max_abs'] < 2e-06 and difference['relative_l2'] < 2e-05
torch.testing.assert_close(risk.detach(), replayed, atol=1e-06, rtol=1e-06)
model.eval()
before = model(x, valid)
state = fit_core.trainable_state(model)
fit_core.load_trainable(other, state)
other.eval()
torch.testing.assert_close(before, other(x, valid), atol=0, rtol=0)
padded = torch.cat([x, torch.randn(4, 2, 769) * 1000000.0], 1)
mask = torch.cat([valid, torch.zeros(4, 2, dtype=torch.bool)], 1)
torch.testing.assert_close(before, model(padded, mask), atol=0, rtol=0)
from cancer_kg.models.calibrated import calibrate
from cancer_kg.training.auxiliary import make_aux
from cancer_kg.training.joint import joint_backward
raw = torch.randn(2, 3, 768, requires_grad=True)
u = torch.randn_like(raw)
mapped = torch.ones(2, 3, dtype=torch.bool)
calibrated, k = calibrate(raw, u, mapped, torch.tensor(0.01))
ratio = (calibrated - raw).square().mean(-1).sqrt() / raw.square().mean(-1).sqrt()
torch.testing.assert_close(ratio, torch.full_like(ratio, 0.01), atol=1e-07, rtol=1e-05)
calibrated.sum().backward()
torch.testing.assert_close(raw.grad, torch.ones_like(raw), atol=0, rtol=0)
aux = make_aux(data)
model.train()
ids = torch.tensor(data['saved_token_ids'][:4])
target = torch.tensor(data['target']['targets_train'])
available = torch.tensor(data['aux_available'])
_, _, joint = joint_backward(model, aux, x, valid, t, e, ids, target, available, 2, 0.1)
assert joint['auxiliary_gradient_health']['passed'] and joint['main_gradient_health']['passed']
risk, main, auxstate, info, _, _ = fit_core.fit(Encoder(), data, 'aligned_real', 1, 2, True, device='cpu')
assert info['best_epoch'] == 1 and np.isfinite(risk).all()
print(json.dumps(dict(passed=True, synthetic_only=True, native_Tahoe_executed=False, negative_input_cases=rejected, reader_reference_passed=reader_check['passed'], gradient_error=difference, small_checkpoint_replay=True, alignment=ALIGNMENT, numpy=np.__version__, torch=torch.__version__)))
