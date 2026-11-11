# KG pathway alignment

## Repository layout

```text
src/cancer_kg/   Python package, grouped by function
  cli/          Preparation, encoding and training entry points
  models/       Model definitions
configs/        Example configurations (where needed)
requirements/   Training, encoder and test dependencies
tests/          Synthetic checks; no patient data
docs/           Input contracts, upstream assets and source provenance
```

Additional package folders contain `data/`, `encoders/`, `training/`,
`evaluation/` or `runtime/` components as required by this method.
Install the package once from the repository root, then use the module commands
below. Choose the appropriate dependency file in `requirements/` for your host.

```bash
python -m pip install --no-deps -e .
```

This branch publishes the selected **aligned_real** model from the executed 2026-10-27 experiment. Its historical reported C-index is **0.631130**. This is a selection from completed project experiments, not a new independent demonstration of superiority. The calibrated KG reader subtracts log slot counts so local/global groups have balanced prior mass, normalizes KG output RMS and adds a learned residual initialized to 1% of detached native-token RMS. A shared pathway head predicts **RNA-derived training-only proxy activity from the KG branch only**. These are not independent molecular measurements.

The only supported model uses real gene-to-KG grounding and Cox + **0.1 masked pathway MSE**. Maximum24 epochs, patience6 on inner Cox, AdamW1e-4/wd.01, clip1, one optimizer step after complete-fit Cox plus globally normalized auxiliary backward. It has9 inner +3 outer fits; outer epochs are the integer median of3 inner best epochs.

```bash
pip install -r requirements/smoke.txt
python -m tests.smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m cancer_kg.cli.train --manifest /private/inputs.json --assets /external/assets.json --output /private/results/aligned_real
```

The entry point always runs aligned_real in a fresh output directory. No automatic resume or hyperparameter search is added.

## Prepared private input

This package starts at the actual saved scFoundation token cache and frozen, per-fit KG bank. It does not rerun feature selection, pretrain a KG or generate a new patient split. All input paths are supplied by the user and SHA-256 checked; no patient data or weights are included.

`inputs.json` has `{"fits": [{"fit_id": "r0_f0_inner0", "data": {"file": "private.npz", "sha256": "..."}, "bank": {"file": "bank.npz", "sha256": "..."}}]}`. Supply exactly `r0_f{0,1,2}_{inner0,inner1,inner2,outer}`. Paths are relative to the JSON or absolute. Original 133 patients, 106 events, outer sizes 45/44/44, disjoint nested roles and 2345 outer comparable pairs are retained.

Each **data NPZ**, loaded with `allow_pickle=False`:

- `patient_ids`: unique nonempty Unicode `[133]`; `time`: finite positive `[133]`; `event`: binary `[133]`.
- `fit`, `held`: nonempty unique integer row indices, disjoint, each with an event. Every artifact shares the exact patient/endpoint axis.
- `tokens`: original frozen scFoundation hidden states promoted to float32 `[133,L,768]`; `valid`: bool `[133,L]`; `token_ids`: exact integers `[133,L]`. Only invalid padding is ignored. Each patient includes exactly one T=19264 and S=19265 token, with at most 1024 biological tokens.
- `selected_token_ids`: original **fit-only Cox-selected** 1024 unique biological IDs. Do not recreate this selection using all patients. Preserve original saved order, values and roles when consolidating files.

Each original **bank NPZ** has `facts float32[16384,192]`, `anchors float32[G,64]`, `local_index int64[G,16]` (-1 absent), `local_role int64[G,16]` (0/1 canonical endpoint, 2 absent), `bank int64[16384,3]`, sorted unique `entity_rows int64[G]`, `true_entity int64[19266]`. G is the fit's mapped selected genes. A fact is pretrained `[entity64,relation64,entity64]`. T/S are unmapped. The loader verifies incident endpoints, embedding equality, and the real grounding map. `src/cancer_kg/data/bank.py` preserves the original public-only deterministic bank construction algorithm; it requires the user's original public KG triples, entity/relation embedding assets and relation-ID axis.

This is an **RNA-token + KG knowledge** route. The scFoundation cache is the input; Tahoe is a frozen downstream transformer, not a second independently encoded RNA/protein modality. The original public embeddings had seen relations 43/44/45; bank construction excluded those relation types. Do not describe the whole embedding pipeline as never seeing them. Canonical DistMult endpoint roles are not biological causal directions.

## Official backbone and environment

Use the original CUDA environment for the full run. Install the dependencies of [Tahoe-x1](https://github.com/tahoebio/tahoe-x1/tree/167cc19fce5888c7738b0f73f61cb4a8d3d1d457), revision `167cc19fce5888c7738b0f73f61cb4a8d3d1d457`, then install this branch's requirements. Native FlashAttention **2.8.3** is required. This CPU smoke does not validate CUDA, FlashAttention or pretrained Tahoe. The portable CLI retains full-risk replay, BF16/cache-disabled Tahoe forwards, FP32 reader, TF32 off, microbatch 4, deterministic Flash calls and original-function restoration.

Download external official Tahoe assets from [tahoebio/Tahoe-x1](https://huggingface.co/tahoebio/Tahoe-x1/tree/d218a580b9c2500ae9dfc8367a398545e6f017a8/3b-model), revision `d218a580b9c2500ae9dfc8367a398545e6f017a8`, subdirectory `3b-model`. Supply `assets.json`:

```json
{"repo_dir":"/external/tahoe-x1","model_dir":"/external/Tahoe-x1/3b-model","device":"cuda","asset_sha256":{"best-model.pt":"<trusted SHA256>","model_config.yml":"<trusted SHA256>","collator_config.yml":"<trusted SHA256>","vocab.json":"<trusted SHA256>"}}
```

Use trusted hashes of the exact revision's assets. `docs/UPSTREAM.json` binds the original two official model source files. The loader requires the **complete 32-layer, width-2560, 20-head pretrained encoder**, freezes its parameters and rejects missing or extra nonchemical checkpoint keys. It never substitutes an untrained/shallow encoder. Official trusted model checkpoints are loaded using their native PyTorch format; do not supply untrusted checkpoint files.

scFoundation input-cache generation follows [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation), with its model license and original expression/gene-axis convention. That encoder and its weights are not bundled.

## Scope, licensing and portability

`docs/SOURCE.json` lists the executed source and candidate hashes. Model classes and scientific fit functions are copied verbatim or extracted with unchanged AST. Unused shuffled-grounding utilities and alternative arm entry points are omitted; the fixed aligned_real coefficient remains0.1. New code is limited to explicit prepared-input I/O, integrity checks, CLI, output packaging and synthetic tests. Historical UNC paths, sequential Slurm/approval/release-chain checks and site-specific resource receipts are removed; this is not a claim that the portable wrapper was rerun on the patient cohort. The original fit optimizer, selection, seed, precision, loss and model are retained. Native numerical checks were executed historically; a different GPU/software stack still needs its own numerical/resource validation before patient use. Output checkpoints contain only trainable small modules and provenance, never pretrained weights. Original diagnostic report aggregation is retained inside the fit functions; historical paper/audit report pipelines are not bundled.

No third-party source tree, pretrained weights, patient data or publication license is granted by this extraction. Tahoe's repository declares Apache-2.0 for code; the [pinned model card](https://huggingface.co/tahoebio/Tahoe-x1/blob/d218a580b9c2500ae9dfc8367a398545e6f017a8/README.md) also declares Apache-2.0 for that model revision. scFoundation's model assets have their own research/noncommercial terms, separate from source-code terms. Each public KG source retains its own license; supplied derived KG embeddings/banks do not waive those terms. This delivery does not assign a project-wide license; the upstream licenses remain unchanged.

Private outputs (`*.pt`, `private_oof.tsv`, fit traces and inputs) are ignored by `.gitignore`; keep them outside the public checkout.

## Additional alignment target artifact

Each manifest fit also has `"targets":{"file":"fit_targets.npz","sha256":"..."}`. The NPZ contains `fit_indices`, Unicode `train_patient_ids`, `selected_token_ids[1024]`, bool `mapped_gene_mask[1024]`, bool `membership[P,1024]`, and float32 `targets_train[len(fit),P]`, with1<=P<=64. Preserve the original full fitted-statistics fields as provenance. No held target is used or generated by training.

`src/cancer_kg/data/targets.py` is the original target-building algorithm: verify saved RNA columns against canonical **training** rows; use original relation30/34 (prepared indices29/33), require at least5 distinct mapped genes, select up to64 pathways by gene count then raw-nodeID **before** variance filtering, with no backfill. Fit gene mean/std on training rows (constant threshold1e-6), mean their standardized expression within pathway, retain training pathway variance>1e-6, then standardize each pathway on training rows. The original real-grounding target cache is used. The portable loader verifies axes, train IDs, membership, target scaling and availability; SHA/shape checks do not independently prove an externally supplied target was prepared without leakage. Preserve and audit original preparation provenance.

The original descriptive inner-held target analysis is omitted; it never participated in training or model selection. Main and auxiliary checkpoints and training-only knowledge summaries remain available.

## Optional original preparation helpers

`src/cancer_kg/data/bank.py` is standalone NumPy code, with no path-dependent imports. Its original callable is:

```python
from cancer_kg.data.bank import build_bank
bank, local, report = build_bank(expanded_triples, selected_entity_rows, original_relation_ids)
```

`expanded_triples[M,3]` uses **embedding-row** entity/relation indices; `selected_entity_rows` comes from this fit's saved selected gene IDs. `original_relation_ids[R]` maps relation rows to original KG relation IDs. The result is16384 unique ordered triples, an entity-row-to-local-slot-list dictionary (up to16 slots), and an aggregate preparation report. Entity embeddings `[E,64]`, relation embeddings `[R,64]`, the exact scFoundation gene axis and its verified entity-row map remain user-provided frozen assets. Compose each fact as `[entity_embedding[a], relation_embedding[r], entity_embedding[b]]`; construct the compact anchor and true-entity maps and local roles using the exact original preparation convention documented above. Public raw node IDs must never be silently treated as embedding-row indices. This helper is optional for the portable trainer: the original bank NPZ can be supplied unchanged.

`src/cancer_kg/data/targets.py` is also standalone NumPy code. Once `bind_columns(...)` has verified saved selection indices/symbols against canonical training expression and the scFoundation gene axis, its original preparation calls are:

```python
from cancer_kg.data.targets import memberships, fit_targets
member, pathway_rows, report = memberships(
    bank, selected_token_ids, true_entity, entity_rows,
    node_types, node_ids, original_relation_ids)
stats = fit_targets(raw_selected_rna, fit, member, pathway_rows)
stats.update(selected_token_ids=selected_token_ids,
             mapped_gene_mask=true_entity[selected_token_ids] >= 0,
             train_patient_ids=patient_ids[fit])
```

`raw_selected_rna` has the exact saved selected-column order; only rows `fit` are read by `fit_targets`. `node_types` and `node_ids` use the same embedding-row axis as the bank; the former uses the original `gene/protein` and `pathway` types, the latter stores original raw node IDs for the frozen tie break. `original_relation_ids[29]==30` and `[33]==34` are checked. Save `stats` as the target NPZ for that fit; retain the separate selection/canonical-axis provenance. No dynamic imports or old deployment files are required by these helper functions.
