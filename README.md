# RNA/protein pooled feature fusion (C_pool)

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

This is the executed 2026-10-31 **feature fusion before one MLP**, not risk-score
ensembling: OmicsFM256 occupies `[0:256]`, `[256:1024]` stays zero, scFoundation768
occupies `[1024:1792]`. Preserve these 1792 slots to preserve historical parameter
initialization. Head: 1792→64/ReLU/dropout0.2→16/ReLU/dropout0.2→1(no output bias).
There is no KG in this historical branch.

## Run

```bash
python -m tests.smoke
python -m cancer_kg.cli.train --manifest /path/to/fits.json --output /new/private/results
```

The output directory must not already exist. Training runs 54 inner candidates
plus 3 outer refits. Six candidates, in order: lr `[1e-4,1e-3]` × weight decay
`[1e-4,0.01,0.1]`; AdamW, seed42, full-fit Breslow Cox, clip1, maximum150 epochs,
patience15. Inner early stopping uses validation Cox loss; candidate selection
uses sum of three inner concordance credits / sum of comparable pairs, first tie.
Outer epochs are the integer median of the selected three inner best epochs.
Only within-fold risks are compared; the original cohort has2345 comparable pairs.

## Prepared input schema

One JSON manifest: `{"fits":[{"fit_id":"r0_f0_inner0","file":"fit.npz",
"sha256":"<64-hex>"}, ...]}`. Exactly12 names:
`r0_f{0,1,2}_{inner0,inner1,inner2,outer}`. Paths are relative to the manifest.
Each NPZ uses only non-object arrays (`allow_pickle=False`):

| key | shape/type | meaning |
|---|---|---|
| patient_ids | Unicode[133] | common unique row IDs, never model input |
| time,event | [133], float/binary | time and event;106 events |
| fit,held | integer index vectors | fixed disjoint training/evaluation roles |
| scf | finite float[133,768] | **already standardized using this fit's training RNA** |
| scf_mean,scf_sd | float[768] | provenance of that RNA standardization |
| omics | finite float[133,256] | original pooled protein; worker standardizes using fit only |

Do not standardize scf a second time. Do not replace original max pools with token
means. Inner roles must partition the corresponding outer-training patients.
The script writes checkpoints and private prediction tables only to your output.
No historical outcomes or reported scores are encoded as training inputs.

The pretrained encoders are **frozen**. Official upstream projects:
- scFoundation: https://github.com/biomap-research/scFoundation ; the historical
  checkpoint distribution was https://huggingface.co/genbio-ai/scFoundation
  (not claimed byte-identical to the author's original SharePoint download).
- OmicsFM: https://github.com/CompOmics/OmicsFM ; checkpoint distribution
  https://huggingface.co/rednaSander/omicsfm . The run used `use_esmc=False` and
  no groups, original vocabulary, per-patient rank/bin inputs, and observed-protein
  max pooling excluding SST/padding.

This package starts from prepared frozen embeddings; it does not reimplement the
encoders, redistribute their weights, download data, or include Tahoe/KG. Generate
inputs with the upstream runtimes and the original fit-specific preprocessing.
Selections (1024 candidate genes/proteins), normalization and any learned feature
processing must be fitted within each training role, never across all patients.
Native RNA and protein vocabularies are independent: do not join their integer IDs
by position. Embedding extraction on a held individual is inference, not fitting.

Install PyTorch appropriate for the host/CUDA separately, then
`python -m pip install -r requirements/base.txt`. Historical GPU runtime: PyTorch
2.7.1+cu126, FP32, TF32 disabled. Candidate CPU smoke runtime is recorded in the
delivery QA; matching a different runtime does not guarantee identical patient
predictions. No patient data, weights, predictions or historical logs are bundled.
All runtime outputs are private artifacts, not files to commit to this branch.

`docs/SOURCE.json` binds original sources, extraction changes and candidate files.
Scientific computations are extracted from executed code; the portable file I/O
and command-line wrapper are new and tested only on synthetic data. Historical
UNC path checks, approval receipts and deployment manifest chains are omitted.
The original scientific evaluation remains 133 patients/106 events, repeat 0,
3 outer folds each with 3 inner folds; this is a reproduction interface, not a
generic split generator. Do not use held labels to select preprocessing or epochs.


Third-party licenses remain separate: scFoundation assets follow their original research/noncommercial model terms; OmicsFM source is Apache-2.0 and the [pinned model card](https://huggingface.co/rednaSander/omicsfm/blob/71d857da1db9de896514322aa8b8d7a85396a698/README.md) declares CC-BY-4.0 for weights. No upstream source tree or weights are bundled. This delivery does not assign a project-wide license; the upstream licenses remain unchanged.
