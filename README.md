# Four-modal KG pathway tensor kernels

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

The retained executed configuration is **real_additive_selfquadratic**: four
measured modalities organized by real pathway knowledge, without the unsuccessful
explicit cross-modal product addition. This branch keeps kernels.py and cox.py
byte-for-byte; run.py replaces UNC authorization/scheduler dependencies with
a SHA-bound twelve-role private input manifest. No GPU is required.

```bash
pip install -r requirements/base.txt
pip install -r requirements/test.txt  # independent autograd reference tests
python -m unittest discover -s tests -v
python -m cancer_kg.cli.run --manifest /secure/prepared/PREPARED.json --output /secure/fresh_tensor_run
```

See docs/INPUTS.md. Reuses corrected graph preparation; RNA/protein native token
values, PTM/CNV values and effective observation masks are all included.
All12 support/effective gates run before survival fitting. Each block is
fit-centered/scaled; raw + pathway-linear + self-quadratic + coverage are common.
The original common real/shuffled support and effective-pair checks remain in
kernel preparation: removing them would change the executed pathway inventory.
Only the selected real kernel is saved and trained. Other model arms are not run.
This is not a new CUDA kernel or a bigger pretrained model.
No final Gram renormalization. Full-risk FP64 Breslow Cox, alpha grid
.001/.01/.1/1, three inner held-loss selection, 39 solves for one model.
The original solver convergence threshold is unchanged.

Historical CPTAC133/106 C-index: **0.621322**, the highest observed real-KG
four-modal configuration among the compared deliveries. This is retrospective
code curation on an explored development cohort, not new external validation
or evidence of a KG-specific gain. SUMMARY.json reports the selected model with
2000 patient resamples within folds; expected comparable pairs2345. It does not
include the historical cluster's independent checkpoint replay audit.
This portable packaging is synthetic-tested, not a new patient-data rerun.
Patient caches/checkpoints/predictions are private and excluded from Git.
