# scFoundation pooled RNA with the historical small MLP

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

This is the small-MLP embedding cell from the3representation by3head factorial. models.py is unchanged. The embedding occupies the last768 of1792 slots; the first1024 slots are zero. The64/16 ReLU/dropout0.2 head has no output bias. AdamW uses six ordered lr脳wd candidates, max150epochs/patience15, clip1 and full-training-risk Cox. Inner validation Cox chooses epochs, pooled inner concordance chooses the candidate with first-tie selection, and the outer refit uses median3inner epochs. Registered seeds are separate calls.

    pip install -r requirements/base.txt
    python -m tests.smoke

The generated18-subject smoke exercises12roles, all6candidates, saved-state replay, padding slots and invalid endpoint/leakage rejection. It uses2epochs only for synthetic testing.

For each externally supplied role, input NPZ fields are values[N,19264] raw counts, observed bool[N,19264], gene_ids in the pinned official order, patient_ids unique nonempty Unicode[N], positive finite time[N], binary event[N], and nonempty unique integer fit/held indices. Preserve original nested splits. Common structural-zero missing columns are supported. Mixed patient-specific missing RNA is rejected.

    python -m cancer_kg.cli.prepare --config configs/prepare.json
    python -m cancer_kg.cli.encode --config configs/encode.json
    python -m cancer_kg.cli.assemble --config configs/assemble.json

Preparation computes the training median full-library target, normalizes every complete library before selecting exactly1024 genes using the original univariate lifelines Cox p-ranking and stable source-column ties. Nonselected encoder expression becomes zero. The native scFoundation encoder uses log10(sum(selected log1p expression)+1e-8) for both T/S, FP32 efficient-only SDPA and max768 pooling including T/S. Tokens retain their native identities and FP16 storage. Assemble joins exact sample IDs and fit-standardizes pooled features with the original helper.

The external official scFoundation source is fixed to397631c495eddf9ad6644fc00c6ea8139e651245. docs/UPSTREAM.json checks the retained source/vocabulary hashes. Provide the trusted checkpoint SHA yourself. Source is Apache2.0, while model parameters use the separate noncommercial-research scFoundation MODEL_LICENSE. Direct native dependencies are declared in requirements/encoder.txt. Record the resolved Torch/CUDA environment for numerical replay.

After producing all12role files, construct configs/manifest.json with relative/current absolute paths and actual SHA256 values.

    python -m cancer_kg.cli.train --manifest private/manifest.json --manifest-sha SHA256 --output private/results --seed 42

The entry also accepts the original factorial bundle manifest with bundles instead of fits. For prepared-only use, embedding[N,768] is already fit-standardized, and embedding_mean/embedding_sd[768] preserve the original preprocessing receipt. Endpoints and all12role partitions are checked. Patient outputs belong in private storage.

The portable loaders have not been clinically rerun for this delivery. Historical numerical replay additionally requires the original fit-selection, source normalization, encoder cache, private manifest/checkpoint and runtime receipts. A newly prepared dataset or environment does not inherit a historical score. docs/SOURCE_PROVENANCE.json records exact copies, AST extraction and new portable I/O.

Install the direct scFoundation encoder dependency closure with `pip install -r requirements/encoder.txt`. The adapter imports official load/pretrainmodels APIs, which require einops and local_attention. The optional upstream get_embedding CLI additionally uses scanpy. Its command-line preprocessing is separate from this direct adapter.
