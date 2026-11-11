# Frozen OmicsFM proteomics with the historical small MLP

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

The executed backbone uses learned individual protein identities, native10-bin inputs, six transformer layers of width256 and final contextual real-protein-token max pooling. SST and padding are excluded. The frozen backbone is FP32 with TF32 disabled. docs/UPSTREAM.json pins CompOmics/OmicsFM4c12a4debf2544e460c2862f0c220b8afdf22d9f and Hugging Face rednaSander/omicsfm71d857da1db9de896514322aa8b8d7a85396a698.

    pip install -r requirements/base.txt
    python -m tests.smoke

Raw private NPZ fields are values[N,G] median-normalized protein intensity, gene_ids Unicode[G], patient_ids unique nonempty Unicode[N], positive finite time[N], binary event[N], and original fit/held integer indices. NaN denotes missing intensity. Finite signed values and zero are observed. The source is an intensity-domain adaptation, not reconstructed absolute abundance.

    python -m cancer_kg.cli.prepare --config configs/prepare.json
    python -m cancer_kg.cli.encode --config configs/encode.json

Prepare exact-uniquely maps source symbols through the supplied SHA-bound FASTA to the checkpoint's unchanged accession vocabulary. Each patient ranks all mapped finite proteins before fit selection using the original random-tie seed0 and ceil(rank/N脳10). Eligibility requires at least10 and50percent training observations, train median and nonzero filled variance. The original lifelines Cox p-ranking selects exactly1024. Encoder input retains only observed selected tokens, in source order. Empty observed selections fail.

After assembling12role files in the SHA-bound manifest, run the downstream head.

    python -m cancer_kg.cli.train --manifest private/manifest.json --manifest-sha SHA256 --output private/results --seed 42

Each embedding[N,256] is fit-standardized and placed in the first256 of1792 slots, with the remaining1536 zero. models.py is the unchanged historical64/16 ReLU/dropout0.2 smallMLP, no output bias, six lr脳wd candidates, max150/patience15, clip1, full-risk Cox. Inner validation loss chooses epochs, pooled inner concordance chooses candidates with first ties, and outer epochs use median3inner. Seeds42/20261022/20261023 are separate calls. The generated-data smoke covers all12roles and6candidates at2synthetic epochs plus malformed-input rejection and checkpoint replay.

Code license is Apache2.0. The pinned model-card metadata declares weights CC-BY4.0, requiring the corresponding attribution. External checkpoints and source are acquired separately. Public pretraining-project overlap remains unresolved in the historical experiment. docs/SOURCE_PROVENANCE.json records the original implementation and new I/O boundary.

This portable rehost was tested synthetically. Clinical scores were not recomputed for this delivery. Exact historical replay needs the original mapping, fitted selection, native-bin axis, private embedding/checkpoint manifests and resolved runtime. Rehosted source paths alone do not establish numerical identity.

Use `pip install -r requirements/encoder.txt` for native encoding and preparation. The pinned architecture imports Torch only, and encoder.py reads the configuration dictionary from the checkpoint. YAML and einops are not runtime imports of this entrypoint.

Optional token_output in encode.json exports actual FP32 native protein tokens with stable checkpoint IDs, valid/special/biological masks and matched max pool, excluding SST. RNA encoder shards expose corresponding masks and a pooled alias for downstream compatibility.
