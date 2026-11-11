# Cancer_KG

Minimal runnable research code for pancreatic-cancer survival prediction,
foundation-model representations, multi-omics fusion, and knowledge graphs.

Each method has its own branch. Check out **one branch** to get its model,
portable training entry point, dependency list, input contract and synthetic
checks. No other research branch needs to be checked out at runtime.

| Branch | Executed method line |
|---|---|
| [single/rna-scfoundation](https://github.com/DongyangLi-Lab/Cancer_KG/tree/single/rna-scfoundation) | RNA → scFoundation → max pooling → small MLP |
| [single/protein-omicsfm](https://github.com/DongyangLi-Lab/Cancer_KG/tree/single/protein-omicsfm) | Protein → OmicsFM → max pooling → small MLP |
| [fusion/rna-protein-pooled](https://github.com/DongyangLi-Lab/Cancer_KG/tree/fusion/rna-protein-pooled) | RNA and protein pooled features fused before one MLP |
| [kg/pathway-alignment](https://github.com/DongyangLi-Lab/Cancer_KG/tree/kg/pathway-alignment) | RNA + real KG with calibrated reader and pathway auxiliary loss |
| [kg/fourmodal-pathway](https://github.com/DongyangLi-Lab/Cancer_KG/tree/kg/fourmodal-pathway) | RNA + protein + PTM + CNV: real pathway additive/self-quadratic kernel Cox |

## Start

```bash
git clone https://github.com/DongyangLi-Lab/Cancer_KG.git
cd Cancer_KG
git switch single/rna-scfoundation
# Follow this branch's README for dependencies, inputs and checks.
```

The master branch is only this index. Methods share Git objects where source
files are identical; old deployment releases and duplicate debug versions are
not included. Only the strongest observed executed configuration in each of the
five retained method categories is delivered. Superseded GNNs, larger-transformer
trials, token-fusion trials and ablation-only branches are not published.

| Retained category | Historical within-fold C-index |
|---|---:|
| RNA: scFoundation + small MLP | 0.640512 |
| Protein: OmicsFM + small MLP | 0.574840 |
| RNA + protein: pooled feature fusion | 0.639659 |
| RNA + real KG: pathway alignment | 0.631130 |
| Four measured modalities + real KG: pathway kernel | 0.621322 |

These are seed42 development-cohort point estimates used to curate the code
delivery, not a new held-out method selection experiment or proof of superiority.
RNA + KG contains one measured modality plus biological knowledge; only the
four-modal branch combines measured RNA, protein, PTM and CNV.

## Reproduction scope

The main PDAC experiments use CPTAC133 patients/106 events and fixed nested
three-fold partitions. Previously explored development results are not new
external validation. These branches are **minimal portable extractions of
executed methods**, with source hashes and documented I/O/launch adaptations.
Synthetic checks validate packaging; they do not constitute a fresh clinical
rerun or guarantee identical floating-point results on different hardware.

Patient tables, patient-level predictions, pretrained weights, fitted models,
private manifests, caches and experiment logs are excluded. Obtain data and
upstream weights under their respective access terms. Single-modality branches
describe encoder preparation; fusion/KG branches explicitly document the
private per-fit embedding/token boundary. Preprocessing, feature selection and
hyperparameter selection belong inside training folds. Never generate fake
tokens by repeating pooled embeddings.

Only the retained real-KG configurations are exposed as training methods.
Necessary frozen input/support checks remain so that extraction does not change
the executed mathematics. Historical results do not establish KG superiority;
a statistical kernel is not itself a new CUDA kernel.

See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency and model-license boundaries.
