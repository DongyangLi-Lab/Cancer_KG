# Private input contract

This minimal delivery starts at **per-fit prepared features**, not raw downloads.
It does not fabricate features, silently impute raw data, or reconstruct the
historical patient split. Provide the original approved private cache or prepare
equivalent inputs under your data access permissions. Historical numerical
reproduction needs the original cache and environment; no patient files ship here.

`manifest.json` has `fits`, a list of 12 objects with `fit_id`, `file` (relative
to manifest or absolute), and file `sha256`. IDs are `r0_f0_inner0` through
`r0_f2_inner2` plus `r0_f0_outer`, `r0_f1_outer`, `r0_f2_outer`.
The original PREPARED.json fits list has the same interface.

Every NPZ uses allow_pickle=False and identical patient ordering:

| Field | Shape / semantics |
|---|---|
| patient_ids | Unicode [133], unique; never committed |
| time, event | [133], positive follow-up, binary event; total 106 events |
| fit, held | Integer index vectors, disjoint; inner folds partition outer train |
| rna, rna_mask, rna_pool | [133,R,768], bool [133,R], [133,768] |
| protein, protein_mask, protein_pool | [133,P,256], bool [133,P], [133,256] |
| ptm, ptm_mask | float32 [133,T], bool [133,T] |
| cnv, cnv_mask | float32 [133,C], bool [133,C] |
| node_ids, node_type | Unicode [V], integer [V]; gene=0, orphan assay=1, pathway=2 |
| {rna,protein,ptm,cnv}_node | Integer feature-to-node maps |
| {real,shuffled}_{member,member_reverse,parent,parent_reverse} | Integer [E,2] source,destination edges |

Original R/P/T/C counts: 1024/1024/1024/512. RNA native storage FP16,
protein FP32. RNA/protein axes were selected separately inside each fit with Cox;
PTM/CNV axes used fit-only observation eligibility and variance ranking. Native
token dimensions must never be pooled or truncated silently.
PTM/CNV values are **already standardized using fit-only statistics**, including
training-median imputation; their missing masks remain false. Native absent
tokens are zero; RNA mask indicates returned positive-input token availability,
not necessarily missing raw measurements. Pooled RNA/protein vectors are
unstandardized original max pools; GNN concat/readout branches standardize them
inside make_batch, tensor kernels use their own fit-only block transforms.

Graph mapping must use the corrected generic PTM parent identity, not the old
kinase-supported subset. `src/cancer_kg/data/graph.py` implements public assay-to-gene joining,
Reactome release 97 All_Levels membership, hierarchy and degree-preserving
shuffle. Supply `assays` in the exact selected feature order, symbol-to-NCBI Gene
mapping, stored pathway membership and hierarchy to `build_graph`; merge its
arrays into each prepared role before calculating the NPZ hash. An identity
join is not evidence of a regulatory interaction. No patient outcome may be
used to construct graph topology or shared preprocessing.
