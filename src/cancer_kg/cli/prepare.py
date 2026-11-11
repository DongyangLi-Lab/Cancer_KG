"""One declared fit's historical full-library normalization and Cox1024 selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cancer_kg.data.selection import rank_cox
from cancer_kg.cli.train import sha


def validate(data):
    n = len(data["patient_ids"])
    ids, time, event = (data[key] for key in ("patient_ids", "time", "event"))
    if (ids.shape != (n,) or ids.dtype.kind != "U" or len(set(ids)) != n or any(not value.strip() for value in ids)
            or time.shape != (n,) or event.shape != (n,) or not np.isfinite(time).all()
            or np.any(time <= 0) or not np.isin(event, [0, 1]).all()):
        raise ValueError("Malformed patient/endpoint axes")
    for key in ("fit", "held"):
        indices = data[key]
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n)
                or not event[indices].any()):
            raise ValueError("Malformed fit/held indices")
    if set(data["fit"]) & set(data["held"]):
        raise ValueError("Fit and held overlap")


def normalized_counts(values, observed, fit):
    if (values.ndim != 2 or values.shape != observed.shape or observed.dtype != bool
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Raw-count values and mask differ")
    structural_missing = (~observed).all(0)
    if not np.all(observed.all(0) | structural_missing) or np.any(values[:, structural_missing] != 0):
        raise ValueError("Only common structural-zero absent columns are supported")
    totals = values.sum(1)
    target = float(np.median(totals[fit]))
    if np.any(totals <= 0) or target <= 0:
        raise ValueError("Positive RNA library sizes required")
    return np.log1p(values / totals[:, None] * target), target


def run(config):
    if config.get("input_unit") != "raw_counts":
        raise ValueError("Explicit raw-count units are required")
    if sha(config["input"]) != config["input_sha256"]:
        raise ValueError("Raw source SHA differs")
    with np.load(config["input"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("values", "observed", "gene_ids", "patient_ids", "time", "event", "fit", "held")}
    validate(data)
    upstream = json.loads((Path(__file__).resolve().parents[3] / "docs" / "UPSTREAM.json").read_text(encoding="utf-8"))
    gene_index = Path(config["gene_index"])
    if sha(gene_index) != upstream["source_files_sha256"]["model/OS_scRNA_gene_index.19264.tsv"]:
        raise ValueError("Pinned official vocabulary differs")
    with gene_index.open(encoding="utf-8") as stream:
        genes = np.array([row["gene_name"] for row in csv.DictReader(stream, delimiter="\t")])
    if (len(genes) != 19264 or not np.array_equal(data["gene_ids"], genes)
            or data["values"].shape != (len(data["patient_ids"]), 19264)):
        raise ValueError("Counts require the complete official19264 vocabulary order")
    normalized, target = normalized_counts(data["values"].astype(np.float64), data["observed"], data["fit"])
    selected, pvalues, failures = rank_cox(normalized[data["fit"]], data["time"][data["fit"]], data["event"][data["fit"]])
    if not data["observed"][:, selected].all():
        raise ValueError("Selected an absent gene")
    masked = np.zeros_like(normalized, dtype=np.float32)
    masked[:, selected] = normalized[:, selected]
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, values=masked, observed=data["observed"], gene_ids=genes,
            sample_ids=data["patient_ids"], raw=normalized[:, selected].astype(np.float32),
            selected_gene_indices=selected, target_sum_train_median=target,
            **{key: data[key] for key in ("patient_ids", "time", "event", "fit", "held")})
    output.with_suffix(".json").write_text(json.dumps(dict(source_sha256=config["input_sha256"],
        output_sha256=sha(output), selected=1024, Cox_failures=len(failures),
        normalization_target_sum_train_median=target, held_outcomes_used=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
