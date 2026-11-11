"""Original exact-unique protein mapping, full-axis bins and fit-only Cox1024."""
import argparse
import json
from pathlib import Path
import numpy as np
from cancer_kg.data.binning import fasta_mapping, native_bins, fit_candidates
from cancer_kg.data.contracts import validate
from cancer_kg.encoders.encoder import load
from cancer_kg.data.selection import rank_cox
from cancer_kg.cli.train import sha


def run(config):
    upstream = json.loads((Path(__file__).resolve().parents[3] / "docs" / "UPSTREAM.json").read_text(encoding="utf-8"))
    architecture = Path(config["repo_dir"]) / "omicsfm/architecture.py"
    if sha(architecture) != upstream["source_files_sha256"]["omicsfm/architecture.py"]:
        raise ValueError("Pinned native architecture differs")
    for key in ("input", "checkpoint", "fasta"):
        if sha(config[key]) != config[key + "_sha256"]:
            raise ValueError("Source/asset SHA differs")
    with np.load(config["input"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("values", "gene_ids", "patient_ids", "time", "event", "fit", "held")}
    validate(data)
    values, genes = data["values"].astype(np.float64), data["gene_ids"]
    if values.shape != (len(data["patient_ids"]), len(genes)) or genes.dtype.kind != "U" or len(set(genes)) != len(genes):
        raise ValueError("Protein matrix/source gene axis differs")
    _, vocabulary, _ = load(config["checkpoint"], architecture, "cpu")
    mapping, ambiguous = fasta_mapping(config["fasta"])
    keep = np.array([i for i, gene in enumerate(genes) if gene in mapping and mapping[gene] in vocabulary], dtype=np.int64)
    accessions = np.array([mapping[genes[i]] for i in keep])
    if len(set(accessions)) != len(accessions):
        raise ValueError("Several source genes map to one accession")
    values = values[:, keep]
    native_ids = np.array([vocabulary[accession] for accession in accessions])
    bins = np.stack([native_bins(row) for row in values])
    eligible, median, filled = fit_candidates(values, data["fit"])
    columns = np.flatnonzero(eligible)
    if len(columns) < 1024:
        raise ValueError("Fewer than1024 eligible observed mapped proteins")
    chosen, pvalues, failures = rank_cox(filled[:, columns], data["time"][data["fit"]], data["event"][data["fit"]])
    selected = columns[chosen]
    if not (bins[:, selected] > 0).any(1).all():
        raise ValueError("A patient has no observed selected protein")
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, feature_ids=native_ids[selected], bins=bins[:, selected], train_median=median[selected],
            selected_accessions=accessions[selected], selected_source_symbols=genes[keep][selected],
            cox_p=pvalues[chosen], **{key: data[key] for key in ("patient_ids", "time", "event", "fit", "held")})
    output.with_suffix(".json").write_text(json.dumps(dict(source_sha256=config["input_sha256"], output_sha256=sha(output),
        selected=1024, mapped=len(keep), ambiguous_symbols=len(ambiguous), Cox_failures=len(failures),
        binning_before_selection=True, held_outcomes_used=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
