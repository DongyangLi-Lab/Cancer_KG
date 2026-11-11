"""Frozen scFoundation encoder for previously selected normalized-log1p inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cancer_kg.encoders.native import ScFoundation
from cancer_kg.cli.train import sha


def run(config):
    upstream = json.loads((Path(__file__).resolve().parents[3] / "docs" / "UPSTREAM.json").read_text(encoding="utf-8"))
    for relative, digest in upstream["source_files_sha256"].items():
        if sha(Path(config["repo_dir"]) / relative) != digest:
            raise ValueError("Pinned official source differs")
    for key in ("checkpoint", "input"):
        if sha(config[key]) != config[key + "_sha256"]:
            raise ValueError("Trusted input/checkpoint SHA differs")
    if (config["pre_normalized"] != "T" or config["pool_type"] != "max"
            or config["precision"] != "fp32" or config.get("input_unit") != "selected_normalized_log1p"):
        raise ValueError("The historical selected-log1p T/max recipe is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = ScFoundation(config)
    target = Path(config["output"])
    target.mkdir(parents=True, exist_ok=False)
    with np.load(config["input"], allow_pickle=False) as source:
        values, observed, sample_ids, genes = (source[key] for key in ("values", "observed", "sample_ids", "gene_ids"))
        if values.ndim != 2 or observed.shape != values.shape or observed.dtype != bool:
            raise ValueError("Canonical input shape/mask differs")
        if len(sample_ids) != len(values) or len(set(sample_ids)) != len(values) or genes.shape != (values.shape[1],):
            raise ValueError("Canonical identity axes differ")
        records = []
        for index, sample in enumerate(sample_ids):
            with torch.inference_mode():
                pool, ids, valid, mask, tokens = encoder.encode(values[index], observed[index], genes, str(sample), True)
            path = target / (str(index) + ".npz")
            special = (ids >= 19264) | ~valid
            saved_pool = pool.astype(np.float16)
            np.savez(path, sample_id=sample, tokens=tokens.astype(np.float16), token_ids=ids,
                token_valid_mask=valid, token_special_mask=special, token_biological_mask=valid & ~special,
                input_observed_mask=mask, pooled_max=saved_pool, pooled=saved_pool)
            records.append(dict(file=path.name, sha256=sha(path)))
    (target / "manifest.json").write_text(json.dumps(dict(status="complete", patients=records, config=config,
        token_dim=768, storage="float16", selection_performed=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
