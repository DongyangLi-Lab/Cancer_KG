"""Frozen final native real-protein-token max from previously prepared bins."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cancer_kg.encoders.encoder import load, batch
from cancer_kg.cli.train import sha


@torch.inference_mode()
def native_outputs(model, inputs):
    context = model(inputs)["transformer_out"]
    real = (inputs["role"] == 1) & (inputs["feature_ids"][..., 0] > 0)
    if context.ndim != 3 or context.shape[0] != 1 or not real.any(1).all():
        raise ValueError("A serial observed-protein native batch is required")
    pool = context.masked_fill(~real[..., None], -torch.inf).amax(1)[0].float().cpu().numpy()
    tokens = context[0, real[0]].float().cpu().numpy()
    ids = inputs["feature_ids"][0, real[0], 0].cpu().numpy()
    if not np.isfinite(tokens).all() or not np.isfinite(pool).all():
        raise ValueError("Native encoder output is nonfinite")
    return dict(pooled=pool, tokens=tokens, token_ids=ids,
        token_valid_mask=np.ones(len(ids), dtype=bool), token_special_mask=np.zeros(len(ids), dtype=bool),
        token_biological_mask=np.ones(len(ids), dtype=bool))


def run(config):
    upstream = json.loads((Path(__file__).resolve().parents[3] / "docs" / "UPSTREAM.json").read_text(encoding="utf-8"))
    architecture = Path(config["repo_dir"]) / "omicsfm/architecture.py"
    if sha(architecture) != upstream["source_files_sha256"]["omicsfm/architecture.py"]:
        raise ValueError("Pinned architecture differs")
    for key in ("checkpoint", "input"):
        if sha(config[key]) != config[key + "_sha256"]:
            raise ValueError("Trusted input/checkpoint SHA differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, _, _ = load(config["checkpoint"], architecture, config["device"])
    if any(p.dtype != torch.float32 or p.requires_grad for p in model.parameters()):
        raise ValueError("The historical frozen FP32 backbone is required")
    token_root = Path(config["token_output"]) if config.get("token_output") else None
    if token_root is not None:
        token_root.mkdir(parents=True, exist_ok=False)
    records = []
    with np.load(config["input"], allow_pickle=False) as source:
        feature_ids, bins = source["feature_ids"], source["bins"]
        if bins.ndim != 2 or feature_ids.shape != (bins.shape[1],) or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Prepared ID/bin axes differ")
        vectors = []
        for index, row in enumerate(bins):
            observed = row > 0
            native = native_outputs(model, batch(feature_ids[observed], row[observed], config["device"]))
            vectors.append(native["pooled"])
            if token_root is not None:
                shard = token_root / (str(index) + ".npz")
                np.savez(shard, sample_id=source["patient_ids"][index], input_observed_mask=observed, **native)
                records.append(dict(file=shard.name, sha256=sha(shard)))
        arrays = {key: source[key] for key in ("patient_ids", "time", "event", "fit", "held")}
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, embedding=np.stack(vectors), **arrays)
    if token_root is not None:
        (token_root / "manifest.json").write_text(json.dumps(dict(status="complete", token_dim=256,
            patients=records, source_sha256=config["input_sha256"], checkpoint_sha256=config["checkpoint_sha256"]), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
