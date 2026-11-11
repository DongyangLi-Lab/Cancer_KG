"""Join native max pools to one fit's source order and fit-standardize the768 block."""
import argparse
import json
from pathlib import Path
import numpy as np
import models
from train import sha
from prepare import validate


def run(config):
    for key in ("source", "cache_manifest"):
        if sha(config[key]) != config[key + "_sha256"]:
            raise ValueError("Source/cache manifest SHA differs")
    with np.load(config["source"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("patient_ids", "time", "event", "fit", "held")}
    validate(data)
    path = Path(config["cache_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for entry in manifest["patients"]:
        member = (path.parent / entry["file"]).resolve()
        if not member.is_relative_to(path.parent.resolve()) or sha(member) != entry["sha256"]:
            raise ValueError("Shard provenance differs")
        with np.load(member, allow_pickle=False) as shard:
            sample = str(shard["sample_id"].item())
            pool = shard["pooled_max"].astype(np.float32)
            if sample in rows or pool.shape != (768,) or not np.isfinite(pool).all():
                raise ValueError("Pool identity/shape differs")
            rows[sample] = pool
    if set(rows) != set(data["patient_ids"]):
        raise ValueError("Cache/source patient sets differ")
    standardized, stats = models.standardize(np.stack([rows[value] for value in data["patient_ids"]]), data["fit"])
    with Path(config["output"]).open("xb") as stream:
        np.savez(stream, embedding=standardized, embedding_mean=stats["mean"], embedding_sd=stats["sd"], **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))
