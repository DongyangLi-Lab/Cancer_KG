"""Portable small-head cell from the historical nine-cell head factorial."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
import models

MODE = "scfoundation"
CANDIDATES = [dict(lr=lr, wd=wd) for lr in (1e-4, 1e-3) for wd in (1e-4, .01, .1)]
ROLES = tuple(f"r0_f{fold}_{role}" for fold in range(3) for role in ("inner0", "inner1", "inner2", "outer"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(data):
    value = np.asarray(data["embedding"])
    width = 768 if MODE == "scfoundation" else 256
    if value.ndim != 2 or value.shape != (len(data["patient_ids"]), width) or not np.isfinite(value).all():
        raise ValueError("Embedding width/finite-value contract differs")
    if MODE == "scfoundation":
        for key in ("embedding_mean", "embedding_sd"):
            if np.asarray(data[key]).shape != (768,) or not np.isfinite(data[key]).all():
                raise ValueError("Historical standardized RNA bundle requires its fit normalization")
        stats = {"mean": data["embedding_mean"], "sd": data["embedding_sd"]}
    else:
        value, stats = models.standardize(value, data["fit"])
    slots = np.zeros((len(value), 1792), dtype=np.float64)
    slots[:, 1024:] = value if MODE == "scfoundation" else 0.
    if MODE == "omicsfm":
        slots[:, :256] = value
    return slots, stats


def counts(time, event, risk):
    comparable = event[:, None].astype(bool) & ((time[:, None] < time[None, :])
        | ((time[:, None] == time[None, :]) & ~event[None, :].astype(bool)))
    difference = risk[:, None] - risk[None, :]
    return float(((difference > 1e-8) + .5 * (np.abs(difference) <= 1e-8))[comparable].sum()), int(comparable.sum())


def validate_roles(fits):
    if set(fits) != set(ROLES):
        raise ValueError("Exactly twelve original nested roles are required")
    reference = fits[ROLES[0]]
    n = len(reference["patient_ids"])
    if (reference["patient_ids"].dtype.kind != "U" or len(set(reference["patient_ids"].tolist())) != n
            or any(not str(value).strip() for value in reference["patient_ids"])):
        raise ValueError("Duplicate patient IDs")
    for data in fits.values():
        for key in ("patient_ids", "time", "event"):
            if not np.array_equal(reference[key], data[key]) or data[key].shape != (n,):
                raise ValueError("Patient/endpoint axes differ between roles")
        if (not np.isfinite(data["time"]).all() or np.any(data["time"] <= 0)
                or not np.isin(data["event"], [0, 1]).all()):
            raise ValueError("Endpoint schema differs")
        for key in ("fit", "held"):
            indices = data[key]
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n)
                    or not data["event"][indices].any()):
                raise ValueError("Malformed partition or no events")
        if set(data["fit"]) & set(data["held"]):
            raise ValueError("Fit/held overlap")
    outer_held = []
    for fold in range(3):
        outer = fits[f"r0_f{fold}_outer"]
        if set(outer["fit"]) | set(outer["held"]) != set(range(n)):
            raise ValueError("Outer cohort partition differs")
        outer_held.extend(outer["held"])
        inner_held = []
        for inner in range(3):
            data = fits[f"r0_f{fold}_inner{inner}"]
            if set(data["fit"]) | set(data["held"]) != set(outer["fit"]):
                raise ValueError("Inner role leaks outer-held patients")
            inner_held.extend(data["held"])
        if sorted(inner_held) != sorted(outer["fit"]):
            raise ValueError("Inner held partitions differ")
    if sorted(outer_held) != list(range(n)):
        raise ValueError("Outer held partitions differ")


def load_manifest(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError("Manifest SHA mismatch")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("fits", doc.get("bundles", []))
    if len(rows) != 12:
        raise ValueError("Twelve fit files are required")
    fits, sources = {}, {}
    for row in rows:
        file = Path(row["file"])
        file = file if file.is_absolute() else path.parent / file
        if sha(file) != row["sha256"] or row["fit_id"] in fits:
            raise ValueError("Fit SHA mismatch or repeated role")
        with np.load(file, allow_pickle=False) as archive:
            names = ["patient_ids", "time", "event", "fit", "held", "embedding"]
            if MODE == "scfoundation":
                names += ["embedding_mean", "embedding_sd"]
            fits[row["fit_id"]] = {key: archive[key] for key in names}
        sources[row["fit_id"]] = row["sha256"]
    validate_roles(fits)
    return fits, sources


def run(fits, sources, output, seed=42, synthetic_epochs=None):
    validate_roles(fits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    choices, oof = [], []
    for fold in range(3):
        rows = []
        for candidate_index, candidate in enumerate(CANDIDATES):
            for inner in range(3):
                name = f"r0_f{fold}_inner{inner}"
                data = fits[name]
                x, stats = matrix(data)
                risk, state, info = models.mlp(x, data["time"], data["event"], data["fit"], data["held"],
                    "small", seed, **candidate, epochs=synthetic_epochs or 150, early=True, patience=15)
                np.testing.assert_array_equal(models.replay(state, x[data["held"]]), risk)
                credit, pairs = counts(data["time"][data["held"]], data["event"][data["held"]], risk)
                rows.append(dict(candidate_index=candidate_index, credit=credit, pairs=pairs, **info))
                with (output / f"{name}_candidate{candidate_index}.pkl").open("xb") as stream:
                    pickle.dump(dict(state=state, normalization=stats, source_sha256=sources[name]), stream)
        scores = [sum(row["credit"] for row in rows if row["candidate_index"] == i)
                  / sum(row["pairs"] for row in rows if row["candidate_index"] == i) for i in range(6)]
        best = int(np.argmax(scores))
        epochs = int(np.median([row["best_epoch"] for row in rows if row["candidate_index"] == best]))
        name = f"r0_f{fold}_outer"
        data = fits[name]
        x, stats = matrix(data)
        risk, state, info = models.mlp(x, data["time"], data["event"], data["fit"], data["held"],
            "small", seed, **CANDIDATES[best], epochs=epochs, early=False, patience=15)
        np.testing.assert_array_equal(models.replay(state, x[data["held"]]), risk)
        with (output / (name + ".pkl")).open("xb") as stream:
            pickle.dump(dict(state=state, normalization=stats, source_sha256=sources[name]), stream)
        choices.append(dict(fold=fold, candidate_index=best, candidate=CANDIDATES[best],
                            scores=scores, epochs=epochs, inner=rows))
        for index, value in zip(data["held"], risk):
            oof.append((data["patient_ids"][index], fold, data["time"][index], int(data["event"][index]), float(value)))
    with (output / "private_oof.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("case_id", "fold", "time", "event", "risk"))
        writer.writerows(oof)
    (output / "selection.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    (output / "COMPLETE.json").write_text(json.dumps(dict(mode=MODE, seed=seed, source_sha256=sources,
        synthetic=synthetic_epochs is not None, source_model_sha256=sha(Path(__file__).with_name("models.py")),
        torch=torch.__version__, numpy=np.__version__, slots=1792, candidates=CANDIDATES), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 20261022, 20261023), default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, sources = load_manifest(args.manifest, args.manifest_sha)
    run(data, sources, args.output, args.seed)
