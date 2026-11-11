import hashlib, json
from pathlib import Path
import numpy as np

ARMS = ("G_real", "G_shuffled", "G_identity_only")


MODALITIES = ("rna", "protein", "ptm", "cnv")


ROLES = tuple(f"r0_f{fold}_{role}" for fold in range(3)
              for role in ("inner0", "inner1", "inner2", "outer"))


RECIPE = dict(seed=42, max_epochs=24, patience=6, lr=1e-3, weight_decay=.01,
              microbatch=4, width=64, graph_layers=2, clip=1.)


RELATIONS = ("member", "member_reverse", "parent", "parent_reverse")


ENDPOINT = ("patient_ids", "time", "event", "fit", "held")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def checked(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_roles(fits):
    if set(fits) != set(ROLES):
        raise ValueError("Exactly the twelve frozen fit roles are required")
    ref = fits[ROLES[0]]
    ids, time, event = (np.asarray(ref[k]) for k in ENDPOINT[:3])
    n = len(ids)
    if (ids.shape != (133,) or len(set(ids.astype(str))) != n or time.shape != (n,)
            or event.shape != (n,) or not np.isfinite(time).all() or not np.isin(event, [0, 1]).all()
            or event.sum() != 106):
        raise ValueError("Expected 133 unique patients and 106 events")
    for z in fits.values():
        for key in ENDPOINT[:3]:
            np.testing.assert_array_equal(z[key], ref[key])
        for key in ENDPOINT[3:]:
            ix = np.asarray(z[key])
            if (ix.ndim != 1 or ix.dtype.kind not in "iu" or not len(ix) or len(set(ix)) != len(ix)
                    or (ix < 0).any() or (ix >= n).any() or not event[ix].any()):
                raise ValueError("Invalid fit/held index or no observed event")
        if set(z["fit"]) & set(z["held"]):
            raise ValueError("Fit and held indices intersect")
    outer_held = []
    for fold in range(3):
        outer = fits[f"r0_f{fold}_outer"]
        if set(outer["fit"]) | set(outer["held"]) != set(range(n)):
            raise ValueError("Outer fold is not a cohort partition")
        inner_held = []
        for inner in range(3):
            z = fits[f"r0_f{fold}_inner{inner}"]
            if set(z["fit"]) | set(z["held"]) != set(outer["fit"]):
                raise ValueError("Inner fold touches outer held indices")
            inner_held.extend(z["held"])
        if len(inner_held) != len(set(inner_held)) or set(inner_held) != set(outer["fit"]):
            raise ValueError("Inner held rows must partition outer fit")
        outer_held.extend(outer["held"])
    if len(outer_held) != n or len(set(outer_held)) != n:
        raise ValueError("Outer held rows must partition cohort")


def c_counts(time, event, risk):
    event = np.asarray(event, bool)
    if not np.isfinite(risk).all():
        raise ValueError("Nonfinite OOF risk")
    comparable = event[:, None] & ((time[:, None] < time[None, :])
                                  | ((time[:, None] == time[None, :]) & ~event[None, :]))
    delta = risk[:, None] - risk[None, :]
    return float(((delta > 1e-8) + .5 * (abs(delta) <= 1e-8))[comparable].sum()), int(comparable.sum())
