"""Portable manifest loading: private precomputed inputs are supplied separately."""
import json
from pathlib import Path
import numpy as np
from common import ARMS, ENDPOINT, ROLES, checked, sha, validate_roles
from schema import validate_features

def load_manifest(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = doc["fits"]
    if len(rows) != len(ROLES) or {r["fit_id"] for r in rows} != set(ROLES):
        raise ValueError("Exactly twelve unique role files are required")
    index, roles = {}, {}
    for row in rows:
        file = Path(row["file"])
        if not file.is_absolute(): file = path.parent / file
        file = checked(file.resolve(), row["sha256"])
        with np.load(file, allow_pickle=False) as z:
            roles[row["fit_id"]] = {k:z[k].copy() for k in ENDPOINT}
        index[row["fit_id"]] = dict(row, file=str(file))
    validate_roles(roles)
    if any(np.any(role["time"]<=0) for role in roles.values()):
        raise ValueError("Positive follow-up times required")
    if any(role["patient_ids"].dtype.kind!="U" or not all(role["patient_ids"]) for role in roles.values()):
        raise ValueError("Nonempty Unicode patient IDs required")
    return index, roles, sha(path)

def load_fit(row):
    with np.load(checked(row["file"], row["sha256"]), allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    validate_features(data,require_graph=tuple(ARMS)!=("F_concat",))
    return data
