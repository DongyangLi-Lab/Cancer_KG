import numpy as np

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

