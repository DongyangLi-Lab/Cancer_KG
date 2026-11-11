"""Generated data only, full twelve-role/six-candidate control-flow smoke."""
from pathlib import Path
from copy import deepcopy
import tempfile
import numpy as np
import torch
import cancer_kg.models.models as models
import cancer_kg.cli.train as train


def fixture():
    rng = np.random.default_rng(42)
    n, width = 18, 768 if train.MODE == "scfoundation" else 256
    raw = rng.normal(size=(n, width))
    fits = {}
    for fold in range(3):
        outer_held = np.arange(6 * fold, 6 * fold + 6)
        outer_fit = np.setdiff1d(np.arange(n), outer_held)
        for role in ("inner0", "inner1", "inner2", "outer"):
            held = outer_held if role == "outer" else np.array_split(outer_fit, 3)[int(role[-1])]
            fit = outer_fit if role == "outer" else np.setdiff1d(outer_fit, held)
            embedding, stats = models.standardize(raw, fit)
            fits[f"r0_f{fold}_{role}"] = dict(patient_ids=np.array([f"synthetic{i}" for i in range(n)]),
                time=np.arange(n, dtype=float) + 1, event=np.ones(n, dtype=bool), fit=fit, held=held,
                embedding=embedding if train.MODE == "scfoundation" else raw,
                embedding_mean=stats["mean"], embedding_sd=stats["sd"])
    return fits


if __name__ == "__main__":
    from tests.smoke_contracts import run_checks
    run_checks(train.validate_roles)
    torch.set_num_threads(1)
    fits = fixture()
    if train.MODE == "omicsfm":
        from cancer_kg.cli.encode import native_outputs
        from cancer_kg.encoders.encoder import batch, pooled
        class SyntheticProtein(torch.nn.Module):
            def forward(self, inputs):
                values = torch.arange(3 * 256, dtype=torch.float32).reshape(1, 3, 256)
                values[:, 0] = 9999.
                return dict(transformer_out=values)
        native_batch = batch([3, 7], [1, 10])
        native = native_outputs(SyntheticProtein(), native_batch)
        np.testing.assert_array_equal(native["pooled"], pooled(SyntheticProtein(), native_batch).numpy()[0])
        np.testing.assert_array_equal(native["pooled"], native["tokens"].max(0))
        assert native["token_ids"].tolist() == [3, 7]
        assert not native["token_special_mask"].any()
    else:
        from core import egsp_bulk
        result = egsp_bulk(np.array([2., 3.]), np.ones(2, dtype=bool), "T")
        assert result[-1] == result[-2]
        np.testing.assert_allclose(result[-1], np.log10(5. + 1e-8), atol=1e-7)
    for field, value in (("time", 0.), ("event", 2), ("patient_ids", "")):
        bad = deepcopy(fits)
        if field == "event":
            for data in bad.values():
                data[field] = data[field].astype(float)
        for data in bad.values():
            data[field][0] = value
        try:
            train.validate_roles(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed endpoint accepted")
    bad = deepcopy(fits)
    bad["r0_f0_inner0"]["fit"][0] = bad["r0_f0_outer"]["held"][0]
    try:
        train.validate_roles(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Outer-held leakage accepted")
    x, _ = train.matrix(fits["r0_f0_outer"])
    assert x.shape == (18, 1792)
    if train.MODE == "scfoundation":
        assert np.array_equal(x[:, :1024], np.zeros((18, 1024)))
    else:
        assert np.array_equal(x[:, 256:], np.zeros((18, 1536)))
    with tempfile.TemporaryDirectory() as directory:
        train.run(fits, {name: "synthetic" for name in fits}, Path(directory) / "run", synthetic_epochs=2)
        assert len((Path(directory) / "run" / "private_oof.tsv").read_text().splitlines()) == 19
    print("PASS synthetic nested smallMLP smoke, checkpoint replay and slot layout")
