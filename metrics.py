"""Selected real-additive OOF scoring; original counts and bootstrap draw sequence."""
import numpy as np

ARMS = ("real_additive_selfquadratic",)
TIED_TOL = 1e-8
BOOTSTRAP_SEED = 20261022
EXPECTED_COHORT = (133, 106, 2345)
HISTORICAL_CONTEXT = (
    dict(name="scFoundation_small_MLP_seed42", credit=1502, pairs=2345, c=1502 / 2345),
    dict(name="strong_RNA_repeat0", credit=1431, pairs=2345, c=1431 / 2345),
)


def scalar_counts(time, event, risk):
    credit, pairs = 0., 0
    for i, first in enumerate(time):
        if not event[i]:
            continue
        for j, second in enumerate(time):
            if first < second or (first == second and not event[j]):
                pairs += 1
                difference = float(risk[i]) - float(risk[j])
                credit += 1. if difference > TIED_TOL else .5 if abs(difference) <= TIED_TOL else 0.
    return credit, pairs


def _inputs(case_ids, time, event, fold, risks):
    ids, time, event, fold = map(np.asarray, (case_ids, time, event, fold))
    if (ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids)
            or len(set(ids)) != len(ids) or any(not value for value in ids)
            or time.shape != ids.shape or event.shape != ids.shape or fold.shape != ids.shape
            or not np.isfinite(time).all() or np.any(time <= 0)
            or not np.isin(event, (0, 1)).all() or fold.dtype.kind not in "iu"
            or set(fold.tolist()) != {0, 1, 2} or set(risks) != set(ARMS)):
        raise ValueError("Unique aligned identities, valid endpoints, three folds and the selected real-additive arm required")
    values = {}
    for arm in ARMS:
        value = np.asarray(risks[arm], dtype=np.float64)
        if value.shape != ids.shape or not np.isfinite(value).all():
            raise ValueError("Malformed aligned OOF risk")
        values[arm] = value
    # Canonical identity order makes finite Monte Carlo draws invariant to TSV ordering.
    order = np.argsort(ids, kind="stable")
    return ids[order], time[order], event[order].astype(bool), fold[order], {a: values[a][order] for a in ARMS}


def _paired_samples(case_ids, time, event, fold, risks, replicates=2000,
                    expected_cohort=EXPECTED_COHORT):
    ids, time, event, fold, risks = _inputs(case_ids, time, event, fold, risks)
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("A positive integer bootstrap count is required")
    expected_n, expected_events, expected_pairs = expected_cohort
    if len(ids) != expected_n or int(event.sum()) != expected_events:
        raise ValueError("Frozen cohort patient/event counts differ")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    totals = np.zeros(len(ARMS), dtype=np.float64)
    boot_credit = np.zeros((replicates, len(ARMS)), dtype=np.float64)
    boot_pairs = np.zeros(replicates, dtype=np.float64)
    total_pairs = 0
    folds = {arm: [] for arm in ARMS}
    for outer in range(3):
        ix = np.flatnonzero(fold == outer)
        t, e = time[ix], event[ix]
        comparable = e[:, None] & ((t[:, None] < t[None, :])
                                   | ((t[:, None] == t[None, :]) & ~e[None, :]))
        pairs = int(comparable.sum())
        if pairs == 0:
            raise ValueError("Outer fold has no comparable pairs")
        total_pairs += pairs
        n = len(ix)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
        boot_pairs += np.einsum("bi,ij,bj->b", weights, comparable, weights, optimize=True)
        for column, arm in enumerate(ARMS):
            r = risks[arm][ix]
            delta = r[:, None] - r[None, :]
            pair_credit = ((delta > TIED_TOL) + .5 * (np.abs(delta) <= TIED_TOL)) * comparable
            credit = float(pair_credit.sum())
            if (credit, pairs) != scalar_counts(t, e, r):
                raise ValueError("Independent scalar/vector concordance mismatch")
            totals[column] += credit
            boot_credit[:, column] += np.einsum("bi,ij,bj->b", weights, pair_credit, weights, optimize=True)
            folds[arm].append(dict(fold=outer, patients=n, events=int(e.sum()),
                                   credit=credit, pairs=pairs, c=credit / pairs))
    if total_pairs != expected_pairs or not np.all(boot_pairs > 0):
        raise ValueError("Frozen comparable-pair count differs or a bootstrap replicate has no pairs")
    return dict(totals=totals, pairs=total_pairs, boot_credit=boot_credit,
                boot_pairs=boot_pairs, folds=folds)


def score_oof(case_ids, time, event, fold, risks, replicates=2000,
              expected_cohort=EXPECTED_COHORT):
    """Return aggregates only; a future clinical driver must use frozen defaults."""
    samples = _paired_samples(case_ids, time, event, fold, risks, replicates, expected_cohort)
    totals, pairs = samples["totals"], samples["pairs"]
    draws = samples["boot_credit"] / samples["boot_pairs"][:, None]
    arms = [dict(arm=arm, credit=float(totals[j]), pairs=pairs, c=float(totals[j] / pairs),
                 ci95=np.quantile(draws[:, j], [.025, .975]).tolist(), folds=samples["folds"][arm])
            for j, arm in enumerate(ARMS)]
    return dict(patients=int(expected_cohort[0]), events=int(expected_cohort[1]),
        within_fold_pairs=pairs, arms=arms,
        bootstrap=dict(replicates=replicates, seed=BOOTSTRAP_SEED, paired=False,
            unit="patient resampled within each fixed outer fold; original draw sequence retained",
            canonical_order="ascending patient identity within outer fold",
            intervals="percentile 95%; single selected model, no contrast family",
            scope="conditional on fitted models, frozen folds and selection; no method-search correction"),
        historical_context=[dict(row) for row in HISTORICAL_CONTEXT],
        historical_scope="descriptive only; not added to bootstrap, ranking gates or model selection",
        cohort_scope="repeatedly explored development cohort; not independent external confirmation")
