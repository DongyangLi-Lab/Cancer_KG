"""FP64 Cox numerical routines, exposed for synthetic equivalence verification."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ALPHAS = (.001, .01, .1, 1.)
SOLVES = 4 * (9 * len(ALPHAS) + 3)
SOLVER_OPTIONS = dict(maxiter=500, maxls=50, ftol=1e-15, gtol=1e-9)


def risk_loss_gradient(risk, time, event):
    risk, time, event = np.asarray(risk), np.asarray(time), np.asarray(event)
    if (risk.ndim != 1 or time.shape != risk.shape or event.shape != risk.shape
            or not np.isfinite(risk).all() or not np.isfinite(time).all()
            or not np.isin(event, (False, True)).all() or not event.any()):
        raise ValueError("Aligned finite Cox vectors and at least one event are required")
    event = event.astype(bool)
    failures = np.flatnonzero(event)
    in_set = time[None, :] >= time[failures, None]
    log_weights = np.where(in_set, risk[None, :], -np.inf)
    log_denominator = logsumexp(log_weights, axis=1)
    loss = float((log_denominator - risk[failures]).mean())
    derivative = np.exp(log_weights - log_denominator[:, None]).sum(0) / len(failures)
    derivative[failures] -= 1 / len(failures)
    return loss, derivative


def objective(beta, features, time, event, alpha):
    features = np.asarray(features, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if alpha <= 0 or not np.isfinite(alpha) or features.ndim != 2 or features.shape[1] != len(beta):
        raise ValueError("A positive finite regularizer and aligned feature matrix are required")
    loss, derivative = risk_loss_gradient(features @ beta, time, event)
    return loss + alpha / 2 * float(beta @ beta), features.T @ derivative + alpha * beta


def eigensystem(all_train, fit):
    all_train = np.asarray(all_train, dtype=np.float64)
    fit = np.asarray(fit)
    if (all_train.ndim != 2 or all_train.shape[1] != len(fit) or not np.isfinite(all_train).all()
            or fit.ndim != 1 or not len(fit) or fit.dtype.kind not in "iu"
            or len(np.unique(fit)) != len(fit) or np.any(fit < 0) or np.any(fit >= len(all_train))):
        raise ValueError("Aligned finite rectangular Gram and unique fit indices are required")
    train = all_train[fit]
    if not np.allclose(train, train.T, rtol=0, atol=1e-12 * max(1., float(np.max(np.abs(train))))):
        raise ValueError("Fit Gram is asymmetric")
    eigenvalues, vectors = np.linalg.eigh((train + train.T) / 2)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    if minimum < -1e-9 * max(1., maximum):
        raise ValueError("Fit Gram fails the frozen PSD tolerance")
    threshold = max(1e-12, 1e-12 * maximum)
    retained = eigenvalues > threshold
    if not retained.any():
        raise ValueError("The global fit Gram has rank zero")
    values, basis = eigenvalues[retained], vectors[:, retained]
    projection = basis / np.sqrt(values)[None, :]
    features = all_train @ projection
    features[fit] = basis * np.sqrt(values)[None, :]
    return dict(features=features, projection=projection, eigenvalues=values,
        rank=int(retained.sum()), dropped=int((~retained).sum()),
        minimum_eigenvalue=minimum, maximum_eigenvalue=maximum, threshold=threshold)


def solve(features, time, event, alpha):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError("A positive-rank feature space is required")
    result = minimize(objective, np.zeros(features.shape[1], dtype=np.float64),
                      args=(features, time, event, alpha), jac=True, method="L-BFGS-B", options=SOLVER_OPTIONS)
    loss, derivative = objective(result.x, features, time, event, alpha)
    gradient_maximum = float(np.max(np.abs(derivative)))
    if not result.success or not np.isfinite(loss) or gradient_maximum > 1e-6:
        raise ValueError("Cox solver convergence gate failed")
    return dict(beta=result.x, objective=loss, gradient_maximum=gradient_maximum,
                iterations=int(result.nit), evaluations=int(result.nfev), message=str(result.message))


def select_alpha(validation):
    """Validation entries contain held-only unpenalized losses and event counts."""
    if set(validation) != set(ALPHAS):
        raise ValueError("The fixed four alpha candidates are required")
    scores = {}
    for alpha, rows in validation.items():
        losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
        events = np.asarray([row["events"] for row in rows], dtype=np.int64)
        if len(rows) != 3 or not np.isfinite(losses).all() or np.any(events <= 0):
            raise ValueError("Three finite held-only validation losses and positive event counts are required")
        scores[alpha] = float(np.sum(losses * events) / events.sum())
    selected = min(scores, key=lambda alpha: (scores[alpha], -alpha))
    return selected, scores
