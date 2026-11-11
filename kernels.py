"""Streaming, fit-only exact within-pathway tensor-product kernels.

This module has no file loader, endpoint access, job submission or clinical CLI.
The caller supplies already verified prepared arrays and fixed fit indices.
"""
import itertools

import numpy as np

MODALITIES = ("rna", "protein", "ptm", "cnv")
TOPOLOGIES = ("real", "shuffled")
PAIRS = tuple(itertools.combinations(MODALITIES, 2))
ARMS = ("real_additive_selfquadratic", "real_additive_selfquadratic_cross",
        "shuffled_additive_selfquadratic", "shuffled_additive_selfquadratic_cross")
EPS = 1e-12


def fit_rows(fit, rows):
    fit = np.asarray(fit)
    if (fit.ndim != 1 or fit.dtype.kind not in "iu" or not len(fit)
            or len(np.unique(fit)) != len(fit) or np.any(fit < 0) or np.any(fit >= rows)):
        raise ValueError("Unique in-range fit indices are required")
    return fit


def gram(features, fit):
    """All-row by training-row Gram, with independently calculated train block."""
    features = np.asarray(features, dtype=np.float64)
    result = features @ features[fit].T
    result[fit] = features[fit] @ features[fit].T
    return result


def center_normalize(block, fit):
    """Fit-reference kernel centering and mean fit diagonal normalization."""
    block = np.asarray(block, dtype=np.float64)
    if block.shape[1] != len(fit) or not np.isfinite(block).all():
        raise ValueError("A finite all-row by fit-row kernel is required")
    train = block[fit]
    column_mean = train.mean(0)
    grand_mean = float(train.mean())
    centered = block - block.mean(1, keepdims=True) - column_mean[None, :] + grand_mean
    scale_squared = float(np.trace(centered[fit]) / len(fit))
    if scale_squared < -EPS:
        raise ValueError("A feature Gram has a negative centered mean diagonal")
    if scale_squared <= EPS:
        return np.zeros_like(block), dict(scale_squared=scale_squared, nondegenerate=False)
    return centered / scale_squared, dict(scale_squared=scale_squared, nondegenerate=True)


def observed_base(values, observed, fit):
    """Observed-only fit mean, remask, scale over all fit rows including zeros."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed)
    if (values.ndim != 2 or observed.shape != (len(values),) or observed.dtype != bool
            or not np.isfinite(values[observed]).all()):
        raise ValueError("Observed path vectors must be finite and mask-aligned")
    observed_fit = observed[fit]
    mean = values[fit][observed_fit].mean(0) if observed_fit.any() else np.zeros(values.shape[1])
    residual = np.where(observed[:, None], values - mean, 0.)
    scale_squared = float(np.square(residual[fit]).sum() / len(fit))
    if scale_squared <= EPS:
        return np.zeros_like(residual), dict(mean=mean, scale_squared=scale_squared, nondegenerate=False)
    return residual / np.sqrt(scale_squared), dict(mean=mean, scale_squared=scale_squared, nondegenerate=True)


def gene_groups(data):
    groups = {}
    types = np.asarray(data["node_type"])
    for modality in MODALITIES:
        nodes = np.asarray(data[modality + "_node"])
        mask = np.asarray(data[modality + "_mask"])
        if (nodes.ndim != 1 or nodes.dtype.kind not in "iu" or mask.dtype != bool
                or mask.ndim != 2 or mask.shape[1] != len(nodes)
                or np.any(nodes < 0) or np.any(nodes >= len(types))):
            raise ValueError("Prepared assay-node/mask axes differ")
        groups[modality] = {int(gene): np.flatnonzero(nodes == gene)
                            for gene in np.unique(nodes) if types[gene] == 0}
    return groups


def inventories(data, groups, fit, min_genes=3, min_patients=10):
    """Freeze pair-specific common support before any molecular values are read."""
    nodes = len(data["node_type"])
    paths = np.flatnonzero(data["node_type"] == 2)
    count_rows = data["rna_mask"].shape[0]
    fit = fit_rows(fit, count_rows)
    gene_observed = {}
    for modality in MODALITIES:
        mask = data[modality + "_mask"]
        if mask.shape[0] != count_rows:
            raise ValueError("Prepared modality patient axes differ")
        observed = np.zeros((len(fit), nodes), dtype=bool)
        for gene, features in groups[modality].items():
            observed[:, gene] = mask[np.ix_(fit, features)].any(1)
        gene_observed[modality] = observed
    members, static, observed = {}, {}, {}
    for topology in TOPOLOGIES:
        edges = np.asarray(data[topology + "_member"])
        if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu"
                or len(np.unique(edges, axis=0)) != len(edges)
                or np.any(edges < 0) or np.any(edges >= nodes)
                or not np.all(data["node_type"][edges[:, 0]] == 0)
                or not np.all(data["node_type"][edges[:, 1]] == 2)):
            raise ValueError("Unique typed direct membership edges are required")
        members[topology] = {int(path): edges[edges[:, 1] == path, 0] for path in paths}
        static[topology] = np.zeros((len(paths), 4), dtype=np.int64)
        observed[topology] = np.zeros((len(fit), len(paths), 4), dtype=bool)
        for p, path in enumerate(paths):
            genes = members[topology][int(path)]
            for m, modality in enumerate(MODALITIES):
                selected = [int(gene) for gene in genes if int(gene) in groups[modality]]
                static[topology][p, m] = len(selected)
                observed[topology][:, p, m] = gene_observed[modality][:, selected].any(1)
    modal = ((static["real"] >= min_genes) & (static["shuffled"] >= min_genes)
             & (observed["real"].sum(0) >= min_patients)
             & (observed["shuffled"].sum(0) >= min_patients))
    pair = {}
    for first, second in PAIRS:
        left, right = MODALITIES.index(first), MODALITIES.index(second)
        eligible = modal[:, left] & modal[:, right]
        for topology in TOPOLOGIES:
            jointly_observed = observed[topology][:, :, left] & observed[topology][:, :, right]
            eligible &= jointly_observed.sum(0) >= min_patients
        pair[first, second] = eligible
    return dict(paths=paths, members=members, modal=modal, pair=pair,
                thresholds=dict(min_genes=min_genes, min_patients=min_patients))


def pool_path(data, modality, path_genes, groups):
    """Observed assay mean within gene, then observed gene mean within pathway."""
    values = data[modality]
    masks = data[modality + "_mask"]
    width = values.shape[2] if values.ndim == 3 else 1
    if values.ndim not in (2, 3) or values.shape[:2] != masks.shape:
        raise ValueError("Prepared native/numerical value dimensions differ")
    numerator = np.zeros((len(values), width), dtype=np.float64)
    observed_genes = np.zeros(len(values), dtype=np.int64)
    selected_genes = [int(gene) for gene in path_genes if int(gene) in groups]
    for gene in selected_genes:
        features = groups[gene]
        gene_sum = np.zeros_like(numerator)
        counts = masks[:, features].sum(1)
        for feature in features:
            value = values[:, feature].reshape(len(values), width).astype(np.float64)
            observed = masks[:, feature]
            if not np.isfinite(value[observed]).all():
                raise ValueError("Observed molecular token is nonfinite")
            gene_sum += np.where(observed[:, None], value, 0.)
        gene_mean = np.divide(gene_sum, counts[:, None], out=np.zeros_like(gene_sum), where=counts[:, None] > 0)
        numerator += gene_mean
        observed_genes += counts > 0
    pooled = np.divide(numerator, observed_genes[:, None], out=np.zeros_like(numerator),
                       where=observed_genes[:, None] > 0)
    coverage = observed_genes / len(selected_genes) if selected_genes else np.zeros(len(values))
    return pooled, observed_genes > 0, coverage


def raw_kernel(data, fit):
    result = np.zeros((len(data["rna_pool"]), len(fit)), dtype=np.float64)
    states = {}
    for modality in MODALITIES:
        features = (np.asarray(data[modality + "_pool"], dtype=np.float64)
                    if modality in ("rna", "protein")
                    else np.concatenate((data[modality], data[modality + "_mask"].astype(np.float64)), axis=1))
        if not np.isfinite(features).all():
            raise ValueError("Raw pooled/prestandardized modality block is nonfinite")
        block, states[modality] = center_normalize(gram(features, fit), fit)
        result += block / 4
    return result, states


def pair_partner_gate(counts, minimum_paths):
    partners = {modality: False for modality in MODALITIES}
    for pair, count in counts.items():
        if count >= minimum_paths:
            for modality in pair:
                partners[modality] = True
    return dict(pair_counts={"__".join(pair): int(value) for pair, value in counts.items()},
                modality_has_supported_partner=partners, passed=all(partners.values()),
                minimum_paths=minimum_paths)


def build_kernels(data, fit=None, min_genes=3, min_patients=10, min_paths=10):
    """Build four fixed arms without labels; one pathway's native vectors live at a time."""
    fit = fit_rows(data["fit"] if fit is None else fit, len(data["rna_mask"]))
    groups = gene_groups(data)
    support = inventories(data, groups, fit, min_genes, min_patients)
    raw, raw_states = raw_kernel(data, fit)
    shape = raw.shape
    components = {topology: dict(raw=raw, linear=np.zeros(shape), self=np.zeros(shape),
                                coverage=np.zeros(shape), cross=np.zeros(shape)) for topology in TOPOLOGIES}
    modal_denominators = support["modal"].sum(0)
    pair_denominators = {pair: int(mask.sum()) for pair, mask in support["pair"].items()}
    block_scales = {topology: [] for topology in TOPOLOGIES}
    effective_counts = {pair: 0 for pair in PAIRS}
    implicit_dimensions = {topology: dict(raw=0, linear=0, self=0, cross=0, coverage=0) for topology in TOPOLOGIES}
    for topology in TOPOLOGIES:
        implicit_dimensions[topology]["raw"] = sum(
            data[modality + "_pool"].shape[1] if modality in ("rna", "protein") else 2 * data[modality].shape[1]
            for modality in MODALITIES)
    for p, path in enumerate(support["paths"]):
        nondegenerate = {topology: {} for topology in TOPOLOGIES}
        for topology in TOPOLOGIES:
            bases = {}
            states = dict(base={}, self={}, cross={}, coverage={})
            for m, modality in enumerate(MODALITIES):
                if not support["modal"][p, m]:
                    continue
                pooled, observed, coverage = pool_path(data, modality, support["members"][topology][int(path)], groups[modality])
                features, base_state = observed_base(pooled, observed, fit)
                bases[modality] = gram(features, fit)
                states["base"][modality] = {key: value for key, value in base_state.items() if key != "mean"}
                denominator = 4 * int(modal_denominators[m])
                components[topology]["linear"] += bases[modality] / denominator
                squared, states["self"][modality] = center_normalize(bases[modality] * bases[modality], fit)
                components[topology]["self"] += squared / denominator
                coverage_states = []
                for exponent in (1, 2):
                    coverage_features = (coverage ** exponent)[:, None]
                    coverage_block, coverage_state = center_normalize(gram(coverage_features, fit), fit)
                    components[topology]["coverage"] += coverage_block / (2 * denominator)
                    coverage_states.append(coverage_state)
                states["coverage"][modality] = coverage_states
                implicit_dimensions[topology]["linear"] += features.shape[1]
                implicit_dimensions[topology]["self"] += features.shape[1] ** 2
                implicit_dimensions[topology]["coverage"] += 2
            for pair in PAIRS:
                if not support["pair"][pair][p]:
                    continue
                block, state = center_normalize(bases[pair[0]] * bases[pair[1]], fit)
                components[topology]["cross"] += block / (6 * pair_denominators[pair])
                states["cross"]["__".join(pair)] = state
                nondegenerate[topology][pair] = state["nondegenerate"]
                dimensions = [data[key].shape[2] if data[key].ndim == 3 else 1 for key in pair]
                implicit_dimensions[topology]["cross"] += dimensions[0] * dimensions[1]
            block_scales[topology].append(states)
        for pair in PAIRS:
            effective_counts[pair] += bool(nondegenerate["real"].get(pair, False)
                                           and nondegenerate["shuffled"].get(pair, False))
    common, arms = {}, {}
    for topology in TOPOLOGIES:
        part = components[topology]
        common[topology] = part["raw"] + part["linear"] + part["self"] + part["coverage"]
        arms[topology + "_additive_selfquadratic"] = common[topology]
        arms[topology + "_additive_selfquadratic_cross"] = common[topology] + part["cross"]
    geometry_diagnostics = {}
    for name in ("raw", "linear", "self", "coverage", "cross", "common"):
        first, second = ((common[topology] if name == "common" else components[topology][name])[fit]
                         for topology in TOPOLOGIES)
        first_norm, second_norm = float(np.linalg.norm(first)), float(np.linalg.norm(second))
        difference = float(np.linalg.norm(first - second))
        label = "path_" + name if name in ("linear", "self", "cross") else name
        geometry_diagnostics[label] = dict(frobenius_difference=difference,
            relative_frobenius_to_real=difference / first_norm if first_norm else None,
            cosine=float(np.vdot(first, second) / (first_norm * second_norm)) if first_norm and second_norm else None,
            real_zero=first_norm == 0., shuffled_zero=second_norm == 0.,
            bitwise_equal=bool(np.array_equal(first, second)))
    return dict(arms=arms, common=common, components=components, inventories=support,
        support_gate=pair_partner_gate(pair_denominators, min_paths),
        effective_gate=pair_partner_gate(effective_counts, min_paths), raw_states=raw_states,
        block_scales=block_scales, implicit_dimensions=implicit_dimensions,
        geometry_diagnostics=geometry_diagnostics,
        normalization="Per-block fit-only transforms; fixed inventory denominators; no final renormalization")
