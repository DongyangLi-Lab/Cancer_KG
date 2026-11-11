import sys
from pathlib import Path
"""Pure synthetic numerical QA. This file reads no patient assets or outcomes."""
from copy import deepcopy
import unittest

import numpy as np

from cancer_kg.training.cox import ALPHAS, SOLVES, eigensystem, objective, risk_loss_gradient, select_alpha, solve
from cancer_kg.models.kernels import (ARMS, EPS, MODALITIES, PAIRS, TOPOLOGIES, build_kernels,
                     center_normalize, gene_groups, gram, inventories, observed_base, pool_path)


def fixture():
    rng = np.random.default_rng(42)
    n = 16
    data = dict(fit=np.arange(10), node_type=np.array([0] * 8 + [2] * 4 + [1, 1]),
        rna_node=np.array([0, 1, 2, 3, 4, 5, 12]),
        protein_node=np.array([0, 2, 4, 5, 6, 7]),
        ptm_node=np.array([0, 0, 1, 3, 5, 6, 13]),
        cnv_node=np.array([0, 2, 3, 4, 6, 7]))
    for m, modality in enumerate(MODALITIES):
        features = len(data[modality + "_node"])
        data[modality + "_mask"] = rng.random((n, features)) > .2
        shape = (n, features, 3 if modality == "rna" else 2) if modality in ("rna", "protein") else (n, features)
        data[modality] = rng.normal(size=shape) + (m + 1) * 3
        if modality in ("rna", "protein"):
            data[modality + "_pool"] = rng.normal(size=(n, shape[-1])) + (m + 1) * 5
    data["rna_mask"][0, :6] = False
    data["protein_mask"][1] = False
    data["ptm_mask"][2, :6] = False
    data["cnv_mask"][3] = False
    members = [(gene, 8 + p) for p, genes in enumerate(((0, 1, 2, 3), (2, 3, 4, 5),
                                                         (4, 5, 6, 7), (0, 1, 6, 7))) for gene in genes]
    data["real_member"] = np.asarray(members)
    data["shuffled_member"] = np.asarray([((gene + 1) % 8, path) for gene, path in members])
    return data


def normalized_features(features, fit):
    centered = features - features[fit].mean(0)
    variance = np.square(centered[fit]).sum() / len(fit)
    return centered / np.sqrt(variance) if variance > EPS else np.zeros_like(centered)


def explicit_features(data, computed):
    """Independent explicit outer-product feature construction for the tiny fixture."""
    fit = data["fit"]
    support = computed["inventories"]
    groups = gene_groups(data)
    raw = []
    for modality in MODALITIES:
        values = data[modality + "_pool"] if modality in ("rna", "protein") else np.concatenate(
            (data[modality], data[modality + "_mask"].astype(float)), axis=1)
        raw.append(normalized_features(values, fit) / 2)
    raw = np.concatenate(raw, axis=1)
    reference = {}
    for topology in TOPOLOGIES:
        blocks = dict(raw=[raw], linear=[], self=[], cross=[], coverage=[])
        for p, path in enumerate(support["paths"]):
            vectors = {}
            for m, modality in enumerate(MODALITIES):
                if not support["modal"][p, m]:
                    continue
                pooled, observed, coverage = pool_path(data, modality, support["members"][topology][int(path)], groups[modality])
                observed_fit = fit[observed[fit]]
                mean = pooled[observed_fit].mean(0)
                vector = np.zeros_like(pooled)
                vector[observed] = pooled[observed] - mean
                variance = np.sum(vector[fit] ** 2) / len(fit)
                if variance > EPS:
                    vector /= np.sqrt(variance)
                else:
                    vector[:] = 0
                vectors[modality] = vector
                denominator = 4 * support["modal"][:, m].sum()
                blocks["linear"].append(vector / np.sqrt(denominator))
                outer = np.einsum("ni,nj->nij", vector, vector).reshape(len(raw), -1)
                blocks["self"].append(normalized_features(outer, fit) / np.sqrt(denominator))
                for exponent in (1, 2):
                    blocks["coverage"].append(normalized_features((coverage ** exponent)[:, None], fit)
                                               / np.sqrt(2 * denominator))
            for pair in PAIRS:
                if support["pair"][pair][p]:
                    outer = np.einsum("ni,nj->nij", vectors[pair[0]], vectors[pair[1]]).reshape(len(raw), -1)
                    blocks["cross"].append(normalized_features(outer, fit)
                                            / np.sqrt(6 * support["pair"][pair].sum()))
        reference[topology] = {key: np.concatenate(values, axis=1) if values else np.zeros((len(raw), 0))
                               for key, values in blocks.items()}
    return reference


class TestCompute(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.result = build_kernels(self.data, min_genes=1, min_patients=2, min_paths=1)

    def test_observed_only_mean_scale_remask_and_constant(self):
        values = np.array([[8., 2.], [12., 6.], [100., -100.], [99., 99.]])
        observed = np.array([True, True, False, True])
        q, state = observed_base(values, observed, np.array([0, 1, 2]))
        np.testing.assert_array_equal(state["mean"], [10., 4.])
        self.assertEqual(state["scale_squared"], 16 / 3)
        np.testing.assert_array_equal(q[2], [0., 0.])
        self.assertAlmostEqual(np.square(q[:3]).sum() / 3, 1.)
        values[:2] = 4
        constant, state = observed_base(values, observed, np.array([0, 1, 2]))
        self.assertFalse(state["nondegenerate"])
        np.testing.assert_array_equal(constant, np.zeros_like(values))
        empty, state = observed_base(values, np.zeros(4, dtype=bool), np.array([0, 1, 2]))
        np.testing.assert_array_equal(empty, np.zeros_like(values))
        self.assertFalse(state["nondegenerate"])

    def test_gene_pool_observed_site_mean_and_path_observed_gene_mean(self):
        data = dict(ptm=np.array([[2., 8., 20.], [9., 30., 40.], [4., 6., 10.]]),
                    ptm_mask=np.array([[1, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool))
        pooled, observed, coverage = pool_path(data, "ptm", [0, 1], {0: np.array([0, 1]), 1: np.array([2])})
        np.testing.assert_array_equal(pooled[:, 0], [11., 40., 0.])
        np.testing.assert_array_equal(observed, [True, True, False])
        np.testing.assert_array_equal(coverage, [1., .5, 0.])

    def test_explicit_implicit_all_components_rectangular_and_psd(self):
        explicit = explicit_features(self.data, self.result)
        fit = self.data["fit"]
        for topology in TOPOLOGIES:
            for component, features in explicit[topology].items():
                np.testing.assert_allclose(self.result["components"][topology][component],
                                           features @ features[fit].T, atol=2e-12, rtol=2e-12)
            common_features = np.concatenate([explicit[topology][key] for key in ("raw", "linear", "self", "coverage")], axis=1)
            on_features = np.concatenate((common_features, explicit[topology]["cross"]), axis=1)
            for arm, features in ((topology + "_additive_selfquadratic", common_features),
                                  (topology + "_additive_selfquadratic_cross", on_features)):
                np.testing.assert_allclose(self.result["arms"][arm], features @ features[fit].T, atol=3e-12, rtol=3e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(self.result["arms"][arm][fit]).min(), -1e-10)
        self.assertEqual(set(self.result["arms"]), set(ARMS))
        self.assertIs(self.result["components"]["real"]["raw"], self.result["components"]["shuffled"]["raw"])
        for topology in TOPOLOGIES:
            self.assertIs(self.result["arms"][topology + "_additive_selfquadratic"], self.result["common"][topology])
            np.testing.assert_array_equal(self.result["arms"][topology + "_additive_selfquadratic_cross"],
                                          self.result["common"][topology] + self.result["components"][topology]["cross"])
        self.assertEqual(set(self.result["geometry_diagnostics"]),
                         {"raw", "path_linear", "path_self", "path_cross", "coverage", "common"})
        self.assertTrue(self.result["geometry_diagnostics"]["raw"]["bitwise_equal"])

    def test_nonzero_product_mean_requires_its_own_centering(self):
        x = np.array([[-2.], [-1.], [1.], [2.], [9.]])
        fit = np.arange(4)
        base = gram(x, fit)
        raw_product = base * base
        self.assertGreater(abs(raw_product[fit].mean()), 1.)
        actual, _ = center_normalize(raw_product, fit)
        expected = normalized_features(x * x, fit)
        np.testing.assert_allclose(actual, expected @ expected[fit].T, atol=1e-12)
        np.testing.assert_allclose(actual[fit].mean(0), 0., atol=1e-12)

    def test_no_products_between_different_pathways(self):
        x = np.array([[-1.], [0.], [1.]])
        y = np.array([[1.], [-1.], [0.]])
        a, b = x @ x.T, y @ y.T
        correct = a * a + b * b
        incorrect = (a + b) * (a + b)
        self.assertGreater(np.linalg.norm(correct - incorrect), 1.)
        explicit = np.concatenate((x * x, y * y), axis=1)
        np.testing.assert_array_equal(correct, explicit @ explicit.T)

    def test_held_perturbation_preserves_train_transforms_and_kernels(self):
        changed = deepcopy(self.data)
        for modality in MODALITIES:
            changed[modality][10:] += 1000
            changed[modality + "_mask"][10:] = ~changed[modality + "_mask"][10:]
            if modality in ("rna", "protein"):
                changed[modality + "_pool"][10:] -= 500
        altered = build_kernels(changed, min_genes=1, min_patients=2, min_paths=1)
        for arm in ARMS:
            np.testing.assert_array_equal(self.result["arms"][arm][:10], altered["arms"][arm][:10])
            original_eigen = eigensystem(self.result["arms"][arm], self.data["fit"])
            changed_eigen = eigensystem(altered["arms"][arm], changed["fit"])
            np.testing.assert_array_equal(original_eigen["projection"], changed_eigen["projection"])
        self.assertEqual(self.result["raw_states"], altered["raw_states"])
        self.assertEqual(self.result["block_scales"], altered["block_scales"])

    def test_empty_pair_family_and_zero_blocks_keep_slots(self):
        data = deepcopy(self.data)
        data["cnv_node"][:] = 13
        result = build_kernels(data, min_genes=1, min_patients=2, min_paths=1)
        self.assertFalse(result["support_gate"]["passed"])
        for pair in PAIRS:
            if "cnv" in pair:
                self.assertEqual(result["inventories"]["pair"][pair].sum(), 0)
        constant = deepcopy(self.data)
        for modality in MODALITIES:
            constant[modality][:] = 7
        result = build_kernels(constant, min_genes=1, min_patients=2, min_paths=1)
        self.assertFalse(result["effective_gate"]["passed"])
        np.testing.assert_array_equal(result["inventories"]["modal"], self.result["inventories"]["modal"])
        for pair in PAIRS:
            np.testing.assert_array_equal(result["inventories"]["pair"][pair], self.result["inventories"]["pair"][pair])
        for topology in TOPOLOGIES:
            np.testing.assert_allclose(result["components"][topology]["cross"], 0., atol=0.)
        # Denominator remains two slots when a normalized block degenerates.
        first, _ = center_normalize(gram(np.arange(16.)[:, None], data["fit"]), data["fit"])
        zero, _ = center_normalize(np.ones_like(first), data["fit"])
        self.assertAlmostEqual(np.trace(((first + zero) / 2)[data["fit"]]) / 10, .5)

    def test_effective_gate_is_same_path_intersection(self):
        from cancer_kg.models.kernels import pair_partner_gate
        real = np.array([True] * 10 + [False] * 10)
        shuffled = ~real
        counts = {pair: int((real & shuffled).sum()) for pair in PAIRS}
        self.assertFalse(pair_partner_gate(counts, 10)["passed"])

    def test_fit_only_eigen_rectangular_projection_and_rank_zero_failure(self):
        fit = np.array([0, 2, 4])
        values = np.array([[1., 0.], [9., 7.], [0., 1.], [5., 3.], [-1., -1.]])
        result = eigensystem(gram(values, fit), fit)
        np.testing.assert_allclose(result["features"] @ result["features"][fit].T, gram(values, fit), atol=2e-13)
        self.assertEqual(result["rank"], 2)
        changed = values.copy()
        changed[[1, 3]] *= 100
        altered = eigensystem(gram(changed, fit), fit)
        np.testing.assert_array_equal(result["eigenvalues"], altered["eigenvalues"])
        np.testing.assert_array_equal(result["projection"], altered["projection"])
        with self.assertRaisesRegex(ValueError, "rank zero"):
            eigensystem(np.zeros((5, 3)), fit)
        with self.assertRaisesRegex(ValueError, "PSD"):
            eigensystem(np.diag([1., -1.]), np.arange(2))

    def test_cox_breslow_ties_gradient_and_extreme_risk(self):
        import torch
        time = np.array([1., 1., 2., 3., 4.])
        event = np.array([True, True, False, True, False])
        risk = np.array([2., -1., .5, 4., -2.])
        value, derivative = risk_loss_gradient(risk, time, event)
        r = torch.tensor(risk, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(time)
        failures = np.flatnonzero(event)
        reference = torch.stack([torch.logsumexp(r[t >= t[index]], 0) - r[index] for index in failures]).mean()
        reference.backward()
        self.assertAlmostEqual(value, float(reference), places=13)
        np.testing.assert_allclose(derivative, r.grad.numpy(), atol=2e-15)
        shifted_value, shifted_gradient = risk_loss_gradient(risk + 1000, time, event)
        self.assertAlmostEqual(value, shifted_value, places=12)
        np.testing.assert_allclose(derivative, shifted_gradient, atol=1e-13)
        features = np.arange(15.).reshape(5, 3) / 10
        beta = np.array([.1, -.2, .3])
        loss, gradient = objective(beta, features, time, event, .1)
        finite_difference = []
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = 1e-5
            finite_difference.append((objective(beta + delta, features, time, event, .1)[0]
                                      - objective(beta - delta, features, time, event, .1)[0]) / 2e-5)
        np.testing.assert_allclose(gradient, finite_difference, atol=3e-11, rtol=3e-10)

    def test_explicit_implicit_cox_gradients_solutions_and_predictions(self):
        fit = self.data["fit"]
        explicit = explicit_features(self.data, self.result)
        time = np.array([1., 1., 2., 3., 4., 4., 5., 6., 7., 8.])
        event = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
        rng = np.random.default_rng(42)
        for arm in ARMS:
            topology = "shuffled" if arm.startswith("shuffled") else "real"
            keys = ["raw", "linear", "self", "coverage"] + (["cross"] if arm.endswith("_cross") else [])
            x = np.concatenate([explicit[topology][key] for key in keys], axis=1)
            space = eigensystem(self.result["arms"][arm], fit)
            transform = x[fit].T @ space["projection"]
            beta = rng.normal(size=space["rank"])
            weight = transform @ beta
            explicit_loss, explicit_gradient = objective(weight, x[fit], time, event, .1)
            implicit_loss, implicit_gradient = objective(beta, space["features"][fit], time, event, .1)
            self.assertAlmostEqual(explicit_loss, implicit_loss, places=11)
            np.testing.assert_allclose(transform.T @ explicit_gradient, implicit_gradient, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(x @ weight, space["features"] @ beta, atol=2e-12, rtol=2e-12)
            a = solve(x[fit], time, event, .1)
            b = solve(space["features"][fit], time, event, .1)
            self.assertAlmostEqual(a["objective"], b["objective"], places=10)
            np.testing.assert_allclose(x @ a["beta"], space["features"] @ b["beta"], atol=3e-6, rtol=3e-6)

    def test_inner_event_weighted_loss_tie_and_solve_budget(self):
        self.assertEqual(SOLVES, 156)
        rows = [{"loss": 1., "events": 1}, {"loss": 2., "events": 2}, {"loss": 3., "events": 3}]
        selected, scores = select_alpha({alpha: rows for alpha in ALPHAS})
        self.assertEqual(selected, 1.)
        self.assertAlmostEqual(scores[1.], 14 / 6)
        risk = np.zeros(3)
        time = np.array([1., 2., 3.])
        held_loss, _ = risk_loss_gradient(risk, time, np.array([True, False, True]))
        self.assertAlmostEqual(held_loss, np.log(3) / 2)


if __name__ == "__main__":
    unittest.main()
