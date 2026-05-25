"""
Unit tests for survival utilities.
"""

import unittest

import torch

from vaelong.survival import (
    gauss_kronrod_15,
    rescale_quadrature,
    subject_quadrature_grid,
    compose_log_hazard,
    cumulative_hazard_from_hazard,
    cumulative_hazard_from_log_terms,
    cumulative_hazard_from_log_hazard,
    survival_log_likelihood,
    log_survival_probability,
    survival_probability,
)


class TestSurvivalUtilities(unittest.TestCase):
    """Tests for quadrature and survival-likelihood helpers."""

    def test_gauss_kronrod_weights_sum_to_two(self):
        """The reference-rule weights on [-1, 1] should sum to 2."""
        _, weights = gauss_kronrod_15()
        self.assertAlmostEqual(weights.sum().item(), 2.0, places=6)

    def test_rescale_quadrature_weights_sum_to_interval_length(self):
        """Rescaled weights should sum to the width of each interval."""
        nodes, weights = gauss_kronrod_15()
        lower = torch.tensor([0.0, 1.0])
        upper = torch.tensor([2.0, 5.0])
        scaled_nodes, scaled_weights = rescale_quadrature(nodes, weights, lower, upper)

        self.assertEqual(scaled_nodes.shape, (2, 15))
        self.assertEqual(scaled_weights.shape, (2, 15))
        self.assertTrue(torch.all(scaled_nodes[0] >= 0.0))
        self.assertTrue(torch.all(scaled_nodes[0] <= 2.0))
        self.assertTrue(torch.all(scaled_nodes[1] >= 1.0))
        self.assertTrue(torch.all(scaled_nodes[1] <= 5.0))
        self.assertTrue(torch.allclose(scaled_weights.sum(dim=-1), upper - lower, atol=1e-6))

    def test_subject_quadrature_grid_defaults_to_zero_lower_bound(self):
        """Subject grids should default to integration over [0, T_i]."""
        event_times = torch.tensor([1.5, 3.0])
        scaled_nodes, scaled_weights = subject_quadrature_grid(event_times)

        self.assertEqual(scaled_nodes.shape, (2, 15))
        self.assertEqual(scaled_weights.shape, (2, 15))
        self.assertTrue(torch.allclose(scaled_weights.sum(dim=-1), event_times, atol=1e-6))

    def test_cumulative_hazard_constant_hazard(self):
        """A constant hazard should integrate to lambda * T."""
        event_times = torch.tensor([2.0, 4.5], dtype=torch.float64)
        _, weights = subject_quadrature_grid(event_times, dtype=torch.float64)
        lam = torch.tensor([0.7, 0.7], dtype=torch.float64)
        log_h = torch.log(lam).unsqueeze(-1).expand(-1, weights.shape[-1])

        cumulative = cumulative_hazard_from_log_hazard(log_h, weights)
        expected = lam * event_times
        self.assertTrue(torch.allclose(cumulative, expected, atol=1e-10))

    def test_cumulative_hazard_linear_hazard(self):
        """A linear hazard h(t)=t should integrate to 0.5 * T^2."""
        event_times = torch.tensor([1.5, 3.0], dtype=torch.float64)
        nodes, weights = subject_quadrature_grid(event_times, dtype=torch.float64)
        hazard_values = nodes

        cumulative = cumulative_hazard_from_hazard(hazard_values, weights)
        expected = 0.5 * event_times ** 2
        self.assertTrue(torch.allclose(cumulative, expected, atol=1e-10))

    def test_cumulative_hazard_from_large_log_terms_is_finite(self):
        """The log-sum-exp helper should remain finite for very large inputs."""
        log_terms = torch.full((3, 15), 1000.0)
        cumulative = cumulative_hazard_from_log_terms(log_terms, max_cumulative_hazard=1e9)

        self.assertTrue(torch.isfinite(cumulative).all())
        self.assertTrue(torch.all(cumulative <= 1e9))

    def test_survival_log_likelihood_handles_censoring(self):
        """delta * log h(T) - H(T) should reduce correctly for censoring."""
        log_hazard_event = torch.log(torch.tensor([2.0, 2.0]))
        cumulative_hazard = torch.tensor([3.0, 3.0])
        event_indicator = torch.tensor([1.0, 0.0])

        loglik = survival_log_likelihood(
            log_hazard_event,
            cumulative_hazard,
            event_indicator,
        )

        expected = torch.tensor([torch.log(torch.tensor(2.0)) - 3.0, -3.0])
        self.assertTrue(torch.allclose(loglik, expected))

    def test_survival_probability_matches_negative_cumulative_hazard(self):
        """S(t)=exp(-H(t)) and log S(t)=-H(t)."""
        cumulative_hazard = torch.tensor([0.0, 1.5, 10.0])

        log_surv = log_survival_probability(cumulative_hazard)
        surv = survival_probability(cumulative_hazard)

        self.assertTrue(torch.allclose(log_surv, -cumulative_hazard))
        self.assertTrue(torch.allclose(surv, torch.exp(-cumulative_hazard), atol=1e-6))

    def test_compose_log_hazard_adds_components(self):
        """The composed log hazard should be additive on the log scale."""
        log_baseline = torch.tensor([0.2, -0.4])
        log_rr = torch.tensor([1.0, 0.5])

        combined = compose_log_hazard(log_baseline, log_rr)
        self.assertTrue(torch.allclose(combined, log_baseline + log_rr))


if __name__ == "__main__":
    unittest.main()
