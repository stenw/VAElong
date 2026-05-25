"""
Unit tests for the joint longitudinal-survival model.
"""

import unittest

import torch

from vaelong.data import generate_synthetic_joint_longitudinal_survival_data
from vaelong.joint_model import JointLongitudinalSurvivalVAE


class TestJointLongitudinalSurvivalVAE(unittest.TestCase):
    """Test cases for the additive joint model."""

    def setUp(self):
        generated = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=6,
            seq_len=8,
            n_features=2,
            n_baseline_features=2,
            n_time_varying_covariates=1,
            n_event_covariates=2,
            seed=42,
        )
        self.x = torch.from_numpy(generated["data"])
        self.baseline = torch.from_numpy(generated["baseline_covariates"])
        self.times = torch.from_numpy(generated["times"])
        self.time_varying_covariates = torch.from_numpy(
            generated["time_varying_covariates"]
        )
        self.event_times = torch.from_numpy(generated["event_times"])
        self.event_indicators = torch.from_numpy(generated["event_indicators"])
        self.event_covariates = torch.from_numpy(generated["event_covariates"])

        self.model = JointLongitudinalSurvivalVAE(
            input_dim=2,
            hidden_dim=16,
            latent_dim=5,
            encoder_type="dense",
            seq_len=8,
            n_baseline=2,
            n_event_covariates=2,
            n_time_varying_covariates=1,
            time_in_encoder=True,
            time_in_decoder=True,
        )

    def test_forward_with_survival_terms(self):
        recon_x, mu, posterior_params, survival_outputs = self.model(
            self.x,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            event_covariates=self.event_covariates,
            return_survival_terms=True,
        )

        self.assertEqual(recon_x.shape, self.x.shape)
        self.assertEqual(mu.shape, (self.x.shape[0], self.model.latent_dim))
        self.assertEqual(
            posterior_params.shape,
            (self.x.shape[0], self.model.latent_dim),
        )
        self.assertIsNotNone(survival_outputs)
        self.assertEqual(
            survival_outputs["log_hazard_event"].shape,
            (self.x.shape[0],),
        )
        self.assertEqual(
            survival_outputs["cumulative_hazard"].shape,
            (self.x.shape[0],),
        )
        self.assertTrue(torch.isfinite(survival_outputs["survival_log_likelihood"]).all())

    def test_encoder_depends_on_survival_context(self):
        mu_a, _ = self.model.encode(
            self.x,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            event_covariates=self.event_covariates,
        )
        mu_b, _ = self.model.encode(
            self.x,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_time=self.event_times + 2.0,
            event_indicator=self.event_indicators,
            event_covariates=self.event_covariates,
        )

        self.assertFalse(torch.allclose(mu_a, mu_b))

    def test_survival_log_likelihood_handles_censoring(self):
        z = torch.randn(self.x.shape[0], self.model.latent_dim)
        common_event_time = torch.full((self.x.shape[0],), 1.5)
        terms_event = self.model.survival_terms(
            z,
            event_time=common_event_time,
            event_indicator=torch.ones_like(common_event_time),
            event_covariates=self.event_covariates,
            baseline=self.baseline,
        )
        terms_censor = self.model.survival_terms(
            z,
            event_time=common_event_time,
            event_indicator=torch.zeros_like(common_event_time),
            event_covariates=self.event_covariates,
            baseline=self.baseline,
        )

        self.assertTrue(torch.isfinite(terms_event["log_hazard_event"]).all())
        self.assertTrue(torch.isfinite(terms_event["cumulative_hazard"]).all())
        self.assertTrue(torch.isfinite(terms_event["survival_log_likelihood"]).all())
        self.assertTrue(torch.isfinite(terms_censor["survival_log_likelihood"]).all())
        diff = (
            terms_event["survival_log_likelihood"]
            - terms_censor["survival_log_likelihood"]
        )
        self.assertTrue(
            torch.allclose(diff, terms_event["log_hazard_event"], atol=1e-5)
        )

    def test_predict_survival_curve_and_interval_probabilities(self):
        z = torch.randn(self.x.shape[0], self.model.latent_dim)
        prediction_times = torch.linspace(0.25, 2.0, steps=5)

        curve = self.model.predict_survival_curve(
            z,
            prediction_times,
            event_covariates=self.event_covariates,
            baseline=self.baseline,
        )
        event_prob = self.model.predict_event_probability(
            z,
            prediction_times[:-1],
            prediction_times[1:],
            event_covariates=self.event_covariates,
            baseline=self.baseline,
        )

        self.assertEqual(curve.shape, (self.x.shape[0], 5))
        self.assertEqual(event_prob.shape, (self.x.shape[0], 4))
        self.assertTrue(torch.isfinite(curve).all())
        self.assertTrue(torch.isfinite(event_prob).all())
        self.assertTrue(torch.all(curve >= 0.0))
        self.assertTrue(torch.all(curve <= 1.0))
        self.assertTrue(torch.all(event_prob >= 0.0))
        self.assertTrue(torch.all(event_prob <= 1.0))
        self.assertTrue(torch.all(curve[:, 1:] <= curve[:, :-1] + 1e-6))


if __name__ == "__main__":
    unittest.main()
