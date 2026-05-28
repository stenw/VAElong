"""
Unit tests for dynamic prediction APIs in the joint model.
"""

import unittest

import torch

from vaelong.data import generate_synthetic_joint_longitudinal_survival_data
from vaelong.joint_model import JointLongitudinalSurvivalVAE


class TestJointPredictionAPIs(unittest.TestCase):
    """Test dynamic landmark predictions for the joint model."""

    def setUp(self):
        generated = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=5,
            seq_len=12,
            n_features=2,
            n_baseline_features=2,
            n_time_varying_covariates=1,
            n_event_covariates=2,
            seed=123,
        )
        self.x = torch.from_numpy(generated["data"])
        self.mask = torch.ones_like(self.x)
        self.baseline = torch.from_numpy(generated["baseline_covariates"])
        self.times = torch.from_numpy(generated["times"])
        self.time_varying_covariates = torch.from_numpy(
            generated["time_varying_covariates"]
        )
        self.event_covariates = torch.from_numpy(generated["event_covariates"])

        self.model = JointLongitudinalSurvivalVAE(
            input_dim=2,
            hidden_dim=16,
            latent_dim=5,
            encoder_type="dense",
            seq_len=12,
            n_baseline=2,
            n_event_covariates=2,
            n_time_varying_covariates=1,
            time_in_encoder=True,
            time_in_decoder=True,
        )

        self.observed_len = 5
        self.x_obs = self.x[:, :self.observed_len]
        self.mask_obs = self.mask[:, :self.observed_len]
        self.landmark_time = self.times[:, self.observed_len - 1]

    def test_predict_longitudinal_from_landmark_with_partial_history(self):
        predicted = self.model.predict_longitudinal_from_landmark(
            self.x_obs,
            self.mask_obs,
            total_seq_len=self.x.shape[1],
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )

        self.assertEqual(predicted.shape, self.x.shape)
        self.assertTrue(torch.isfinite(predicted).all())

    def test_predict_survival_from_landmark_is_monotone(self):
        prediction_times = self.times[:, self.observed_len - 1 :]
        survival = self.model.predict_survival_from_landmark(
            self.x_obs,
            self.mask_obs,
            prediction_times=prediction_times,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )

        self.assertEqual(survival.shape, prediction_times.shape)
        self.assertTrue(torch.isfinite(survival).all())
        self.assertTrue(torch.all(survival >= 0.0))
        self.assertTrue(torch.all(survival <= 1.0))
        self.assertTrue(torch.allclose(survival[:, 0], torch.ones_like(survival[:, 0]), atol=1e-5))
        self.assertTrue(torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-6))

    def test_predict_event_probability_from_landmark_is_bounded(self):
        start_times = self.times[:, self.observed_len - 1 : -1]
        end_times = self.times[:, self.observed_len :]
        event_prob = self.model.predict_event_probability_from_landmark(
            self.x_obs,
            self.mask_obs,
            start_times=start_times,
            end_times=end_times,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )

        self.assertEqual(event_prob.shape, start_times.shape)
        self.assertTrue(torch.isfinite(event_prob).all())
        self.assertTrue(torch.all(event_prob >= 0.0))
        self.assertTrue(torch.all(event_prob <= 1.0))

    def test_predict_hazard_from_landmark_is_nonnegative(self):
        prediction_times = self.times[:, self.observed_len - 1 :]
        hazard = self.model.predict_hazard_from_landmark(
            self.x_obs,
            self.mask_obs,
            prediction_times=prediction_times,
            baseline=self.baseline,
            times=self.times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )

        self.assertEqual(hazard.shape, prediction_times.shape)
        self.assertTrue(torch.isfinite(hazard).all())
        self.assertTrue(torch.all(hazard >= 0.0))

    def test_landmark_prediction_apis_accept_shared_time_grid(self):
        shared_times = self.times[0]
        prediction_times = shared_times[self.observed_len - 1 :]

        survival = self.model.predict_survival_from_landmark(
            self.x_obs,
            self.mask_obs,
            prediction_times=prediction_times,
            baseline=self.baseline,
            times=shared_times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )
        hazard = self.model.predict_hazard_from_landmark(
            self.x_obs,
            self.mask_obs,
            prediction_times=prediction_times,
            baseline=self.baseline,
            times=shared_times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )
        event_prob = self.model.predict_event_probability_from_landmark(
            self.x_obs,
            self.mask_obs,
            start_times=shared_times[self.observed_len - 1 : -1],
            end_times=shared_times[self.observed_len :],
            baseline=self.baseline,
            times=shared_times,
            time_varying_covariates=self.time_varying_covariates,
            event_covariates=self.event_covariates,
        )

        self.assertEqual(survival.shape, (self.x.shape[0], prediction_times.shape[0]))
        self.assertEqual(hazard.shape, (self.x.shape[0], prediction_times.shape[0]))
        self.assertEqual(event_prob.shape, (self.x.shape[0], prediction_times.shape[0] - 1))
        self.assertTrue(torch.isfinite(survival).all())
        self.assertTrue(torch.isfinite(hazard).all())
        self.assertTrue(torch.isfinite(event_prob).all())


if __name__ == "__main__":
    unittest.main()
