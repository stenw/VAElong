"""
Unit tests for the joint longitudinal-survival trainer.
"""

import unittest

from torch.utils.data import DataLoader

from vaelong.data import (
    JointLongitudinalSurvivalDataset,
    create_missing_mask,
    generate_synthetic_joint_longitudinal_survival_data,
)
from vaelong.joint_model import JointLongitudinalSurvivalVAE
from vaelong.joint_trainer import JointVAETrainer


class TestJointVAETrainer(unittest.TestCase):
    """Test cases for JointVAETrainer."""

    def setUp(self):
        generated = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=24,
            seq_len=10,
            n_features=2,
            n_baseline_features=2,
            n_time_varying_covariates=1,
            n_event_covariates=2,
            seed=42,
        )
        self.input_dim = 2
        self.seq_len = 10
        self.batch_size = 6

        self.dataset = JointLongitudinalSurvivalDataset(
            data=generated["data"],
            mask=generated["mask"],
            baseline_covariates=generated["baseline_covariates"],
            times=generated["times"],
            time_varying_covariates=generated["time_varying_covariates"],
            event_times=generated["event_times"],
            event_indicators=generated["event_indicators"],
            event_covariates=generated["event_covariates"],
            normalize=True,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model = JointLongitudinalSurvivalVAE(
            input_dim=self.input_dim,
            hidden_dim=16,
            latent_dim=5,
            encoder_type="dense",
            seq_len=self.seq_len,
            n_baseline=2,
            n_event_covariates=2,
            n_time_varying_covariates=1,
            time_in_encoder=True,
            time_in_decoder=True,
        )
        self.trainer = JointVAETrainer(
            self.model,
            learning_rate=1e-3,
            device="cpu",
            survival_loss_weight=0.5,
        )

    def _make_missing_dataloader(self):
        generated = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=24,
            seq_len=10,
            n_features=2,
            n_baseline_features=2,
            n_time_varying_covariates=1,
            n_event_covariates=2,
            seed=7,
        )
        mask = create_missing_mask(
            generated["data"].shape,
            missing_rate=0.15,
            pattern="random",
            seed=7,
        )
        dataset = JointLongitudinalSurvivalDataset(
            data=generated["data"] * mask,
            mask=mask,
            baseline_covariates=generated["baseline_covariates"],
            times=generated["times"],
            time_varying_covariates=generated["time_varying_covariates"],
            event_times=generated["event_times"],
            event_indicators=generated["event_indicators"],
            event_covariates=generated["event_covariates"],
            normalize=True,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_train_epoch(self):
        """One training epoch should return all joint loss components."""
        loss, recon_loss, survival_loss, kld_loss = self.trainer.train_epoch(
            self.dataloader
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(recon_loss, float)
        self.assertIsInstance(survival_loss, float)
        self.assertIsInstance(kld_loss, float)
        self.assertGreater(loss, 0.0)
        self.assertGreater(survival_loss, 0.0)

    def test_validate(self):
        """Validation should return all joint loss components."""
        loss, recon_loss, survival_loss, kld_loss = self.trainer.validate(
            self.dataloader
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(recon_loss, float)
        self.assertIsInstance(survival_loss, float)
        self.assertIsInstance(kld_loss, float)
        self.assertGreater(loss, 0.0)

    def test_fit(self):
        """Fit should track joint loss components in history."""
        history = self.trainer.fit(
            self.dataloader,
            val_loader=self.dataloader,
            epochs=3,
            verbose=False,
        )

        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["train_recon"]), 3)
        self.assertEqual(len(history["train_survival"]), 3)
        self.assertEqual(len(history["train_kld"]), 3)
        self.assertEqual(len(history["val_loss"]), 3)
        self.assertEqual(len(history["val_survival"]), 3)

    def test_train_with_em_imputation_latent(self):
        """Latent-space EM imputation should work with the joint trainer."""
        dataloader = self._make_missing_dataloader()
        loss, recon_loss, survival_loss, kld_loss = self.trainer.train_epoch(
            dataloader,
            use_em_imputation=True,
            em_iterations=2,
            imputation_method="latent",
        )

        self.assertGreater(loss, 0.0)
        self.assertGreater(recon_loss, 0.0)
        self.assertGreater(survival_loss, 0.0)
        self.assertGreater(kld_loss, 0.0)

    def test_train_with_em_imputation_rwmh(self):
        """Missing-value RWMH should work with the joint trainer."""
        dataloader = self._make_missing_dataloader()
        loss, recon_loss, survival_loss, kld_loss = self.trainer.train_epoch(
            dataloader,
            use_em_imputation=True,
            em_iterations=2,
            imputation_method="rwmh",
            mh_steps=1,
        )

        self.assertGreater(loss, 0.0)
        self.assertGreater(recon_loss, 0.0)
        self.assertGreater(survival_loss, 0.0)
        self.assertGreater(kld_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
