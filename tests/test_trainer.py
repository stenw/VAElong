"""
Unit tests for VAE trainer.
"""

import unittest
import torch
from torch.utils.data import DataLoader

from vaelong.model import LongitudinalVAE, CNNLongitudinalVAE
from vaelong.trainer import VAETrainer
from vaelong.data import LongitudinalDataset, generate_synthetic_longitudinal_data, create_missing_mask
import numpy as np


class TestVAETrainer(unittest.TestCase):
    """Test cases for VAETrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.hidden_dim = 32
        self.latent_dim = 10
        self.batch_size = 8

        # Generate small dataset for testing
        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=20,
            n_features=self.input_dim,
            seed=42
        )

        self.dataset = LongitudinalDataset(data, normalize=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Create model
        self.model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )

        # Create trainer
        self.trainer = VAETrainer(self.model, learning_rate=1e-3, device='cpu')

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        self.assertEqual(self.trainer.beta, 1.0)
        self.assertEqual(self.trainer.device, torch.device('cpu'))

    def test_train_epoch(self):
        """Test training for one epoch."""
        loss, recon_loss, kld_loss = self.trainer.train_epoch(self.dataloader)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(recon_loss, float)
        self.assertIsInstance(kld_loss, float)

        self.assertGreater(loss, 0)

    def test_validate(self):
        """Test validation."""
        loss, recon_loss, kld_loss = self.trainer.validate(self.dataloader)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(recon_loss, float)
        self.assertIsInstance(kld_loss, float)

        self.assertGreater(loss, 0)

    def test_fit(self):
        """Test fitting the model."""
        history = self.trainer.fit(
            self.dataloader,
            val_loader=self.dataloader,
            epochs=5,
            verbose=False
        )

        self.assertIn('train_loss', history)
        self.assertIn('train_recon', history)
        self.assertIn('train_kld', history)
        self.assertIn('val_loss', history)

        self.assertEqual(len(history['train_loss']), 5)
        self.assertEqual(len(history['val_loss']), 5)

    def test_fit_without_validation(self):
        """Test fitting without validation set."""
        history = self.trainer.fit(
            self.dataloader,
            val_loader=None,
            epochs=3,
            verbose=False
        )

        self.assertEqual(len(history['train_loss']), 3)
        self.assertEqual(len(history['val_loss']), 0)

    def test_save_and_load_model(self):
        """Test saving and loading model."""
        import tempfile
        import os

        self.trainer.fit(self.dataloader, epochs=2, verbose=False)

        params_before = {name: param.clone() for name, param in self.model.named_parameters()}

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            self.trainer.save_model(temp_path)

            new_model = LongitudinalVAE(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim
            )
            new_trainer = VAETrainer(new_model, device='cpu')
            new_trainer.load_model(temp_path)

            params_after = {name: param for name, param in new_model.named_parameters()}

            for name in params_before:
                self.assertTrue(
                    torch.allclose(params_before[name], params_after[name], atol=1e-6),
                    f"Parameter {name} differs after loading"
                )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_beta_parameter(self):
        """Test beta parameter affects training."""
        trainer_beta = VAETrainer(self.model, learning_rate=1e-3, beta=0.5, device='cpu')

        self.assertEqual(trainer_beta.beta, 0.5)

    def test_train_with_missing_data(self):
        """Test training with missing data."""
        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=20,
            n_features=self.input_dim,
            seed=42
        )

        mask = create_missing_mask(
            data.shape,
            missing_rate=0.2,
            pattern='random',
            seed=42
        )

        data_masked = data * mask

        dataset = LongitudinalDataset(data_masked, mask=mask, normalize=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss, recon_loss, kld_loss = self.trainer.train_epoch(dataloader)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_train_with_em_imputation(self):
        """Test training with EM imputation."""
        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=20,
            n_features=self.input_dim,
            seed=42
        )

        mask = create_missing_mask(
            data.shape,
            missing_rate=0.2,
            pattern='random',
            seed=42
        )

        data_masked = data * mask

        dataset = LongitudinalDataset(data_masked, mask=mask, normalize=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss, recon_loss, kld_loss = self.trainer.train_epoch(
            dataloader,
            use_em_imputation=True,
            em_iterations=3
        )

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_fit_with_em_imputation(self):
        """Test fitting with EM imputation."""
        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=20,
            n_features=self.input_dim,
            seed=42
        )

        mask = create_missing_mask(
            data.shape,
            missing_rate=0.2,
            pattern='random',
            seed=42
        )

        data_masked = data * mask

        dataset = LongitudinalDataset(data_masked, mask=mask, normalize=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = self.trainer.fit(
            dataloader,
            epochs=3,
            verbose=False,
            use_em_imputation=True,
            em_iterations=2
        )

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)


class TestCNNVAETrainer(unittest.TestCase):
    """Test cases for VAETrainer with CNN model."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.seq_len = 64
        self.latent_dim = 10
        self.batch_size = 8

        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=self.seq_len,
            n_features=self.input_dim,
            seed=42
        )

        self.dataset = LongitudinalDataset(data, normalize=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.model = CNNLongitudinalVAE(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            latent_dim=self.latent_dim,
            hidden_channels=[16, 32],
            kernel_size=3
        )

        self.trainer = VAETrainer(self.model, learning_rate=1e-3, device='cpu')

    def test_cnn_trainer_train_epoch(self):
        """Test training CNN model for one epoch."""
        loss, recon_loss, kld_loss = self.trainer.train_epoch(self.dataloader)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_cnn_fit(self):
        """Test fitting CNN model."""
        history = self.trainer.fit(
            self.dataloader,
            epochs=3,
            verbose=False
        )

        self.assertEqual(len(history['train_loss']), 3)

    def test_cnn_with_missing_data(self):
        """Test CNN model with missing data."""
        data = generate_synthetic_longitudinal_data(
            n_samples=50,
            seq_len=self.seq_len,
            n_features=self.input_dim,
            seed=42
        )

        mask = create_missing_mask(
            data.shape,
            missing_rate=0.2,
            pattern='random',
            seed=42
        )

        data_masked = data * mask

        dataset = LongitudinalDataset(data_masked, mask=mask, normalize=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = self.trainer.fit(
            dataloader,
            epochs=3,
            verbose=False,
            use_em_imputation=True,
            em_iterations=2
        )

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)


if __name__ == '__main__':
    unittest.main()
