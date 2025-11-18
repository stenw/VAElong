"""
Unit tests for CNN-based VAE model and missing data handling.
"""

import unittest
import torch
import numpy as np
from vaelong import CNNLongitudinalVAE
from vaelong.data import create_missing_mask
from vaelong.model import vae_loss_function


class TestCNNLongitudinalVAE(unittest.TestCase):
    """Test CNN-based VAE model."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.seq_len = 64
        self.latent_dim = 10
        self.batch_size = 8

        self.model = CNNLongitudinalVAE(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            latent_dim=self.latent_dim,
            hidden_channels=[16, 32],
            kernel_size=3
        )

    def test_model_creation(self):
        """Test model instantiation."""
        self.assertIsInstance(self.model, CNNLongitudinalVAE)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.seq_len, self.seq_len)
        self.assertEqual(self.model.latent_dim, self.latent_dim)

    def test_encode(self):
        """Test encoder output shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mu, logvar = self.model.encode(x)

        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_encode_with_mask(self):
        """Test encoder with mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones(self.batch_size, self.seq_len, self.input_dim)
        mask[:, :10, :] = 0  # Set first 10 timesteps as missing

        mu, logvar = self.model.encode(x, mask)

        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_decode(self):
        """Test decoder output shape."""
        z = torch.randn(self.batch_size, self.latent_dim)
        recon_x = self.model.decode(z)

        self.assertEqual(recon_x.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon_x, mu, logvar = self.model(x)

        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones_like(x)
        mask[:, :10, :] = 0

        recon_x, mu, logvar = self.model(x, mask)

        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_sample(self):
        """Test sampling."""
        num_samples = 5
        samples = self.model.sample(num_samples, device='cpu')

        self.assertEqual(samples.shape, (num_samples, self.seq_len, self.input_dim))

    def test_impute_missing(self):
        """Test missing data imputation."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones_like(x)
        mask[:, 10:20, :] = 0  # Set timesteps 10-20 as missing

        # Apply mask
        x_masked = x * mask

        # Impute
        imputed = self.model.impute_missing(x_masked, mask, num_iterations=3)

        self.assertEqual(imputed.shape, x.shape)
        # Check that observed values are preserved
        self.assertTrue(torch.allclose(imputed * mask, x_masked, atol=1e-5))
        # Check that missing values are filled
        self.assertFalse(torch.allclose(imputed * (1 - mask), torch.zeros_like(imputed * (1 - mask))))


class TestMissingDataUtilities(unittest.TestCase):
    """Test missing data utilities."""

    def test_create_missing_mask_random(self):
        """Test random missing mask creation."""
        shape = (100, 50, 5)
        missing_rate = 0.2
        mask = create_missing_mask(shape, missing_rate=missing_rate, pattern='random', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))
        # Check that approximately the right amount of data is missing
        actual_missing_rate = 1 - mask.mean()
        self.assertAlmostEqual(actual_missing_rate, missing_rate, delta=0.05)

    def test_create_missing_mask_block(self):
        """Test block missing mask creation."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='block', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_create_missing_mask_monotone(self):
        """Test monotone missing mask creation."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='monotone', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

        # Check monotone property for a few samples
        for i in range(10):
            for j in range(shape[2]):
                seq = mask[i, :, j]
                # If value at time t is missing, all subsequent should be missing
                first_missing = np.where(seq == 0)[0]
                if len(first_missing) > 0:
                    first_missing_idx = first_missing[0]
                    self.assertTrue(np.all(seq[first_missing_idx:] == 0))

    def test_vae_loss_with_mask(self):
        """Test VAE loss computation with mask."""
        batch_size = 8
        seq_len = 64
        input_dim = 5
        latent_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        recon_x = torch.randn(batch_size, seq_len, input_dim)
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)

        # Create mask with some missing values
        mask = torch.ones_like(x)
        mask[:, 10:20, :] = 0  # 10 timesteps missing

        # Compute loss with mask
        loss_masked, recon_loss_masked, kld_loss_masked = vae_loss_function(
            recon_x, x, mu, logvar, beta=1.0, mask=mask
        )

        # Compute loss without mask
        loss_full, recon_loss_full, kld_loss_full = vae_loss_function(
            recon_x, x, mu, logvar, beta=1.0, mask=None
        )

        # KLD should be the same
        self.assertAlmostEqual(kld_loss_masked.item(), kld_loss_full.item(), places=5)

        # Reconstruction loss should be different (masked should typically be different)
        # But we can't make strong assertions about the relationship


if __name__ == '__main__':
    unittest.main()
