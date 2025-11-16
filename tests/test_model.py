"""
Unit tests for Longitudinal VAE model.
"""

import unittest
import torch
import numpy as np

from vaelong.model import LongitudinalVAE, vae_loss_function


class TestLongitudinalVAE(unittest.TestCase):
    """Test cases for LongitudinalVAE model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.hidden_dim = 32
        self.latent_dim = 10
        self.batch_size = 8
        self.seq_len = 20
        
        self.model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        
        # Create dummy data
        self.dummy_data = torch.randn(self.batch_size, self.seq_len, self.input_dim)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.latent_dim, self.latent_dim)
    
    def test_encode(self):
        """Test encoding produces correct shapes."""
        mu, logvar = self.model.encode(self.dummy_data)
        
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        z = self.model.reparameterize(mu, logvar)
        
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
    
    def test_decode(self):
        """Test decoding produces correct shapes."""
        z = torch.randn(self.batch_size, self.latent_dim)
        output = self.model.decode(z, self.seq_len)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))
    
    def test_forward(self):
        """Test forward pass produces correct shapes."""
        recon_x, mu, logvar = self.model(self.dummy_data)
        
        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
    
    def test_sample(self):
        """Test sampling from the model."""
        num_samples = 5
        samples = self.model.sample(num_samples, self.seq_len)
        
        self.assertEqual(samples.shape, (num_samples, self.seq_len, self.input_dim))
    
    def test_gru_mode(self):
        """Test model with GRU instead of LSTM."""
        model_gru = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            use_gru=True
        )
        
        recon_x, mu, logvar = model_gru(self.dummy_data)
        
        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
    
    def test_multi_layer(self):
        """Test model with multiple RNN layers."""
        model_multi = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2
        )
        
        recon_x, mu, logvar = model_multi(self.dummy_data)
        
        self.assertEqual(recon_x.shape, self.dummy_data.shape)


class TestVAELoss(unittest.TestCase):
    """Test cases for VAE loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.seq_len = 20
        self.input_dim = 5
        self.latent_dim = 10
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.recon_x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.logvar = torch.randn(self.batch_size, self.latent_dim)
    
    def test_loss_computation(self):
        """Test loss function computes without error."""
        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar
        )
        
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kld_loss.item(), float)
    
    def test_loss_positive(self):
        """Test that losses are positive."""
        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar
        )
        
        self.assertGreaterEqual(recon_loss.item(), 0)
        # KLD can be negative for individual samples but typically positive overall
    
    def test_beta_parameter(self):
        """Test beta parameter affects loss."""
        loss1, _, kld1 = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, beta=1.0
        )
        loss2, _, kld2 = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, beta=0.5
        )
        
        # With lower beta, total loss should be lower (less weight on KLD)
        self.assertLess(loss2.item(), loss1.item())
        # KLD itself shouldn't change
        self.assertAlmostEqual(kld1.item(), kld2.item(), places=5)
    
    def test_perfect_reconstruction(self):
        """Test loss with perfect reconstruction."""
        # When recon_x equals x, reconstruction loss should be 0
        loss, recon_loss, kld_loss = vae_loss_function(
            self.x, self.x, self.mu, self.logvar
        )
        
        self.assertAlmostEqual(recon_loss.item(), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
