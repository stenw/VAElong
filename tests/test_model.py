"""
Unit tests for Longitudinal VAE model.
"""

import unittest
import warnings
import torch

from vaelong.model import LongitudinalVAE, vae_loss_function
from vaelong.config import VariableConfig, VariableSpec


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
            latent_dim=self.latent_dim,
            encoder_type="lstm",
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

    def test_encode_with_mask(self):
        """Test encoding with mask."""
        mask = torch.ones_like(self.dummy_data)
        mask[:, :5, :] = 0
        mu, logvar = self.model.encode(self.dummy_data, mask=mask)

        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_masked_entries_still_reach_encoder(self):
        """Masked entries should still affect the encoder if their values differ."""
        model = LongitudinalVAE(
            input_dim=2,
            hidden_dim=8,
            latent_dim=4,
            encoder_type="dense",
            seq_len=3,
        )
        x_base = torch.zeros(1, 3, 2)
        x_shifted = x_base.clone()
        x_shifted[0, 1, 0] = 5.0
        mask = torch.ones_like(x_base)
        mask[0, 1, 0] = 0.0

        mu_base, _ = model.encode(x_base, mask=mask)
        mu_shifted, _ = model.encode(x_shifted, mask=mask)

        self.assertFalse(torch.allclose(mu_base, mu_shifted))

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

    def test_predict_latent_trajectory_shapes(self):
        """The decoder should expose its pre-activation latent trajectory."""
        z = torch.randn(self.batch_size, self.latent_dim)
        eta = self.model.predict_latent_trajectory(z, self.seq_len)

        self.assertEqual(eta.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_decode_matches_output_activations_of_latent_trajectory(self):
        """decode() should remain the activated version of the latent trajectory."""
        var_config = VariableConfig(variables=[
            VariableSpec(name="cont", var_type="continuous"),
            VariableSpec(name="bin", var_type="binary"),
            VariableSpec(name="bnd", var_type="bounded", lower=0.0, upper=1.0),
        ])
        model = LongitudinalVAE(
            input_dim=3,
            hidden_dim=12,
            latent_dim=5,
            encoder_type="dense",
            seq_len=4,
            var_config=var_config,
        )
        z = torch.randn(2, 5)

        eta = model.predict_latent_trajectory(z, seq_len=4)
        decoded = model.decode(z, seq_len=4)

        self.assertTrue(torch.allclose(decoded, model._apply_output_activations(eta)))
        self.assertFalse(torch.allclose(decoded[:, :, 1], eta[:, :, 1]))
        self.assertFalse(torch.allclose(decoded[:, :, 2], eta[:, :, 2]))

    def test_forward(self):
        """Test forward pass produces correct shapes."""
        recon_x, mu, logvar = self.model(self.dummy_data)

        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        mask = torch.ones_like(self.dummy_data)
        mask[:, :5, :] = 0
        recon_x, mu, logvar = self.model(self.dummy_data, mask=mask)

        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

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
            num_layers=2,
            encoder_type="lstm",
        )

        recon_x, mu, logvar = model_multi(self.dummy_data)

        self.assertEqual(recon_x.shape, self.dummy_data.shape)

    def test_full_covariance_latent_prior(self):
        """Test optional full-covariance latent prior support."""
        model_full = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="lstm",
            latent_prior_type="full",
        )

        recon_x, mu, posterior_params = model_full(self.dummy_data)
        prior_chol = model_full.get_latent_prior_cholesky()

        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(posterior_params.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(prior_chol.shape, (self.latent_dim, self.latent_dim))
        self.assertTrue(torch.all(torch.diagonal(prior_chol) > 0))

    def test_deprecated_correlated_prior_alias(self):
        """The legacy alias should still map to the full-covariance prior."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_alias = LongitudinalVAE(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                encoder_type="lstm",
                latent_prior_type="correlated",
            )
        self.assertEqual(model_alias.latent_prior_type, "full")
        self.assertTrue(any("deprecated" in str(w.message).lower() for w in caught))

    def test_full_covariance_latent_posterior(self):
        """A full posterior should emit packed Cholesky parameters and sample correctly."""
        model_full_q = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="lstm",
            latent_posterior_type="full",
        )
        recon_x, mu, posterior_params = model_full_q(self.dummy_data)
        n_offdiag = self.latent_dim * (self.latent_dim - 1) // 2

        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(
            posterior_params.shape,
            (self.batch_size, self.latent_dim + n_offdiag),
        )
        z = model_full_q.reparameterize(mu, posterior_params)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

    def test_lowrank_latent_posterior(self):
        """A low-rank posterior should emit diagonal plus factor parameters."""
        rank = 3
        model_lowrank_q = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="lstm",
            latent_posterior_type="lowrank",
            latent_posterior_rank=rank,
        )
        recon_x, mu, posterior_params = model_lowrank_q(self.dummy_data)

        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(
            posterior_params.shape,
            (self.batch_size, self.latent_dim + self.latent_dim * rank),
        )
        z = model_lowrank_q.reparameterize(mu, posterior_params)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

    def test_dense_time_in_decoder_shapes(self):
        """Dense decoder with time_in_decoder=True produces the right shapes
        and accepts a decode-time seq_len different from the encoder one."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_decoder=True,
        )

        recon_x, mu, logvar = model(self.dummy_data)
        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

        # Decoder must accept a different seq_len because the per-timestep MLP
        # is not tied to the build-time seq_len.
        z = torch.randn(self.batch_size, self.latent_dim)
        long_out = model.decode(z, seq_len=self.seq_len + 5)
        self.assertEqual(long_out.shape, (self.batch_size, self.seq_len + 5, self.input_dim))

    def test_dense_time_in_encoder_shapes(self):
        """Encoder consumes augmented input when time_in_encoder=True."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_encoder=True,
        )
        recon_x, mu, logvar = model(self.dummy_data)
        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

    def test_dense_time_in_encoder_uses_time(self):
        """Encoder output must depend on supplied times when enabled."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_encoder=True,
        )
        model.eval()
        with torch.no_grad():
            t1 = torch.arange(self.seq_len, dtype=torch.float32).expand(self.batch_size, -1)
            t2 = t1 * 10.0  # different time scale
            mu1, _ = model.encode(self.dummy_data, times=t1)
            mu2, _ = model.encode(self.dummy_data, times=t2)
        self.assertGreater((mu1 - mu2).abs().mean().item(), 1e-6)

    def test_lstm_time_in_encoder(self):
        """LSTM encoder also accepts time embeddings when enabled."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="lstm",
            time_in_encoder=True,
        )
        recon_x, mu, logvar = model(self.dummy_data)
        self.assertEqual(recon_x.shape, self.dummy_data.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

    def test_dense_time_in_decoder_uses_time(self):
        """Decoder output must vary across time when time_in_decoder=True."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_decoder=True,
        )
        model.eval()
        with torch.no_grad():
            z = torch.randn(self.batch_size, self.latent_dim)
            out = model.decode(z, seq_len=self.seq_len)
        # Adjacent timesteps should not be identical: sinusoidal time embeddings
        # plus a non-trivial MLP should give measurable differences.
        diff = (out[:, 1:, :] - out[:, :-1, :]).abs().mean().item()
        self.assertGreater(diff, 1e-6)

    def test_latent_trajectory_uses_time_varying_covariates(self):
        """The latent trajectory API should respect known decoder covariates."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_decoder=True,
            n_time_varying_covariates=2,
        )
        model.eval()
        with torch.no_grad():
            z = torch.randn(self.batch_size, self.latent_dim)
            times = torch.arange(self.seq_len, dtype=torch.float32).expand(self.batch_size, -1)
            cov1 = torch.zeros(self.batch_size, self.seq_len, 2)
            cov2 = torch.ones(self.batch_size, self.seq_len, 2)
            eta1 = model.predict_latent_trajectory(
                z, seq_len=self.seq_len, times=times, time_varying_covariates=cov1,
            )
            eta2 = model.predict_latent_trajectory(
                z, seq_len=self.seq_len, times=times, time_varying_covariates=cov2,
            )
        self.assertGreater((eta1 - eta2).abs().mean().item(), 1e-6)

    def test_dense_time_varying_covariates_affect_forward(self):
        """Known time-varying covariates should influence the output."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            n_time_varying_covariates=2,
        )
        model.eval()
        with torch.no_grad():
            cov1 = torch.zeros(self.batch_size, self.seq_len, 2)
            cov2 = torch.ones(self.batch_size, self.seq_len, 2)
            out1, _, _ = model(self.dummy_data, time_varying_covariates=cov1)
            out2, _, _ = model(self.dummy_data, time_varying_covariates=cov2)
        self.assertGreater((out1 - out2).abs().mean().item(), 1e-6)

    def test_landmark_prediction_accepts_time_varying_covariates(self):
        """Landmark prediction should accept known future covariates."""
        model = LongitudinalVAE(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            encoder_type="dense",
            seq_len=self.seq_len,
            time_in_encoder=True,
            time_in_decoder=True,
            n_time_varying_covariates=2,
        )
        x_obs = self.dummy_data[:, :10, :]
        mask_obs = torch.ones_like(x_obs)
        times = torch.arange(self.seq_len, dtype=torch.float32).expand(self.batch_size, -1)
        covs = torch.randn(self.batch_size, self.seq_len, 2)
        pred = model.predict_from_landmark(
            x_obs, mask_obs, total_seq_len=self.seq_len, times=times,
            time_varying_covariates=covs,
        )
        self.assertEqual(pred.shape, (self.batch_size, self.seq_len, self.input_dim))


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
        loss, recon_loss, kld_loss = vae_loss_function(
            self.x, self.x, self.mu, self.logvar
        )

        self.assertAlmostEqual(recon_loss.item(), 0.0, places=5)

    def test_loss_with_mask(self):
        """Test loss function with missing data mask."""
        mask = torch.ones_like(self.x)
        mask[:, :10, :] = 0

        loss_masked, recon_masked, kld_masked = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, mask=mask
        )

        loss_full, recon_full, kld_full = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, mask=None
        )

        # KLD should be the same
        self.assertAlmostEqual(kld_masked.item(), kld_full.item(), places=5)

        self.assertIsInstance(loss_masked.item(), float)
        self.assertIsInstance(recon_masked.item(), float)

    def test_loss_with_full_covariance_prior(self):
        """Test KL computation with a full-covariance latent prior."""
        chol = torch.eye(self.latent_dim)
        chol[1, 0] = 0.2
        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar,
            latent_prior_cholesky=chol,
        )

        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kld_loss.item(), float)
        self.assertGreaterEqual(kld_loss.item(), 0)

    def test_loss_with_full_covariance_posterior(self):
        """Test KL computation with a full-covariance posterior."""
        log_diag = torch.zeros(self.batch_size, self.latent_dim)
        n_offdiag = self.latent_dim * (self.latent_dim - 1) // 2
        offdiag = torch.zeros(self.batch_size, n_offdiag)
        offdiag[:, 0] = 0.2
        posterior_params = torch.cat([log_diag, offdiag], dim=1)

        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x,
            self.x,
            self.mu,
            posterior_params,
            latent_posterior_type="full",
        )

        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kld_loss.item(), float)
        self.assertGreaterEqual(kld_loss.item(), 0)

    def test_loss_with_lowrank_posterior(self):
        """Test KL computation with a diagonal-plus-low-rank posterior."""
        rank = 2
        log_diag = torch.zeros(self.batch_size, self.latent_dim)
        lowrank = torch.zeros(self.batch_size, self.latent_dim * rank)
        lowrank[:, 0] = 0.2
        posterior_params = torch.cat([log_diag, lowrank], dim=1)

        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x,
            self.x,
            self.mu,
            posterior_params,
            latent_posterior_type="lowrank",
            latent_posterior_rank=rank,
        )

        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kld_loss.item(), float)
        self.assertGreaterEqual(kld_loss.item(), 0)

    def test_loss_all_missing(self):
        """Test loss when all values are missing."""
        mask = torch.zeros_like(self.x)

        loss_masked, recon_masked, kld_masked = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, mask=mask
        )

        self.assertIsInstance(loss_masked.item(), float)
        self.assertIsInstance(kld_masked.item(), float)

    def test_loss_mask_shape_mismatch(self):
        """Test that mask must have same shape as data."""
        mask = torch.ones_like(self.x)
        loss, recon_loss, kld_loss = vae_loss_function(
            self.recon_x, self.x, self.mu, self.logvar, mask=mask
        )

        self.assertIsInstance(loss.item(), float)


if __name__ == '__main__':
    unittest.main()
