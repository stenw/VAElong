"""Tests for the Time-Parameterized CNN VAE model."""

import unittest

import torch
import numpy as np

from vaelong.model import TPConv1d, TPConvTranspose1d, TPCNNLongitudinalVAE
from vaelong.config import VariableConfig, VariableSpec


class TestTPConv1d(unittest.TestCase):
    """Tests for the TPConv1d helper module."""

    def setUp(self):
        self.layer = TPConv1d(in_channels=4, out_channels=16, kernel_size=3, stride=2)

    def test_output_shape(self):
        x = torch.randn(2, 4, 50)  # (batch, channels, length)
        out = self.layer(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 16)
        # With stride=2, padding=1: length = ceil(50/2) = 25
        self.assertEqual(out.shape[2], 25)

    def test_custom_offsets(self):
        x = torch.randn(2, 4, 50)
        offsets = torch.tensor([-0.5, 0.0, 0.5])
        out = self.layer(x, offsets=offsets)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 16)

    def test_gradient_flow(self):
        x = torch.randn(2, 4, 50, requires_grad=True)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # MLP parameters should have gradients
        for p in self.layer.kernel_mlp.parameters():
            self.assertIsNotNone(p.grad)


class TestTPConvTranspose1d(unittest.TestCase):
    """Tests for the TPConvTranspose1d helper module."""

    def test_output_shape(self):
        layer = TPConvTranspose1d(in_channels=16, out_channels=4, kernel_size=3,
                                  stride=2, output_padding=1)
        x = torch.randn(2, 16, 25)
        out = layer(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 4)
        # Transposed conv with stride=2 roughly doubles the length
        self.assertGreater(out.shape[2], 25)

    def test_gradient_flow(self):
        layer = TPConvTranspose1d(in_channels=16, out_channels=4, kernel_size=3)
        x = torch.randn(2, 16, 10, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestTPCNNLongitudinalVAE(unittest.TestCase):
    """Tests for the TPCNNLongitudinalVAE class."""

    def setUp(self):
        self.input_dim = 5
        self.seq_len = 64
        self.latent_dim = 10
        self.batch_size = 8
        self.hidden_channels = [16, 32]

        self.var_config = VariableConfig(variables=[
            VariableSpec(name='feat1', var_type='continuous'),
            VariableSpec(name='feat2', var_type='continuous'),
            VariableSpec(name='feat3', var_type='binary'),
            VariableSpec(name='feat4', var_type='bounded', lower=0.0, upper=1.0),
            VariableSpec(name='feat5', var_type='continuous'),
        ])

        self.model = TPCNNLongitudinalVAE(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            latent_dim=self.latent_dim,
            hidden_channels=self.hidden_channels,
            kernel_size=3,
            n_baseline=3,
            var_config=self.var_config,
        )

    def test_model_creation(self):
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.seq_len, self.seq_len)
        self.assertEqual(self.model.latent_dim, self.latent_dim)

    def test_encode_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        bl = torch.randn(self.batch_size, 3)
        mu, logvar = self.model.encode(x, baseline=bl)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_encode_with_mask(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones_like(x)
        mask[:, 30:, :] = 0  # mask out second half
        bl = torch.randn(self.batch_size, 3)
        mu, logvar = self.model.encode(x, mask=mask, baseline=bl)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

    def test_decode_shape(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        bl = torch.randn(self.batch_size, 3)
        output = self.model.decode(z, baseline=bl)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_forward_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        bl = torch.randn(self.batch_size, 3)
        recon, mu, logvar = self.model(x, baseline=bl)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_forward_with_mask(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.ones_like(x)
        mask[:, :, 2] = 0  # mask out one feature
        bl = torch.randn(self.batch_size, 3)
        recon, mu, logvar = self.model(x, mask=mask, baseline=bl)
        self.assertEqual(recon.shape, x.shape)

    def test_sample(self):
        bl = torch.randn(4, 3)
        samples = self.model.sample(4, device='cpu', baseline=bl)
        self.assertEqual(samples.shape, (4, self.seq_len, self.input_dim))

    def test_sample_no_baseline(self):
        """Sample should work when n_baseline=0."""
        model = TPCNNLongitudinalVAE(
            input_dim=3, seq_len=32, latent_dim=8,
            hidden_channels=[16], kernel_size=3,
        )
        samples = model.sample(4, device='cpu')
        self.assertEqual(samples.shape, (4, 32, 3))

    def test_log_noise_var_exists(self):
        self.assertTrue(hasattr(self.model, 'log_noise_var'))
        # 3 continuous variables
        self.assertEqual(self.model.log_noise_var.shape, (3,))

    def test_output_activations(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        bl = torch.randn(self.batch_size, 3)
        recon, _, _ = self.model(x, baseline=bl)
        # Binary feature should be in [0, 1]
        self.assertTrue((recon[:, :, 2] >= 0).all())
        self.assertTrue((recon[:, :, 2] <= 1).all())
        # Bounded feature should be in [0, 1]
        self.assertTrue((recon[:, :, 3] >= 0).all())
        self.assertTrue((recon[:, :, 3] <= 1).all())

    def test_predict_from_landmark(self):
        x_padded = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask_padded = torch.ones_like(x_padded)
        mask_padded[:, self.seq_len // 2:, :] = 0  # future is masked
        bl = torch.randn(self.batch_size, 3)
        pred = self.model.predict_from_landmark(x_padded, mask_padded, baseline=bl)
        self.assertEqual(pred.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_no_var_config(self):
        """Model should work without var_config (all continuous)."""
        model = TPCNNLongitudinalVAE(
            input_dim=3, seq_len=32, latent_dim=8,
            hidden_channels=[16], kernel_size=3,
        )
        x = torch.randn(4, 32, 3)
        recon, mu, logvar = model(x)
        self.assertEqual(recon.shape, x.shape)

    def test_backward(self):
        """Verify gradients flow through the full model."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        bl = torch.randn(self.batch_size, 3)
        recon, mu, logvar = self.model(x, baseline=bl)
        loss = recon.sum() + mu.sum() + logvar.sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            # log_noise_var only gets gradients through the loss function, not forward()
            if p.requires_grad and name != 'log_noise_var':
                self.assertIsNotNone(p.grad, f"No gradient for {name}")


if __name__ == '__main__':
    unittest.main()
