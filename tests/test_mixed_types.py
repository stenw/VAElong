"""
Unit tests for mixed-type multivariate data support.
"""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader

from vaelong.config import VariableSpec, VariableConfig
from vaelong.model import (
    LongitudinalVAE, CNNLongitudinalVAE,
    TPCNNLongitudinalVAE, TransformerLongitudinalVAE,
    vae_loss_function, mixed_vae_loss_function,
)
from vaelong.data import (
    LongitudinalDataset, generate_mixed_longitudinal_data, create_missing_mask,
)
from vaelong.trainer import VAETrainer


class TestVariableConfig(unittest.TestCase):
    """Test VariableSpec and VariableConfig."""

    def test_variable_spec_creation(self):
        """Test creating variable specs."""
        spec = VariableSpec(name='x', var_type='continuous')
        self.assertEqual(spec.var_type, 'continuous')

        spec_bin = VariableSpec(name='y', var_type='binary')
        self.assertEqual(spec_bin.var_type, 'binary')

        spec_bnd = VariableSpec(name='z', var_type='bounded', lower=0.0, upper=10.0)
        self.assertEqual(spec_bnd.lower, 0.0)
        self.assertEqual(spec_bnd.upper, 10.0)

    def test_variable_spec_invalid_type(self):
        """Test that invalid var_type raises error."""
        with self.assertRaises(ValueError):
            VariableSpec(name='bad', var_type='invalid')

    def test_variable_spec_invalid_bounds(self):
        """Test that lower >= upper raises error for bounded."""
        with self.assertRaises(ValueError):
            VariableSpec(name='bad', var_type='bounded', lower=5.0, upper=3.0)

    def test_variable_config_indices(self):
        """Test index properties of VariableConfig."""
        config = VariableConfig(variables=[
            VariableSpec(name='c1', var_type='continuous'),
            VariableSpec(name='b1', var_type='binary'),
            VariableSpec(name='c2', var_type='continuous'),
            VariableSpec(name='bnd1', var_type='bounded'),
            VariableSpec(name='b2', var_type='binary'),
        ])

        self.assertEqual(config.n_features, 5)
        self.assertEqual(config.continuous_indices, [0, 2])
        self.assertEqual(config.binary_indices, [1, 4])
        self.assertEqual(config.bounded_indices, [3])

    def test_all_continuous_factory(self):
        """Test all_continuous factory method."""
        config = VariableConfig.all_continuous(3)
        self.assertEqual(config.n_features, 3)
        self.assertEqual(config.continuous_indices, [0, 1, 2])
        self.assertEqual(config.binary_indices, [])
        self.assertEqual(config.bounded_indices, [])

    def test_get_bounds(self):
        """Test get_bounds method."""
        config = VariableConfig(variables=[
            VariableSpec(name='c1', var_type='continuous'),
            VariableSpec(name='bnd1', var_type='bounded', lower=0.0, upper=5.0),
            VariableSpec(name='bnd2', var_type='bounded', lower=-1.0, upper=1.0),
        ])

        bounds = config.get_bounds()
        self.assertEqual(bounds[1], (0.0, 5.0))
        self.assertEqual(bounds[2], (-1.0, 1.0))
        self.assertNotIn(0, bounds)


class TestMixedSyntheticData(unittest.TestCase):
    """Test mixed-type synthetic data generation."""

    def test_default_generation(self):
        """Test generation with default config."""
        data, baseline = generate_mixed_longitudinal_data(
            n_samples=100, seq_len=30, seed=42
        )

        self.assertEqual(data.shape, (100, 30, 5))
        self.assertIsNone(baseline)

    def test_custom_config(self):
        """Test generation with custom config."""
        config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded', lower=0.0, upper=10.0),
        ])

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=200, seq_len=40, var_config=config, seed=42
        )

        self.assertEqual(data.shape, (200, 40, 3))

        # Binary values should be 0 or 1
        binary_vals = data[:, :, 1]
        self.assertTrue(np.all((binary_vals == 0) | (binary_vals == 1)))

        # Bounded values should be in [0, 10]
        bounded_vals = data[:, :, 2]
        self.assertTrue(np.all(bounded_vals >= 0.0))
        self.assertTrue(np.all(bounded_vals <= 10.0))

    def test_baseline_generation(self):
        """Test baseline covariate generation."""
        data, baseline = generate_mixed_longitudinal_data(
            n_samples=100, seq_len=30, n_baseline_features=4, seed=42
        )

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.shape, (100, 4))

    def test_deterministic(self):
        """Test deterministic generation."""
        data1, bl1 = generate_mixed_longitudinal_data(n_samples=50, seq_len=20, seed=42)
        data2, bl2 = generate_mixed_longitudinal_data(n_samples=50, seq_len=20, seed=42)

        np.testing.assert_array_equal(data1, data2)


class TestMixedDataset(unittest.TestCase):
    """Test LongitudinalDataset with mixed types and baselines."""

    def setUp(self):
        """Set up test fixtures."""
        self.var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded', lower=0.0, upper=10.0),
        ])

        self.data, self.baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=30, var_config=self.var_config,
            n_baseline_features=3, seed=42
        )

    def test_dataset_with_var_config(self):
        """Test dataset creation with var_config."""
        dataset = LongitudinalDataset(
            self.data, var_config=self.var_config, normalize=True
        )

        self.assertEqual(len(dataset), 50)
        item, mask, length, baseline = dataset[0]
        self.assertEqual(item.shape, torch.Size([30, 3]))
        self.assertEqual(baseline.shape, torch.Size([0]))  # no baseline provided

    def test_dataset_with_baseline(self):
        """Test dataset with baseline covariates."""
        dataset = LongitudinalDataset(
            self.data, var_config=self.var_config,
            baseline_covariates=self.baseline, normalize=True
        )

        item, mask, length, baseline = dataset[0]
        self.assertEqual(baseline.shape, torch.Size([3]))

    def test_normalization_by_type(self):
        """Test that normalization is type-aware."""
        dataset = LongitudinalDataset(
            self.data, var_config=self.var_config, normalize=True
        )

        # Continuous variable (idx 0) should be z-scored
        cont_data = dataset.data[:, :, 0]
        cont_mask = dataset.mask[:, :, 0]
        observed = cont_data * cont_mask
        n_obs = cont_mask.sum()
        mean_val = observed.sum() / n_obs
        self.assertAlmostEqual(mean_val.item(), 0.0, places=4)

        # Binary variable (idx 1) should still be 0/1
        bin_data = dataset.data[:, :, 1]
        unique_vals = torch.unique(bin_data)
        self.assertTrue(all(v in [0.0, 1.0] for v in unique_vals.tolist()))

        # Bounded variable (idx 2) should be in [0, 1] after normalization
        bnd_data = dataset.data[:, :, 2]
        self.assertTrue(torch.all(bnd_data >= -0.01))
        self.assertTrue(torch.all(bnd_data <= 1.01))

    def test_inverse_transform(self):
        """Test type-aware inverse transform."""
        dataset = LongitudinalDataset(
            self.data, var_config=self.var_config, normalize=True
        )

        normalized, mask, _, _ = dataset[0]
        denormalized = dataset.inverse_transform(normalized)

        original = torch.FloatTensor(self.data[0])

        # Continuous: should match original
        self.assertTrue(torch.allclose(
            denormalized[..., 0], original[..., 0], atol=1e-4
        ))

        # Bounded: should match original (within tolerance for float precision)
        self.assertTrue(torch.allclose(
            denormalized[..., 2], original[..., 2], atol=0.1
        ))

    def test_dataset_with_mask(self):
        """Test dataset with missing data mask and mixed types."""
        mask = create_missing_mask(self.data.shape, missing_rate=0.2, pattern='random', seed=42)

        dataset = LongitudinalDataset(
            self.data * mask, mask=mask, var_config=self.var_config, normalize=True
        )

        item, item_mask, length, baseline = dataset[0]
        self.assertEqual(item.shape, torch.Size([30, 3]))
        self.assertEqual(item_mask.shape, torch.Size([30, 3]))


class TestMixedLossFunction(unittest.TestCase):
    """Test mixed_vae_loss_function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.seq_len = 20
        self.latent_dim = 10

        self.var_config = VariableConfig(variables=[
            VariableSpec(name='c1', var_type='continuous'),
            VariableSpec(name='c2', var_type='continuous'),
            VariableSpec(name='b1', var_type='binary'),
            VariableSpec(name='bnd1', var_type='bounded'),
        ])

        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.logvar = torch.randn(self.batch_size, self.latent_dim)

    def test_fallback_to_vae_loss(self):
        """Test that var_config=None falls back to standard loss."""
        x = torch.randn(self.batch_size, self.seq_len, 4)
        recon_x = torch.randn(self.batch_size, self.seq_len, 4)

        loss1, r1, k1 = vae_loss_function(recon_x, x, self.mu, self.logvar)
        loss2, r2, k2 = mixed_vae_loss_function(recon_x, x, self.mu, self.logvar, var_config=None)

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)

    def test_mixed_loss_computes(self):
        """Test mixed loss computes without error."""
        # Create data appropriate for mixed types
        x = torch.randn(self.batch_size, self.seq_len, 4)
        x[:, :, 2] = torch.bernoulli(torch.ones(self.batch_size, self.seq_len) * 0.5)  # binary
        x[:, :, 3] = torch.sigmoid(torch.randn(self.batch_size, self.seq_len))  # bounded [0,1]

        recon_x = torch.randn(self.batch_size, self.seq_len, 4)
        recon_x[:, :, 2] = torch.sigmoid(recon_x[:, :, 2])  # model outputs sigmoid for binary
        recon_x[:, :, 3] = torch.sigmoid(recon_x[:, :, 3])  # model outputs sigmoid for bounded

        loss, recon_loss, kld_loss = mixed_vae_loss_function(
            recon_x, x, self.mu, self.logvar, var_config=self.var_config
        )

        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kld_loss.item(), float)

    def test_mixed_loss_with_mask(self):
        """Test mixed loss with missing data mask."""
        x = torch.randn(self.batch_size, self.seq_len, 4)
        x[:, :, 2] = torch.bernoulli(torch.ones(self.batch_size, self.seq_len) * 0.5)
        x[:, :, 3] = torch.sigmoid(torch.randn(self.batch_size, self.seq_len))

        recon_x = torch.randn(self.batch_size, self.seq_len, 4)
        recon_x[:, :, 2] = torch.sigmoid(recon_x[:, :, 2])
        recon_x[:, :, 3] = torch.sigmoid(recon_x[:, :, 3])

        mask = torch.ones_like(x)
        mask[:, :5, :] = 0

        loss, recon_loss, kld_loss = mixed_vae_loss_function(
            recon_x, x, self.mu, self.logvar, mask=mask, var_config=self.var_config
        )

        self.assertIsInstance(loss.item(), float)


class TestMixedTypeModel(unittest.TestCase):
    """Test models with mixed types and baseline conditioning."""

    def setUp(self):
        """Set up test fixtures."""
        self.var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded'),
        ])
        self.input_dim = 3
        self.batch_size = 8
        self.seq_len = 20
        self.latent_dim = 10
        self.n_baseline = 4

    def test_lstm_with_var_config(self):
        """Test LSTM model with var_config."""
        model = LongitudinalVAE(
            input_dim=self.input_dim, hidden_dim=32,
            latent_dim=self.latent_dim, var_config=self.var_config
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        recon_x, mu, logvar = model(x)

        self.assertEqual(recon_x.shape, x.shape)

        # Binary output should be in [0, 1]
        self.assertTrue(torch.all(recon_x[:, :, 1] >= 0))
        self.assertTrue(torch.all(recon_x[:, :, 1] <= 1))

        # Bounded output should be in [0, 1]
        self.assertTrue(torch.all(recon_x[:, :, 2] >= 0))
        self.assertTrue(torch.all(recon_x[:, :, 2] <= 1))

    def test_lstm_with_baseline(self):
        """Test LSTM model with baseline covariates."""
        model = LongitudinalVAE(
            input_dim=self.input_dim, hidden_dim=32,
            latent_dim=self.latent_dim, n_baseline=self.n_baseline,
            var_config=self.var_config
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        baseline = torch.randn(self.batch_size, self.n_baseline)

        recon_x, mu, logvar = model(x, baseline=baseline)

        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

    def test_cnn_with_var_config(self):
        """Test CNN model with var_config."""
        seq_len = 64
        model = CNNLongitudinalVAE(
            input_dim=self.input_dim, seq_len=seq_len,
            latent_dim=self.latent_dim, hidden_channels=[16, 32],
            var_config=self.var_config
        )

        x = torch.randn(self.batch_size, seq_len, self.input_dim)
        recon_x, mu, logvar = model(x)

        self.assertEqual(recon_x.shape, x.shape)

        # Binary output should be in [0, 1]
        self.assertTrue(torch.all(recon_x[:, :, 1] >= 0))
        self.assertTrue(torch.all(recon_x[:, :, 1] <= 1))

    def test_cnn_with_baseline(self):
        """Test CNN model with baseline covariates."""
        seq_len = 64
        model = CNNLongitudinalVAE(
            input_dim=self.input_dim, seq_len=seq_len,
            latent_dim=self.latent_dim, hidden_channels=[16, 32],
            n_baseline=self.n_baseline, var_config=self.var_config
        )

        x = torch.randn(self.batch_size, seq_len, self.input_dim)
        baseline = torch.randn(self.batch_size, self.n_baseline)

        recon_x, mu, logvar = model(x, baseline=baseline)

        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))

    def test_backward_compat_no_config(self):
        """Test models work without var_config (backward compat)."""
        model = LongitudinalVAE(
            input_dim=5, hidden_dim=32, latent_dim=10
        )
        x = torch.randn(4, 20, 5)
        recon_x, mu, logvar = model(x)
        self.assertEqual(recon_x.shape, x.shape)

    def test_backward_compat_cnn_no_config(self):
        """Test CNN model works without var_config (backward compat)."""
        model = CNNLongitudinalVAE(
            input_dim=5, seq_len=64, latent_dim=10, hidden_channels=[16, 32]
        )
        x = torch.randn(4, 64, 5)
        recon_x, mu, logvar = model(x)
        self.assertEqual(recon_x.shape, x.shape)


class TestLandmarkPrediction(unittest.TestCase):
    """Test landmark prediction functionality."""

    def test_lstm_landmark_prediction(self):
        """Test LSTM landmark prediction."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
        ])

        model = LongitudinalVAE(
            input_dim=2, hidden_dim=32, latent_dim=10,
            var_config=var_config
        )

        batch_size = 4
        observed_len = 15
        total_len = 30

        x_observed = torch.randn(batch_size, observed_len, 2)
        mask_observed = torch.ones(batch_size, observed_len, 2)

        predicted = model.predict_from_landmark(
            x_observed, mask_observed, total_seq_len=total_len
        )

        self.assertEqual(predicted.shape, (batch_size, total_len, 2))

        # Binary output should be in [0, 1]
        self.assertTrue(torch.all(predicted[:, :, 1] >= 0))
        self.assertTrue(torch.all(predicted[:, :, 1] <= 1))

    def test_lstm_landmark_with_baseline(self):
        """Test LSTM landmark prediction with baseline."""
        model = LongitudinalVAE(
            input_dim=2, hidden_dim=32, latent_dim=10, n_baseline=3
        )

        batch_size = 4
        x_observed = torch.randn(batch_size, 15, 2)
        mask_observed = torch.ones(batch_size, 15, 2)
        baseline = torch.randn(batch_size, 3)

        predicted = model.predict_from_landmark(
            x_observed, mask_observed, total_seq_len=30, baseline=baseline
        )

        self.assertEqual(predicted.shape, (batch_size, 30, 2))

    def test_cnn_landmark_prediction(self):
        """Test CNN landmark prediction."""
        seq_len = 64
        model = CNNLongitudinalVAE(
            input_dim=3, seq_len=seq_len, latent_dim=10,
            hidden_channels=[16, 32],
            var_config=VariableConfig(variables=[
                VariableSpec(name='c', var_type='continuous'),
                VariableSpec(name='b', var_type='binary'),
                VariableSpec(name='bnd', var_type='bounded'),
            ])
        )

        batch_size = 4
        landmark_t = 30

        # Pad to seq_len, mask future as missing
        x_padded = torch.randn(batch_size, seq_len, 3)
        mask = torch.ones(batch_size, seq_len, 3)
        mask[:, landmark_t:, :] = 0
        x_padded = x_padded * mask

        predicted = model.predict_from_landmark(x_padded, mask)

        self.assertEqual(predicted.shape, (batch_size, seq_len, 3))

    def test_tpcnn_landmark_prediction(self):
        """Test TPCNN landmark prediction."""
        seq_len = 64
        model = TPCNNLongitudinalVAE(
            input_dim=3, seq_len=seq_len, latent_dim=10,
            hidden_channels=[16, 32],
            var_config=VariableConfig(variables=[
                VariableSpec(name='c', var_type='continuous'),
                VariableSpec(name='b', var_type='binary'),
                VariableSpec(name='bnd', var_type='bounded'),
            ])
        )

        batch_size = 4
        landmark_t = 30

        x_padded = torch.randn(batch_size, seq_len, 3)
        mask = torch.ones(batch_size, seq_len, 3)
        mask[:, landmark_t:, :] = 0
        x_padded = x_padded * mask

        predicted = model.predict_from_landmark(x_padded, mask)
        self.assertEqual(predicted.shape, (batch_size, seq_len, 3))

    def test_transformer_landmark_prediction(self):
        """Test Transformer landmark prediction."""
        seq_len = 64
        model = TransformerLongitudinalVAE(
            input_dim=3, seq_len=seq_len, latent_dim=10,
            d_model=16, nhead=4, num_layers=1,
            var_config=VariableConfig(variables=[
                VariableSpec(name='c', var_type='continuous'),
                VariableSpec(name='b', var_type='binary'),
                VariableSpec(name='bnd', var_type='bounded'),
            ])
        )

        batch_size = 4
        landmark_t = 30

        x_padded = torch.randn(batch_size, seq_len, 3)
        mask = torch.ones(batch_size, seq_len, 3)
        mask[:, landmark_t:, :] = 0
        x_padded = x_padded * mask

        predicted = model.predict_from_landmark(x_padded, mask)
        self.assertEqual(predicted.shape, (batch_size, seq_len, 3))


class TestMixedTrainerIntegration(unittest.TestCase):
    """Integration tests for trainer with mixed-type data."""

    def test_full_training_pipeline(self):
        """Test full training pipeline with mixed types and baselines."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded'),
        ])

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=20, var_config=var_config,
            n_baseline_features=3, seed=42
        )

        dataset = LongitudinalDataset(
            data, var_config=var_config, baseline_covariates=baseline,
            normalize=True
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = LongitudinalVAE(
            input_dim=3, hidden_dim=32, latent_dim=10,
            n_baseline=3, var_config=var_config
        )

        trainer = VAETrainer(model, learning_rate=1e-3, device='cpu', var_config=var_config)

        history = trainer.fit(dataloader, epochs=3, verbose=False)

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)

    def test_training_with_missing_data_mixed(self):
        """Test training with missing data and mixed types."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
        ])

        data, _ = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=20, var_config=var_config, seed=42
        )

        mask = create_missing_mask(data.shape, missing_rate=0.2, pattern='random', seed=42)
        data_masked = data * mask

        dataset = LongitudinalDataset(
            data_masked, mask=mask, var_config=var_config, normalize=True
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = LongitudinalVAE(
            input_dim=2, hidden_dim=32, latent_dim=10, var_config=var_config
        )

        trainer = VAETrainer(model, learning_rate=1e-3, device='cpu', var_config=var_config)

        history = trainer.fit(
            dataloader, epochs=3, verbose=False,
            use_em_imputation=True, em_iterations=2
        )

        self.assertEqual(len(history['train_loss']), 3)

    def test_cnn_training_pipeline_mixed(self):
        """Test CNN training with mixed types."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded'),
        ])

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=64, var_config=var_config,
            n_baseline_features=2, seed=42
        )

        dataset = LongitudinalDataset(
            data, var_config=var_config, baseline_covariates=baseline,
            normalize=True
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = CNNLongitudinalVAE(
            input_dim=3, seq_len=64, latent_dim=10,
            hidden_channels=[16, 32], n_baseline=2, var_config=var_config
        )

        trainer = VAETrainer(model, learning_rate=1e-3, device='cpu', var_config=var_config)

        history = trainer.fit(dataloader, epochs=3, verbose=False)

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)

    def test_tpcnn_training_pipeline_mixed(self):
        """Test TPCNN training with mixed types."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded'),
        ])

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=64, var_config=var_config,
            n_baseline_features=2, seed=42
        )

        dataset = LongitudinalDataset(
            data, var_config=var_config, baseline_covariates=baseline,
            normalize=True
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = TPCNNLongitudinalVAE(
            input_dim=3, seq_len=64, latent_dim=10,
            hidden_channels=[16, 32], n_baseline=2, var_config=var_config
        )

        trainer = VAETrainer(model, learning_rate=1e-3, device='cpu', var_config=var_config)

        history = trainer.fit(dataloader, epochs=3, verbose=False)

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)

    def test_transformer_training_pipeline_mixed(self):
        """Test Transformer training with mixed types."""
        var_config = VariableConfig(variables=[
            VariableSpec(name='c', var_type='continuous'),
            VariableSpec(name='b', var_type='binary'),
            VariableSpec(name='bnd', var_type='bounded'),
        ])

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=64, var_config=var_config,
            n_baseline_features=2, seed=42
        )

        dataset = LongitudinalDataset(
            data, var_config=var_config, baseline_covariates=baseline,
            normalize=True
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = TransformerLongitudinalVAE(
            input_dim=3, seq_len=64, latent_dim=10,
            d_model=16, nhead=4, num_layers=1,
            n_baseline=2, var_config=var_config
        )

        trainer = VAETrainer(model, learning_rate=1e-3, device='cpu', var_config=var_config)

        history = trainer.fit(dataloader, epochs=3, verbose=False)

        self.assertEqual(len(history['train_loss']), 3)
        self.assertGreater(history['train_loss'][0], 0)


if __name__ == '__main__':
    unittest.main()
