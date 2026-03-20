"""Tests for bounded variable loss options (BCE, Beta, logit-normal)."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from vaelong import (
    VariableConfig, VariableSpec,
    LongitudinalVAE, LongitudinalDataset, VAETrainer,
    generate_mixed_longitudinal_data, create_missing_mask,
)
from vaelong.model import (
    mixed_vae_loss_function,
    TPCNNLongitudinalVAE, TransformerLongitudinalVAE,
    CNNLongitudinalVAE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_var_config(bounded_loss="bce", bounded_eps=0.0):
    return VariableConfig(
        variables=[
            VariableSpec(name='cont1', var_type='continuous'),
            VariableSpec(name='bounded1', var_type='bounded', lower=0.0, upper=1.0),
            VariableSpec(name='binary1', var_type='binary'),
        ],
        bounded_loss=bounded_loss,
        bounded_eps=bounded_eps,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_bounded_loss_options(self):
        for loss in ("bce", "beta", "logit_normal"):
            vc = _make_var_config(bounded_loss=loss)
            assert vc.bounded_loss == loss

    def test_invalid_bounded_loss_raises(self):
        with pytest.raises(ValueError, match="bounded_loss"):
            _make_var_config(bounded_loss="invalid")

    def test_valid_bounded_eps(self):
        vc = _make_var_config(bounded_eps=1e-4)
        assert vc.bounded_eps == 1e-4

    def test_invalid_bounded_eps_raises(self):
        with pytest.raises(ValueError, match="bounded_eps"):
            _make_var_config(bounded_eps=0.5)
        with pytest.raises(ValueError, match="bounded_eps"):
            _make_var_config(bounded_eps=-0.1)

    def test_default_is_bce(self):
        vc = VariableConfig(variables=[
            VariableSpec(name='x', var_type='bounded', lower=0, upper=1),
        ])
        assert vc.bounded_loss == "bce"
        assert vc.bounded_eps == 0.0


# ---------------------------------------------------------------------------
# Forward pass shape preservation
# ---------------------------------------------------------------------------

class TestForwardPassShape:
    @pytest.fixture(params=["bce", "beta", "logit_normal"])
    def loss_type(self, request):
        return request.param

    def test_longitudinal_vae_shape(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        x = torch.randn(4, 10, vc.n_features)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape

    def test_cnn_vae_shape(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = CNNLongitudinalVAE(
            input_dim=vc.n_features, seq_len=10, latent_dim=4,
            var_config=vc,
        )
        x = torch.randn(4, 10, vc.n_features)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape

    def test_tpcnn_vae_shape(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = TPCNNLongitudinalVAE(
            input_dim=vc.n_features, seq_len=10, latent_dim=4,
            hidden_channels=[8], kernel_size=3, var_config=vc,
        )
        x = torch.randn(4, 10, vc.n_features)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape

    def test_transformer_vae_shape(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = TransformerLongitudinalVAE(
            input_dim=vc.n_features, seq_len=10, latent_dim=4,
            d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
            var_config=vc,
        )
        x = torch.randn(4, 10, vc.n_features)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TestLossComputation:
    @pytest.fixture(params=["bce", "beta", "logit_normal"])
    def loss_type(self, request):
        return request.param

    def test_loss_is_finite(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        x = torch.rand(4, 10, vc.n_features).clamp(0.01, 0.99)
        recon, mu, logvar = model(x)
        loss, recon_loss, kld = mixed_vae_loss_function(
            recon, x, mu, logvar, beta=1.0, var_config=vc,
            log_noise_var=model.log_noise_var,
            log_bounded_precision=getattr(model, 'log_bounded_precision', None),
            log_bounded_var=getattr(model, 'log_bounded_var', None),
        )
        assert torch.isfinite(loss), f"Loss not finite for {loss_type}: {loss}"
        assert torch.isfinite(recon_loss)
        assert torch.isfinite(kld)

    def test_loss_with_mask(self, loss_type):
        vc = _make_var_config(bounded_loss=loss_type)
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        x = torch.rand(4, 10, vc.n_features).clamp(0.01, 0.99)
        mask = (torch.rand(4, 10, vc.n_features) > 0.3).float()
        recon, mu, logvar = model(x * mask, mask)
        loss, _, _ = mixed_vae_loss_function(
            recon, x, mu, logvar, beta=1.0, mask=mask, var_config=vc,
            log_noise_var=model.log_noise_var,
            log_bounded_precision=getattr(model, 'log_bounded_precision', None),
            log_bounded_var=getattr(model, 'log_bounded_var', None),
        )
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Epsilon clamping
# ---------------------------------------------------------------------------

class TestEpsilonClamping:
    def test_clamping_removes_exact_boundaries(self):
        vc = _make_var_config(bounded_eps=1e-4)
        data = np.zeros((10, 5, 3))
        data[:, :, 0] = np.random.randn(10, 5)  # continuous
        data[:, :, 1] = np.array([0.0, 0.5, 1.0, 0.0, 1.0] * 10).reshape(10, 5)  # bounded with 0s and 1s
        data[:, :, 2] = np.random.randint(0, 2, (10, 5)).astype(float)  # binary
        mask = np.ones_like(data)

        dataset = LongitudinalDataset(data, mask=mask, var_config=vc, normalize=True)
        # Check bounded column (index 1) has no exact 0 or 1
        all_bounded = torch.stack([dataset[i][0][:, 1] for i in range(len(dataset))])
        observed_mask = torch.stack([dataset[i][1][:, 1] for i in range(len(dataset))])
        observed_vals = all_bounded[observed_mask > 0]
        assert observed_vals.min() >= 1e-4
        assert observed_vals.max() <= 1 - 1e-4


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    @pytest.fixture(params=["bce", "beta", "logit_normal"])
    def loss_type(self, request):
        return request.param

    def test_loss_decreases(self, loss_type):
        torch.manual_seed(42)
        np.random.seed(42)

        eps = 1e-4 if loss_type != "bce" else 0.0
        vc = _make_var_config(bounded_loss=loss_type, bounded_eps=eps)

        data, baseline = generate_mixed_longitudinal_data(
            n_samples=50, seq_len=10, var_config=vc,
            n_baseline_features=2, seed=42,
        )
        mask = create_missing_mask(data.shape, missing_rate=0.15, seed=42)

        dataset = LongitudinalDataset(
            data * mask, mask=mask, var_config=vc,
            baseline_covariates=baseline, normalize=True,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=32, latent_dim=8,
            seq_len=10, n_baseline=2, var_config=vc,
        )
        trainer = VAETrainer(model, learning_rate=1e-3, beta=0.1, var_config=vc)

        first_loss, _, _ = trainer.train_epoch(loader)
        for _ in range(9):
            last_loss, _, _ = trainer.train_epoch(loader)

        assert last_loss < first_loss, (
            f"Loss did not decrease for {loss_type}: {first_loss:.4f} -> {last_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Model parameters exist
# ---------------------------------------------------------------------------

class TestModelParameters:
    def test_beta_has_precision_param(self):
        vc = _make_var_config(bounded_loss="beta")
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        assert hasattr(model, 'log_bounded_precision')
        assert model.log_bounded_precision.shape == (1,)  # 1 bounded variable

    def test_logit_normal_has_var_param(self):
        vc = _make_var_config(bounded_loss="logit_normal")
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        assert hasattr(model, 'log_bounded_var')
        assert model.log_bounded_var.shape == (1,)

    def test_bce_has_no_extra_params(self):
        vc = _make_var_config(bounded_loss="bce")
        model = LongitudinalVAE(
            input_dim=vc.n_features, hidden_dim=16, latent_dim=4,
            seq_len=10, var_config=vc,
        )
        assert not hasattr(model, 'log_bounded_precision')
        assert not hasattr(model, 'log_bounded_var')
