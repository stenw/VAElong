"""
Training utilities for VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .model import vae_loss_function, mixed_vae_loss_function


class VAETrainer:
    """
    Trainer class for Longitudinal VAE.

    Args:
        model: LongitudinalVAE or CNNLongitudinalVAE model instance
        learning_rate: Learning rate for optimizer (default: 1e-3)
        beta: Weight for KL divergence term (default: 1.0)
        device: Device to train on (default: 'cuda' if available else 'cpu')
        var_config: Optional VariableConfig for mixed-type loss computation
    """

    def __init__(self, model, learning_rate=1e-3, beta=1.0, device=None, var_config=None):
        self.model = model
        self.beta = beta
        self.var_config = var_config

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def _get_baseline_arg(self, batch_baseline):
        """Return baseline tensor or None if no baseline features."""
        if batch_baseline.shape[-1] > 0:
            return batch_baseline.to(self.device)
        return None

    def _get_log_noise_var(self):
        """Return the model's learned log_noise_var or None."""
        return getattr(self.model, 'log_noise_var', None)

    def _compute_loss(self, recon_batch, batch_data, mu, logvar, mask_arg):
        """Compute mixed VAE loss, passing through learned noise variance."""
        return mixed_vae_loss_function(
            recon_batch, batch_data, mu, logvar, self.beta, mask_arg,
            self.var_config, self._get_log_noise_var()
        )

    def train_epoch(self, train_loader, use_em_imputation=False, em_iterations=3):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            use_em_imputation: Whether to use EM-like imputation for missing data
            em_iterations: Number of EM iterations per batch (default: 3)

        Returns:
            avg_loss: Average loss for the epoch
            avg_recon_loss: Average reconstruction loss
            avg_kld_loss: Average KL divergence loss
        """
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0

        for batch in train_loader:
            batch_data, batch_mask, _, batch_baseline = batch
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            baseline_arg = self._get_baseline_arg(batch_baseline)

            # Check if there's any missing data
            has_missing = (batch_mask.sum() < batch_mask.numel())

            if use_em_imputation and has_missing:
                # EM-like approach: alternate between imputation and parameter estimation
                for em_iter in range(em_iterations):
                    # E-step: Impute missing values
                    if em_iter > 0:  # Skip first iteration, use initial values
                        with torch.no_grad():
                            recon_batch, mu_temp, logvar_temp = self.model(
                                batch_data, batch_mask, baseline_arg
                            )
                            # Type-aware imputation
                            imputed = recon_batch.clone()
                            if self.var_config is not None:
                                for idx in self.var_config.binary_indices:
                                    imputed[:, :, idx] = (imputed[:, :, idx] > 0.5).float()
                                for idx in self.var_config.bounded_indices:
                                    imputed[:, :, idx] = imputed[:, :, idx].clamp(0, 1)
                            # Update missing values with predictions
                            batch_data = batch_mask * batch_data + (1 - batch_mask) * imputed

                    # M-step: Update model parameters
                    recon_batch, mu, logvar = self.model(batch_data, batch_mask, baseline_arg)
                    loss, recon_loss, kld_loss = self._compute_loss(
                        recon_batch, batch_data, mu, logvar, batch_mask
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                # Standard training (with or without missing data mask)
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, logvar = self.model(batch_data, mask_arg, baseline_arg)
                loss, recon_loss, kld_loss = self._compute_loss(
                    recon_batch, batch_data, mu, logvar, mask_arg
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches

        return avg_loss, avg_recon, avg_kld

    def validate(self, val_loader):
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            avg_loss: Average validation loss
            avg_recon_loss: Average reconstruction loss
            avg_kld_loss: Average KL divergence loss
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_data, batch_mask, _, batch_baseline = batch
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)
                baseline_arg = self._get_baseline_arg(batch_baseline)

                # Check if there's any missing data
                has_missing = (batch_mask.sum() < batch_mask.numel())

                # Forward pass
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, logvar = self.model(batch_data, mask_arg, baseline_arg)

                # Compute loss
                loss, recon_loss, kld_loss = self._compute_loss(
                    recon_batch, batch_data, mu, logvar, mask_arg
                )

                # Accumulate losses
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches

        return avg_loss, avg_recon, avg_kld

    def fit(self, train_loader, val_loader=None, epochs=100, verbose=True,
            use_em_imputation=False, em_iterations=3, patience=0):
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            verbose: Whether to print progress
            use_em_imputation: Whether to use EM-like imputation for missing data
            em_iterations: Number of EM iterations per batch (default: 3)
            patience: Early-stopping patience (0 = disabled). Training stops
                when validation loss has not improved for ``patience`` epochs
                and the best model weights are restored.

        Returns:
            history: Dictionary containing training history
        """
        import copy

        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kld': [],
            'val_loss': [],
            'val_recon': [],
            'val_kld': []
        }

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_recon, train_kld = self.train_epoch(train_loader, use_em_imputation, em_iterations)
            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_kld'].append(train_kld)

            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kld = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_recon'].append(val_recon)
                history['val_kld'].append(val_kld)

                # Early stopping bookkeeping
                if patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f})'
                if val_loader is not None:
                    msg += f' | Val Loss: {val_loss:.4f}'
                print(msg)

            # Early stopping trigger
            if patience > 0 and epochs_no_improve >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)')
                break

        # Restore best weights
        if patience > 0 and best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f'Restored best model (val loss: {best_val_loss:.4f})')

        return history

    def save_model(self, path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
