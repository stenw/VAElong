"""
Training utilities for VAE.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import mixed_vae_loss_function, gaussian_kl_divergence


class VAETrainer:
    """
    Trainer class for Longitudinal VAE.

    Args:
        model: LongitudinalVAE or CNNLongitudinalVAE model instance
        learning_rate: Learning rate for optimizer (default: 1e-3)
        beta: Weight for KL divergence term (default: 1.0)
        device: Device to train on (default: 'cuda' if available else 'cpu')
        var_config: Optional VariableConfig for mixed-type loss computation
        noise_var_penalty: L2 penalty weight on log_noise_var (default: 1.0,
            mild regularisation). Set to 0.0 for no penalty, or higher
            (e.g. 10.0) for stronger anchoring toward σ²=1.
        weight_decay: L2 regularisation on model weights via AdamW-style
            decay (default: 0.0, no regularisation).
    """

    def __init__(self, model, learning_rate=1e-3, beta=1.0, device=None,
                 var_config=None, noise_var_penalty=1.0, weight_decay=0.0):
        self.model = model
        self.beta = beta
        self.var_config = var_config
        self.noise_var_penalty = noise_var_penalty

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

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
        """Compute mixed VAE loss, passing through learned parameters."""
        return mixed_vae_loss_function(
            recon_batch, batch_data, mu, logvar, self.beta, mask_arg,
            self.var_config, self._get_log_noise_var(),
            noise_var_penalty=self.noise_var_penalty,
            log_bounded_precision=getattr(self.model, 'log_bounded_precision', None),
            log_bounded_var=getattr(self.model, 'log_bounded_var', None),
            latent_prior_cholesky=getattr(self.model, 'get_latent_prior_cholesky', lambda **_: None)(
                device=batch_data.device, dtype=batch_data.dtype
            ),
        )

    def _sample_from_observation_model(self, recon_batch):
        """Sample from the observation model p(y | z) given decoder output.

        For continuous variables, samples y ~ N(mean, sigma_y^2) using the
        learned per-variable noise variance.  For binary variables, samples
        y ~ Bernoulli(p).  For bounded variables the treatment depends on the
        loss type (BCE -> clamp, logit-normal -> sigmoid then clamp).

        Args:
            recon_batch: (batch, seq_len, n_features) decoder output (means /
                probabilities).

        Returns:
            sampled: tensor of same shape with stochastic draws.
        """
        sampled = recon_batch.clone()

        if self.var_config is not None:
            # Continuous: y ~ N(m, sigma_y^2)
            log_nv = self._get_log_noise_var()
            if log_nv is not None and len(self.var_config.continuous_indices) > 0:
                sigma = (0.5 * log_nv.clamp(-6.0, 6.0)).exp()  # (n_cont,)
                for k, idx in enumerate(self.var_config.continuous_indices):
                    noise = torch.randn_like(sampled[:, :, idx]) * sigma[k]
                    sampled[:, :, idx] = sampled[:, :, idx] + noise

            # Binary: y ~ Bernoulli(p)
            for idx in self.var_config.binary_indices:
                prob = sampled[:, :, idx].clamp(1e-6, 1 - 1e-6)
                sampled[:, :, idx] = torch.bernoulli(prob)

            # Bounded: clamp to [0, 1] (optionally via sigmoid for logit-normal)
            for idx in self.var_config.bounded_indices:
                if getattr(self.var_config, 'bounded_loss', 'bce') == 'logit_normal':
                    sampled[:, :, idx] = torch.sigmoid(sampled[:, :, idx])
                else:
                    sampled[:, :, idx] = sampled[:, :, idx].clamp(0, 1)
        return sampled

    def _deterministic_reconstruction(self, batch_data, batch_mask, baseline_arg):
        """Decode from the posterior mean for a deterministic imputation score."""
        mu, logvar = self.model.encode(batch_data, batch_mask, baseline_arg)
        try:
            recon_batch = self.model.decode(mu, batch_data.shape[1], baseline_arg)
        except TypeError:
            recon_batch = self.model.decode(mu, baseline_arg)
        return recon_batch, mu, logvar

    def _reconstruction_nll_per_sample(self, recon_batch, batch_data):
        """Per-sample reconstruction NLL on the full imputed sequence."""
        if self.var_config is None:
            return ((recon_batch - batch_data) ** 2).sum(dim=(1, 2))

        recon_loss = torch.zeros(batch_data.shape[0], device=batch_data.device)

        cont_idx = self.var_config.continuous_indices
        if cont_idx:
            cont_recon = recon_batch[:, :, cont_idx]
            cont_x = batch_data[:, :, cont_idx]
            log_nv = self._get_log_noise_var()
            if log_nv is not None:
                lnv = log_nv.clamp(-6.0, 6.0).view(1, 1, -1)
                cont_nll = 0.5 * (lnv + (cont_recon - cont_x) ** 2 / lnv.exp())
            else:
                cont_nll = (cont_recon - cont_x) ** 2
            recon_loss = recon_loss + cont_nll.sum(dim=(1, 2))

        bin_idx = self.var_config.binary_indices
        if bin_idx:
            bin_recon = recon_batch[:, :, bin_idx].clamp(1e-7, 1 - 1e-7)
            bin_x = batch_data[:, :, bin_idx]
            bin_nll = F.binary_cross_entropy(bin_recon, bin_x, reduction='none')
            recon_loss = recon_loss + bin_nll.sum(dim=(1, 2))

        bnd_idx = self.var_config.bounded_indices
        if bnd_idx:
            bnd_recon = recon_batch[:, :, bnd_idx]
            bnd_x = batch_data[:, :, bnd_idx].clamp(1e-6, 1 - 1e-6)
            bounded_loss_type = self.var_config.bounded_loss

            if bounded_loss_type == "bce":
                bnd_recon_c = bnd_recon.clamp(1e-7, 1 - 1e-7)
                bnd_nll = F.binary_cross_entropy(bnd_recon_c, bnd_x, reduction='none')
            elif bounded_loss_type == "beta":
                log_phi = getattr(self.model, 'log_bounded_precision', None)
                if log_phi is None:
                    raise ValueError("Beta bounded loss requires log_bounded_precision.")
                mu_b = bnd_recon.clamp(1e-4, 1 - 1e-4)
                log_phi = log_phi.clamp(-4.0, 6.0).view(1, 1, -1)
                phi = log_phi.exp()
                alpha = mu_b * phi
                beta_param = (1 - mu_b) * phi
                bnd_nll = (
                    torch.lgamma(alpha)
                    + torch.lgamma(beta_param)
                    - torch.lgamma(alpha + beta_param)
                    - (alpha - 1) * torch.log(bnd_x)
                    - (beta_param - 1) * torch.log(1 - bnd_x)
                )
            elif bounded_loss_type == "logit_normal":
                log_bounded_var = getattr(self.model, 'log_bounded_var', None)
                if log_bounded_var is None:
                    raise ValueError("Logit-normal bounded loss requires log_bounded_var.")
                logit_x = torch.log(bnd_x / (1 - bnd_x))
                lnv = log_bounded_var.clamp(-6.0, 6.0).view(1, 1, -1)
                bnd_nll = 0.5 * (lnv + (logit_x - bnd_recon) ** 2 / lnv.exp())
                bnd_nll = bnd_nll - torch.log(bnd_x) - torch.log(1 - bnd_x)
            else:
                raise ValueError(f"Unsupported bounded_loss '{bounded_loss_type}'.")

            recon_loss = recon_loss + bnd_nll.sum(dim=(1, 2))

        return recon_loss

    def _imputation_log_target(self, batch_data, batch_mask, baseline_arg):
        """Approximate log target for missing-data RWMH updates.

        Uses a deterministic ELBO-style score on the fully imputed sequence:
        log target ∝ -reconstruction_nll(x_tilde | z=mu(x_tilde)) - beta * KL(q||p).
        """
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            recon_batch, mu, logvar = self._deterministic_reconstruction(
                batch_data, batch_mask, baseline_arg
            )
            recon_nll = self._reconstruction_nll_per_sample(recon_batch, batch_data)
            prior_chol = getattr(
                self.model, 'get_latent_prior_cholesky', lambda **_: None
            )(device=batch_data.device, dtype=batch_data.dtype)
            if prior_chol is None:
                kld = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(),
                    dim=1,
                )
            else:
                prior_precision = torch.cholesky_inverse(prior_chol)
                q_var = logvar.exp()
                trace_term = torch.sum(
                    q_var * torch.diagonal(prior_precision, dim1=-2, dim2=-1),
                    dim=1,
                )
                quad_term = torch.sum((mu @ prior_precision) * mu, dim=1)
                latent_dim = mu.size(1)
                logdet_prior = 2.0 * torch.sum(torch.log(torch.diagonal(prior_chol)))
                logdet_q = torch.sum(logvar, dim=1)
                kld = 0.5 * (
                    trace_term + quad_term - latent_dim + logdet_prior - logdet_q
                )
            score = -(recon_nll + self.beta * kld)
        if was_training:
            self.model.train()
        return score

    @staticmethod
    def _reflect_to_interval(values, lower, upper):
        """Reflect proposals back into [lower, upper] to preserve symmetry."""
        width = upper - lower
        if width <= 0:
            raise ValueError("upper must be greater than lower for reflection.")
        shifted = values - lower
        period = 2 * width
        reflected = torch.remainder(shifted, period)
        reflected = torch.where(reflected <= width, reflected, period - reflected)
        return reflected + lower

    def _propose_missing_values(
        self,
        current_data,
        batch_mask,
        continuous_step_size,
        bounded_step_size,
        binary_flip_prob,
    ):
        """Symmetric random-walk proposals for missing entries only."""
        proposal = current_data.clone()
        missing = batch_mask == 0

        if self.var_config is None:
            if missing.any():
                proposal[missing] = (
                    current_data[missing]
                    + torch.randn_like(current_data[missing]) * continuous_step_size
                )
            return proposal

        for idx in self.var_config.continuous_indices:
            idx_missing = missing[:, :, idx]
            if idx_missing.any():
                proposal[:, :, idx][idx_missing] = (
                    current_data[:, :, idx][idx_missing]
                    + torch.randn_like(current_data[:, :, idx][idx_missing]) * continuous_step_size
                )

        for idx in self.var_config.binary_indices:
            idx_missing = missing[:, :, idx]
            if idx_missing.any():
                flip_mask = (
                    torch.rand_like(current_data[:, :, idx]) < binary_flip_prob
                ) & idx_missing
                proposal[:, :, idx][flip_mask] = 1.0 - current_data[:, :, idx][flip_mask]

        for idx in self.var_config.bounded_indices:
            idx_missing = missing[:, :, idx]
            if idx_missing.any():
                proposed = current_data[:, :, idx][idx_missing] + (
                    torch.randn_like(current_data[:, :, idx][idx_missing]) * bounded_step_size
                )
                eps = getattr(self.var_config, 'bounded_eps', 0.0)
                lower = eps
                upper = 1.0 - eps if eps > 0 else 1.0
                proposal[:, :, idx][idx_missing] = self._reflect_to_interval(
                    proposed, lower, upper
                )

        return proposal

    def _rwmh_impute_missing(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        mh_steps=1,
        continuous_step_size=0.1,
        bounded_step_size=0.05,
        binary_flip_prob=0.1,
    ):
        """Run random-walk Metropolis-Hastings updates for missing entries."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(current, batch_mask, baseline_arg)

        for _ in range(max(int(mh_steps), 1)):
            proposal = self._propose_missing_values(
                current,
                batch_mask,
                continuous_step_size=continuous_step_size,
                bounded_step_size=bounded_step_size,
                binary_flip_prob=binary_flip_prob,
            )
            proposal_score = self._imputation_log_target(proposal, batch_mask, baseline_arg)
            log_alpha = proposal_score - current_score
            accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
            accept = torch.rand_like(accept_prob) < accept_prob
            current[accept] = proposal[accept]
            current_score[accept] = proposal_score[accept]

        return current

    def train_epoch(self, train_loader, use_em_imputation=False,
                    em_iterations=3, stochastic_impute=True,
                    imputation_method='rwmh', mh_steps=1,
                    mh_continuous_step_size=0.1,
                    mh_bounded_step_size=0.05,
                    mh_binary_flip_prob=0.1):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            use_em_imputation: Whether to use EM-like imputation for missing data
            em_iterations: Number of EM iterations per batch (default: 3)
            stochastic_impute: If True (default), the E-step samples from the
                full observation model p(y|z) — i.e. y ~ N(m, sigma_y^2)
                for continuous and y ~ Bernoulli(p) for binary variables.
                If False, uses the deterministic mean/threshold as before.
            imputation_method: One of 'rwmh' (default) or 'direct'. 'rwmh'
                runs a random-walk Metropolis-Hastings update over missing
                values using a deterministic ELBO-style target. 'direct' keeps
                the older direct sampling from p(y|z).
            mh_steps: Number of MH proposal/accept steps per E-step.
            mh_continuous_step_size: Proposal SD for continuous variables on
                the normalized scale.
            mh_bounded_step_size: Proposal SD for bounded variables on the
                normalized [0, 1] scale.
            mh_binary_flip_prob: Proposal probability for flipping a missing
                binary state at each MH step.

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
                            if stochastic_impute:
                                if imputation_method == 'rwmh':
                                    imputed = self._rwmh_impute_missing(
                                        batch_data,
                                        batch_mask,
                                        baseline_arg,
                                        mh_steps=mh_steps,
                                        continuous_step_size=mh_continuous_step_size,
                                        bounded_step_size=mh_bounded_step_size,
                                        binary_flip_prob=mh_binary_flip_prob,
                                    )
                                elif imputation_method == 'direct':
                                    # Sample from p(y|z): older direct generative draw
                                    imputed = self._sample_from_observation_model(
                                        recon_batch
                                    )
                                else:
                                    raise ValueError(
                                        "imputation_method must be 'rwmh' or 'direct', "
                                        f"got '{imputation_method}'."
                                    )
                            else:
                                # Deterministic: use mean / threshold
                                imputed = recon_batch.clone()
                                if self.var_config is not None:
                                    for idx in self.var_config.binary_indices:
                                        imputed[:, :, idx] = (
                                            imputed[:, :, idx] > 0.5
                                        ).float()
                                    for idx in self.var_config.bounded_indices:
                                        bl = getattr(
                                            self.var_config, 'bounded_loss', 'bce'
                                        )
                                        if bl == 'logit_normal':
                                            imputed[:, :, idx] = torch.sigmoid(
                                                imputed[:, :, idx]
                                            )
                                        else:
                                            imputed[:, :, idx] = (
                                                imputed[:, :, idx].clamp(0, 1)
                                            )
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
            use_em_imputation=False, em_iterations=3, patience=0,
            stochastic_impute=True, imputation_method='rwmh', mh_steps=1,
            mh_continuous_step_size=0.1, mh_bounded_step_size=0.05,
            mh_binary_flip_prob=0.1):
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
            stochastic_impute: If True (default), the EM E-step samples from
                the observation model p(y|z) rather than using the
                deterministic mean.  This properly propagates observation-level
                uncertainty into the imputed values.
            imputation_method: One of 'rwmh' (default) or 'direct'.
            mh_steps: Number of MH updates per E-step when using RWMH.
            mh_continuous_step_size: Proposal SD for continuous variables.
            mh_bounded_step_size: Proposal SD for bounded variables.
            mh_binary_flip_prob: Proposal flip probability for binary variables.

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
            train_loss, train_recon, train_kld = self.train_epoch(
                train_loader, use_em_imputation, em_iterations,
                stochastic_impute=stochastic_impute,
                imputation_method=imputation_method,
                mh_steps=mh_steps,
                mh_continuous_step_size=mh_continuous_step_size,
                mh_bounded_step_size=mh_bounded_step_size,
                mh_binary_flip_prob=mh_binary_flip_prob,
            )
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
                msg = (
                    f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} '
                    f'(Recon: {train_recon:.4f}, KLD: {train_kld:.4f})'
                )
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
