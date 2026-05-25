"""
Training utilities for VAE.
"""

import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .model import mixed_vae_loss_function, gaussian_kl_divergence_per_sample


class _IndexedDataset(Dataset):
    """Wrap a dataset so each sample also carries its dataset index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, tuple):
            return item + (idx,)
        return (item, idx)


class _MHAdaptiveState:
    """Per-variable adaptive state for the missing-value RWMH sampler."""

    def __init__(self, var_config, init_continuous_step, init_bounded_step,
                 init_binary_flip, target_rate, rm_decay, rm_offset,
                 step_min, step_max, flip_min, flip_max,
                 adaptive, n_individuals=None, n_features=None):
        self.var_config = var_config
        self.target_rate = float(target_rate)
        self.rm_decay = float(rm_decay)
        self.rm_offset = float(rm_offset)
        self.step_min = float(step_min)
        self.step_max = float(step_max)
        self.flip_min = float(flip_min)
        self.flip_max = float(flip_max)
        self.adaptive = bool(adaptive)
        self.t = 0

        self.var_accepts = {}
        self.var_proposes = {}
        self.cont_steps = {}
        self.bnd_steps = {}
        self.bin_flips = {}

        if var_config is not None:
            for idx in var_config.continuous_indices:
                self.cont_steps[idx] = float(init_continuous_step)
                self.var_accepts[idx] = 0
                self.var_proposes[idx] = 0
            for idx in var_config.bounded_indices:
                self.bnd_steps[idx] = float(init_bounded_step)
                self.var_accepts[idx] = 0
                self.var_proposes[idx] = 0
            for idx in var_config.binary_indices:
                self.bin_flips[idx] = float(init_binary_flip)
                self.var_accepts[idx] = 0
                self.var_proposes[idx] = 0
        else:
            self.cont_steps[None] = float(init_continuous_step)
            self.var_accepts[None] = 0
            self.var_proposes[None] = 0

        if n_individuals is not None and n_features is not None:
            self.ind_accepts = torch.zeros(n_individuals, n_features, dtype=torch.long)
            self.ind_proposes = torch.zeros(n_individuals, n_features, dtype=torch.long)
        else:
            self.ind_accepts = None
            self.ind_proposes = None

    def get_step(self, var_idx, var_type):
        if var_type == "continuous":
            return self.cont_steps[var_idx]
        if var_type == "bounded":
            return self.bnd_steps[var_idx]
        if var_type == "binary":
            return self.bin_flips[var_idx]
        raise ValueError(f"Unknown var_type '{var_type}'")

    def record(self, var_idx, var_type, accept_mask, propose_mask, indices=None):
        n_accept = int(accept_mask.sum().item())
        n_propose = int(propose_mask.sum().item())
        self.var_accepts[var_idx] = self.var_accepts.get(var_idx, 0) + n_accept
        self.var_proposes[var_idx] = self.var_proposes.get(var_idx, 0) + n_propose

        if self.ind_accepts is not None and indices is not None and var_idx is not None:
            idx_cpu = indices.detach().cpu().long()
            accept_cpu = accept_mask.detach().cpu().long()
            propose_cpu = propose_mask.detach().cpu().long()
            self.ind_accepts[idx_cpu, var_idx] += accept_cpu
            self.ind_proposes[idx_cpu, var_idx] += propose_cpu

        if self.adaptive and n_propose > 0:
            self._rm_update(var_idx, var_type, n_accept / n_propose)
        self.t += 1

    def _rm_update(self, var_idx, var_type, accept_rate):
        gamma = 1.0 / ((self.t + self.rm_offset) ** self.rm_decay)
        delta = accept_rate - self.target_rate
        if var_type in ("continuous", "bounded"):
            store = self.cont_steps if var_type == "continuous" else self.bnd_steps
            current = max(store[var_idx], 1e-12)
            new_step = math.exp(math.log(current) + gamma * delta)
            store[var_idx] = max(self.step_min, min(self.step_max, new_step))
        elif var_type == "binary":
            current = min(max(self.bin_flips[var_idx], 1e-6), 1 - 1e-6)
            logit = math.log(current / (1.0 - current))
            new_p = 1.0 / (1.0 + math.exp(-(logit + gamma * delta)))
            self.bin_flips[var_idx] = max(self.flip_min, min(self.flip_max, new_p))

    def reset_counters(self):
        for k in self.var_accepts:
            self.var_accepts[k] = 0
            self.var_proposes[k] = 0
        if self.ind_accepts is not None:
            self.ind_accepts.zero_()
            self.ind_proposes.zero_()

    def summary(self):
        per_var = {}
        for idx, n_acc in self.var_accepts.items():
            n_prop = self.var_proposes.get(idx, 0)
            rate = (n_acc / n_prop) if n_prop > 0 else float("nan")
            per_var[idx] = {"rate": rate, "accept": n_acc, "propose": n_prop}
        result = {
            "per_variable": per_var,
            "step_sizes": {
                "continuous": dict(self.cont_steps),
                "bounded": dict(self.bnd_steps),
                "binary_flip_prob": dict(self.bin_flips),
            },
            "updates": self.t,
        }
        if self.ind_accepts is not None:
            prop = self.ind_proposes.float()
            acc = self.ind_accepts.float()
            rates = torch.where(
                prop > 0,
                acc / prop.clamp(min=1.0),
                torch.full_like(prop, float("nan")),
            )
            result["per_individual"] = {
                "accepts": self.ind_accepts.clone(),
                "proposes": self.ind_proposes.clone(),
                "rates": rates,
            }
        return result


class VAETrainer:
    """
    Trainer class for Longitudinal VAE.

    Args:
        model: LongitudinalVAE or CNNLongitudinalVAE model instance
        learning_rate: Learning rate for optimizer (default: 1e-3)
        beta: Weight for KL divergence term (default: 1.0)
        device: Device to train on (default: 'cuda' if available else 'cpu')
        var_config: Optional VariableConfig for mixed-type loss computation
        noise_var_penalty: L2 penalty weight on log_noise_var
        weight_decay: L2 regularisation on model weights via AdamW-style decay
    """

    def __init__(self, model, learning_rate=1e-3, beta=1.0, device=None,
                 var_config=None, noise_var_penalty=1.0, weight_decay=0.0):
        self.model = model
        self.beta = beta
        self.var_config = var_config
        self.noise_var_penalty = noise_var_penalty

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.train_losses = []
        self.val_losses = []
        self.mh_state = None
        self.mh_history = []

    def _get_baseline_arg(self, batch_baseline):
        """Return baseline tensor or None if no baseline features."""
        if batch_baseline.shape[-1] > 0:
            return batch_baseline.to(self.device)
        return None

    def _get_log_noise_var(self):
        """Return the model's learned log_noise_var or None."""
        return getattr(self.model, "log_noise_var", None)

    @staticmethod
    def _resolve_imputation_method(imputation_method):
        """Validate the requested EM imputation strategy."""
        if imputation_method not in {"rwmh", "latent"}:
            raise ValueError(
                "imputation_method must be 'rwmh' or 'latent'; direct "
                "observation-model sampling is no longer supported."
            )
        return imputation_method

    def _compute_loss(self, recon_batch, batch_data, mu, posterior_params, mask_arg):
        """Compute mixed VAE loss, passing through learned parameters."""
        return mixed_vae_loss_function(
            recon_batch, batch_data, mu, posterior_params, self.beta, mask_arg,
            self.var_config, self._get_log_noise_var(),
            noise_var_penalty=self.noise_var_penalty,
            log_bounded_precision=getattr(self.model, "log_bounded_precision", None),
            log_bounded_var=getattr(self.model, "log_bounded_var", None),
            latent_prior_cholesky=getattr(
                self.model, "get_latent_prior_cholesky", lambda **_: None
            )(device=batch_data.device, dtype=batch_data.dtype),
            latent_posterior_type=getattr(
                self.model, "latent_posterior_type", "diagonal"
            ),
            latent_posterior_rank=getattr(self.model, "latent_posterior_rank", 0),
        )

    def _sample_from_observation_model(self, recon_batch):
        """Sample one draw from p(y | z) given decoder output parameters."""
        sampled = recon_batch.clone()

        if self.var_config is None:
            log_nv = self._get_log_noise_var()
            if log_nv is None:
                return sampled + torch.randn_like(sampled)
            sigma = (0.5 * log_nv.clamp(-6.0, 6.0)).exp().view(1, 1, -1)
            return sampled + torch.randn_like(sampled) * sigma

        log_nv = self._get_log_noise_var()
        if len(self.var_config.continuous_indices) > 0:
            if log_nv is None:
                for idx in self.var_config.continuous_indices:
                    sampled[:, :, idx] = sampled[:, :, idx] + torch.randn_like(
                        sampled[:, :, idx]
                    )
            else:
                sigma = (0.5 * log_nv.clamp(-6.0, 6.0)).exp()
                for k, idx in enumerate(self.var_config.continuous_indices):
                    sampled[:, :, idx] = sampled[:, :, idx] + torch.randn_like(
                        sampled[:, :, idx]
                    ) * sigma[k]

        for idx in self.var_config.binary_indices:
            prob = sampled[:, :, idx].clamp(1e-6, 1 - 1e-6)
            sampled[:, :, idx] = torch.bernoulli(prob)

        for k, idx in enumerate(self.var_config.bounded_indices):
            bounded_loss = getattr(self.var_config, "bounded_loss", "bce")
            if bounded_loss == "beta":
                log_phi = getattr(self.model, "log_bounded_precision", None)
                if log_phi is None:
                    raise ValueError("Beta bounded loss requires log_bounded_precision.")
                mu_b = sampled[:, :, idx].clamp(1e-4, 1 - 1e-4)
                phi = log_phi.clamp(-4.0, 6.0).exp()[k]
                alpha = mu_b * phi
                beta_param = (1 - mu_b) * phi
                sampled[:, :, idx] = torch.distributions.Beta(alpha, beta_param).sample()
            elif bounded_loss == "logit_normal":
                log_bounded_var = getattr(self.model, "log_bounded_var", None)
                if log_bounded_var is None:
                    raise ValueError(
                        "Logit-normal bounded loss requires log_bounded_var."
                    )
                sigma = torch.exp(0.5 * log_bounded_var.clamp(-6.0, 6.0))[k]
                latent_draw = sampled[:, :, idx] + torch.randn_like(
                    sampled[:, :, idx]
                ) * sigma
                sampled[:, :, idx] = torch.sigmoid(latent_draw)
            else:
                sampled[:, :, idx] = sampled[:, :, idx].clamp(0, 1)

        return sampled

    def _model_forward(self, x, mask, baseline, times=None,
                       time_varying_covariates=None):
        """Call ``self.model`` while tolerating newer optional kwargs."""
        try:
            return self.model(
                x, mask, baseline, times=times,
                time_varying_covariates=time_varying_covariates,
            )
        except TypeError:
            try:
                return self.model(x, mask, baseline, times=times)
            except TypeError:
                return self.model(x, mask, baseline)

    def _decode_from_latent(self, z, seq_len, baseline_arg, times=None,
                            time_varying_covariates=None):
        """Decode a provided latent sample across model variants."""
        try:
            return self.model.decode(
                z, seq_len, baseline_arg, times=times,
                time_varying_covariates=time_varying_covariates,
            )
        except TypeError:
            try:
                return self.model.decode(z, seq_len, baseline_arg)
            except TypeError:
                return self.model.decode(z, baseline_arg)

    @staticmethod
    def _unpack_batch(batch):
        """Return ``(data, mask, lengths, baseline, times, tv_covs, indices)``."""
        if len(batch) == 7:
            return batch
        if len(batch) == 6:
            data, mask, lengths, baseline, fifth, sixth = batch
            if isinstance(sixth, torch.Tensor) and sixth.dim() >= 2:
                return data, mask, lengths, baseline, fifth, sixth, None
            n = data.shape[0]
            T = data.shape[1]
            tv_covs = data.new_zeros((n, T, 0))
            return data, mask, lengths, baseline, fifth, tv_covs, sixth
        if len(batch) == 5:
            data, mask, lengths, baseline, times = batch
            n = data.shape[0]
            T = data.shape[1]
            tv_covs = data.new_zeros((n, T, 0))
            return data, mask, lengths, baseline, times, tv_covs, None
        if len(batch) == 4:
            data, mask, lengths, baseline = batch
            n, T = data.shape[0], data.shape[1]
            times = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(n, -1)
            tv_covs = data.new_zeros((n, T, 0))
            return data, mask, lengths, baseline, times, tv_covs, None
        raise ValueError(f"Unexpected batch tuple of length {len(batch)}")

    def _deterministic_reconstruction(self, batch_data, batch_mask, baseline_arg,
                                      times=None, time_varying_covariates=None):
        """Decode from the posterior mean for a deterministic imputation score."""
        try:
            mu, posterior_params = self.model.encode(
                batch_data, batch_mask, baseline_arg, times=times,
                time_varying_covariates=time_varying_covariates,
            )
        except TypeError:
            try:
                mu, posterior_params = self.model.encode(
                    batch_data, batch_mask, baseline_arg, times=times,
                )
            except TypeError:
                mu, posterior_params = self.model.encode(
                    batch_data, batch_mask, baseline_arg
                )
        recon_batch = self._decode_from_latent(
            mu,
            batch_data.shape[1],
            baseline_arg,
            times=times,
            time_varying_covariates=time_varying_covariates,
        )
        return recon_batch, mu, posterior_params

    def _latent_space_impute_missing(self, batch_data, batch_mask, baseline_arg,
                                     stochastic_impute=True, times=None,
                                     time_varying_covariates=None):
        """Algorithm 1-style E-step: sample z, then sample missing y | z."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        try:
            mu, posterior_params = self.model.encode(
                batch_data, batch_mask, baseline_arg, times=times,
                time_varying_covariates=time_varying_covariates,
            )
        except TypeError:
            try:
                mu, posterior_params = self.model.encode(
                    batch_data, batch_mask, baseline_arg, times=times,
                )
            except TypeError:
                mu, posterior_params = self.model.encode(
                    batch_data, batch_mask, baseline_arg
                )

        z = self.model.reparameterize(mu, posterior_params) if stochastic_impute else mu
        recon_batch = self._decode_from_latent(
            z,
            batch_data.shape[1],
            baseline_arg,
            times=times,
            time_varying_covariates=time_varying_covariates,
        )
        imputed = (
            self._sample_from_observation_model(recon_batch)
            if stochastic_impute else recon_batch
        )

        if not stochastic_impute and self.var_config is not None:
            for idx in self.var_config.binary_indices:
                imputed[:, :, idx] = (imputed[:, :, idx] > 0.5).float()
            for idx in self.var_config.bounded_indices:
                if getattr(self.var_config, "bounded_loss", "bce") == "logit_normal":
                    imputed[:, :, idx] = torch.sigmoid(imputed[:, :, idx])
                else:
                    imputed[:, :, idx] = imputed[:, :, idx].clamp(0, 1)

        return batch_mask * batch_data + (1 - batch_mask) * imputed

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
            bin_nll = F.binary_cross_entropy(bin_recon, bin_x, reduction="none")
            recon_loss = recon_loss + bin_nll.sum(dim=(1, 2))

        bnd_idx = self.var_config.bounded_indices
        if bnd_idx:
            bnd_recon = recon_batch[:, :, bnd_idx]
            bnd_x = batch_data[:, :, bnd_idx].clamp(1e-6, 1 - 1e-6)
            bounded_loss_type = self.var_config.bounded_loss

            if bounded_loss_type == "bce":
                bnd_recon_c = bnd_recon.clamp(1e-7, 1 - 1e-7)
                bnd_nll = F.binary_cross_entropy(bnd_recon_c, bnd_x, reduction="none")
            elif bounded_loss_type == "beta":
                log_phi = getattr(self.model, "log_bounded_precision", None)
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
                log_bounded_var = getattr(self.model, "log_bounded_var", None)
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

    def _imputation_log_target(self, batch_data, batch_mask, baseline_arg,
                               times=None, time_varying_covariates=None):
        """Approximate log target for missing-value RWMH updates."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            recon_batch, mu, posterior_params = self._deterministic_reconstruction(
                batch_data, batch_mask, baseline_arg, times=times,
                time_varying_covariates=time_varying_covariates,
            )
            recon_nll = self._reconstruction_nll_per_sample(recon_batch, batch_data)
            prior_chol = getattr(
                self.model, "get_latent_prior_cholesky", lambda **_: None
            )(device=batch_data.device, dtype=batch_data.dtype)
            kld = gaussian_kl_divergence_per_sample(
                mu,
                posterior_params,
                prior_cholesky=prior_chol,
                posterior_type=getattr(self.model, "latent_posterior_type", "diagonal"),
                posterior_rank=getattr(self.model, "latent_posterior_rank", 0),
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
                eps = getattr(self.var_config, "bounded_eps", 0.0)
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
        times=None,
        time_varying_covariates=None,
    ):
        """Run random-walk Metropolis-Hastings updates for missing entries."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(
            current, batch_mask, baseline_arg, times=times,
            time_varying_covariates=time_varying_covariates,
        )

        for _ in range(max(int(mh_steps), 1)):
            proposal = self._propose_missing_values(
                current,
                batch_mask,
                continuous_step_size=continuous_step_size,
                bounded_step_size=bounded_step_size,
                binary_flip_prob=binary_flip_prob,
            )
            proposal_score = self._imputation_log_target(
                proposal, batch_mask, baseline_arg, times=times,
                time_varying_covariates=time_varying_covariates,
            )
            log_alpha = proposal_score - current_score
            accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
            accept = torch.rand_like(accept_prob) < accept_prob
            current[accept] = proposal[accept]
            current_score[accept] = proposal_score[accept]

        return current

    def _propose_single_variable(self, current, batch_mask, var_idx, var_type, step):
        """Symmetric proposal for missing entries of a single variable."""
        proposal = current.clone()
        if var_idx is None:
            missing = batch_mask == 0
            if not missing.any():
                return (
                    proposal,
                    torch.zeros(current.shape[0], dtype=torch.bool, device=current.device),
                )
            noise = torch.randn_like(current) * step
            proposal = torch.where(missing, current + noise, current)
            return proposal, missing.view(current.shape[0], -1).any(dim=1)

        missing = batch_mask[:, :, var_idx] == 0
        has_missing_per_indiv = missing.any(dim=1)
        if not missing.any():
            return proposal, has_missing_per_indiv

        cur_slice = current[:, :, var_idx]
        if var_type == "continuous":
            new_slice = cur_slice + torch.randn_like(cur_slice) * step
        elif var_type == "bounded":
            candidate = cur_slice + torch.randn_like(cur_slice) * step
            eps = getattr(self.var_config, "bounded_eps", 0.0) if self.var_config else 0.0
            lower = eps
            upper = 1.0 - eps if eps > 0 else 1.0
            new_slice = self._reflect_to_interval(candidate, lower, upper)
        elif var_type == "binary":
            flip = (torch.rand_like(cur_slice) < step) & missing
            new_slice = torch.where(flip, 1.0 - cur_slice, cur_slice)
        else:
            raise ValueError(f"Unknown var_type '{var_type}'")

        proposal[:, :, var_idx] = torch.where(missing, new_slice, cur_slice)
        return proposal, has_missing_per_indiv

    def _variable_specs(self):
        """Return list of ``(var_idx, var_type)`` covered by per-variable MH."""
        if self.var_config is None:
            return [(None, "continuous")]
        specs = []
        for idx in self.var_config.continuous_indices:
            specs.append((idx, "continuous"))
        for idx in self.var_config.bounded_indices:
            specs.append((idx, "bounded"))
        for idx in self.var_config.binary_indices:
            specs.append((idx, "binary"))
        return specs

    def _rwmh_impute_missing_per_variable(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        state,
        mh_steps=1,
        indices=None,
        times=None,
        time_varying_covariates=None,
    ):
        """Per-variable Metropolis-within-Gibbs RWMH over missing entries."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(
            current, batch_mask, baseline_arg, times=times,
            time_varying_covariates=time_varying_covariates,
        )
        var_specs = self._variable_specs()

        for _ in range(max(int(mh_steps), 1)):
            for var_idx, var_type in var_specs:
                step = state.get_step(var_idx, var_type)
                proposal, has_missing = self._propose_single_variable(
                    current, batch_mask, var_idx, var_type, step,
                )
                if not has_missing.any():
                    continue
                proposal_score = self._imputation_log_target(
                    proposal, batch_mask, baseline_arg, times=times,
                    time_varying_covariates=time_varying_covariates,
                )
                log_alpha = proposal_score - current_score
                accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
                accept = (torch.rand_like(accept_prob) < accept_prob) & has_missing
                if accept.any():
                    current[accept] = proposal[accept]
                    current_score[accept] = proposal_score[accept]
                state.record(var_idx, var_type, accept, has_missing, indices=indices)

        return current

    def train_epoch(self, train_loader, use_em_imputation=False,
                    em_iterations=3, stochastic_impute=True,
                    imputation_method="rwmh", mh_steps=1,
                    mh_continuous_step_size=0.1,
                    mh_bounded_step_size=0.05,
                    mh_binary_flip_prob=0.1,
                    mh_adaptive=True):
        """
        Train for one epoch.

        ``imputation_method='rwmh'`` samples missing values directly via
        Metropolis-Hastings. ``imputation_method='latent'`` follows Algorithm 1
        and samples in latent space before drawing missing outcomes.
        """
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0

        imputation_method = self._resolve_imputation_method(imputation_method)

        for batch in train_loader:
            batch_data, batch_mask, _, batch_baseline, batch_times, batch_tv_covs, batch_indices = (
                self._unpack_batch(batch)
            )
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            batch_times = batch_times.to(self.device)
            batch_tv_covs = batch_tv_covs.to(self.device)
            baseline_arg = self._get_baseline_arg(batch_baseline)

            has_missing = batch_mask.sum() < batch_mask.numel()

            if use_em_imputation and has_missing:
                for em_iter in range(em_iterations):
                    if em_iter > 0:
                        with torch.no_grad():
                            if stochastic_impute:
                                if imputation_method == "latent":
                                    batch_data = self._latent_space_impute_missing(
                                        batch_data,
                                        batch_mask,
                                        baseline_arg,
                                        stochastic_impute=True,
                                        times=batch_times,
                                        time_varying_covariates=batch_tv_covs,
                                    )
                                elif mh_adaptive and self.mh_state is not None:
                                    batch_data = self._rwmh_impute_missing_per_variable(
                                        batch_data,
                                        batch_mask,
                                        baseline_arg,
                                        state=self.mh_state,
                                        mh_steps=mh_steps,
                                        indices=batch_indices,
                                        times=batch_times,
                                        time_varying_covariates=batch_tv_covs,
                                    )
                                else:
                                    batch_data = self._rwmh_impute_missing(
                                        batch_data,
                                        batch_mask,
                                        baseline_arg,
                                        mh_steps=mh_steps,
                                        continuous_step_size=mh_continuous_step_size,
                                        bounded_step_size=mh_bounded_step_size,
                                        binary_flip_prob=mh_binary_flip_prob,
                                        times=batch_times,
                                        time_varying_covariates=batch_tv_covs,
                                    )
                            else:
                                recon_batch, _, _ = self._model_forward(
                                    batch_data, batch_mask, baseline_arg, times=batch_times,
                                    time_varying_covariates=batch_tv_covs,
                                )
                                imputed = recon_batch.clone()
                                if self.var_config is not None:
                                    for idx in self.var_config.binary_indices:
                                        imputed[:, :, idx] = (imputed[:, :, idx] > 0.5).float()
                                    for idx in self.var_config.bounded_indices:
                                        bl = getattr(self.var_config, "bounded_loss", "bce")
                                        if bl == "logit_normal":
                                            imputed[:, :, idx] = torch.sigmoid(imputed[:, :, idx])
                                        else:
                                            imputed[:, :, idx] = imputed[:, :, idx].clamp(0, 1)
                                batch_data = batch_mask * batch_data + (1 - batch_mask) * imputed

                    recon_batch, mu, posterior_params = self._model_forward(
                        batch_data, batch_mask, baseline_arg, times=batch_times,
                        time_varying_covariates=batch_tv_covs,
                    )
                    loss, recon_loss, kld_loss = self._compute_loss(
                        recon_batch, batch_data, mu, posterior_params, batch_mask
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, posterior_params = self._model_forward(
                    batch_data, mask_arg, baseline_arg, times=batch_times,
                    time_varying_covariates=batch_tv_covs,
                )
                loss, recon_loss, kld_loss = self._compute_loss(
                    recon_batch, batch_data, mu, posterior_params, mask_arg
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        return avg_loss, avg_recon, avg_kld

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_data, batch_mask, _, batch_baseline, batch_times, batch_tv_covs, _ = (
                    self._unpack_batch(batch)
                )
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_times = batch_times.to(self.device)
                batch_tv_covs = batch_tv_covs.to(self.device)
                baseline_arg = self._get_baseline_arg(batch_baseline)

                has_missing = batch_mask.sum() < batch_mask.numel()
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, posterior_params = self._model_forward(
                    batch_data, mask_arg, baseline_arg, times=batch_times,
                    time_varying_covariates=batch_tv_covs,
                )
                loss, recon_loss, kld_loss = self._compute_loss(
                    recon_batch, batch_data, mu, posterior_params, mask_arg
                )

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        return avg_loss, avg_recon, avg_kld

    def _wrap_loader_with_indices(self, train_loader):
        """Return a DataLoader yielding each batch plus dataset indices."""
        sampler = getattr(train_loader, "sampler", None)
        shuffle = sampler is None or isinstance(sampler, torch.utils.data.RandomSampler)
        return DataLoader(
            _IndexedDataset(train_loader.dataset),
            batch_size=train_loader.batch_size,
            shuffle=shuffle,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
            drop_last=getattr(train_loader, "drop_last", False),
        )

    def fit(self, train_loader, val_loader=None, epochs=100, verbose=True,
            use_em_imputation=False, em_iterations=3, patience=0,
            stochastic_impute=True, imputation_method="rwmh", mh_steps=1,
            mh_continuous_step_size=0.1, mh_bounded_step_size=0.05,
            mh_binary_flip_prob=0.1,
            mh_adaptive=True, mh_target_accept=0.234,
            mh_rm_decay=0.6, mh_rm_offset=10.0,
            mh_step_min=1e-4, mh_step_max=2.0,
            mh_flip_min=1e-3, mh_flip_max=0.5,
            mh_track_per_individual=True):
        """Train the model with optional EM-style missing-data updates."""
        import copy

        imputation_method = self._resolve_imputation_method(imputation_method)

        history = {
            "train_loss": [],
            "train_recon": [],
            "train_kld": [],
            "val_loss": [],
            "val_recon": [],
            "val_kld": [],
        }

        adaptive_active = (
            mh_adaptive
            and imputation_method == "rwmh"
            and use_em_imputation
        )
        if adaptive_active:
            n_features = self.var_config.n_features if self.var_config is not None else None
            n_individuals = None
            if mh_track_per_individual:
                try:
                    n_individuals = len(train_loader.dataset)
                except Exception:
                    n_individuals = None
                if n_individuals is not None and n_features is not None:
                    train_loader = self._wrap_loader_with_indices(train_loader)

            self.mh_state = _MHAdaptiveState(
                var_config=self.var_config,
                init_continuous_step=mh_continuous_step_size,
                init_bounded_step=mh_bounded_step_size,
                init_binary_flip=mh_binary_flip_prob,
                target_rate=mh_target_accept,
                rm_decay=mh_rm_decay,
                rm_offset=mh_rm_offset,
                step_min=mh_step_min,
                step_max=mh_step_max,
                flip_min=mh_flip_min,
                flip_max=mh_flip_max,
                adaptive=True,
                n_individuals=n_individuals,
                n_features=n_features,
            )
            history["mh_acceptance"] = []
        else:
            self.mh_state = None
        self.mh_history = []

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            if adaptive_active and self.mh_state is not None:
                self.mh_state.reset_counters()

            train_loss, train_recon, train_kld = self.train_epoch(
                train_loader,
                use_em_imputation,
                em_iterations,
                stochastic_impute=stochastic_impute,
                imputation_method=imputation_method,
                mh_steps=mh_steps,
                mh_continuous_step_size=mh_continuous_step_size,
                mh_bounded_step_size=mh_bounded_step_size,
                mh_binary_flip_prob=mh_binary_flip_prob,
                mh_adaptive=adaptive_active,
            )
            history["train_loss"].append(train_loss)
            history["train_recon"].append(train_recon)
            history["train_kld"].append(train_kld)

            if adaptive_active and self.mh_state is not None:
                snapshot = self.mh_state.summary()
                history["mh_acceptance"].append(snapshot)
                self.mh_history.append(snapshot)

            if val_loader is not None:
                val_loss, val_recon, val_kld = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_recon"].append(val_recon)
                history["val_kld"].append(val_kld)

                if patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(self.model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if verbose and (epoch + 1) % 10 == 0:
                msg = (
                    f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f} "
                    f"(Recon: {train_recon:.4f}, KLD: {train_kld:.4f})"
                )
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

            if patience > 0 and epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {patience} epochs)"
                    )
                break

        if patience > 0 and best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"Restored best model (val loss: {best_val_loss:.4f})")

        return history

    def save_model(self, path):
        """Save model state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
