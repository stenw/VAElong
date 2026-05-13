"""
Training utilities for VAE.
"""

import math

import torch
import torch.optim as optim
import torch.nn.functional as F
from .model import mixed_vae_loss_function, gaussian_kl_divergence
from torch.utils.data import DataLoader, Dataset


class _IndexedDataset(Dataset):
    """Wraps a base dataset to also yield the sample's dataset index.

    Used internally by ``VAETrainer.fit`` when adaptive RWMH monitoring is
    enabled so that per-individual acceptance counts can be aggregated against
    a stable identifier rather than a per-epoch batch position.
    """

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
    """Per-variable RWMH proposal state with optional Robbins-Monro adaptation.

    Maintains independent step sizes (continuous / bounded SDs and binary flip
    probabilities) per variable, plus accept/propose counters at both the
    per-variable and per-individual level. When ``adaptive`` is True the step
    sizes are updated after each MH step via a Robbins-Monro recursion that
    targets ``target_rate``.

    The recursion runs on log-scale for SDs (to stay positive) and on
    logit-scale for flip probabilities (to stay in (0, 1)). Updates are
    clamped to [step_min, step_max] / [flip_min, flip_max].
    """

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
        self.t = 0  # global update counter (across all variables)

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
            # Single global continuous step when no var_config is supplied.
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
        if var_type == 'continuous':
            return self.cont_steps[var_idx]
        if var_type == 'bounded':
            return self.bnd_steps[var_idx]
        if var_type == 'binary':
            return self.bin_flips[var_idx]
        raise ValueError(f"Unknown var_type '{var_type}'")

    def record(self, var_idx, var_type, accept_mask, propose_mask, indices=None):
        """Update counters and optionally the per-individual matrix.

        ``accept_mask`` and ``propose_mask`` are bool tensors of shape
        ``(batch_size,)``. ``indices`` (when not None) are the dataset indices
        for each row in the batch.
        """
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
        if var_type in ('continuous', 'bounded'):
            store = self.cont_steps if var_type == 'continuous' else self.bnd_steps
            current = max(store[var_idx], 1e-12)
            new_step = math.exp(math.log(current) + gamma * delta)
            store[var_idx] = max(self.step_min, min(self.step_max, new_step))
        elif var_type == 'binary':
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
            rate = (n_acc / n_prop) if n_prop > 0 else float('nan')
            per_var[idx] = {'rate': rate, 'accept': n_acc, 'propose': n_prop}
        result = {
            'per_variable': per_var,
            'step_sizes': {
                'continuous': dict(self.cont_steps),
                'bounded': dict(self.bnd_steps),
                'binary_flip_prob': dict(self.bin_flips),
            },
            'updates': self.t,
        }
        if self.ind_accepts is not None:
            prop = self.ind_proposes.float()
            acc = self.ind_accepts.float()
            rates = torch.where(prop > 0, acc / prop.clamp(min=1.0),
                                torch.full_like(prop, float('nan')))
            result['per_individual'] = {
                'accepts': self.ind_accepts.clone(),
                'proposes': self.ind_proposes.clone(),
                'rates': rates,
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

        # Optional adaptive RWMH state — populated by ``fit`` when
        # ``mh_adaptive=True``. ``mh_history`` collects per-epoch acceptance
        # snapshots.
        self.mh_state = None
        self.mh_history = []

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

    def _model_forward(self, x, mask, baseline, times=None):
        """Call ``self.model`` while tolerating models without ``times`` kwarg.

        Only ``LongitudinalVAE`` (with ``time_in_decoder=True``) currently uses
        the per-batch times tensor; CNN/TPCNN variants take only positional
        information and reject the kwarg.
        """
        try:
            return self.model(x, mask, baseline, times=times)
        except TypeError:
            return self.model(x, mask, baseline)

    @staticmethod
    def _unpack_batch(batch):
        """Return ``(data, mask, lengths, baseline, times, indices)``.

        Supports both 4-tuple (legacy datasets without times), 5-tuple
        (current dataset including times) and 6-tuple (5-tuple wrapped by
        ``_IndexedDataset``). When times are missing they are synthesised
        as positional indices broadcast across the batch.
        """
        if len(batch) == 6:
            return batch
        if len(batch) == 5:
            data, mask, lengths, baseline, times = batch
            return data, mask, lengths, baseline, times, None
        if len(batch) == 4:
            data, mask, lengths, baseline = batch
            n, T = data.shape[0], data.shape[1]
            times = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(n, -1)
            return data, mask, lengths, baseline, times, None
        raise ValueError(f"Unexpected batch tuple of length {len(batch)}")

    def _deterministic_reconstruction(self, batch_data, batch_mask, baseline_arg, times=None):
        """Decode from the posterior mean for a deterministic imputation score."""
        mu, logvar = self.model.encode(batch_data, batch_mask, baseline_arg)
        try:
            recon_batch = self.model.decode(mu, batch_data.shape[1], baseline_arg, times=times)
        except TypeError:
            # CNN/TPCNN decoders take (z, baseline) and ignore times.
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

    def _imputation_log_target(self, batch_data, batch_mask, baseline_arg, times=None):
        """Approximate log target for missing-data RWMH updates.

        Uses a deterministic ELBO-style score on the fully imputed sequence:
        log target ∝ -reconstruction_nll(x_tilde | z=mu(x_tilde)) - beta * KL(q||p).
        """
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            recon_batch, mu, logvar = self._deterministic_reconstruction(
                batch_data, batch_mask, baseline_arg, times=times,
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
        times=None,
    ):
        """Run random-walk Metropolis-Hastings updates for missing entries."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(current, batch_mask, baseline_arg, times=times)

        for _ in range(max(int(mh_steps), 1)):
            proposal = self._propose_missing_values(
                current,
                batch_mask,
                continuous_step_size=continuous_step_size,
                bounded_step_size=bounded_step_size,
                binary_flip_prob=binary_flip_prob,
            )
            proposal_score = self._imputation_log_target(proposal, batch_mask, baseline_arg, times=times)
            log_alpha = proposal_score - current_score
            accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
            accept = torch.rand_like(accept_prob) < accept_prob
            current[accept] = proposal[accept]
            current_score[accept] = proposal_score[accept]

        return current

    def _propose_single_variable(self, current, batch_mask, var_idx, var_type, step):
        """Symmetric proposal for missing entries of a single variable.

        Returns ``(proposal, has_missing_per_individual)`` where the second
        tensor is a (batch,) bool flagging individuals with at least one
        missing entry for this variable (so acceptance is only counted where
        a real proposal was made).
        """
        proposal = current.clone()
        if var_idx is None:
            # No var_config: treat all entries as continuous Gaussian proposals.
            missing = batch_mask == 0
            if not missing.any():
                return (
                    proposal,
                    torch.zeros(current.shape[0], dtype=torch.bool, device=current.device),
                )
            noise = torch.randn_like(current) * step
            proposal = torch.where(missing, current + noise, current)
            return proposal, missing.view(current.shape[0], -1).any(dim=1)

        missing = batch_mask[:, :, var_idx] == 0  # (batch, seq_len)
        has_missing_per_indiv = missing.any(dim=1)
        if not missing.any():
            return proposal, has_missing_per_indiv

        cur_slice = current[:, :, var_idx]
        if var_type == 'continuous':
            new_slice = cur_slice + torch.randn_like(cur_slice) * step
        elif var_type == 'bounded':
            candidate = cur_slice + torch.randn_like(cur_slice) * step
            eps = getattr(self.var_config, 'bounded_eps', 0.0) if self.var_config else 0.0
            lower = eps
            upper = 1.0 - eps if eps > 0 else 1.0
            new_slice = self._reflect_to_interval(candidate, lower, upper)
        elif var_type == 'binary':
            flip = (torch.rand_like(cur_slice) < step) & missing
            new_slice = torch.where(flip, 1.0 - cur_slice, cur_slice)
        else:
            raise ValueError(f"Unknown var_type '{var_type}'")

        proposal[:, :, var_idx] = torch.where(missing, new_slice, cur_slice)
        return proposal, has_missing_per_indiv

    def _variable_specs(self):
        """Return list of ``(var_idx, var_type)`` covered by per-variable MH."""
        if self.var_config is None:
            return [(None, 'continuous')]
        specs = []
        for idx in self.var_config.continuous_indices:
            specs.append((idx, 'continuous'))
        for idx in self.var_config.bounded_indices:
            specs.append((idx, 'bounded'))
        for idx in self.var_config.binary_indices:
            specs.append((idx, 'binary'))
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
    ):
        """Per-variable Metropolis-within-Gibbs RWMH over missing entries.

        Sweeps over each variable separately, proposing+accepting one variable
        at a time. Step sizes come from ``state``; per-variable and (when
        ``indices`` is provided) per-individual acceptance counts are recorded
        on ``state``. When ``state.adaptive`` is True the step sizes are also
        updated via Robbins-Monro after each per-variable step.
        """
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(current, batch_mask, baseline_arg, times=times)
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
                    imputation_method='rwmh', mh_steps=1,
                    mh_continuous_step_size=0.1,
                    mh_bounded_step_size=0.05,
                    mh_binary_flip_prob=0.1,
                    mh_adaptive=True):
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
            batch_data, batch_mask, _, batch_baseline, batch_times, batch_indices = (
                self._unpack_batch(batch)
            )
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            batch_times = batch_times.to(self.device)
            baseline_arg = self._get_baseline_arg(batch_baseline)

            # Check if there's any missing data
            has_missing = (batch_mask.sum() < batch_mask.numel())

            if use_em_imputation and has_missing:
                # EM-like approach: alternate between imputation and parameter estimation
                for em_iter in range(em_iterations):
                    # E-step: Impute missing values
                    if em_iter > 0:  # Skip first iteration, use initial values
                        with torch.no_grad():
                            recon_batch, mu_temp, logvar_temp = self._model_forward(
                                batch_data, batch_mask, baseline_arg, times=batch_times,
                            )
                            if stochastic_impute:
                                if imputation_method == 'rwmh':
                                    if mh_adaptive and self.mh_state is not None:
                                        imputed = self._rwmh_impute_missing_per_variable(
                                            batch_data,
                                            batch_mask,
                                            baseline_arg,
                                            state=self.mh_state,
                                            mh_steps=mh_steps,
                                            indices=batch_indices,
                                            times=batch_times,
                                        )
                                    else:
                                        imputed = self._rwmh_impute_missing(
                                            batch_data,
                                            batch_mask,
                                            baseline_arg,
                                            mh_steps=mh_steps,
                                            continuous_step_size=mh_continuous_step_size,
                                            bounded_step_size=mh_bounded_step_size,
                                            binary_flip_prob=mh_binary_flip_prob,
                                            times=batch_times,
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
                    recon_batch, mu, logvar = self._model_forward(
                        batch_data, batch_mask, baseline_arg, times=batch_times,
                    )
                    loss, recon_loss, kld_loss = self._compute_loss(
                        recon_batch, batch_data, mu, logvar, batch_mask
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                # Standard training (with or without missing data mask)
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, logvar = self._model_forward(
                    batch_data, mask_arg, baseline_arg, times=batch_times,
                )
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
                batch_data, batch_mask, _, batch_baseline, batch_times, _ = (
                    self._unpack_batch(batch)
                )
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_times = batch_times.to(self.device)
                baseline_arg = self._get_baseline_arg(batch_baseline)

                # Check if there's any missing data
                has_missing = (batch_mask.sum() < batch_mask.numel())

                # Forward pass
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, logvar = self._model_forward(
                    batch_data, mask_arg, baseline_arg, times=batch_times,
                )

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

    def _wrap_loader_with_indices(self, train_loader):
        """Return a DataLoader yielding (data, mask, lengths, baseline, idx).

        Preserves the original loader's batch size, shuffle behaviour, worker
        count and pin_memory setting. Used when adaptive MH monitoring is on
        so per-individual acceptance can be aggregated against stable dataset
        indices.
        """
        sampler = getattr(train_loader, 'sampler', None)
        shuffle = (
            sampler is None or isinstance(sampler, torch.utils.data.RandomSampler)
        )
        return DataLoader(
            _IndexedDataset(train_loader.dataset),
            batch_size=train_loader.batch_size,
            shuffle=shuffle,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=getattr(train_loader, 'pin_memory', False),
            drop_last=getattr(train_loader, 'drop_last', False),
        )

    def fit(self, train_loader, val_loader=None, epochs=100, verbose=True,
            use_em_imputation=False, em_iterations=3, patience=0,
            stochastic_impute=True, imputation_method='rwmh', mh_steps=1,
            mh_continuous_step_size=0.1, mh_bounded_step_size=0.05,
            mh_binary_flip_prob=0.1,
            mh_adaptive=True, mh_target_accept=0.234,
            mh_rm_decay=0.6, mh_rm_offset=10.0,
            mh_step_min=1e-4, mh_step_max=2.0,
            mh_flip_min=1e-3, mh_flip_max=0.5,
            mh_track_per_individual=True):
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
            mh_adaptive: If True (default) switch the RWMH update to a
                per-variable Metropolis-within-Gibbs scheme and adapt each
                variable's proposal scale on the fly via a Robbins-Monro
                recursion aimed at ``mh_target_accept``. Acceptance counts are
                tracked per variable and (when ``mh_track_per_individual`` is
                True) per individual. Only takes effect when
                ``imputation_method='rwmh'`` and ``use_em_imputation=True``.
            mh_target_accept: Target acceptance rate for the Robbins-Monro
                update. ``0.234`` is the Roberts/Rosenthal optimum for
                high-dimensional RWMH; use ~0.44 for very low-dimensional
                proposals.
            mh_rm_decay: Robbins-Monro decay exponent (alpha). Standard
                guidance: alpha in (0.5, 1.0]; smaller values adapt faster but
                with more noise. Default 0.6.
            mh_rm_offset: Robbins-Monro offset constant added to the iteration
                counter so initial step sizes do not change too aggressively.
            mh_step_min, mh_step_max: Clamps on continuous / bounded SDs.
            mh_flip_min, mh_flip_max: Clamps on binary flip probabilities.
            mh_track_per_individual: If True (default) the trainer wraps the
                training dataset so each batch carries its dataset indices,
                enabling per-individual acceptance bookkeeping. Disable if
                wrapping the dataset is incompatible with a custom sampler.

        Returns:
            history: Dictionary containing training history. When
                ``mh_adaptive=True`` an additional ``mh_acceptance`` entry
                holds a per-epoch list of summaries returned by
                ``_MHAdaptiveState.summary()``.
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

        # Set up adaptive MH state (per-variable step sizes + counters)
        # and optionally wrap the loader so batches carry dataset indices.
        adaptive_active = (
            mh_adaptive
            and imputation_method == 'rwmh'
            and use_em_imputation
        )
        if adaptive_active:
            n_features = None
            if self.var_config is not None:
                n_features = self.var_config.n_features
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
            history['mh_acceptance'] = []
        else:
            self.mh_state = None
        self.mh_history = []

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            if adaptive_active and self.mh_state is not None:
                # Reset counters between epochs so each entry in
                # ``history['mh_acceptance']`` reflects that epoch only.
                self.mh_state.reset_counters()

            # Train
            train_loss, train_recon, train_kld = self.train_epoch(
                train_loader, use_em_imputation, em_iterations,
                stochastic_impute=stochastic_impute,
                imputation_method=imputation_method,
                mh_steps=mh_steps,
                mh_continuous_step_size=mh_continuous_step_size,
                mh_bounded_step_size=mh_bounded_step_size,
                mh_binary_flip_prob=mh_binary_flip_prob,
                mh_adaptive=adaptive_active,
            )
            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_kld'].append(train_kld)

            if adaptive_active and self.mh_state is not None:
                snapshot = self.mh_state.summary()
                history['mh_acceptance'].append(snapshot)
                self.mh_history.append(snapshot)

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
