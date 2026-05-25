"""
Training utilities for the joint longitudinal-survival VAE.
"""

from __future__ import annotations

import copy

import torch

from .trainer import VAETrainer, _MHAdaptiveState


class JointVAETrainer(VAETrainer):
    """
    Trainer for :class:`JointLongitudinalSurvivalVAE`.

    This extends the longitudinal ELBO with a right-censored survival
    log-likelihood term while preserving the existing longitudinal-only trainer
    behavior in :class:`VAETrainer`.
    """

    def __init__(
        self,
        model,
        learning_rate=1e-3,
        beta=1.0,
        device=None,
        var_config=None,
        noise_var_penalty=1.0,
        weight_decay=0.0,
        survival_loss_weight=1.0,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            beta=beta,
            device=device,
            var_config=var_config,
            noise_var_penalty=noise_var_penalty,
            weight_decay=weight_decay,
        )
        self.survival_loss_weight = float(survival_loss_weight)

    @staticmethod
    def _unpack_joint_batch(batch):
        """Return joint batch fields plus an optional dataset index."""
        if len(batch) == 10:
            (
                data,
                mask,
                lengths,
                baseline,
                times,
                time_varying_covariates,
                event_time,
                event_indicator,
                event_covariates,
                indices,
            ) = batch
            return (
                data,
                mask,
                lengths,
                baseline,
                times,
                time_varying_covariates,
                event_time,
                event_indicator,
                event_covariates,
                indices,
            )

        if len(batch) == 9:
            (
                data,
                mask,
                lengths,
                baseline,
                times,
                time_varying_covariates,
                event_time,
                event_indicator,
                event_covariates,
            ) = batch
            return (
                data,
                mask,
                lengths,
                baseline,
                times,
                time_varying_covariates,
                event_time,
                event_indicator,
                event_covariates,
                None,
            )

        raise ValueError(
            "Joint batches must be 9-tuples from JointLongitudinalSurvivalDataset "
            "or 10-tuples when wrapped with dataset indices"
        )

    def _get_event_covariates_arg(self, batch_event_covariates):
        """Return event covariates tensor or None if absent."""
        if batch_event_covariates.shape[-1] > 0:
            return batch_event_covariates.to(self.device)
        return None

    def _model_forward(
        self,
        x,
        mask,
        baseline,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        times=None,
        time_varying_covariates=None,
        return_survival_terms=False,
    ):
        """Forward through the joint model."""
        return self.model(
            x,
            mask,
            baseline,
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
            return_survival_terms=return_survival_terms,
        )

    def _deterministic_reconstruction(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        times=None,
        time_varying_covariates=None,
    ):
        """Decode from the posterior mean with survival conditioning included."""
        mu, posterior_params = self.model.encode(
            batch_data,
            batch_mask,
            baseline_arg,
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
        )
        recon_batch = self._decode_from_latent(
            mu,
            batch_data.shape[1],
            baseline_arg,
            times=times,
            time_varying_covariates=time_varying_covariates,
        )
        return recon_batch, mu, posterior_params

    def _latent_space_impute_missing(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        stochastic_impute=True,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        times=None,
        time_varying_covariates=None,
    ):
        """Algorithm 1-style E-step with the joint posterior."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        mu, posterior_params = self.model.encode(
            batch_data,
            batch_mask,
            baseline_arg,
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
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

    def _imputation_log_target(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        times=None,
        time_varying_covariates=None,
    ):
        """Approximate joint log target for missing-value RWMH updates."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            recon_batch, mu, posterior_params = self._deterministic_reconstruction(
                batch_data,
                batch_mask,
                baseline_arg,
                event_time=event_time,
                event_indicator=event_indicator,
                event_covariates=event_covariates,
                times=times,
                time_varying_covariates=time_varying_covariates,
            )
            recon_nll = self._reconstruction_nll_per_sample(recon_batch, batch_data)
            prior_chol = getattr(
                self.model, "get_latent_prior_cholesky", lambda **_: None
            )(device=batch_data.device, dtype=batch_data.dtype)
            kld = self._per_sample_kld(mu, posterior_params, prior_chol)
            survival_terms = self.model.survival_terms(
                mu,
                event_time=event_time,
                event_indicator=event_indicator,
                event_covariates=event_covariates,
                baseline=baseline_arg,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )
            survival_nll = -survival_terms["survival_log_likelihood"]
            score = -(recon_nll + self.survival_loss_weight * survival_nll + self.beta * kld)
        if was_training:
            self.model.train()
        return score

    def _per_sample_kld(self, mu, posterior_params, prior_cholesky):
        from .model import gaussian_kl_divergence_per_sample

        return gaussian_kl_divergence_per_sample(
            mu,
            posterior_params,
            prior_cholesky=prior_cholesky,
            posterior_type=getattr(self.model, "latent_posterior_type", "diagonal"),
            posterior_rank=getattr(self.model, "latent_posterior_rank", 0),
        )

    def _rwmh_impute_missing(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        mh_steps=1,
        continuous_step_size=0.1,
        bounded_step_size=0.05,
        binary_flip_prob=0.1,
        times=None,
        time_varying_covariates=None,
    ):
        """Run RWMH imputation using the joint log target."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(
            current,
            batch_mask,
            baseline_arg,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
            times=times,
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
                proposal,
                batch_mask,
                baseline_arg,
                event_time=event_time,
                event_indicator=event_indicator,
                event_covariates=event_covariates,
                times=times,
                time_varying_covariates=time_varying_covariates,
            )
            log_alpha = proposal_score - current_score
            accept_prob = torch.exp(torch.clamp(log_alpha, max=0.0))
            accept = torch.rand_like(accept_prob) < accept_prob
            current[accept] = proposal[accept]
            current_score[accept] = proposal_score[accept]

        return current

    def _rwmh_impute_missing_per_variable(
        self,
        batch_data,
        batch_mask,
        baseline_arg,
        state,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        mh_steps=1,
        indices=None,
        times=None,
        time_varying_covariates=None,
    ):
        """Per-variable RWMH imputation using the joint log target."""
        missing = batch_mask == 0
        if not missing.any():
            return batch_data

        current = batch_data.clone()
        current_score = self._imputation_log_target(
            current,
            batch_mask,
            baseline_arg,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
            times=times,
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
                    proposal,
                    batch_mask,
                    baseline_arg,
                    event_time=event_time,
                    event_indicator=event_indicator,
                    event_covariates=event_covariates,
                    times=times,
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

    def _compute_joint_loss(
        self,
        recon_batch,
        batch_data,
        mu,
        posterior_params,
        survival_outputs,
        mask_arg,
    ):
        """Combine longitudinal loss, survival loss, and KL."""
        longitudinal_loss, recon_loss, kld_loss = self._compute_loss(
            recon_batch,
            batch_data,
            mu,
            posterior_params,
            mask_arg,
        )
        if survival_outputs is None:
            survival_loss = recon_batch.new_tensor(0.0)
        else:
            survival_loss = -survival_outputs["survival_log_likelihood"].sum()
        total_loss = longitudinal_loss + self.survival_loss_weight * survival_loss
        return total_loss, recon_loss, survival_loss, kld_loss

    def train_epoch(
        self,
        train_loader,
        use_em_imputation=False,
        em_iterations=3,
        stochastic_impute=True,
        imputation_method="rwmh",
        mh_steps=1,
        mh_continuous_step_size=0.1,
        mh_bounded_step_size=0.05,
        mh_binary_flip_prob=0.1,
        mh_adaptive=True,
    ):
        """Train for one epoch with the joint ELBO."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_survival = 0.0
        total_kld = 0.0
        n_batches = 0

        imputation_method = self._resolve_imputation_method(imputation_method)

        for batch in train_loader:
            (
                batch_data,
                batch_mask,
                _,
                batch_baseline,
                batch_times,
                batch_tv_covs,
                batch_event_time,
                batch_event_indicator,
                batch_event_covariates,
                batch_indices,
            ) = self._unpack_joint_batch(batch)

            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            batch_times = batch_times.to(self.device)
            batch_tv_covs = batch_tv_covs.to(self.device)
            batch_event_time = batch_event_time.to(self.device)
            batch_event_indicator = batch_event_indicator.to(self.device)
            baseline_arg = self._get_baseline_arg(batch_baseline)
            event_covariates_arg = self._get_event_covariates_arg(batch_event_covariates)

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
                                        event_time=batch_event_time,
                                        event_indicator=batch_event_indicator,
                                        event_covariates=event_covariates_arg,
                                        times=batch_times,
                                        time_varying_covariates=batch_tv_covs,
                                    )
                                elif mh_adaptive and self.mh_state is not None:
                                    batch_data = self._rwmh_impute_missing_per_variable(
                                        batch_data,
                                        batch_mask,
                                        baseline_arg,
                                        state=self.mh_state,
                                        event_time=batch_event_time,
                                        event_indicator=batch_event_indicator,
                                        event_covariates=event_covariates_arg,
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
                                        event_time=batch_event_time,
                                        event_indicator=batch_event_indicator,
                                        event_covariates=event_covariates_arg,
                                        mh_steps=mh_steps,
                                        continuous_step_size=mh_continuous_step_size,
                                        bounded_step_size=mh_bounded_step_size,
                                        binary_flip_prob=mh_binary_flip_prob,
                                        times=batch_times,
                                        time_varying_covariates=batch_tv_covs,
                                    )
                            else:
                                recon_batch, _, _ = self._model_forward(
                                    batch_data,
                                    batch_mask,
                                    baseline_arg,
                                    event_time=batch_event_time,
                                    event_indicator=batch_event_indicator,
                                    event_covariates=event_covariates_arg,
                                    times=batch_times,
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

                    recon_batch, mu, posterior_params, survival_outputs = self._model_forward(
                        batch_data,
                        batch_mask,
                        baseline_arg,
                        event_time=batch_event_time,
                        event_indicator=batch_event_indicator,
                        event_covariates=event_covariates_arg,
                        times=batch_times,
                        time_varying_covariates=batch_tv_covs,
                        return_survival_terms=True,
                    )
                    loss, recon_loss, survival_loss, kld_loss = self._compute_joint_loss(
                        recon_batch,
                        batch_data,
                        mu,
                        posterior_params,
                        survival_outputs,
                        batch_mask,
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, posterior_params, survival_outputs = self._model_forward(
                    batch_data,
                    mask_arg,
                    baseline_arg,
                    event_time=batch_event_time,
                    event_indicator=batch_event_indicator,
                    event_covariates=event_covariates_arg,
                    times=batch_times,
                    time_varying_covariates=batch_tv_covs,
                    return_survival_terms=True,
                )
                loss, recon_loss, survival_loss, kld_loss = self._compute_joint_loss(
                    recon_batch,
                    batch_data,
                    mu,
                    posterior_params,
                    survival_outputs,
                    mask_arg,
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_survival += survival_loss.item()
            total_kld += kld_loss.item()
            n_batches += 1

        return (
            total_loss / n_batches,
            total_recon / n_batches,
            total_survival / n_batches,
            total_kld / n_batches,
        )

    def validate(self, val_loader):
        """Validate the joint model."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_survival = 0.0
        total_kld = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                (
                    batch_data,
                    batch_mask,
                    _,
                    batch_baseline,
                    batch_times,
                    batch_tv_covs,
                    batch_event_time,
                    batch_event_indicator,
                    batch_event_covariates,
                    _,
                ) = self._unpack_joint_batch(batch)

                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_times = batch_times.to(self.device)
                batch_tv_covs = batch_tv_covs.to(self.device)
                batch_event_time = batch_event_time.to(self.device)
                batch_event_indicator = batch_event_indicator.to(self.device)
                baseline_arg = self._get_baseline_arg(batch_baseline)
                event_covariates_arg = self._get_event_covariates_arg(batch_event_covariates)

                has_missing = batch_mask.sum() < batch_mask.numel()
                mask_arg = batch_mask if has_missing else None
                recon_batch, mu, posterior_params, survival_outputs = self._model_forward(
                    batch_data,
                    mask_arg,
                    baseline_arg,
                    event_time=batch_event_time,
                    event_indicator=batch_event_indicator,
                    event_covariates=event_covariates_arg,
                    times=batch_times,
                    time_varying_covariates=batch_tv_covs,
                    return_survival_terms=True,
                )
                loss, recon_loss, survival_loss, kld_loss = self._compute_joint_loss(
                    recon_batch,
                    batch_data,
                    mu,
                    posterior_params,
                    survival_outputs,
                    mask_arg,
                )
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_survival += survival_loss.item()
                total_kld += kld_loss.item()
                n_batches += 1

        return (
            total_loss / n_batches,
            total_recon / n_batches,
            total_survival / n_batches,
            total_kld / n_batches,
        )

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        verbose=True,
        use_em_imputation=False,
        em_iterations=3,
        patience=0,
        stochastic_impute=True,
        imputation_method="rwmh",
        mh_steps=1,
        mh_continuous_step_size=0.1,
        mh_bounded_step_size=0.05,
        mh_binary_flip_prob=0.1,
        mh_adaptive=True,
        mh_target_accept=0.234,
        mh_rm_decay=0.6,
        mh_rm_offset=10.0,
        mh_step_min=1e-4,
        mh_step_max=2.0,
        mh_flip_min=1e-3,
        mh_flip_max=0.5,
        mh_track_per_individual=True,
    ):
        """Train the joint model with optional EM-style missing-data updates."""
        imputation_method = self._resolve_imputation_method(imputation_method)

        history = {
            "train_loss": [],
            "train_recon": [],
            "train_survival": [],
            "train_kld": [],
            "val_loss": [],
            "val_recon": [],
            "val_survival": [],
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

            train_loss, train_recon, train_survival, train_kld = self.train_epoch(
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
            history["train_survival"].append(train_survival)
            history["train_kld"].append(train_kld)

            if adaptive_active and self.mh_state is not None:
                snapshot = self.mh_state.summary()
                history["mh_acceptance"].append(snapshot)
                self.mh_history.append(snapshot)

            if val_loader is not None:
                val_loss, val_recon, val_survival, val_kld = self.validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_recon"].append(val_recon)
                history["val_survival"].append(val_survival)
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
                    f"(Long: {train_recon:.4f}, Survival: {train_survival:.4f}, "
                    f"KLD: {train_kld:.4f})"
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


__all__ = ["JointVAETrainer"]
