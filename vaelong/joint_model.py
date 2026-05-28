"""
Joint longitudinal-survival model built additively on top of LongitudinalVAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import align_time_varying_covariates_to_grid
from .model import (
    LongitudinalVAE,
    _build_latent_posterior_heads,
    _compute_latent_posterior_params,
)
from .survival import (
    compose_log_hazard,
    cumulative_hazard_from_log_hazard,
    subject_quadrature_grid,
    survival_log_likelihood as _survival_log_likelihood,
    survival_probability,
)


class JointLongitudinalSurvivalVAE(LongitudinalVAE):
    """
    Joint VAE for longitudinal and right-censored time-to-event outcomes.

    The class reuses the existing longitudinal encoder/decoder while adding:

    - a survival-aware posterior context built from event information
    - a neural log-baseline hazard head
    - a neural association head linking the latent trajectory to event risk
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        latent_dim=20,
        num_layers=1,
        encoder_type="dense",
        seq_len=None,
        use_gru=False,
        n_baseline=0,
        n_event_covariates=0,
        var_config=None,
        latent_prior_type="identity",
        latent_posterior_type="diagonal",
        latent_posterior_rank=None,
        time_in_decoder=False,
        time_in_encoder=False,
        n_time_varying_covariates=0,
        survival_context_dim=None,
        hazard_hidden_dim=None,
        hazard_uses_time=True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            encoder_type=encoder_type,
            seq_len=seq_len,
            use_gru=use_gru,
            n_baseline=n_baseline,
            var_config=var_config,
            latent_prior_type=latent_prior_type,
            latent_posterior_type=latent_posterior_type,
            latent_posterior_rank=latent_posterior_rank,
            time_in_decoder=time_in_decoder,
            time_in_encoder=time_in_encoder,
            n_time_varying_covariates=n_time_varying_covariates,
        )

        self.n_event_covariates = int(n_event_covariates)
        self.survival_context_dim = int(survival_context_dim or hidden_dim)
        self.hazard_hidden_dim = int(hazard_hidden_dim or hidden_dim)
        self.hazard_uses_time = bool(hazard_uses_time)
        self.hazard_time_feature_dim = 4

        # Rebuild the posterior heads for a joint hidden state that includes
        # the survival context in addition to the longitudinal representation
        # and any baseline covariates.
        _build_latent_posterior_heads(
            self,
            hidden_dim + n_baseline + self.survival_context_dim,
        )

        self.survival_context_net = nn.Sequential(
            nn.Linear(3 + self.n_event_covariates, self.survival_context_dim),
            nn.ReLU(),
            nn.Linear(self.survival_context_dim, self.survival_context_dim),
            nn.ReLU(),
        )

        self.log_baseline_hazard_net = nn.Sequential(
            nn.Linear(self.hazard_time_feature_dim, self.hazard_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hazard_hidden_dim, self.hazard_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hazard_hidden_dim, 1),
        )

        association_input_dim = input_dim + self.n_event_covariates
        if self.hazard_uses_time:
            association_input_dim += self.hazard_time_feature_dim
        self.hazard_association_net = nn.Sequential(
            nn.Linear(association_input_dim, self.hazard_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hazard_hidden_dim, self.hazard_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hazard_hidden_dim, 1),
        )

    def _time_features(self, times):
        """Build smooth nonnegative features for the hazard networks."""
        times = torch.as_tensor(times, dtype=torch.float32)
        times = torch.clamp(times, min=0.0)
        return torch.stack(
            [
                times,
                torch.log1p(times),
                torch.sqrt(times + 1e-6),
                times * times,
            ],
            dim=-1,
        )

    def _prepare_event_covariates(self, event_covariates, batch_size, device, dtype):
        if self.n_event_covariates == 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if event_covariates is None:
            return torch.zeros(
                batch_size,
                self.n_event_covariates,
                device=device,
                dtype=dtype,
            )
        event_covariates = torch.as_tensor(
            event_covariates,
            device=device,
            dtype=dtype,
        )
        if event_covariates.dim() != 2 or event_covariates.shape != (
            batch_size,
            self.n_event_covariates,
        ):
            raise ValueError(
                "event_covariates must have shape "
                f"({batch_size}, {self.n_event_covariates}), got "
                f"{tuple(event_covariates.shape)}"
            )
        return event_covariates

    def _build_survival_context(
        self,
        batch_size,
        device,
        dtype,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
    ):
        if (
            event_time is None
            and event_indicator is None
            and event_covariates is None
        ):
            return torch.zeros(
                batch_size,
                self.survival_context_dim,
                device=device,
                dtype=dtype,
            )

        if event_time is None:
            event_time = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            event_time = torch.as_tensor(
                event_time, device=device, dtype=dtype
            ).reshape(batch_size)
        event_time = torch.clamp(event_time, min=0.0)

        if event_indicator is None:
            event_indicator = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            event_indicator = torch.as_tensor(
                event_indicator, device=device, dtype=dtype
            ).reshape(batch_size)

        covariates = self._prepare_event_covariates(
            event_covariates,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        context_input = torch.cat(
            [
                event_time.unsqueeze(-1),
                torch.log1p(event_time).unsqueeze(-1),
                event_indicator.unsqueeze(-1),
                covariates,
            ],
            dim=-1,
        )
        return self.survival_context_net(context_input)

    def _encode_longitudinal_hidden(
        self,
        x,
        times=None,
        time_varying_covariates=None,
    ):
        if self.n_time_varying_covariates > 0 and time_varying_covariates is None:
            time_varying_covariates = x.new_zeros(
                x.shape[0], x.shape[1], self.n_time_varying_covariates
            )
        if time_varying_covariates is not None:
            x = torch.cat([x, time_varying_covariates], dim=-1)

        if self.time_in_encoder:
            batch_size, seq_len = x.shape[0], x.shape[1]
            time_emb = self._sinusoidal_embedding(seq_len, x.device, times=times)
            if time_emb.shape[0] == 1:
                time_emb = time_emb.expand(batch_size, -1, -1)
            x = torch.cat([x, time_emb], dim=-1)

        if self.encoder_type == "dense":
            return self.encoder_mlp(x.reshape(x.size(0), -1))

        _, hidden = self.encoder_rnn(x)
        if self.encoder_type == "gru":
            return hidden[-1]
        return hidden[0][-1]

    def encode(
        self,
        x,
        mask=None,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
    ):
        """
        Encode the longitudinal sequence and optional survival information.

        Missingness masks are reserved for downstream loss masking only, just
        like in the longitudinal-only model.
        """
        del mask  # kept for API compatibility

        h_longitudinal = self._encode_longitudinal_hidden(
            x,
            times=times,
            time_varying_covariates=time_varying_covariates,
        )

        pieces = [h_longitudinal]
        if baseline is not None and self.n_baseline > 0:
            pieces.append(baseline)

        survival_context = self._build_survival_context(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
        )
        pieces.append(survival_context)

        hidden_state = torch.cat(pieces, dim=-1)
        return _compute_latent_posterior_params(self, hidden_state)

    def log_baseline_hazard(self, times):
        """Evaluate the neural log-baseline hazard on a time grid."""
        times = torch.as_tensor(times, dtype=torch.float32)
        features = self._time_features(times).to(device=times.device, dtype=times.dtype)
        return self.log_baseline_hazard_net(features).squeeze(-1)

    def hazard_log_relative_risk(self, eta, event_covariates=None, times=None):
        """Evaluate the association network linking trajectory to hazard."""
        squeeze_time_dim = False
        if eta.dim() == 2:
            eta = eta.unsqueeze(1)
            squeeze_time_dim = True
        if eta.dim() != 3:
            raise ValueError("eta must be a 2D or 3D tensor")

        batch_size, seq_len, _ = eta.shape
        covariates = self._prepare_event_covariates(
            event_covariates,
            batch_size=batch_size,
            device=eta.device,
            dtype=eta.dtype,
        )
        covariates = covariates.unsqueeze(1).expand(-1, seq_len, -1)

        pieces = [eta, covariates]
        if self.hazard_uses_time:
            if times is None:
                times = torch.arange(
                    seq_len, device=eta.device, dtype=eta.dtype
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                times = torch.as_tensor(times, device=eta.device, dtype=eta.dtype)
                if times.dim() == 1:
                    times = times.unsqueeze(0).expand(batch_size, -1)
                if times.shape != (batch_size, seq_len):
                    raise ValueError(
                        f"times must have shape ({batch_size}, {seq_len}), "
                        f"got {tuple(times.shape)}"
                    )
            pieces.append(self._time_features(times))

        association_input = torch.cat(pieces, dim=-1)
        log_relative_risk = self.hazard_association_net(association_input).squeeze(-1)
        if squeeze_time_dim:
            log_relative_risk = log_relative_risk.squeeze(1)
        return log_relative_risk

    def _align_decoder_covariates(
        self,
        target_times,
        measurement_times=None,
        time_varying_covariates=None,
    ):
        if self.n_time_varying_covariates == 0:
            return None
        if time_varying_covariates is None:
            return None

        if measurement_times is None:
            if time_varying_covariates.shape[1] != target_times.shape[1]:
                raise ValueError(
                    "measurement_times are required when aligning decoder covariates "
                    "to a target grid with a different sequence length"
                )
            return time_varying_covariates

        measurement_times = torch.as_tensor(
            measurement_times,
            device=target_times.device,
            dtype=target_times.dtype,
        )
        if measurement_times.dim() == 1:
            measurement_times = measurement_times.unsqueeze(0).expand(
                time_varying_covariates.shape[0], -1
            )
        if measurement_times.dim() != 2:
            raise ValueError("measurement_times must be a 1D or 2D tensor")
        if measurement_times.shape != time_varying_covariates.shape[:2]:
            raise ValueError(
                "measurement_times must match the leading dimensions of "
                "time_varying_covariates, got "
                f"{tuple(measurement_times.shape)} and "
                f"{tuple(time_varying_covariates.shape[:2])}"
            )

        aligned = align_time_varying_covariates_to_grid(
            values=time_varying_covariates.detach().cpu().numpy(),
            source_times=measurement_times.detach().cpu().numpy(),
            target_times=target_times.detach().cpu().numpy(),
        )
        return torch.as_tensor(
            aligned,
            device=target_times.device,
            dtype=time_varying_covariates.dtype,
        )

    def _normalize_prediction_times(self, prediction_times, batch_size, device, dtype,
                                    name="prediction_times"):
        """Normalize a scalar/grid of prediction times to shape ``(B, T)``."""
        prediction_times = torch.as_tensor(
            prediction_times,
            device=device,
            dtype=dtype,
        )
        if prediction_times.dim() == 1:
            prediction_times = prediction_times.unsqueeze(0).expand(batch_size, -1)
        if prediction_times.dim() != 2 or prediction_times.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have shape (batch_size, n_times) or (n_times,), "
                f"got {tuple(prediction_times.shape)}"
            )
        return prediction_times

    def _slice_time_grid_prefix(self, times, batch_size, observed_len, device, dtype):
        """Slice the observed prefix from a shared or subject-specific time grid."""
        if times is None:
            return None

        times = torch.as_tensor(times, device=device, dtype=dtype)
        if times.dim() == 1:
            if times.shape[0] < observed_len:
                raise ValueError(
                    f"times must have at least {observed_len} entries, got {times.shape[0]}"
                )
            return times[:observed_len].unsqueeze(0).expand(batch_size, -1)

        if times.dim() == 2:
            if times.shape[0] != batch_size or times.shape[1] < observed_len:
                raise ValueError(
                    "times must have shape (batch_size, >= observed_len), "
                    f"got {tuple(times.shape)}"
                )
            return times[:, :observed_len]

        raise ValueError("times must be a 1D or 2D tensor")

    def _slice_time_varying_covariate_prefix(
        self,
        time_varying_covariates,
        batch_size,
        observed_len,
        device,
        dtype,
    ):
        """Slice the observed prefix of known time-varying covariates."""
        if time_varying_covariates is None:
            return None

        time_varying_covariates = torch.as_tensor(
            time_varying_covariates,
            device=device,
            dtype=dtype,
        )
        if (
            time_varying_covariates.dim() != 3
            or time_varying_covariates.shape[0] != batch_size
            or time_varying_covariates.shape[1] < observed_len
        ):
            raise ValueError(
                "time_varying_covariates must have shape "
                "(batch_size, >= observed_len, n_covariates)"
            )
        return time_varying_covariates[:, :observed_len]

    def _prepare_decode_time_grid(self, times, batch_size, total_seq_len, device, dtype):
        """Prepare the decode-time grid for landmark trajectory prediction."""
        if times is None:
            return None
        times = torch.as_tensor(times, device=device, dtype=dtype)
        if times.dim() == 1:
            if times.shape[0] < total_seq_len:
                raise ValueError(
                    f"times must have at least {total_seq_len} entries, got {times.shape[0]}"
                )
            return times[:total_seq_len].unsqueeze(0).expand(batch_size, -1)
        if times.dim() == 2:
            if times.shape[0] != batch_size or times.shape[1] < total_seq_len:
                raise ValueError(
                    "times must have shape (batch_size, >= total_seq_len), "
                    f"got {tuple(times.shape)}"
                )
            return times[:, :total_seq_len]
        raise ValueError("times must be a 1D or 2D tensor")

    def _prepare_decode_time_varying_covariates(
        self,
        time_varying_covariates,
        batch_size,
        total_seq_len,
        device,
        dtype,
    ):
        """Prepare known decoder covariates for landmark trajectory prediction."""
        if time_varying_covariates is None:
            return None
        time_varying_covariates = torch.as_tensor(
            time_varying_covariates,
            device=device,
            dtype=dtype,
        )
        if (
            time_varying_covariates.dim() != 3
            or time_varying_covariates.shape[0] != batch_size
            or time_varying_covariates.shape[1] < total_seq_len
        ):
            raise ValueError(
                "time_varying_covariates must have shape "
                "(batch_size, >= total_seq_len, n_covariates)"
            )
        return time_varying_covariates[:, :total_seq_len]

    def _resolve_landmark_time(self, x_observed, landmark_time=None, times=None):
        """Resolve the landmark time from an explicit value or the observed grid."""
        batch_size = x_observed.shape[0]
        observed_len = x_observed.shape[1]
        device = x_observed.device
        dtype = x_observed.dtype

        if landmark_time is not None:
            landmark_time = torch.as_tensor(
                landmark_time,
                device=device,
                dtype=dtype,
            )
            if landmark_time.dim() == 0:
                landmark_time = landmark_time.repeat(batch_size)
            else:
                landmark_time = landmark_time.reshape(batch_size)
            return landmark_time

        times_obs = self._slice_time_grid_prefix(
            times,
            batch_size=batch_size,
            observed_len=observed_len,
            device=device,
            dtype=dtype,
        )
        if times_obs is not None:
            return times_obs[:, observed_len - 1]

        return torch.full(
            (batch_size,),
            float(observed_len - 1),
            device=device,
            dtype=dtype,
        )

    def infer_latent_from_landmark(
        self,
        x_observed,
        mask_observed,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
        landmark_event_indicator=None,
    ):
        """
        Infer the latent posterior from history observed up to a landmark time.

        By default the survival context encodes survival up to the landmark
        (`event_indicator=0`), which is the standard dynamic-prediction case.
        """
        batch_size, observed_len = x_observed.shape[0], x_observed.shape[1]
        landmark_time = self._resolve_landmark_time(
            x_observed,
            landmark_time=landmark_time,
            times=times,
        )
        if landmark_event_indicator is None:
            landmark_event_indicator = torch.zeros_like(landmark_time)
        else:
            landmark_event_indicator = torch.as_tensor(
                landmark_event_indicator,
                device=x_observed.device,
                dtype=x_observed.dtype,
            ).reshape(batch_size)

        times_obs = self._slice_time_grid_prefix(
            times,
            batch_size=batch_size,
            observed_len=observed_len,
            device=x_observed.device,
            dtype=x_observed.dtype,
        )
        tvc_obs = self._slice_time_varying_covariate_prefix(
            time_varying_covariates,
            batch_size=batch_size,
            observed_len=observed_len,
            device=x_observed.device,
            dtype=x_observed.dtype,
        )

        if self.encoder_type == "dense" and observed_len < self.seq_len:
            pad_len = self.seq_len - observed_len
            x_observed = F.pad(x_observed, (0, 0, 0, pad_len))
            mask_observed = F.pad(mask_observed, (0, 0, 0, pad_len))
            if times_obs is not None:
                pad_times = landmark_time.unsqueeze(-1).expand(-1, pad_len)
                times_obs = torch.cat([times_obs, pad_times], dim=1)
            if tvc_obs is not None:
                tvc_obs = F.pad(tvc_obs, (0, 0, 0, pad_len))

        mu, posterior_params = self.encode(
            x_observed,
            mask_observed,
            baseline=baseline,
            times=times_obs,
            time_varying_covariates=tvc_obs,
            event_time=landmark_time,
            event_indicator=landmark_event_indicator,
            event_covariates=event_covariates,
        )
        return mu, posterior_params, landmark_time

    def survival_terms(
        self,
        z,
        event_time,
        event_indicator,
        event_covariates=None,
        baseline=None,
        measurement_times=None,
        time_varying_covariates=None,
    ):
        """
        Compute hazard and log-likelihood terms for right-censored survival data.
        """
        event_time = torch.as_tensor(
            event_time, device=z.device, dtype=z.dtype
        ).reshape(z.shape[0])
        event_indicator = torch.as_tensor(
            event_indicator, device=z.device, dtype=z.dtype
        ).reshape(z.shape[0])
        event_covariates = self._prepare_event_covariates(
            event_covariates,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )

        event_times = event_time.unsqueeze(-1)
        event_tvc = self._align_decoder_covariates(
            event_times,
            measurement_times=measurement_times,
            time_varying_covariates=time_varying_covariates,
        )
        eta_event = self.predict_latent_trajectory(
            z,
            seq_len=1,
            baseline=baseline,
            times=event_times,
            time_varying_covariates=event_tvc,
        ).squeeze(1)

        log_baseline_event = self.log_baseline_hazard(event_time)
        log_relative_risk_event = self.hazard_log_relative_risk(
            eta_event,
            event_covariates=event_covariates,
            times=event_times,
        )
        log_hazard_event = compose_log_hazard(
            log_baseline_event,
            log_relative_risk_event,
        )

        quadrature_times, quadrature_weights = subject_quadrature_grid(
            event_time,
            device=z.device,
            dtype=z.dtype,
        )
        quadrature_tvc = self._align_decoder_covariates(
            quadrature_times,
            measurement_times=measurement_times,
            time_varying_covariates=time_varying_covariates,
        )
        eta_quadrature = self.predict_latent_trajectory(
            z,
            seq_len=quadrature_times.shape[1],
            baseline=baseline,
            times=quadrature_times,
            time_varying_covariates=quadrature_tvc,
        )
        log_baseline_quadrature = self.log_baseline_hazard(quadrature_times)
        log_relative_risk_quadrature = self.hazard_log_relative_risk(
            eta_quadrature,
            event_covariates=event_covariates,
            times=quadrature_times,
        )
        log_hazard_quadrature = compose_log_hazard(
            log_baseline_quadrature,
            log_relative_risk_quadrature,
        )
        cumulative_hazard = cumulative_hazard_from_log_hazard(
            log_hazard_quadrature,
            quadrature_weights,
        )
        log_likelihood = _survival_log_likelihood(
            log_hazard_event,
            cumulative_hazard,
            event_indicator,
        )

        return {
            "eta_event": eta_event,
            "log_hazard_event": log_hazard_event,
            "quadrature_times": quadrature_times,
            "quadrature_weights": quadrature_weights,
            "log_hazard_quadrature": log_hazard_quadrature,
            "cumulative_hazard": cumulative_hazard,
            "survival_log_likelihood": log_likelihood,
        }

    def predict_survival_curve(
        self,
        z,
        prediction_times,
        event_covariates=None,
        baseline=None,
        measurement_times=None,
        time_varying_covariates=None,
    ):
        """Predict marginal survival probabilities on a subject-specific grid."""
        if z.dim() != 2:
            raise ValueError("z must have shape (batch_size, latent_dim)")

        prediction_times = self._normalize_prediction_times(
            prediction_times,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )

        batch_size, n_times = prediction_times.shape
        flat_times = prediction_times.reshape(-1)
        quadrature_times, quadrature_weights = subject_quadrature_grid(
            flat_times,
            device=z.device,
            dtype=z.dtype,
        )

        z_repeated = z.unsqueeze(1).expand(-1, n_times, -1).reshape(-1, z.shape[1])
        baseline_repeated = None
        if baseline is not None and self.n_baseline > 0:
            baseline_repeated = baseline.unsqueeze(1).expand(-1, n_times, -1).reshape(
                -1, baseline.shape[1]
            )

        event_covariates = self._prepare_event_covariates(
            event_covariates,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )
        event_covariates_repeated = event_covariates.unsqueeze(1).expand(
            -1, n_times, -1
        ).reshape(-1, event_covariates.shape[1])

        measurement_times_repeated = None
        time_varying_covariates_repeated = None
        if measurement_times is not None:
            measurement_times = self._normalize_prediction_times(
                measurement_times,
                batch_size=batch_size,
                device=z.device,
                dtype=z.dtype,
                name="measurement_times",
            )
            measurement_times_repeated = measurement_times.unsqueeze(1).expand(
                -1, n_times, -1
            ).reshape(-1, measurement_times.shape[1])
        if time_varying_covariates is not None:
            time_varying_covariates_repeated = (
                time_varying_covariates.unsqueeze(1)
                .expand(-1, n_times, -1, -1)
                .reshape(
                    -1,
                    time_varying_covariates.shape[1],
                    time_varying_covariates.shape[2],
                )
            )

        quadrature_tvc = self._align_decoder_covariates(
            quadrature_times,
            measurement_times=measurement_times_repeated,
            time_varying_covariates=time_varying_covariates_repeated,
        )

        eta_quadrature = self.predict_latent_trajectory(
            z_repeated,
            seq_len=quadrature_times.shape[1],
            baseline=baseline_repeated,
            times=quadrature_times,
            time_varying_covariates=quadrature_tvc,
        )
        log_hazard_quadrature = compose_log_hazard(
            self.log_baseline_hazard(quadrature_times),
            self.hazard_log_relative_risk(
                eta_quadrature,
                event_covariates=event_covariates_repeated,
                times=quadrature_times,
            ),
        )
        cumulative_hazard = cumulative_hazard_from_log_hazard(
            log_hazard_quadrature,
            quadrature_weights,
        )
        return survival_probability(cumulative_hazard).reshape(batch_size, n_times)

    def predict_hazard_curve(
        self,
        z,
        prediction_times,
        event_covariates=None,
        baseline=None,
        measurement_times=None,
        time_varying_covariates=None,
    ):
        """Predict the hazard curve on a future time grid for known latent states."""
        if z.dim() != 2:
            raise ValueError("z must have shape (batch_size, latent_dim)")

        prediction_times = self._normalize_prediction_times(
            prediction_times,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
        )
        aligned_tvc = self._align_decoder_covariates(
            prediction_times,
            measurement_times=measurement_times,
            time_varying_covariates=time_varying_covariates,
        )
        eta = self.predict_latent_trajectory(
            z,
            seq_len=prediction_times.shape[1],
            baseline=baseline,
            times=prediction_times,
            time_varying_covariates=aligned_tvc,
        )
        log_hazard = compose_log_hazard(
            self.log_baseline_hazard(prediction_times),
            self.hazard_log_relative_risk(
                eta,
                event_covariates=event_covariates,
                times=prediction_times,
            ),
        )
        return torch.exp(torch.clamp(log_hazard, max=20.0))

    def predict_event_probability(
        self,
        z,
        start_times,
        end_times,
        event_covariates=None,
        baseline=None,
        measurement_times=None,
        time_varying_covariates=None,
    ):
        """
        Predict event probability in ``[start_times, end_times]`` conditional on
        survival to ``start_times``.
        """
        start_times = torch.as_tensor(start_times, device=z.device, dtype=z.dtype)
        end_times = torch.as_tensor(end_times, device=z.device, dtype=z.dtype)
        if start_times.dim() == 1:
            start_times = start_times.unsqueeze(0).expand(z.shape[0], -1)
        if end_times.dim() == 1:
            end_times = end_times.unsqueeze(0).expand(z.shape[0], -1)
        if start_times.shape != end_times.shape:
            raise ValueError("start_times and end_times must have the same shape")

        survival_start = self.predict_survival_curve(
            z,
            start_times,
            event_covariates=event_covariates,
            baseline=baseline,
            measurement_times=measurement_times,
            time_varying_covariates=time_varying_covariates,
        )
        survival_end = self.predict_survival_curve(
            z,
            end_times,
            event_covariates=event_covariates,
            baseline=baseline,
            measurement_times=measurement_times,
            time_varying_covariates=time_varying_covariates,
        )
        conditional_survival = survival_end / survival_start.clamp_min(1e-8)
        return 1.0 - conditional_survival.clamp(min=0.0, max=1.0)

    def predict_longitudinal_from_landmark(
        self,
        x_observed,
        mask_observed,
        total_seq_len,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
    ):
        """
        Predict the longitudinal trajectory given history up to a landmark.

        This is the joint-model analogue of ``LongitudinalVAE.predict_from_landmark``.
        """
        self.eval()
        with torch.no_grad():
            mu, _, _ = self.infer_latent_from_landmark(
                x_observed,
                mask_observed,
                baseline=baseline,
                times=times,
                time_varying_covariates=time_varying_covariates,
                event_covariates=event_covariates,
                landmark_time=landmark_time,
            )
            decode_times = self._prepare_decode_time_grid(
                times,
                batch_size=x_observed.shape[0],
                total_seq_len=total_seq_len,
                device=x_observed.device,
                dtype=x_observed.dtype,
            )
            decode_tvc = self._prepare_decode_time_varying_covariates(
                time_varying_covariates,
                batch_size=x_observed.shape[0],
                total_seq_len=total_seq_len,
                device=x_observed.device,
                dtype=x_observed.dtype,
            )
            return self.decode(
                mu,
                total_seq_len,
                baseline=baseline,
                times=decode_times,
                time_varying_covariates=decode_tvc,
            )

    def predict_from_landmark(
        self,
        x_observed,
        mask_observed,
        total_seq_len,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
    ):
        """Backwards-compatible alias for joint landmark trajectory prediction."""
        return self.predict_longitudinal_from_landmark(
            x_observed,
            mask_observed,
            total_seq_len,
            baseline=baseline,
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_covariates=event_covariates,
            landmark_time=landmark_time,
        )

    def predict_survival_from_landmark(
        self,
        x_observed,
        mask_observed,
        prediction_times,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
    ):
        """
        Predict survival conditional on history and being event-free at landmark.
        """
        self.eval()
        with torch.no_grad():
            mu, _, resolved_landmark = self.infer_latent_from_landmark(
                x_observed,
                mask_observed,
                baseline=baseline,
                times=times,
                time_varying_covariates=time_varying_covariates,
                event_covariates=event_covariates,
                landmark_time=landmark_time,
            )
            prediction_times = self._normalize_prediction_times(
                prediction_times,
                batch_size=x_observed.shape[0],
                device=x_observed.device,
                dtype=x_observed.dtype,
            )
            if torch.any(prediction_times < resolved_landmark.unsqueeze(-1)):
                raise ValueError("prediction_times must be at or after the landmark time")

            survival_curve = self.predict_survival_curve(
                mu,
                prediction_times,
                event_covariates=event_covariates,
                baseline=baseline,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )
            survival_at_landmark = self.predict_survival_curve(
                mu,
                resolved_landmark.unsqueeze(-1),
                event_covariates=event_covariates,
                baseline=baseline,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )
            conditional_survival = survival_curve / survival_at_landmark.clamp_min(1e-8)
            return conditional_survival.clamp(min=0.0, max=1.0)

    def predict_hazard_from_landmark(
        self,
        x_observed,
        mask_observed,
        prediction_times,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
    ):
        """Predict the hazard curve conditional on history up to a landmark."""
        self.eval()
        with torch.no_grad():
            mu, _, resolved_landmark = self.infer_latent_from_landmark(
                x_observed,
                mask_observed,
                baseline=baseline,
                times=times,
                time_varying_covariates=time_varying_covariates,
                event_covariates=event_covariates,
                landmark_time=landmark_time,
            )
            prediction_times = self._normalize_prediction_times(
                prediction_times,
                batch_size=x_observed.shape[0],
                device=x_observed.device,
                dtype=x_observed.dtype,
            )
            if torch.any(prediction_times < resolved_landmark.unsqueeze(-1)):
                raise ValueError("prediction_times must be at or after the landmark time")
            return self.predict_hazard_curve(
                mu,
                prediction_times,
                event_covariates=event_covariates,
                baseline=baseline,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )

    def predict_event_probability_from_landmark(
        self,
        x_observed,
        mask_observed,
        start_times,
        end_times,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
        landmark_time=None,
    ):
        """
        Predict interval event probabilities conditional on history up to a landmark.
        """
        self.eval()
        with torch.no_grad():
            mu, _, resolved_landmark = self.infer_latent_from_landmark(
                x_observed,
                mask_observed,
                baseline=baseline,
                times=times,
                time_varying_covariates=time_varying_covariates,
                event_covariates=event_covariates,
                landmark_time=landmark_time,
            )
            start_times = self._normalize_prediction_times(
                start_times,
                batch_size=x_observed.shape[0],
                device=x_observed.device,
                dtype=x_observed.dtype,
                name="start_times",
            )
            end_times = self._normalize_prediction_times(
                end_times,
                batch_size=x_observed.shape[0],
                device=x_observed.device,
                dtype=x_observed.dtype,
                name="end_times",
            )
            if start_times.shape != end_times.shape:
                raise ValueError("start_times and end_times must have the same shape")
            if torch.any(start_times < resolved_landmark.unsqueeze(-1)):
                raise ValueError("start_times must be at or after the landmark time")
            if torch.any(end_times < start_times):
                raise ValueError("end_times must be greater than or equal to start_times")
            return self.predict_event_probability(
                mu,
                start_times,
                end_times,
                event_covariates=event_covariates,
                baseline=baseline,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )

    def forward(
        self,
        x,
        mask=None,
        baseline=None,
        times=None,
        time_varying_covariates=None,
        event_time=None,
        event_indicator=None,
        event_covariates=None,
        return_survival_terms=False,
    ):
        """
        Joint forward pass.

        By default this mirrors the longitudinal-only return signature. When
        ``return_survival_terms=True`` and event information is provided, the
        method also returns the survival contributions for the sampled latent
        state.
        """
        seq_len = x.size(1)
        mu, posterior_params = self.encode(
            x,
            mask=mask,
            baseline=baseline,
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_time=event_time,
            event_indicator=event_indicator,
            event_covariates=event_covariates,
        )
        z = self.reparameterize(mu, posterior_params)
        recon_x = self.decode(
            z,
            seq_len,
            baseline=baseline,
            times=times,
            time_varying_covariates=time_varying_covariates,
        )

        if not return_survival_terms:
            return recon_x, mu, posterior_params

        survival_outputs = None
        if event_time is not None and event_indicator is not None:
            survival_outputs = self.survival_terms(
                z,
                event_time=event_time,
                event_indicator=event_indicator,
                event_covariates=event_covariates,
                baseline=baseline,
                measurement_times=times,
                time_varying_covariates=time_varying_covariates,
            )
        return recon_x, mu, posterior_params, survival_outputs


__all__ = ["JointLongitudinalSurvivalVAE"]
