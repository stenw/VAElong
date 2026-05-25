"""
Data utilities for longitudinal measurements.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from .config import VariableConfig, VariableSpec


class LongitudinalDataset(Dataset):
    """
    Dataset class for longitudinal measurements with missing data support.

    Args:
        data: Numpy array of shape (n_samples, seq_len, n_features) or list of sequences
        mask: Optional binary mask of same shape as data (1=observed, 0=missing)
        normalize: Whether to normalize the data (default: True)
        padding_value: Value to use for padding shorter sequences (default: 0.0)
        baseline_covariates: Optional numpy array of shape (n_samples, n_baseline_features)
        var_config: Optional VariableConfig specifying variable types
        times: Optional array of measurement times for each (subject, timestep).
            Accepted shapes: ``(seq_len,)`` for a single time grid shared across
            all subjects, or ``(n_samples, seq_len)`` for per-subject times.
            Defaults to position indices ``0..seq_len-1`` broadcast across
            subjects, which is what the model uses when ``time_in_decoder``
            is enabled on the encoder side without explicit times.
        time_varying_covariates: Optional array of known time-dependent
            covariates that should condition the model but should not be
            reconstructed. Must have shape ``(n_samples, seq_len, n_covariates)``.
    """

    def __init__(self, data, mask=None, normalize=True, padding_value=0.0,
                 baseline_covariates=None, var_config=None, times=None,
                 time_varying_covariates=None):
        if isinstance(data, list):
            # Handle variable length sequences
            self.data, self.lengths = self._pad_sequences(data, padding_value)
        else:
            # Fixed length sequences (clone to avoid modifying input array)
            self.data = torch.FloatTensor(np.array(data, copy=True))
            self.lengths = torch.LongTensor([data.shape[1]] * len(data))

        # Handle mask
        if mask is not None:
            if isinstance(mask, list):
                self.mask, _ = self._pad_sequences(mask, 0.0)
            else:
                self.mask = torch.FloatTensor(mask)
        else:
            # Default: all data is observed
            self.mask = torch.ones_like(self.data)

        # Store variable config (default: all continuous for backward compat)
        n_feat = self.data.shape[-1]
        if var_config is None:
            self.var_config = VariableConfig.all_continuous(n_feat)
        else:
            self.var_config = var_config

        # Handle baseline covariates
        if baseline_covariates is not None:
            self.baseline = torch.FloatTensor(baseline_covariates)
        else:
            self.baseline = torch.zeros(len(self.data), 0)

        # Handle measurement times (per-subject, per-timestep). Defaults to
        # positional indices, broadcast across subjects so the model can
        # always rely on ``times`` being present.
        n_samples, seq_len = self.data.shape[0], self.data.shape[1]
        if times is None:
            grid = torch.arange(seq_len, dtype=torch.float32)
            self.times = grid.unsqueeze(0).expand(n_samples, -1).contiguous()
        else:
            times_arr = torch.as_tensor(np.asarray(times), dtype=torch.float32)
            if times_arr.dim() == 1:
                if times_arr.shape[0] != seq_len:
                    raise ValueError(
                        f"times of shape {tuple(times_arr.shape)} does not match seq_len={seq_len}"
                    )
                self.times = times_arr.unsqueeze(0).expand(n_samples, -1).contiguous()
            elif times_arr.dim() == 2:
                if times_arr.shape != (n_samples, seq_len):
                    raise ValueError(
                        f"times of shape {tuple(times_arr.shape)} must equal (n_samples, seq_len)="
                        f"({n_samples}, {seq_len})"
                    )
                self.times = times_arr
            else:
                raise ValueError(
                    "times must be a 1D (seq_len,) or 2D (n_samples, seq_len) array"
                )

        # Handle known time-varying covariates that condition the model but
        # are not part of the reconstruction target.
        if time_varying_covariates is None:
            self.time_varying_covariates = torch.zeros(
                n_samples, seq_len, 0, dtype=torch.float32
            )
        else:
            tvc = torch.as_tensor(
                np.asarray(time_varying_covariates), dtype=torch.float32
            )
            if tvc.dim() != 3:
                raise ValueError(
                    "time_varying_covariates must be a 3D "
                    "(n_samples, seq_len, n_covariates) array"
                )
            expected = (n_samples, seq_len)
            if tvc.shape[:2] != expected:
                raise ValueError(
                    "time_varying_covariates must have leading shape "
                    f"{expected}, got {tuple(tvc.shape[:2])}"
                )
            self.time_varying_covariates = tvc

        if normalize:
            self._normalize_by_type()
        else:
            self.mean = None
            self.std = None
            self.bounds_info = None

    def _normalize_by_type(self):
        """Type-aware normalization.

        - Continuous: z-score using observed values only
        - Bounded: affine transform to [0, 1] using known bounds
        - Binary: no normalization
        """
        cont_idx = self.var_config.continuous_indices
        bounded_idx = self.var_config.bounded_indices

        # Initialize per-feature mean/std (only meaningful for continuous)
        n_feat = self.data.shape[-1]
        self.mean = torch.zeros(1, 1, n_feat)
        self.std = torch.ones(1, 1, n_feat)

        # Continuous: z-score using observed values
        if cont_idx:
            for idx in cont_idx:
                observed = self.data[:, :, idx] * self.mask[:, :, idx]
                n_obs = self.mask[:, :, idx].sum()
                if n_obs > 0:
                    m = observed.sum() / n_obs
                    s = torch.sqrt(((observed - m * self.mask[:, :, idx]) ** 2).sum() / n_obs)
                    if s == 0:
                        s = torch.tensor(1.0)
                    self.mean[0, 0, idx] = m
                    self.std[0, 0, idx] = s
                    self.data[:, :, idx] = ((self.data[:, :, idx] - m) / s) * self.mask[:, :, idx]

        # Bounded: affine transform to [0, 1]
        self.bounds_info = {}
        if bounded_idx:
            bounds = self.var_config.get_bounds()
            for idx in bounded_idx:
                lo, hi = bounds[idx]
                self.bounds_info[idx] = (lo, hi)
                self.data[:, :, idx] = ((self.data[:, :, idx] - lo) / (hi - lo)) * self.mask[:, :, idx]
                # Optional epsilon clamping to [eps, 1-eps]
                if self.var_config.bounded_eps > 0:
                    eps = self.var_config.bounded_eps
                    self.data[:, :, idx] = self.data[:, :, idx].clamp(eps, 1 - eps) * self.mask[:, :, idx]

        # Binary: no normalization needed

    def _pad_sequences(self, sequences, padding_value):
        """Pad sequences to same length."""
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        n_features = sequences[0].shape[-1]

        padded = np.full((len(sequences), max_len, n_features), padding_value, dtype=np.float32)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq

        return torch.FloatTensor(padded), torch.LongTensor(lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.mask[idx],
            self.lengths[idx],
            self.baseline[idx],
            self.times[idx],
            self.time_varying_covariates[idx],
        )

    def inverse_transform(self, data):
        """
        Type-aware inverse transformation.

        Args:
            data: Normalized data tensor

        Returns:
            Denormalized data
        """
        result = data.clone()

        if self.mean is not None and self.std is not None:
            # Continuous: reverse z-score
            for idx in self.var_config.continuous_indices:
                result[..., idx] = result[..., idx] * self.std[0, 0, idx] + self.mean[0, 0, idx]

        if self.bounds_info:
            # Bounded: reverse affine from [0,1] to [lower, upper]
            for idx, (lo, hi) in self.bounds_info.items():
                result[..., idx] = result[..., idx] * (hi - lo) + lo

        # Binary: no inverse needed
        return result


def _pad_vector_sequences(sequences, padding_value=0.0, carry_forward_last=False):
    """Pad a list of 1D sequences to a common length."""
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded = np.full((len(sequences), max_len), padding_value, dtype=np.float32)

    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float32)
        padded[i, :len(arr)] = arr
        if carry_forward_last and len(arr) > 0 and len(arr) < max_len:
            padded[i, len(arr):] = arr[-1]

    return padded


def _pad_matrix_sequences(sequences, padding_value=0.0):
    """Pad a list of 2D arrays with a common feature dimension."""
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    n_features = sequences[0].shape[-1] if sequences else 0
    padded = np.full(
        (len(sequences), max_len, n_features), padding_value, dtype=np.float32
    )

    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float32)
        padded[i, :len(arr)] = arr

    return padded


def _validate_subject_level_vector(name, values, n_samples, allow_negative=False):
    """Convert a subject-level vector to float32 and validate its shape."""
    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1)
    if tensor.shape != (n_samples,):
        raise ValueError(
            f"{name} must have shape (n_samples,)={(n_samples,)}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if not allow_negative and torch.any(tensor < 0):
        raise ValueError(f"{name} must be non-negative")
    return tensor


def _validate_event_indicators(values, n_samples):
    """Convert event indicators to float32 and ensure they are binary."""
    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32).reshape(-1)
    if tensor.shape != (n_samples,):
        raise ValueError(
            "event_indicators must have shape "
            f"(n_samples,)={(n_samples,)}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("event_indicators must contain only finite values")
    is_binary = torch.logical_or(tensor == 0.0, tensor == 1.0)
    if not bool(is_binary.all()):
        raise ValueError("event_indicators must contain only 0/1 values")
    return tensor


def _validate_subject_level_matrix(name, values, n_samples):
    """Convert a subject-level matrix to float32 and validate its shape."""
    if values is None:
        return torch.zeros(n_samples, 0, dtype=torch.float32)

    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2D (n_samples, n_features) array")
    if tensor.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have leading dimension n_samples={n_samples}, "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class JointLongitudinalSurvivalDataset(LongitudinalDataset):
    """
    Dataset for joint longitudinal and time-to-event modeling.

    This extends :class:`LongitudinalDataset` additively by attaching the
    subject-level survival outcome and optional event covariates, while
    preserving the same longitudinal inputs and masking behavior.
    """

    def __init__(
        self,
        data,
        event_times,
        event_indicators,
        mask=None,
        normalize=True,
        padding_value=0.0,
        baseline_covariates=None,
        var_config=None,
        times=None,
        time_varying_covariates=None,
        event_covariates=None,
    ):
        custom_time_padding = isinstance(data, list) and isinstance(times, list)
        custom_tvc_padding = (
            isinstance(data, list) and isinstance(time_varying_covariates, list)
        )

        super().__init__(
            data,
            mask=mask,
            normalize=normalize,
            padding_value=padding_value,
            baseline_covariates=baseline_covariates,
            var_config=var_config,
            times=None if custom_time_padding else times,
            time_varying_covariates=(
                None if custom_tvc_padding else time_varying_covariates
            ),
        )

        n_samples = len(self.data)

        if custom_time_padding:
            if len(times) != n_samples:
                raise ValueError(
                    f"times must have one sequence per subject; expected {n_samples}, got {len(times)}"
                )
            if any(len(seq) != int(self.lengths[i]) for i, seq in enumerate(times)):
                raise ValueError(
                    "Each time sequence must match the corresponding longitudinal sequence length"
                )
            self.times = torch.as_tensor(
                _pad_vector_sequences(
                    times,
                    padding_value=padding_value,
                    carry_forward_last=True,
                ),
                dtype=torch.float32,
            )

        if custom_tvc_padding:
            if len(time_varying_covariates) != n_samples:
                raise ValueError(
                    "time_varying_covariates must have one sequence per subject; "
                    f"expected {n_samples}, got {len(time_varying_covariates)}"
                )
            if any(
                len(seq) != int(self.lengths[i])
                for i, seq in enumerate(time_varying_covariates)
            ):
                raise ValueError(
                    "Each time-varying covariate sequence must match the corresponding "
                    "longitudinal sequence length"
                )
            self.time_varying_covariates = torch.as_tensor(
                _pad_matrix_sequences(
                    time_varying_covariates,
                    padding_value=padding_value,
                ),
                dtype=torch.float32,
            )

        self.event_times = _validate_subject_level_vector(
            "event_times", event_times, n_samples
        )
        self.event_indicators = _validate_event_indicators(
            event_indicators, n_samples
        )
        self.event_covariates = _validate_subject_level_matrix(
            "event_covariates", event_covariates, n_samples
        )

    def __getitem__(self, idx):
        data, mask, length, baseline, times, time_varying_covariates = super().__getitem__(idx)
        return (
            data,
            mask,
            length,
            baseline,
            times,
            time_varying_covariates,
            self.event_times[idx],
            self.event_indicators[idx],
            self.event_covariates[idx],
        )


def align_time_varying_covariates_to_grid(
    values,
    source_times,
    target_times,
    lengths=None,
    fill_value=0.0,
):
    """
    Align known time-varying covariates to a target grid using LOCF.

    Values before the first available measurement are filled with
    ``fill_value`` instead of extrapolating backwards.
    """
    values_arr = np.asarray(values, dtype=np.float32)
    source_times_arr = np.asarray(source_times, dtype=np.float32)
    target_times_arr = np.asarray(target_times, dtype=np.float32)

    if values_arr.ndim != 3:
        raise ValueError("values must be a 3D (n_samples, seq_len, n_covariates) array")
    if source_times_arr.shape != values_arr.shape[:2]:
        raise ValueError(
            "source_times must have shape (n_samples, seq_len) matching values, "
            f"got {tuple(source_times_arr.shape)}"
        )
    if target_times_arr.ndim != 2 or target_times_arr.shape[0] != values_arr.shape[0]:
        raise ValueError(
            "target_times must have shape (n_samples, n_target_times), "
            f"got {tuple(target_times_arr.shape)}"
        )

    n_samples, seq_len, n_covariates = values_arr.shape
    if lengths is None:
        lengths_arr = np.full(n_samples, seq_len, dtype=np.int64)
    else:
        lengths_arr = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if lengths_arr.shape != (n_samples,):
            raise ValueError(
                "lengths must have shape (n_samples,), "
                f"got {tuple(lengths_arr.shape)}"
            )

    aligned = np.full(
        (n_samples, target_times_arr.shape[1], n_covariates),
        fill_value,
        dtype=np.float32,
    )

    for i in range(n_samples):
        valid_len = int(lengths_arr[i])
        if valid_len <= 0:
            continue
        subject_times = source_times_arr[i, :valid_len]
        subject_values = values_arr[i, :valid_len]
        insertion_points = np.searchsorted(
            subject_times, target_times_arr[i], side="right"
        ) - 1
        valid_targets = insertion_points >= 0
        if np.any(valid_targets):
            aligned[i, valid_targets] = subject_values[insertion_points[valid_targets]]

    return aligned


def _encode_feature_block(df, columns, category_maps=None):
    """Encode numeric/categorical covariate blocks into a float32 matrix."""
    import pandas as pd

    if not columns:
        return np.zeros((len(df), 0), dtype=np.float32), {}

    encoded = np.zeros((len(df), len(columns)), dtype=np.float32)
    learned_maps = {}

    for j, col in enumerate(columns):
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            encoded[:, j] = np.nan_to_num(
                series.to_numpy(dtype=np.float32, copy=True),
                nan=0.0,
            )
            continue

        mapping = None if category_maps is None else category_maps.get(col)
        string_values = series.astype("string")
        observed_values = [value for value in string_values.dropna().unique()]
        if mapping is None:
            mapping = {
                str(value): float(idx + 1)
                for idx, value in enumerate(observed_values)
            }

        encoded_col = np.zeros(len(series), dtype=np.float32)
        observed_mask = string_values.notna().to_numpy()
        if np.any(observed_mask):
            observed_strings = string_values[observed_mask].astype(str).to_numpy()
            unknown = sorted(set(observed_strings) - set(mapping))
            if unknown:
                raise ValueError(
                    f"Column '{col}' contains unseen categories: {unknown}. "
                    "Pass category_maps learned on the training set."
                )
            encoded_col[observed_mask] = np.array(
                [mapping[value] for value in observed_strings],
                dtype=np.float32,
            )

        encoded[:, j] = encoded_col
        learned_maps[col] = mapping

    return encoded, learned_maps


def build_joint_dataset_inputs(
    longitudinal_df,
    survival_df,
    subject_col,
    time_col,
    feature_cols,
    event_time_col,
    event_indicator_col,
    baseline_cols=None,
    time_varying_covariate_cols=None,
    event_covariate_cols=None,
    sort_by=None,
    category_maps=None,
):
    """
    Build joint-model dataset inputs from longitudinal and survival tables.

    The returned dictionary can be passed directly into
    :class:`JointLongitudinalSurvivalDataset`. Feature columns and event
    outcomes must be numeric. Baseline, time-varying, and event covariates may
    be numeric or categorical; categorical columns are encoded with integer
    codes starting at 1 so that 0 can represent missing values.
    """
    import pandas as pd

    feature_cols = list(feature_cols)
    baseline_cols = list(baseline_cols or [])
    time_varying_covariate_cols = list(time_varying_covariate_cols or [])
    event_covariate_cols = list(event_covariate_cols or [])
    sort_keys = list(sort_by or [time_col])

    required_long_cols = {
        subject_col,
        time_col,
        *feature_cols,
        *sort_keys,
        *time_varying_covariate_cols,
    }
    required_surv_cols = {
        subject_col,
        event_time_col,
        event_indicator_col,
        *baseline_cols,
        *event_covariate_cols,
    }

    missing_long = sorted(col for col in required_long_cols if col not in longitudinal_df.columns)
    if missing_long:
        raise ValueError(f"longitudinal_df is missing required columns: {missing_long}")

    missing_surv = sorted(col for col in required_surv_cols if col not in survival_df.columns)
    if missing_surv:
        raise ValueError(f"survival_df is missing required columns: {missing_surv}")

    if survival_df[subject_col].duplicated().any():
        raise ValueError("survival_df must contain exactly one row per subject")

    numeric_long_cols = [time_col, *feature_cols]
    numeric_surv_cols = [event_time_col, event_indicator_col]
    for col in numeric_long_cols:
        if not pd.api.types.is_numeric_dtype(longitudinal_df[col]):
            raise ValueError(f"Longitudinal column '{col}' must be numeric")
    for col in numeric_surv_cols:
        if not pd.api.types.is_numeric_dtype(survival_df[col]):
            raise ValueError(f"Survival column '{col}' must be numeric")

    long_sorted = longitudinal_df.sort_values([subject_col, *sort_keys]).reset_index(drop=True)
    surv_sorted = survival_df.sort_values(subject_col).reset_index(drop=True)

    long_subject_ids = long_sorted[subject_col].drop_duplicates().to_numpy()
    surv_subject_ids = surv_sorted[subject_col].to_numpy()
    if not np.array_equal(long_subject_ids, surv_subject_ids):
        raise ValueError(
            "longitudinal_df and survival_df must contain the same subjects "
            "with exactly one survival row per subject"
        )

    learned_maps = {
        "baseline_covariates": {},
        "time_varying_covariates": {},
        "event_covariates": {},
    }
    category_maps = category_maps or {}

    baseline_covariates, baseline_maps = _encode_feature_block(
        surv_sorted,
        baseline_cols,
        category_maps=category_maps.get("baseline_covariates"),
    )
    learned_maps["baseline_covariates"] = baseline_maps

    event_covariates, event_cov_maps = _encode_feature_block(
        surv_sorted,
        event_covariate_cols,
        category_maps=category_maps.get("event_covariates"),
    )
    learned_maps["event_covariates"] = event_cov_maps

    time_varying_covariates_all, tvc_maps = _encode_feature_block(
        long_sorted,
        time_varying_covariate_cols,
        category_maps=category_maps.get("time_varying_covariates"),
    )
    learned_maps["time_varying_covariates"] = tvc_maps

    data_sequences = []
    mask_sequences = []
    time_sequences = []
    tvc_sequences = []

    for _, group in long_sorted.groupby(subject_col, sort=True):
        feature_block = group[feature_cols].to_numpy(dtype=np.float32, copy=True)
        data_sequences.append(np.nan_to_num(feature_block, nan=0.0).astype(np.float32))
        mask_sequences.append((~np.isnan(feature_block)).astype(np.float32))
        time_sequences.append(group[time_col].to_numpy(dtype=np.float32, copy=True))
        tvc_sequences.append(
            time_varying_covariates_all[group.index.to_numpy()].astype(np.float32, copy=True)
        )

    event_times = surv_sorted[event_time_col].to_numpy(dtype=np.float32, copy=True)
    event_indicators = surv_sorted[event_indicator_col].to_numpy(dtype=np.float32, copy=True)
    if not np.all(np.isfinite(event_times)):
        raise ValueError("event_time_col must contain only finite values")
    if np.any(event_times < 0.0):
        raise ValueError("event_time_col must be non-negative")
    if not np.all(np.isin(event_indicators, [0.0, 1.0])):
        raise ValueError("event_indicator_col must contain only 0/1 values")

    return {
        "data": data_sequences,
        "mask": mask_sequences,
        "baseline_covariates": baseline_covariates.astype(np.float32, copy=False),
        "times": time_sequences,
        "time_varying_covariates": tvc_sequences,
        "event_times": event_times,
        "event_indicators": event_indicators,
        "event_covariates": event_covariates.astype(np.float32, copy=False),
        "subject_ids": surv_subject_ids.copy(),
        "categorical_maps": learned_maps,
    }


def split_joint_tables_by_fold(
    longitudinal_df,
    survival_df,
    subject_col,
    fold_col="fold",
    fold_id=0,
):
    """
    Split full joint-model tables into train/test sets using subject-level folds.

    The survival table is treated as the source of truth for fold assignments.
    This is useful when longitudinal exports do not already carry a reliable
    fold column for every row.
    """
    required_surv_cols = {subject_col, fold_col}
    missing_surv = sorted(col for col in required_surv_cols if col not in survival_df.columns)
    if missing_surv:
        raise ValueError(f"survival_df is missing required columns: {missing_surv}")
    if subject_col not in longitudinal_df.columns:
        raise ValueError(f"longitudinal_df is missing required column: '{subject_col}'")

    if survival_df[subject_col].duplicated().any():
        raise ValueError("survival_df must contain exactly one row per subject")

    fold_mapping = survival_df[[subject_col, fold_col]].copy()
    long_with_fold = longitudinal_df.drop(columns=[fold_col], errors="ignore").merge(
        fold_mapping,
        on=subject_col,
        how="left",
    )
    if long_with_fold[fold_col].isna().any():
        missing_subjects = sorted(
            long_with_fold.loc[long_with_fold[fold_col].isna(), subject_col]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Some longitudinal subjects do not have a fold assignment in survival_df: "
            f"{missing_subjects[:10]}"
        )

    train_longitudinal = long_with_fold[long_with_fold[fold_col] != fold_id].copy()
    test_longitudinal = long_with_fold[long_with_fold[fold_col] == fold_id].copy()
    train_survival = survival_df[survival_df[fold_col] != fold_id].copy()
    test_survival = survival_df[survival_df[fold_col] == fold_id].copy()

    return {
        "train_longitudinal": train_longitudinal.reset_index(drop=True),
        "train_survival": train_survival.reset_index(drop=True),
        "test_longitudinal": test_longitudinal.reset_index(drop=True),
        "test_survival": test_survival.reset_index(drop=True),
    }


def generate_synthetic_longitudinal_data(n_samples=1000, seq_len=50, n_features=5,
                                         noise_level=0.1, seed=None):
    """
    Generate synthetic longitudinal data for testing.

    Creates data with temporal patterns (trends, seasonality).

    Args:
        n_samples: Number of samples to generate
        seq_len: Length of each sequence
        n_features: Number of features per time step
        noise_level: Amount of noise to add
        seed: Random seed for reproducibility

    Returns:
        data: Numpy array of shape (n_samples, seq_len, n_features)
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((n_samples, seq_len, n_features))

    for i in range(n_samples):
        # Generate temporal patterns
        t = np.linspace(0, 4*np.pi, seq_len)

        for j in range(n_features):
            # Combine trend, seasonality, and noise
            trend = np.random.randn() * t / (4*np.pi)
            seasonality = np.sin(t + np.random.rand() * 2 * np.pi) * np.random.rand()
            noise = np.random.randn(seq_len) * noise_level

            data[i, :, j] = trend + seasonality + noise

    return data.astype(np.float32)


def generate_synthetic_joint_longitudinal_survival_data(
    n_samples=500,
    seq_len=20,
    n_features=1,
    n_baseline_features=2,
    n_time_varying_covariates=1,
    n_event_covariates=1,
    noise_level=0.1,
    association_strength=0.75,
    baseline_hazard=0.1,
    censoring_rate=0.05,
    seed=None,
):
    """
    Generate a toy joint longitudinal-survival dataset with a shared latent effect.

    The latent subject effect drives both the longitudinal trajectory and the
    event rate, making it suitable for smoke tests of a shared-random-effect
    joint model.
    """
    rng = np.random.default_rng(seed)

    times = np.tile(
        np.linspace(0.0, 1.0, seq_len, dtype=np.float32),
        (n_samples, 1),
    )

    baseline_covariates = rng.normal(
        size=(n_samples, n_baseline_features)
    ).astype(np.float32)
    time_varying_covariates = rng.normal(
        size=(n_samples, seq_len, n_time_varying_covariates)
    ).astype(np.float32)
    event_covariates = rng.normal(
        size=(n_samples, n_event_covariates)
    ).astype(np.float32)

    shared_effect = rng.normal(size=n_samples).astype(np.float32)
    data = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)

    baseline_effect = (
        baseline_covariates[:, 0]
        if n_baseline_features > 0
        else np.zeros(n_samples, dtype=np.float32)
    )
    time_varying_effect = (
        time_varying_covariates[:, :, 0]
        if n_time_varying_covariates > 0
        else np.zeros((n_samples, seq_len), dtype=np.float32)
    )

    for j in range(n_features):
        feature_scale = 1.0 + 0.2 * j
        trend = feature_scale * times
        seasonal = 0.25 * np.sin(2.0 * np.pi * times + 0.4 * j)
        latent_trajectory = (
            shared_effect[:, None]
            + 0.5 * baseline_effect[:, None]
            + 0.3 * time_varying_effect
            + trend
            + seasonal
        )
        data[:, :, j] = latent_trajectory + rng.normal(
            scale=noise_level, size=(n_samples, seq_len)
        )

    linear_predictor = np.log(max(baseline_hazard, 1e-6))
    linear_predictor = linear_predictor + association_strength * shared_effect
    if n_event_covariates > 0:
        linear_predictor = linear_predictor + 0.25 * event_covariates.sum(axis=1)
    linear_predictor = np.clip(linear_predictor, -6.0, 6.0)

    event_rate = np.exp(linear_predictor)
    true_event_times = rng.exponential(
        scale=1.0 / np.maximum(event_rate, 1e-6),
        size=n_samples,
    ).astype(np.float32)
    censor_times = rng.exponential(
        scale=1.0 / max(censoring_rate, 1e-6),
        size=n_samples,
    ).astype(np.float32)

    event_indicators = (true_event_times <= censor_times).astype(np.float32)
    event_times = np.minimum(true_event_times, censor_times).astype(np.float32)

    return {
        "data": data,
        "mask": np.ones_like(data, dtype=np.float32),
        "baseline_covariates": baseline_covariates,
        "times": times,
        "time_varying_covariates": time_varying_covariates,
        "event_times": event_times,
        "event_indicators": event_indicators,
        "event_covariates": event_covariates,
        "shared_effect": shared_effect,
    }


def generate_mixed_longitudinal_data(n_samples=1000, seq_len=50, var_config=None,
                                     n_baseline_features=0, noise_level=0.1,
                                     random_intercept_sd=0.0, seed=None):
    """
    Generate synthetic longitudinal data with mixed variable types.

    Args:
        n_samples: Number of samples to generate
        seq_len: Length of each sequence
        var_config: VariableConfig specifying variable types. If None, creates
                    a default config with 2 continuous, 2 binary, 1 bounded variable.
        n_baseline_features: Number of baseline (time-invariant) features to generate
        noise_level: Amount of noise to add
        random_intercept_sd: Standard deviation of a per-subject random intercept
            added to the latent trajectory. Larger values create more between-subject
            variability (default: 0.0, no intercept).
        seed: Random seed for reproducibility

    Returns:
        data: Numpy array of shape (n_samples, seq_len, n_features)
        baseline: Numpy array of shape (n_samples, n_baseline_features) or None
    """
    if seed is not None:
        np.random.seed(seed)

    if var_config is None:
        var_config = VariableConfig(variables=[
            VariableSpec(name='continuous_1', var_type='continuous'),
            VariableSpec(name='continuous_2', var_type='continuous'),
            VariableSpec(name='binary_1', var_type='binary'),
            VariableSpec(name='binary_2', var_type='binary'),
            VariableSpec(name='bounded_1', var_type='bounded', lower=0.0, upper=1.0),
        ])

    n_features = var_config.n_features
    data = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)

    for i in range(n_samples):
        t = np.linspace(0, 4 * np.pi, seq_len)

        for j, var_spec in enumerate(var_config.variables):
            # Per-subject random intercept
            intercept = np.random.randn() * random_intercept_sd

            # Generate a latent smooth trajectory
            trend = np.random.randn() * t / (4 * np.pi)
            seasonality = np.sin(t + np.random.rand() * 2 * np.pi) * np.random.rand()
            noise = np.random.randn(seq_len) * noise_level
            latent = intercept + trend + seasonality + noise

            if var_spec.var_type == 'continuous':
                data[i, :, j] = latent

            elif var_spec.var_type == 'binary':
                # Sigmoid of latent, then threshold at 0.5
                prob = 1.0 / (1.0 + np.exp(-latent))
                data[i, :, j] = (np.random.rand(seq_len) < prob).astype(np.float32)

            elif var_spec.var_type == 'bounded':
                # Sigmoid to [0, 1], then scale to [lower, upper]
                sig = 1.0 / (1.0 + np.exp(-latent))
                data[i, :, j] = sig * (var_spec.upper - var_spec.lower) + var_spec.lower

    # Generate baseline covariates
    baseline = None
    if n_baseline_features > 0:
        baseline = np.zeros((n_samples, n_baseline_features), dtype=np.float32)
        for j in range(n_baseline_features):
            if j % 2 == 0:
                # Continuous baseline
                baseline[:, j] = np.random.randn(n_samples).astype(np.float32)
            else:
                # Binary baseline
                baseline[:, j] = (np.random.rand(n_samples) > 0.5).astype(np.float32)

    return data, baseline


def create_missing_mask(data_shape, missing_rate=0.2, pattern='random', seed=None):
    """
    Create a binary mask for missing data.

    Args:
        data_shape: Shape of the data (n_samples, seq_len, n_features)
        missing_rate: Proportion of values to mark as missing (0.0 to 1.0)
        pattern: Missing data pattern - 'random', 'block', or 'monotone'
                 - 'random': Random missing values throughout
                 - 'block': Contiguous blocks of missing values in time
                 - 'monotone': Monotone missingness (if t is missing, all t+1, t+2... are missing)
        seed: Random seed for reproducibility

    Returns:
        mask: Binary mask array where 1=observed, 0=missing
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, seq_len, n_features = data_shape
    mask = np.ones(data_shape, dtype=np.float32)

    if pattern == 'random':
        # Random missing values
        missing_indices = np.random.rand(*data_shape) < missing_rate
        mask[missing_indices] = 0.0

    elif pattern == 'block':
        # Contiguous blocks of missing values in time
        for i in range(n_samples):
            for j in range(n_features):
                # Randomly select number of blocks
                n_blocks = max(1, int(missing_rate * seq_len / 5))
                for _ in range(n_blocks):
                    # Random block start and length
                    start = np.random.randint(0, seq_len)
                    length = np.random.randint(1, max(2, int(seq_len * 0.2)))
                    end = min(start + length, seq_len)
                    mask[i, start:end, j] = 0.0

    elif pattern == 'monotone':
        # Monotone missingness pattern
        for i in range(n_samples):
            for j in range(n_features):
                if np.random.rand() < missing_rate:
                    # Random dropout point
                    dropout_point = np.random.randint(0, seq_len)
                    mask[i, dropout_point:, j] = 0.0

    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use 'random', 'block', or 'monotone'")

    return mask
