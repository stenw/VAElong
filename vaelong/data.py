"""
Data utilities for longitudinal measurements.
"""

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
    """

    def __init__(self, data, mask=None, normalize=True, padding_value=0.0,
                 baseline_covariates=None, var_config=None):
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
        return self.data[idx], self.mask[idx], self.lengths[idx], self.baseline[idx]

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
