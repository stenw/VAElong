"""
Data utilities for longitudinal measurements.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LongitudinalDataset(Dataset):
    """
    Dataset class for longitudinal measurements with missing data support.

    Args:
        data: Numpy array of shape (n_samples, seq_len, n_features) or list of sequences
        mask: Optional binary mask of same shape as data (1=observed, 0=missing)
        normalize: Whether to normalize the data (default: True)
        padding_value: Value to use for padding shorter sequences (default: 0.0)
    """

    def __init__(self, data, mask=None, normalize=True, padding_value=0.0):
        if isinstance(data, list):
            # Handle variable length sequences
            self.data, self.lengths = self._pad_sequences(data, padding_value)
        else:
            # Fixed length sequences
            self.data = torch.FloatTensor(data)
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

        if normalize:
            # Compute statistics only on observed values
            observed_data = self.data * self.mask
            n_observed = self.mask.sum(dim=(0, 1), keepdim=True)
            n_observed[n_observed == 0] = 1.0  # Avoid division by zero

            self.mean = observed_data.sum(dim=(0, 1), keepdim=True) / n_observed
            self.std = torch.sqrt(((observed_data - self.mean * self.mask) ** 2).sum(dim=(0, 1), keepdim=True) / n_observed)
            self.std[self.std == 0] = 1.0  # Avoid division by zero

            # Normalize only observed values
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
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
        return self.data[idx], self.mask[idx], self.lengths[idx]
    
    def inverse_transform(self, data):
        """
        Transform normalized data back to original scale.
        
        Args:
            data: Normalized data tensor
            
        Returns:
            Denormalized data
        """
        if self.mean is not None and self.std is not None:
            return data * self.std + self.mean
        return data


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
