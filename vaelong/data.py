"""
Data utilities for longitudinal measurements.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LongitudinalDataset(Dataset):
    """
    Dataset class for longitudinal measurements.
    
    Args:
        data: Numpy array of shape (n_samples, seq_len, n_features) or list of sequences
        normalize: Whether to normalize the data (default: True)
        padding_value: Value to use for padding shorter sequences (default: 0.0)
    """
    
    def __init__(self, data, normalize=True, padding_value=0.0):
        if isinstance(data, list):
            # Handle variable length sequences
            self.data, self.lengths = self._pad_sequences(data, padding_value)
        else:
            # Fixed length sequences
            self.data = torch.FloatTensor(data)
            self.lengths = torch.LongTensor([data.shape[1]] * len(data))
        
        if normalize:
            self.mean = self.data.mean(dim=(0, 1), keepdim=True)
            self.std = self.data.std(dim=(0, 1), keepdim=True)
            self.std[self.std == 0] = 1.0  # Avoid division by zero
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
        return self.data[idx], self.lengths[idx]
    
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
