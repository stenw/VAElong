"""
Unit tests for data utilities.
"""

import unittest
import numpy as np
import torch

from vaelong.data import LongitudinalDataset, generate_synthetic_longitudinal_data


class TestLongitudinalDataset(unittest.TestCase):
    """Test cases for LongitudinalDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_samples = 100
        self.seq_len = 50
        self.n_features = 5
        
        self.data = np.random.randn(self.n_samples, self.seq_len, self.n_features).astype(np.float32)
    
    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        dataset = LongitudinalDataset(self.data, normalize=False)
        
        self.assertEqual(len(dataset), self.n_samples)
        self.assertEqual(dataset.data.shape, torch.Size([self.n_samples, self.seq_len, self.n_features]))
    
    def test_normalization(self):
        """Test data normalization."""
        dataset = LongitudinalDataset(self.data, normalize=True)
        
        # Check that mean is close to 0 and std is close to 1
        data_mean = dataset.data.mean(dim=(0, 1))
        data_std = dataset.data.std(dim=(0, 1))
        
        self.assertTrue(torch.allclose(data_mean, torch.zeros_like(data_mean), atol=1e-5))
        self.assertTrue(torch.allclose(data_std, torch.ones_like(data_std), atol=1e-1))
    
    def test_getitem(self):
        """Test getting items from dataset."""
        dataset = LongitudinalDataset(self.data, normalize=False)
        
        item, length = dataset[0]
        
        self.assertEqual(item.shape, torch.Size([self.seq_len, self.n_features]))
        self.assertEqual(length.item(), self.seq_len)
    
    def test_variable_length_sequences(self):
        """Test dataset with variable length sequences."""
        # Create sequences of different lengths
        sequences = [
            np.random.randn(30, self.n_features).astype(np.float32),
            np.random.randn(40, self.n_features).astype(np.float32),
            np.random.randn(50, self.n_features).astype(np.float32),
        ]
        
        dataset = LongitudinalDataset(sequences, normalize=False)
        
        self.assertEqual(len(dataset), 3)
        # All should be padded to max length (50)
        self.assertEqual(dataset.data.shape[1], 50)
        
        # Check lengths are correct
        self.assertEqual(dataset.lengths[0].item(), 30)
        self.assertEqual(dataset.lengths[1].item(), 40)
        self.assertEqual(dataset.lengths[2].item(), 50)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        dataset = LongitudinalDataset(self.data, normalize=True)
        
        # Get normalized data
        normalized, _ = dataset[0]
        
        # Inverse transform
        denormalized = dataset.inverse_transform(normalized)
        
        # Should be close to original
        original = torch.FloatTensor(self.data[0])
        self.assertTrue(torch.allclose(denormalized, original, atol=1e-5))


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test cases for synthetic data generation."""
    
    def test_generate_data_shape(self):
        """Test generated data has correct shape."""
        n_samples = 100
        seq_len = 50
        n_features = 5
        
        data = generate_synthetic_longitudinal_data(
            n_samples=n_samples,
            seq_len=seq_len,
            n_features=n_features,
            seed=42
        )
        
        self.assertEqual(data.shape, (n_samples, seq_len, n_features))
        self.assertEqual(data.dtype, np.float32)
    
    def test_generate_data_deterministic(self):
        """Test data generation is deterministic with seed."""
        data1 = generate_synthetic_longitudinal_data(
            n_samples=10,
            seq_len=20,
            n_features=3,
            seed=42
        )
        
        data2 = generate_synthetic_longitudinal_data(
            n_samples=10,
            seq_len=20,
            n_features=3,
            seed=42
        )
        
        np.testing.assert_array_equal(data1, data2)
    
    def test_generate_data_different_seeds(self):
        """Test different seeds produce different data."""
        data1 = generate_synthetic_longitudinal_data(
            n_samples=10,
            seq_len=20,
            n_features=3,
            seed=42
        )
        
        data2 = generate_synthetic_longitudinal_data(
            n_samples=10,
            seq_len=20,
            n_features=3,
            seed=43
        )
        
        # Data should be different
        self.assertFalse(np.array_equal(data1, data2))
    
    def test_noise_level(self):
        """Test noise level affects variability."""
        data_low_noise = generate_synthetic_longitudinal_data(
            n_samples=100,
            seq_len=50,
            n_features=5,
            noise_level=0.01,
            seed=42
        )
        
        data_high_noise = generate_synthetic_longitudinal_data(
            n_samples=100,
            seq_len=50,
            n_features=5,
            noise_level=1.0,
            seed=42
        )
        
        # High noise data should have higher variance
        var_low = np.var(data_low_noise)
        var_high = np.var(data_high_noise)
        
        self.assertLess(var_low, var_high)


if __name__ == '__main__':
    unittest.main()
