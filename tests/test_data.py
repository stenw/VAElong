"""
Unit tests for data utilities.
"""

import unittest
import numpy as np
import torch

from vaelong.data import LongitudinalDataset, generate_synthetic_longitudinal_data, create_missing_mask


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

        item, mask, length = dataset[0]

        self.assertEqual(item.shape, torch.Size([self.seq_len, self.n_features]))
        self.assertEqual(mask.shape, torch.Size([self.seq_len, self.n_features]))
        self.assertEqual(length.item(), self.seq_len)
        # Mask should be all ones (no missing data)
        self.assertTrue(torch.all(mask == 1.0))
    
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
        normalized, mask, _ = dataset[0]

        # Inverse transform
        denormalized = dataset.inverse_transform(normalized)

        # Should be close to original
        original = torch.FloatTensor(self.data[0])
        self.assertTrue(torch.allclose(denormalized, original, atol=1e-5))

    def test_dataset_with_mask(self):
        """Test dataset with missing data mask."""
        # Create mask with some missing values
        mask = np.ones((self.n_samples, self.seq_len, self.n_features), dtype=np.float32)
        mask[:, :10, :] = 0  # First 10 timesteps are missing

        dataset = LongitudinalDataset(self.data, mask=mask, normalize=False)

        self.assertEqual(len(dataset), self.n_samples)

        item, item_mask, length = dataset[0]

        # Check mask is correctly loaded
        self.assertEqual(item_mask.shape, torch.Size([self.seq_len, self.n_features]))
        self.assertTrue(torch.all(item_mask[:10, :] == 0))
        self.assertTrue(torch.all(item_mask[10:, :] == 1))

    def test_normalization_with_missing_data(self):
        """Test normalization computes statistics only on observed values."""
        # Create data and mask
        mask = np.ones((self.n_samples, self.seq_len, self.n_features), dtype=np.float32)
        mask[:, :10, :] = 0  # First 10 timesteps are missing

        dataset = LongitudinalDataset(self.data, mask=mask, normalize=True)

        # Mean and std should be computed only on observed values
        self.assertIsNotNone(dataset.mean)
        self.assertIsNotNone(dataset.std)

        # Check that normalization is applied
        observed_data = dataset.data * dataset.mask
        # Mean of observed values should be close to 0
        n_observed = dataset.mask.sum(dim=(0, 1), keepdim=True)
        data_mean = observed_data.sum(dim=(0, 1), keepdim=True) / n_observed

        self.assertTrue(torch.allclose(data_mean, torch.zeros_like(data_mean), atol=1e-5))


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


class TestMissingMaskCreation(unittest.TestCase):
    """Test cases for missing mask creation."""

    def test_random_mask_shape(self):
        """Test random mask has correct shape."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='random', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertEqual(mask.dtype, np.float32)

    def test_random_mask_values(self):
        """Test random mask contains only 0 and 1."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='random', seed=42)

        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_random_mask_missing_rate(self):
        """Test random mask has approximately correct missing rate."""
        shape = (1000, 50, 5)
        missing_rate = 0.2
        mask = create_missing_mask(shape, missing_rate=missing_rate, pattern='random', seed=42)

        actual_missing_rate = 1 - mask.mean()
        # Allow 5% tolerance
        self.assertAlmostEqual(actual_missing_rate, missing_rate, delta=0.05)

    def test_block_mask(self):
        """Test block mask creation."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='block', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_monotone_mask(self):
        """Test monotone mask creation."""
        shape = (100, 50, 5)
        mask = create_missing_mask(shape, missing_rate=0.2, pattern='monotone', seed=42)

        self.assertEqual(mask.shape, shape)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

        # Check monotone property: if value at time t is missing,
        # all subsequent values should be missing
        for i in range(10):  # Check first 10 samples
            for j in range(shape[2]):  # Check all features
                seq = mask[i, :, j]
                first_missing = np.where(seq == 0)[0]
                if len(first_missing) > 0:
                    first_missing_idx = first_missing[0]
                    # All values from first_missing_idx onwards should be 0
                    self.assertTrue(np.all(seq[first_missing_idx:] == 0))

    def test_mask_deterministic(self):
        """Test mask creation is deterministic with seed."""
        shape = (50, 30, 5)
        mask1 = create_missing_mask(shape, missing_rate=0.3, pattern='random', seed=42)
        mask2 = create_missing_mask(shape, missing_rate=0.3, pattern='random', seed=42)

        np.testing.assert_array_equal(mask1, mask2)

    def test_invalid_pattern(self):
        """Test invalid pattern raises error."""
        shape = (10, 20, 5)
        with self.assertRaises(ValueError):
            create_missing_mask(shape, missing_rate=0.2, pattern='invalid')

    def test_zero_missing_rate(self):
        """Test zero missing rate creates all-ones mask."""
        shape = (50, 30, 5)
        mask = create_missing_mask(shape, missing_rate=0.0, pattern='random', seed=42)

        # With missing_rate=0, most values should be 1
        self.assertGreater(mask.mean(), 0.95)

    def test_high_missing_rate(self):
        """Test high missing rate creates mostly zeros."""
        shape = (50, 30, 5)
        mask = create_missing_mask(shape, missing_rate=0.9, pattern='random', seed=42)

        # With missing_rate=0.9, most values should be 0
        self.assertLess(mask.mean(), 0.2)


if __name__ == '__main__':
    unittest.main()
