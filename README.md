# VAElong

A Variational Autoencoder (VAE) implementation for longitudinal (time-series) measurements in Python using PyTorch.

## Overview

VAElong provides a flexible and easy-to-use implementation of VAEs designed specifically for longitudinal data. The package includes both LSTM/GRU-based and CNN-based architectures for handling sequential data, with built-in support for missing data handling through EM-like imputation.

## Features

- **Multiple Architectures**:
  - LSTM/GRU-based VAE for traditional sequential modeling
  - CNN-based VAE for efficient processing of time series
- **Missing Data Handling**:
  - Binary masking for missing values
  - EM-like imputation during training
  - Multiple missing data patterns (random, block, monotone)
- **Flexible latent space** dimension configuration
- **Beta-VAE support** for controlling the trade-off between reconstruction and KL divergence
- **Built-in training utilities** with easy-to-use trainer class
- **Data preprocessing** for longitudinal measurements
- **Synthetic data generation** for testing and experimentation
- **Julia translation** available for high-performance computing
- **Comprehensive examples** and documentation

## Installation

### From source

```bash
git clone https://github.com/stenw/VAElong.git
cd VAElong
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0

## Quick Start

```python
import torch
from torch.utils.data import DataLoader
from vaelong import LongitudinalVAE, VAETrainer, LongitudinalDataset
from vaelong.data import generate_synthetic_longitudinal_data

# Generate synthetic longitudinal data
data = generate_synthetic_longitudinal_data(
    n_samples=1000,
    seq_len=50,
    n_features=5,
    noise_level=0.1,
    seed=42
)

# Create dataset
dataset = LongitudinalDataset(data, normalize=True)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create VAE model
model = LongitudinalVAE(
    input_dim=5,
    hidden_dim=64,
    latent_dim=10,
    num_layers=1,
    use_gru=False
)

# Create trainer and train
trainer = VAETrainer(model, learning_rate=1e-3, beta=1.0)
history = trainer.fit(train_loader, epochs=50, verbose=True)

# Generate new samples
generated_samples = model.sample(num_samples=10, seq_len=50)
```

## Architecture

### LSTM/GRU-based VAE

1. **Encoder**: LSTM/GRU layers that process the input sequence and output parameters (mean and log variance) of the latent distribution
2. **Reparameterization**: Samples from the latent distribution using the reparameterization trick
3. **Decoder**: Transforms latent representation back to sequence space using LSTM/GRU layers

### CNN-based VAE

1. **Encoder**:
   - Multiple 1D convolutional layers with stride 2 for downsampling
   - Batch normalization and ReLU activations
   - Fully connected layers to latent distribution parameters
2. **Reparameterization**: Samples from the latent distribution using the reparameterization trick
3. **Decoder**:
   - Fully connected layer from latent space
   - Multiple 1D transposed convolutional layers for upsampling
   - Batch normalization and ReLU activations
   - Output matches input sequence dimensions

### Loss Function

The VAE loss consists of two components:

```
Loss = Reconstruction Loss + β × KL Divergence
```

- **Reconstruction Loss**: Mean Squared Error (MSE) between input and reconstructed sequences (only on observed values when handling missing data)
- **KL Divergence**: Measures how much the learned latent distribution diverges from a standard normal distribution
- **β**: Weight parameter for KL divergence (β-VAE variant)

### Missing Data Handling

The package supports missing data through:

1. **Binary Masking**: A mask tensor indicates which values are observed (1) and which are missing (0)
2. **Masked Loss**: Reconstruction loss is computed only on observed values
3. **EM-like Imputation**: During training, alternates between:
   - **E-step**: Generate predictions for missing values and sample from them
   - **M-step**: Update model parameters given the imputed data
4. **Missing Data Patterns**:
   - **Random**: Random missing values throughout the data
   - **Block**: Contiguous blocks of missing values in time
   - **Monotone**: If value at time t is missing, all subsequent values are missing

## Usage Examples

### Basic Training (LSTM/GRU-based)

```python
from vaelong import LongitudinalVAE, VAETrainer, LongitudinalDataset
import numpy as np

# Prepare your data (shape: n_samples, seq_len, n_features)
data = np.random.randn(1000, 50, 5)

# Create dataset
dataset = LongitudinalDataset(data, normalize=True)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create and train model
model = LongitudinalVAE(input_dim=5, hidden_dim=64, latent_dim=10)
trainer = VAETrainer(model)
history = trainer.fit(train_loader, epochs=100)
```

### CNN-based VAE

```python
from vaelong import CNNLongitudinalVAE, VAETrainer, LongitudinalDataset

# Create CNN-based model
model = CNNLongitudinalVAE(
    input_dim=5,
    seq_len=64,  # Power of 2 works well for CNNs
    latent_dim=16,
    hidden_channels=[32, 64, 128],  # Channel progression
    kernel_size=3
)

# Train as usual
trainer = VAETrainer(model, learning_rate=1e-3, beta=1.0)
history = trainer.fit(train_loader, epochs=100)

# Generate samples
samples = model.sample(num_samples=10, device='cpu')
```

### Handling Missing Data

```python
from vaelong import CNNLongitudinalVAE, VAETrainer, LongitudinalDataset
from vaelong.data import generate_synthetic_longitudinal_data, create_missing_mask

# Generate data with missing values
data = generate_synthetic_longitudinal_data(
    n_samples=1000,
    seq_len=64,
    n_features=5,
    noise_level=0.1,
    seed=42
)

# Create missing data mask (20% missing)
mask = create_missing_mask(
    data.shape,
    missing_rate=0.2,
    pattern='random',  # or 'block', 'monotone'
    seed=42
)

# Apply mask (zero out missing values)
data_with_missing = data * mask

# Create dataset with mask
dataset = LongitudinalDataset(data_with_missing, mask=mask, normalize=True)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model and trainer
model = CNNLongitudinalVAE(input_dim=5, seq_len=64, latent_dim=16)
trainer = VAETrainer(model, learning_rate=1e-3, beta=1.0)

# Train with EM-like imputation
history = trainer.fit(
    train_loader,
    epochs=100,
    use_em_imputation=True,  # Enable EM imputation
    em_iterations=3  # Number of EM iterations per batch
)

# Impute missing values after training
imputed_data = model.impute_missing(data_tensor, mask_tensor, num_iterations=5)
```

### Variable Length Sequences

```python
# Create sequences of different lengths
sequences = [
    np.random.randn(30, 5),
    np.random.randn(45, 5),
    np.random.randn(60, 5),
]

# Dataset automatically handles padding
dataset = LongitudinalDataset(sequences, normalize=True)
```

### Using GRU Instead of LSTM

```python
model = LongitudinalVAE(
    input_dim=5,
    hidden_dim=64,
    latent_dim=10,
    use_gru=True  # Use GRU instead of LSTM
)
```

### Beta-VAE

```python
# Lower beta emphasizes reconstruction quality
trainer = VAETrainer(model, beta=0.5)

# Higher beta emphasizes learning disentangled representations
trainer = VAETrainer(model, beta=2.0)
```

### Generating New Samples

```python
# Generate 10 new sequences of length 50
new_samples = model.sample(num_samples=10, seq_len=50, device='cpu')
```

### Saving and Loading Models

```python
# Save model
trainer.save_model('my_vae_model.pth')

# Load model
new_trainer = VAETrainer(model)
new_trainer.load_model('my_vae_model.pth')
```

## Examples

See the `examples/` directory for complete examples:

- `basic_example.py`: Complete workflow with LSTM/GRU-based VAE including data generation, training, and visualization
- `cnn_missing_data_example.py`: CNN-based VAE with missing data handling, EM imputation, and comparison

Run the examples:

```bash
cd examples
python basic_example.py
python cnn_missing_data_example.py
```

The CNN missing data example will:
- Generate synthetic longitudinal data with missing values
- Train both baseline and EM-imputation models
- Compare imputation quality
- Create comprehensive visualizations
- Save results and trained models

## Julia Translation

A complete Julia translation of the package is available in the `julia/VAElong/` directory, using Flux.jl for deep learning. See `julia/VAElong/README.md` for installation and usage instructions.

## Testing

Run the test suite:

```bash
cd tests
python -m unittest discover
```

Or run specific test modules:

```bash
python -m unittest test_model
python -m unittest test_data
python -m unittest test_trainer
```

## API Reference

### Models

#### LongitudinalVAE

LSTM/GRU-based VAE model class.

**Parameters:**
- `input_dim` (int): Dimension of input features at each time step
- `hidden_dim` (int): Dimension of LSTM/GRU hidden state (default: 64)
- `latent_dim` (int): Dimension of latent space (default: 20)
- `num_layers` (int): Number of LSTM/GRU layers (default: 1)
- `use_gru` (bool): Use GRU instead of LSTM (default: False)

**Methods:**
- `forward(x, mask=None)`: Forward pass through the VAE
- `encode(x)`: Encode input to latent distribution parameters
- `decode(z, seq_len)`: Decode latent representation to sequence
- `sample(num_samples, seq_len, device)`: Generate new samples

#### CNNLongitudinalVAE

CNN-based VAE model class with missing data support.

**Parameters:**
- `input_dim` (int): Dimension of input features at each time step
- `seq_len` (int): Expected sequence length
- `latent_dim` (int): Dimension of latent space (default: 20)
- `hidden_channels` (list): Channel sizes for encoder (default: [32, 64, 128])
- `kernel_size` (int): Convolution kernel size (default: 3)

**Methods:**
- `forward(x, mask=None)`: Forward pass through the VAE
- `encode(x, mask=None)`: Encode input to latent distribution parameters
- `decode(z)`: Decode latent representation to sequence
- `sample(num_samples, device)`: Generate new samples
- `impute_missing(x, mask, num_iterations=5)`: Impute missing values using EM-like approach

### Training

#### VAETrainer

Training utility class.

**Parameters:**
- `model`: VAE model instance (LongitudinalVAE or CNNLongitudinalVAE)
- `learning_rate` (float): Learning rate for optimizer (default: 1e-3)
- `beta` (float): Weight for KL divergence term (default: 1.0)
- `device` (str): Device to train on (default: auto-detect)

**Methods:**
- `fit(train_loader, val_loader=None, epochs=100, verbose=True, use_em_imputation=False, em_iterations=3)`: Train the model
- `train_epoch(train_loader, use_em_imputation=False, em_iterations=3)`: Train for one epoch
- `validate(val_loader)`: Validate the model
- `save_model(path)`: Save model checkpoint
- `load_model(path)`: Load model checkpoint

### Data Utilities

#### LongitudinalDataset

Dataset class for longitudinal data with missing data support.

**Parameters:**
- `data`: Numpy array or list of sequences
- `mask` (array, optional): Binary mask for missing data (1=observed, 0=missing)
- `normalize` (bool): Whether to normalize the data (default: True)
- `padding_value` (float): Value for padding shorter sequences (default: 0.0)

**Methods:**
- `__getitem__(idx)`: Get item at index (returns data, mask, length)
- `inverse_transform(data)`: Transform normalized data back to original scale

#### Utility Functions

- `generate_synthetic_longitudinal_data(n_samples, seq_len, n_features, noise_level, seed)`: Generate synthetic time-series data
- `create_missing_mask(data_shape, missing_rate, pattern, seed)`: Create binary mask for missing data with specified pattern
- `vae_loss_function(recon_x, x, mu, logvar, beta, mask)`: Compute VAE loss with optional masking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vaelong,
  title = {VAElong: Variational Autoencoder for Longitudinal Measurements},
  author = {},
  year = {2025},
  url = {https://github.com/stenw/VAElong}
}
```

## Acknowledgments

This implementation is based on the Variational Autoencoder framework and extends it for longitudinal/time-series data.
