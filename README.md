# VAElong

A Variational Autoencoder (VAE) implementation for longitudinal (time-series) measurements in Python using PyTorch.

## Overview

VAElong provides a flexible and easy-to-use implementation of VAEs designed specifically for longitudinal data. The model uses LSTM/GRU layers to handle the sequential nature of time-series data and learns a compact latent representation that captures the underlying patterns in the data.

## Features

- **LSTM/GRU-based architecture** for handling sequential data
- **Flexible latent space** dimension configuration
- **Beta-VAE support** for controlling the trade-off between reconstruction and KL divergence
- **Built-in training utilities** with easy-to-use trainer class
- **Data preprocessing** for longitudinal measurements
- **Synthetic data generation** for testing and experimentation
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

### Model Components

1. **Encoder**: LSTM/GRU layers that process the input sequence and output parameters (mean and log variance) of the latent distribution
2. **Reparameterization**: Samples from the latent distribution using the reparameterization trick
3. **Decoder**: Transforms latent representation back to sequence space using LSTM/GRU layers

### Loss Function

The VAE loss consists of two components:

```
Loss = Reconstruction Loss + β × KL Divergence
```

- **Reconstruction Loss**: Mean Squared Error (MSE) between input and reconstructed sequences
- **KL Divergence**: Measures how much the learned latent distribution diverges from a standard normal distribution
- **β**: Weight parameter for KL divergence (β-VAE variant)

## Usage Examples

### Basic Training

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

- `basic_example.py`: Complete workflow including data generation, training, and visualization

Run the basic example:

```bash
cd examples
python basic_example.py
```

This will:
- Generate synthetic longitudinal data
- Train a VAE model
- Create visualizations of:
  - Original vs reconstructed sequences
  - Training history
  - Latent space representation
- Save the trained model

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

### LongitudinalVAE

Main VAE model class.

**Parameters:**
- `input_dim` (int): Dimension of input features at each time step
- `hidden_dim` (int): Dimension of LSTM/GRU hidden state (default: 64)
- `latent_dim` (int): Dimension of latent space (default: 20)
- `num_layers` (int): Number of LSTM/GRU layers (default: 1)
- `use_gru` (bool): Use GRU instead of LSTM (default: False)

**Methods:**
- `forward(x)`: Forward pass through the VAE
- `encode(x)`: Encode input to latent distribution parameters
- `decode(z, seq_len)`: Decode latent representation to sequence
- `sample(num_samples, seq_len, device)`: Generate new samples

### VAETrainer

Training utility class.

**Parameters:**
- `model`: LongitudinalVAE model instance
- `learning_rate` (float): Learning rate for optimizer (default: 1e-3)
- `beta` (float): Weight for KL divergence term (default: 1.0)
- `device` (str): Device to train on (default: auto-detect)

**Methods:**
- `fit(train_loader, val_loader, epochs, verbose)`: Train the model
- `train_epoch(train_loader)`: Train for one epoch
- `validate(val_loader)`: Validate the model
- `save_model(path)`: Save model checkpoint
- `load_model(path)`: Load model checkpoint

### LongitudinalDataset

Dataset class for longitudinal data.

**Parameters:**
- `data`: Numpy array or list of sequences
- `normalize` (bool): Whether to normalize the data (default: True)
- `padding_value` (float): Value for padding shorter sequences (default: 0.0)

**Methods:**
- `__getitem__(idx)`: Get item at index
- `inverse_transform(data)`: Transform normalized data back to original scale

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
