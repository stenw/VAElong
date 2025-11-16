# Quick Start Guide

This guide will help you get started with VAElong in 5 minutes.

## Installation

```bash
git clone https://github.com/stenw/VAElong.git
cd VAElong
pip install -r requirements.txt
```

## Basic Usage (3 Steps)

### 1. Prepare Your Data

```python
import numpy as np
from vaelong import LongitudinalDataset

# Your data should be a numpy array of shape (n_samples, seq_len, n_features)
# For example, 1000 patients, 50 time points, 5 measurements each
data = np.random.randn(1000, 50, 5)

# Create dataset (automatically normalizes)
dataset = LongitudinalDataset(data, normalize=True)
```

### 2. Create and Train Model

```python
from torch.utils.data import DataLoader
from vaelong import LongitudinalVAE, VAETrainer

# Create data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create VAE model
model = LongitudinalVAE(
    input_dim=5,      # Number of features per time point
    hidden_dim=64,    # Size of LSTM hidden state
    latent_dim=10     # Size of latent representation
)

# Train the model
trainer = VAETrainer(model, learning_rate=1e-3)
history = trainer.fit(train_loader, epochs=50)
```

### 3. Use the Trained Model

```python
# Reconstruct data
reconstructed, mu, logvar = model(data_tensor)

# Generate new samples
new_samples = model.sample(num_samples=10, seq_len=50)

# Save model
trainer.save_model('my_vae.pth')
```

## Try the Example

```bash
cd examples
python basic_example.py
```

This will:
- Generate synthetic data
- Train a VAE model
- Create visualizations
- Save the trained model

## Next Steps

- See [README.md](README.md) for full documentation
- Check `examples/basic_example.py` for a complete workflow
- Run tests with `python -m unittest discover tests`

## Common Use Cases

### Variable Length Sequences

```python
# List of sequences with different lengths
sequences = [
    np.random.randn(30, 5),  # 30 time points
    np.random.randn(45, 5),  # 45 time points
    np.random.randn(60, 5),  # 60 time points
]

# Automatically handles padding
dataset = LongitudinalDataset(sequences)
```

### Using GRU Instead of LSTM

```python
model = LongitudinalVAE(
    input_dim=5,
    hidden_dim=64,
    latent_dim=10,
    use_gru=True  # Use GRU cells instead of LSTM
)
```

### Beta-VAE for Better Latent Space

```python
# Lower beta = better reconstruction
trainer = VAETrainer(model, beta=0.5)

# Higher beta = better disentangled features
trainer = VAETrainer(model, beta=2.0)
```

## Need Help?

- Check the [full API documentation](README.md#api-reference)
- Look at the [examples](examples/) directory
- Run the tests to see usage examples: `python -m unittest discover tests`
