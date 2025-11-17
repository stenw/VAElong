# VAElong.jl

Variational Autoencoder for Longitudinal Measurements - Julia Implementation

A Julia package for training Variational Autoencoders (VAEs) on longitudinal/time-series data with support for missing data handling.

## Features

- **LSTM/GRU-based VAE**: Traditional recurrent architecture for sequential data
- **CNN-based VAE**: Convolutional architecture for efficient processing of time series
- **Missing Data Handling**: Built-in support for missing values with:
  - Binary masking
  - EM-like imputation during training
  - Multiple missing data patterns (random, block, monotone)
- **Flexible Training**: Customizable loss functions, β-VAE support, and training utilities

## Installation

```julia
# Add the package (local installation)
using Pkg
Pkg.develop(path="/path/to/VAElong/julia/VAElong")

# Or activate the project
using Pkg
Pkg.activate("/path/to/VAElong/julia/VAElong")
Pkg.instantiate()
```

## Quick Start

```julia
using VAElong
using Random

# Generate synthetic data
data = generate_synthetic_longitudinal_data(
    n_samples=1000,
    seq_len=64,
    n_features=5,
    noise_level=0.1f0,
    seed=42
)

# Create missing data mask
mask = create_missing_mask(
    size(data),
    missing_rate=0.2f0,
    pattern="random",
    seed=42
)

# Apply mask
data_with_missing = data .* mask

# Create dataset
dataset = LongitudinalDataset(data_with_missing, mask=mask, normalize=true)

# Create CNN-based VAE
model = CNNLongitudinalVAE(
    5,      # input_dim
    64,     # seq_len
    latent_dim=16,
    hidden_channels=[32, 64, 128],
    kernel_size=3
)

# Create trainer
trainer = VAETrainer(
    model,
    learning_rate=1e-3f0,
    β=1.0f0
)

# Create data loader
train_loader = create_data_loader(dataset, batch_size=32, shuffle=true)

# Train with EM imputation
history = fit!(
    trainer,
    train_loader,
    epochs=100,
    use_em_imputation=true,
    em_iterations=3
)

# Impute missing values
imputed = impute_missing(model, data_normalized, mask, num_iterations=5)

# Generate new samples
samples = sample(model, 10)
```

## Architecture

### CNN-based VAE

The CNN-based VAE uses 1D convolutional layers to process time series data:

**Encoder:**
- Multiple Conv1D layers with stride 2 for downsampling
- BatchNorm and ReLU activations
- Fully connected layers to latent space (μ and log σ²)

**Decoder:**
- Fully connected layer from latent space
- Multiple ConvTranspose1D layers for upsampling
- BatchNorm and ReLU activations
- Output matches input dimensions

### LSTM/GRU-based VAE

Traditional recurrent architecture:

**Encoder:**
- LSTM/GRU layers process sequential input
- Final hidden state mapped to latent space

**Decoder:**
- Latent vector mapped to hidden dimension
- LSTM/GRU layers generate output sequence

## Missing Data Handling

The package supports three missing data patterns:

1. **Random**: Random missing values throughout the data
2. **Block**: Contiguous blocks of missing values in time
3. **Monotone**: If value at time t is missing, all subsequent values are missing

### EM-like Imputation

During training, the model alternates between:
1. **E-step**: Generate predictions for missing values and sample from them
2. **M-step**: Update model parameters given the imputed data

This approach improves reconstruction quality on missing values.

## Examples

See the `examples/` directory for complete examples:

- `cnn_missing_data_example.jl`: CNN-based VAE with missing data handling

Run an example:

```julia
include("examples/cnn_missing_data_example.jl")
```

## API Reference

### Models

- `CNNLongitudinalVAE(input_dim, seq_len; latent_dim, hidden_channels, kernel_size)`
- `LongitudinalVAE(input_dim; hidden_dim, latent_dim, num_layers, use_gru)`

### Data

- `LongitudinalDataset(data; mask, normalize)`
- `generate_synthetic_longitudinal_data(; n_samples, seq_len, n_features, noise_level, seed)`
- `create_missing_mask(data_shape; missing_rate, pattern, seed)`

### Training

- `VAETrainer(model; learning_rate, β, device)`
- `fit!(trainer, train_loader; val_loader, epochs, verbose, use_em_imputation, em_iterations)`
- `create_data_loader(dataset; batch_size, shuffle)`

### Model Functions

- `encode(model, x, mask)`: Encode input to latent distribution
- `decode(model, z)`: Decode latent vector to output
- `sample(model, num_samples)`: Generate samples from prior
- `impute_missing(model, x, mask; num_iterations)`: Impute missing values
- `vae_loss(recon_x, x, μ, logσ²; β, mask)`: Compute VAE loss

## Requirements

- Julia ≥ 1.6
- Flux.jl ≥ 0.13
- CUDA.jl (optional, for GPU support)

## License

This project is open source and available under the MIT License.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{vaelong_jl,
  title = {VAElong.jl: Variational Autoencoders for Longitudinal Data},
  author = {VAElong Contributors},
  year = {2024},
  url = {https://github.com/yourusername/VAElong}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
