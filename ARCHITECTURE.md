# VAElong Project Architecture

This document explains the structure of the VAElong package, what each file does, and where to start.

## Where to Start

1. **New user?** Read `QUICKSTART.md` for a minimal working example.
2. **Want mixed-type data (continuous + binary + bounded)?** Start with `examples/mixed_type_example.py`.
3. **Only have continuous data?** Start with `examples/basic_example.py`.
4. **Want to understand the code?** Read on below.

## Project Structure

```
vaelong/
├── config.py        # Variable type configuration
├── data.py          # Dataset and synthetic data generation
├── model.py         # VAE model architectures and loss functions
├── trainer.py       # Training loop and utilities
└── __init__.py      # Package exports

examples/
├── basic_example.py              # Simple continuous-only workflow
├── cnn_missing_data_example.py   # CNN model with missing data
└── mixed_type_example.py         # Mixed types, baselines, landmark prediction

tests/
├── test_model.py        # LSTM/GRU model tests
├── test_cnn_model.py    # CNN model and imputation tests
├── test_data.py         # Dataset and data generation tests
├── test_trainer.py      # Training pipeline tests
└── test_mixed_types.py  # Mixed-type, baseline, and landmark tests
```

## Core Modules

### `vaelong/config.py` — Variable Type Configuration

Defines the types of your time-varying variables. This is the first thing you set up when working with mixed-type data.

- **`VariableSpec`**: Describes a single variable (name, type, bounds).
  Types: `'continuous'` (unbounded, real-valued), `'binary'` (0/1), `'bounded'` (constrained to a range).
- **`VariableConfig`**: A list of `VariableSpec` objects. Provides index helpers (`continuous_indices`, `binary_indices`, `bounded_indices`) used throughout the codebase.
- **`VariableConfig.all_continuous(n)`**: Factory for the common case where all variables are continuous (backward-compatible default).

### `vaelong/data.py` — Data Handling

Handles datasets, normalization, and synthetic data generation.

- **`LongitudinalDataset`**: PyTorch `Dataset` wrapping longitudinal data.
  - Accepts data as a numpy array `(n_samples, seq_len, n_features)` or a list of variable-length sequences.
  - Optional `mask` for missing data (1=observed, 0=missing).
  - Optional `baseline_covariates` for time-invariant features per subject.
  - Optional `var_config` for type-aware normalization: z-score for continuous, affine to [0,1] for bounded, no-op for binary.
  - Returns 4-tuple: `(data, mask, length, baseline)`.

- **`generate_synthetic_longitudinal_data()`**: Creates continuous-only synthetic data with trend + seasonality + noise. Good for quick tests.

- **`generate_mixed_longitudinal_data()`**: Creates synthetic data with mixed types. Binary variables are generated via sigmoid-thresholded latent trajectories; bounded variables via sigmoid scaling. Can also generate baseline covariates.

- **`create_missing_mask()`**: Creates binary masks with three patterns: `'random'`, `'block'` (contiguous gaps), `'monotone'` (dropout — once missing, stays missing).

### `vaelong/model.py` — Model Architectures

Contains the two VAE architectures and loss functions.

- **`LongitudinalVAE`**: LSTM/GRU-based VAE.
  - Encoder: RNN processes the sequence, last hidden state maps to `(mu, logvar)`.
  - Decoder: Latent code is repeated across time steps, decoded by a second RNN.
  - Supports `n_baseline` for conditional VAE (baselines concatenated to hidden state before mu/logvar, and to latent before decoding).
  - Supports `var_config` for per-variable output activations (sigmoid for binary/bounded).
  - `predict_from_landmark()`: Encode partial observations, decode a full-length trajectory.

- **`CNNLongitudinalVAE`**: CNN-based VAE.
  - Encoder: 1D convolutions with stride-2 downsampling.
  - Decoder: 1D transposed convolutions for upsampling.
  - Same baseline conditioning and mixed-type support as the LSTM model.
  - `impute_missing()`: Iterative EM-like imputation of missing values.
  - For landmark prediction, pad the input to `seq_len` and mask future time steps.

- **`vae_loss_function()`**: Standard VAE loss (MSE reconstruction + KL divergence). Supports masked loss for missing data.

- **`mixed_vae_loss_function()`**: Extended loss for mixed types. Uses MSE for continuous, BCE for binary and bounded variables. Falls back to the standard loss when `var_config` is `None`.

### `vaelong/trainer.py` — Training

- **`VAETrainer`**: Manages the training loop.
  - `fit()`: Train for multiple epochs with optional validation.
  - `train_epoch()`: Single epoch with optional EM imputation for missing data.
  - `validate()`: Evaluation on a validation set.
  - `save_model()` / `load_model()`: Checkpoint model and optimizer state.
  - Accepts `var_config` to use the mixed-type loss and type-aware EM imputation.
  - Automatically passes baseline covariates from the dataloader to the model.

## Typical Workflow

```
1. Define variables          →  VariableConfig (config.py)
2. Prepare or generate data  →  LongitudinalDataset, generate_mixed_longitudinal_data (data.py)
3. Create model              →  LongitudinalVAE or CNNLongitudinalVAE (model.py)
4. Train                     →  VAETrainer.fit() (trainer.py)
5. Use trained model         →  model.sample(), model.predict_from_landmark(), model.impute_missing()
```

## Key Design Decisions

- **Backward compatible**: All new parameters (`var_config`, `n_baseline`, `baseline`) default to `None`/`0`. Existing code that only uses continuous data works without changes.
- **Conditional VAE via concatenation**: Baseline covariates are concatenated to the encoder hidden state and to the latent code. This is the standard CVAE approach.
- **Single output layer + per-index activation**: Rather than separate decoder heads per variable type, one linear layer outputs all features, then sigmoid is applied to binary/bounded indices. This is simpler and equally expressive since each output dimension has its own weight row.
- **BCE for bounded variables**: Bounded data is affine-transformed to [0,1] during normalization, and the decoder uses sigmoid output. BCE is used as the reconstruction loss, which is equivalent to fitting a distribution on [0,1].
