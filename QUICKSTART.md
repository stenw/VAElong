# Quick Start Guide

This guide will help you get started with VAElong in 5 minutes.

## Installation

```bash
git clone https://github.com/stenw/VAElong.git
cd VAElong
pip install -r requirements.txt
pip install -e .
```

## Basic Usage (Continuous Data)

```python
import numpy as np
from torch.utils.data import DataLoader
from vaelong import LongitudinalVAE, VAETrainer, LongitudinalDataset

# Your data: (n_samples, seq_len, n_features)
data = np.random.randn(1000, 50, 5).astype(np.float32)

# Create dataset and dataloader
dataset = LongitudinalDataset(data, normalize=True)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create and train a model
model = LongitudinalVAE(input_dim=5, hidden_dim=64, latent_dim=10, seq_len=50)
trainer = VAETrainer(model, learning_rate=1e-3)
history = trainer.fit(train_loader, epochs=50)

# Generate new samples
new_samples = model.sample(num_samples=10, seq_len=50)
```

## Mixed-Type Data (Continuous + Binary + Bounded)

```python
from vaelong import (
    VariableConfig, VariableSpec,
    LongitudinalVAE, VAETrainer, LongitudinalDataset,
    generate_mixed_longitudinal_data,
)

# 1. Define your variable types
var_config = VariableConfig(variables=[
    VariableSpec(name='biomarker', var_type='continuous'),
    VariableSpec(name='blood_pressure', var_type='bounded', lower=60.0, upper=200.0),
    VariableSpec(name='symptom', var_type='binary'),
])

# 2. Generate synthetic data (or use your own)
data, baseline = generate_mixed_longitudinal_data(
    n_samples=500, seq_len=50, var_config=var_config,
    n_baseline_features=3, seed=42,
)

# 3. Create dataset with baselines
dataset = LongitudinalDataset(
    data, var_config=var_config,
    baseline_covariates=baseline, normalize=True,
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Create model with baseline conditioning
model = LongitudinalVAE(
    input_dim=3, hidden_dim=64, latent_dim=16,
    seq_len=50, n_baseline=3, var_config=var_config,
)

# 5. Train
trainer = VAETrainer(model, learning_rate=1e-3, var_config=var_config)
history = trainer.fit(train_loader, epochs=50)
```

If you have explicit measurement times, pass them through the dataset and
enable time-aware inputs:

```python
times = np.linspace(0.0, 24.0, 50, dtype=np.float32)

dataset = LongitudinalDataset(
    data, var_config=var_config,
    baseline_covariates=baseline, normalize=True, times=times,
)

model = LongitudinalVAE(
    input_dim=3, hidden_dim=64, latent_dim=16,
    seq_len=50, n_baseline=3, var_config=var_config,
    time_in_encoder=True, time_in_decoder=True,
)
```

## Landmark Prediction

Predict future trajectories from partial observations:

```python
# Observe first 25 time steps, predict all 50
x_observed = data_tensor[:, :25, :]
mask_observed = torch.ones_like(x_observed)
times_tensor = full_times_tensor  # shape: (batch, 50)

predicted = model.predict_from_landmark(
    x_observed, mask_observed,
    total_seq_len=50, baseline=baseline_tensor, times=times_tensor,
)
```

## Missing Data

```python
from vaelong import create_missing_mask

# Create a mask with 20% missing values
mask = create_missing_mask(data.shape, missing_rate=0.2, pattern='random', seed=42)
# Patterns: 'random', 'block', 'monotone'

# Create dataset with mask
dataset = LongitudinalDataset(data * mask, mask=mask, normalize=True)

# Train with EM imputation
trainer.fit(train_loader, epochs=50, use_em_imputation=True, em_iterations=3)

# Or use random-walk Metropolis-Hastings updates for the missing values
trainer.fit(
    train_loader, epochs=50, use_em_imputation=True, em_iterations=3,
    imputation_method="rwmh", mh_steps=2, mh_adaptive=True,
)
```

## Examples

```bash
python examples/basic_example.py              # Continuous data
python examples/cnn_missing_data_example.py   # CNN + missing data
python examples/mixed_type_example.py         # Mixed types + baselines + landmark
```

## YAML Applications

For application-style analyses, prefer the YAML runner:

```bash
python -m vaelong.app --config configs/glucose.yaml
```

You can override file locations and plotted subjects from the command line:

```bash
python -m vaelong.app --config configs/glucose.yaml --data-path /path/to/data.parquet --output-dir /path/to/results --plot-ids 23 48 150
```

Thin wrappers are also available:

```bash
python application/glucose_landmark.py
python application/ema_vae.py --data-path /path/to/ema_data.parquet
```

The bundled YAML files resolve relative paths from the `configs/` directory, so
their default `../...` paths point back to the repo root.

## Processing QMDs And Notebooks

To process the repository's Quarto documents and notebooks in one step:

```bash
python scripts/process_documents.py
```

This renders `.qmd` files with Quarto and executes `.ipynb` files in place with
`nbconvert`. Use `--dry-run` first if you want to preview exactly what will run.

To strip notebook outputs without executing anything:

```bash
python -m jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace application/ema_affect.ipynb
```

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for a guide to the codebase structure
- See [README.md](README.md) for the full API reference
- Run tests with `python -m unittest discover tests`
