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
model = LongitudinalVAE(input_dim=5, hidden_dim=64, latent_dim=10)
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
    n_baseline=3, var_config=var_config,
)

# 5. Train
trainer = VAETrainer(model, learning_rate=1e-3, var_config=var_config)
history = trainer.fit(train_loader, epochs=50)
```

## Landmark Prediction

Predict future trajectories from partial observations:

```python
# Observe first 25 time steps, predict all 50
x_observed = data_tensor[:, :25, :]
mask_observed = torch.ones_like(x_observed)

predicted = model.predict_from_landmark(
    x_observed, mask_observed,
    total_seq_len=50, baseline=baseline_tensor,
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
```

## Examples

```bash
python examples/basic_example.py              # Continuous data
python examples/cnn_missing_data_example.py   # CNN + missing data
python examples/mixed_type_example.py         # Mixed types + baselines + landmark
```

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for a guide to the codebase structure
- See [README.md](README.md) for the full API reference
- Run tests with `python -m unittest discover tests`
