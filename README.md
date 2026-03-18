# VAElong

A Variational Autoencoder framework for **mixed-type longitudinal data** in Python (PyTorch) and Julia (Flux.jl).

## What this code does

VAElong trains variational autoencoders on longitudinal (time-series) measurements so you can:

- Learn a low-dimensional **latent representation** of each subject's trajectory
- **Reconstruct** observed trajectories and **predict** future time points from partial observations (landmark prediction)
- Handle **missing data** via mask-aware training and EM-like imputation
- Model **mixed variable types** (continuous, binary, bounded) with proper per-type likelihoods

## Key features

### Variable types

Each feature is declared with a type via `VariableConfig`:

| Type | Likelihood | Output activation |
|------|-----------|-------------------|
| `continuous` | Gaussian NLL with learned per-variable variance | Linear |
| `binary` | Bernoulli (BCE) | Sigmoid |
| `bounded` | BCE, Beta, or logit-normal (configurable) | Sigmoid or linear |

Bounded variables support three loss functions (`bounded_loss` parameter):
- `"bce"` (default) -- binary cross-entropy on [0,1]-normalised data
- `"beta"` -- Beta distribution NLL with learned per-variable precision
- `"logit_normal"` -- Gaussian NLL in logit space with learned variance

Optional epsilon clamping (`bounded_eps`) prevents exact 0/1 values for numerical stability.

### Model architectures

| Model | Description |
|-------|-------------|
| `LongitudinalVAE` | Dense (MLP) encoder/decoder by default; LSTM/GRU optional via `encoder_type` |
| `CNNLongitudinalVAE` | 1D convolutional encoder with transposed-conv decoder |
| `TPCNNLongitudinalVAE` | Time-Parameterized CNN -- kernels generated from relative time offsets |
| `TransformerLongitudinalVAE` | Encoder-only Transformer with multi-head self-attention |

All models support **baseline covariates** (CVAE conditioning), **missing data masks**, and **landmark prediction**.

### Missing data

- **Binary mask** (1=observed, 0=missing) -- reconstruction loss computed only over observed entries
- **EM-like imputation** -- alternates between predicting missing values (E-step) and updating parameters (M-step)
- Three missingness patterns: `random`, `block`, `monotone`

### Training

`VAETrainer` provides:
- Configurable beta (KL weight) for beta-VAE
- Early stopping with patience
- Learned observation noise variance for continuous variables (with optional L2 penalty via `noise_var_penalty`)
- EM imputation toggle

## Installation

```bash
git clone https://github.com/stenw/VAElong.git
cd VAElong
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from vaelong import (
    VariableConfig, VariableSpec,
    LongitudinalVAE, VAETrainer, LongitudinalDataset,
    generate_mixed_longitudinal_data, create_missing_mask,
)

# Define variable types
var_config = VariableConfig(variables=[
    VariableSpec(name='biomarker',       var_type='continuous'),
    VariableSpec(name='blood_pressure',  var_type='bounded', lower=60.0, upper=200.0),
    VariableSpec(name='symptom_present', var_type='binary'),
])

# Generate synthetic data (100 subjects, 50 time points, 2 baseline covariates)
data, baseline = generate_mixed_longitudinal_data(
    n_samples=100, seq_len=50, var_config=var_config,
    n_baseline_features=2, seed=42,
)

# Introduce 15% missing data
mask = create_missing_mask(data.shape, missing_rate=0.15, seed=42)

# Create dataset and loader
dataset = LongitudinalDataset(
    data * mask, mask=mask, var_config=var_config,
    baseline_covariates=baseline, normalize=True,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train an LSTM VAE with EM imputation
model = LongitudinalVAE(
    input_dim=var_config.n_features, hidden_dim=64, latent_dim=16,
    encoder_type="lstm", seq_len=50, n_baseline=2, var_config=var_config,
)
trainer = VAETrainer(model, learning_rate=1e-3, beta=0.5, var_config=var_config)
history = trainer.fit(loader, epochs=100, use_em_imputation=True, patience=20)
```

## Examples

| File | Description |
|------|-------------|
| `examples/mixed_type_example.py` | Full benchmark: LSTM VAE vs LMM vs Seq2Seq vs TPCNN vs Transformer (15% missing) |
| `examples/mixed_type_example.qmd` | Quarto notebook version of the above |
| `examples/mixed_type_example2.py` | Same benchmark with 50% missing data stress test |
| `examples/mixed_type_example2.qmd` | Quarto notebook version of the stress test |
| `application/ema_affect.py` | Real-data application: EMA affect modelling (EM_PA, EM_NA) |
| `application/ema_affect.ipynb` | Jupyter notebook version with results |

### Rendering Quarto documents

```bash
# Register the Jupyter kernel (once)
python -m ipykernel install --user --name vaelong --display-name "Python (VAElong)"

# Render
quarto render examples/mixed_type_example.qmd
```

## Testing

```bash
pytest tests/ -v
```

## Julia

A Julia translation using Flux.jl is in `julia/VAElong/`. See `julia/VAElong/README.md` for details.

## License

MIT License.

## Citation

```bibtex
@software{vaelong,
  title = {VAElong: Variational Autoencoder for Longitudinal Measurements},
  year = {2025},
  url = {https://github.com/stenw/VAElong}
}
```
