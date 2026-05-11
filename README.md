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

## Notebook output stripping

To prevent Jupyter notebook outputs from being committed, install the repo hooks once:

```bash
pip install pre-commit
pre-commit install
```

This repo uses `pre-commit` with `nbstripout`, so staged `.ipynb` files are cleaned automatically before commit.
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

## YAML-driven applications

Application-style analyses can now be described in YAML rather than hard-coded
Python scripts. The shared runner lives in [vaelong/app.py](/E:/Users/Sten/Documents/codexwork/vaelong/vaelong/app.py:1),
[vaelong/app_config.py](/E:/Users/Sten/Documents/codexwork/vaelong/vaelong/app_config.py:1),
and [vaelong/app_runner.py](/E:/Users/Sten/Documents/codexwork/vaelong/vaelong/app_runner.py:591).

Run a config directly:

```bash
python -m vaelong.app --config configs/glucose.yaml
```

Useful overrides:

```bash
python -m vaelong.app --config configs/glucose.yaml --data-path /path/to/data.parquet --output-dir /path/to/results --plot-ids 23 48 150
```

Current example configs:

- [configs/glucose.yaml](/E:/Users/Sten/Documents/codexwork/vaelong/configs/glucose.yaml): glucose landmark prediction with midpoint landmarking
- [configs/ema_vae.yaml](/E:/Users/Sten/Documents/codexwork/vaelong/configs/ema_vae.yaml): EMA VAE-only workflow

Relative `data.path` and `output.dir` values are resolved relative to the YAML
file itself, so the provided configs use `../...` paths to point back to the
repo root and `application/results/`.

Thin application wrappers are available at:

- [application/glucose_landmark.py](/E:/Users/Sten/Documents/codexwork/vaelong/application/glucose_landmark.py:1)
- [application/ema_vae.py](/E:/Users/Sten/Documents/codexwork/vaelong/application/ema_vae.py:1)

The legacy [application/ema_affect.py](/E:/Users/Sten/Documents/codexwork/vaelong/application/ema_affect.py:1)
script remains for the custom mixed-model benchmark, which has not been
generalized into the YAML runner yet.

## Examples

| File | Description |
|------|-------------|
| `examples/mixed_type_example.py` | Full benchmark: LSTM VAE vs LMM vs Seq2Seq vs TPCNN vs Transformer (15% missing) |
| `examples/mixed_type_example.qmd` | Quarto notebook version of the above |
| `examples/mixed_type_example2.py` | Same benchmark with 50% missing data stress test |
| `examples/mixed_type_example2.qmd` | Quarto notebook version of the stress test |
| `application/ema_affect.py` | Legacy real-data application: EMA VAE + mixed-model benchmark |
| `application/ema_vae.py` | YAML-driven EMA VAE wrapper |
| `application/glucose_landmark.py` | YAML-driven glucose landmark wrapper |
| `application/ema_affect.ipynb` | Jupyter notebook version with results |

### Rendering Quarto documents

```bash
# Register the Jupyter kernel (once)
python -m ipykernel install --user --name vaelong --display-name "Python (VAElong)"

# Render
quarto render examples/mixed_type_example.qmd
```

To process the repo's Quarto files and notebooks from one command:

```bash
python scripts/process_documents.py
```

Useful variants:

```bash
python scripts/process_documents.py --qmd-only
python scripts/process_documents.py --notebooks-only
python scripts/process_documents.py --dry-run
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
