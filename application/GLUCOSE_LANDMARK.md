# Glucose Landmark Application

Primary config: [configs/glucose.yaml](/E:/Users/Sten/Documents/codexwork/vaelong/configs/glucose.yaml)  
Thin wrapper: [application/glucose_landmark.py](/E:/Users/Sten/Documents/codexwork/vaelong/application/glucose_landmark.py)

This application is now config-driven. The glucose-specific Python script is a thin wrapper around the shared YAML runner in [vaelong/app_runner.py](/E:/Users/Sten/Documents/codexwork/vaelong/vaelong/app_runner.py:591).

By default the config looks for `../glucose_data.parquet` relative to `configs/glucose.yaml`, which resolves to a `glucose_data.parquet` file in the repo root. If your data live elsewhere, pass `--data-path`.

## Expected input

Required columns:

- `patient_id`
- `patient_idx`
- `timestamp`
- `time_s`
- `glucose_mmol_l`

Derived in the config:

- `time_of_day`
- `sin_time_of_day`
- `cos_time_of_day`

## Default modelling choices

- Outcome: `glucose_mmol_l`
- Time-varying features: `glucose_mmol_l`, `time_s`, `sin_time_of_day`, `cos_time_of_day`
- Landmark: `seq_len // 2`
- Split: 60% train, 20% validation, 20% test at the patient level
- Default encoder: `lstm`
- EM-like imputation: enabled
- Example profile plot: 4 individuals by default, or specific patients via `--plot-ids`

If patients do not all have the same sequence length, the default config keeps the patients with the modal sequence length. Set `data.strict_seq_len: true` in the YAML file to fail instead.

## Run

Config-first entry point from the repo root:

```powershell
python -m vaelong.app --config configs\glucose.yaml
```

Equivalent thin-wrapper command:

```powershell
python application\glucose_landmark.py
```

To plot specific patients instead of random test subjects:

```powershell
python -m vaelong.app --config configs\glucose.yaml --plot-ids 23 48 150 200
```

The wrapper supports the same overrides:

```powershell
python application\glucose_landmark.py --plot-ids 23 48 150 200
```

Hyperparameter search is now controlled in the YAML file. To enable it, set:

```yaml
tuning:
  enabled: true
```

and then run either command above. The default config samples 9 random combinations over:

- learning rate
- weight decay
- beta
- hidden dimension
- latent dimension

With a custom parquet path or output directory:

```powershell
python -m vaelong.app --config configs\glucose.yaml --data-path "D:\data\glucose_data.parquet" --output-dir "E:\results\glucose_landmark"
```

## Outputs

The default config writes these files under `application/results/glucose_landmark`:

- `model.pt`
- `training_history.csv`
- `training_curve.png`
- `hyperparameter_search_results.csv` when tuning is enabled
- `split_assignments.csv`
- `future_predictions.csv`
- `landmark_prediction_examples.png`
- `metrics_summary.csv`
- `run_metadata.json`
