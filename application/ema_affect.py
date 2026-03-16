"""
EMA Affect Modelling — Proof of Concept
========================================
Model EM_PA and EM_NA (bounded [0, 1]) from ecological momentary assessment
data using the vaelong VAE framework.

Features:
  - EM_PA, EM_NA: bounded outcome variables
  - sin_hrs, cos_hrs: continuous time-varying features
  - AGE, SEX_1, SEX_2, SEX_3: baseline covariates

Models:
  - LSTM VAE (with EM imputation)
  - Linear Mixed Model (benchmark)
"""

import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from torch.utils.data import DataLoader

from vaelong import (
    VariableConfig, VariableSpec,
    LongitudinalVAE,
    VAETrainer, LongitudinalDataset,
    create_missing_mask,
)

torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load and reshape data ─────────────────────────────────────────────────

df = pd.read_parquet("W:/parquetdata/Final_DF.parquet")

# Sort consistently
df = df.sort_values(["id", "hrs_since_start"]).reset_index(drop=True)

subject_ids = sorted(df["id"].unique())
n_subjects = len(subject_ids)
seq_len = 105  # every subject has exactly 105 observations

# Variable columns (time-varying)
outcome_cols = ["EM_PA", "EM_NA"]
time_cols = ["sin_hrs", "cos_hrs"]
feature_cols = outcome_cols + time_cols
n_features = len(feature_cols)

# Baseline covariates (time-invariant)
baseline_cols = ["AGE", "SEX_1", "SEX_2", "SEX_3"]
n_baseline = len(baseline_cols)

print(f"Subjects: {n_subjects}, Sequence length: {seq_len}")
print(f"Features: {feature_cols}")
print(f"Baseline: {baseline_cols}")

# ── 2. Build 3D arrays ───────────────────────────────────────────────────────

data = np.zeros((n_subjects, seq_len, n_features), dtype=np.float32)
mask = np.ones((n_subjects, seq_len, n_features), dtype=np.float32)
baseline = np.zeros((n_subjects, n_baseline), dtype=np.float32)

id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

for sid, grp in df.groupby("id"):
    i = id_to_idx[sid]
    grp = grp.sort_values("hrs_since_start").reset_index(drop=True)

    for j, col in enumerate(feature_cols):
        vals = grp[col].values
        data[i, :, j] = np.nan_to_num(vals, nan=0.0)
        mask[i, :, j] = (~np.isnan(vals)).astype(np.float32)

    baseline[i] = grp[baseline_cols].iloc[0].values

# sin_hrs, cos_hrs are always observed — set mask to 1
for j, col in enumerate(feature_cols):
    if col in time_cols:
        mask[:, :, j] = 1.0

observed_rate = mask[:, :, :len(outcome_cols)].mean()
print(f"Observed rate (outcomes): {observed_rate:.1%}")
print(f"Data shape:     {data.shape}")
print(f"Baseline shape: {baseline.shape}")

# ── 3. Variable configuration ────────────────────────────────────────────────

var_config = VariableConfig(variables=[
    VariableSpec(name="EM_PA",    var_type="bounded", lower=0.0, upper=1.0),
    VariableSpec(name="EM_NA",    var_type="bounded", lower=0.0, upper=1.0),
    VariableSpec(name="sin_hrs",  var_type="continuous"),
    VariableSpec(name="cos_hrs",  var_type="continuous"),
])

print(f"\nBounded indices:    {var_config.bounded_indices}")
print(f"Continuous indices: {var_config.continuous_indices}")

# ── 4. Dataset and splits ────────────────────────────────────────────────────

dataset = LongitudinalDataset(
    data, mask=mask, var_config=var_config,
    baseline_covariates=baseline, normalize=True,
)

train_size = int(0.8 * n_subjects)
val_size = n_subjects - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

print(f"Train: {train_size}, Validation: {val_size}")

# ── 5. Train LSTM VAE ────────────────────────────────────────────────────────

model = LongitudinalVAE(
    input_dim=var_config.n_features,
    hidden_dim=64,
    latent_dim=16,
    n_baseline=n_baseline,
    var_config=var_config,
)

trainer = VAETrainer(model, learning_rate=1e-3, beta=0.5, var_config=var_config)

history = trainer.fit(
    train_loader, val_loader=val_loader, epochs=200, verbose=True,
    use_em_imputation=True, em_iterations=2, patience=20,
)

# ── 6. Training curves ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["train_loss"], label="Train")
ax.plot(history["val_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("LSTM VAE — Training and Validation Loss")
ax.legend()
plt.tight_layout()
plt.savefig("application/vae_training_loss.png", dpi=150)
plt.show()

# ── 7. Landmark prediction ───────────────────────────────────────────────────

landmark_t = seq_len // 2  # predict from midpoint

val_indices = list(val_ds.indices)

# Predict for all validation subjects
all_actual, all_predicted = [], []

for idx in val_indices:
    xi = dataset[idx][0].unsqueeze(0)
    mi = dataset[idx][1].unsqueeze(0)
    bi = dataset[idx][3].unsqueeze(0)

    xi_obs = xi[:, :landmark_t, :]
    mi_obs = mi[:, :landmark_t, :]

    pred_i = model.predict_from_landmark(
        xi_obs, mi_obs, total_seq_len=seq_len, baseline=bi,
    )

    all_actual.append(dataset.inverse_transform(xi).detach())
    all_predicted.append(dataset.inverse_transform(pred_i).detach())

all_actual = torch.cat(all_actual, dim=0).numpy()
all_predicted = torch.cat(all_predicted, dim=0).numpy()

future_actual = all_actual[:, landmark_t:, :]
future_pred = all_predicted[:, landmark_t:, :]

# ── 8. Plot landmark predictions for 3 individuals ───────────────────────────

rng = np.random.default_rng(123)
chosen = sorted(rng.choice(len(val_indices), size=3, replace=False))
time_axis = np.arange(seq_len)

fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

for row, c in enumerate(chosen):
    for col, vname in enumerate(outcome_cols):
        ax = axes[row, col]
        vidx = feature_cols.index(vname)

        actual_vals = all_actual[c, :, vidx]
        pred_vals = all_predicted[c, :, vidx]

        ax.plot(time_axis, actual_vals, "k-", linewidth=1.2, label="Actual")
        ax.plot(time_axis[:landmark_t], pred_vals[:landmark_t],
                "b-", linewidth=1, alpha=0.5)
        ax.plot(time_axis[landmark_t:], pred_vals[landmark_t:],
                "r-", linewidth=1.5, label="Predicted")

        ax.axvspan(landmark_t, seq_len - 1, alpha=0.08, color="red")
        ax.axvline(landmark_t, color="grey", linestyle="--", linewidth=0.8)

        if row == 0:
            ax.set_title(vname, fontsize=12)
        if col == 0:
            ax.set_ylabel(f"Individual {row + 1}", fontsize=10)
        if row == 2:
            ax.set_xlabel("Time step")

axes[0, -1].legend(loc="upper right", fontsize=8)
fig.suptitle(f"Landmark Prediction (observed up to t = {landmark_t})", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("application/vae_landmark_prediction.png", dpi=150)
plt.show()

# ── 9. LMM benchmark ─────────────────────────────────────────────────────────

train_indices = list(train_ds.indices)
n_val = len(val_indices)

lmm_predictions = np.zeros((n_val, seq_len, n_features))

for col_idx, vname in enumerate(outcome_cols):
    print(f"  Fitting LMM for {vname}...", end=" ", flush=True)

    rows = []
    for i in train_indices:
        for t in range(landmark_t):
            if mask[i, t, col_idx] == 1.0:
                row = {
                    "subject": int(i), "time": t,
                    "y": float(data[i, t, col_idx]),
                    "sin_hrs": float(data[i, t, feature_cols.index("sin_hrs")]),
                    "cos_hrs": float(data[i, t, feature_cols.index("cos_hrs")]),
                }
                for b, bcol in enumerate(baseline_cols):
                    row[bcol] = float(baseline[i, b])
                rows.append(row)

    df_train = pd.DataFrame(rows)

    fixed_formula = "y ~ time + sin_hrs + cos_hrs + " + " + ".join(baseline_cols)
    re_formula = "1 + time"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(fixed_formula, df_train, groups=df_train["subject"],
                         re_formula=re_formula)
        mdf = md.fit(reml=True, method="lbfgs")

    beta_hat = np.array(mdf.fe_params)
    D = np.array(mdf.cov_re)
    sigma2_e = mdf.scale
    print(f"done (var_e={sigma2_e:.4f})")

    for j, subj_idx in enumerate(val_indices):
        obs_times, obs_y, obs_sin, obs_cos = [], [], [], []
        for t in range(landmark_t):
            if mask[subj_idx, t, col_idx] == 1.0:
                obs_times.append(t)
                obs_y.append(data[subj_idx, t, col_idx])
                obs_sin.append(data[subj_idx, t, feature_cols.index("sin_hrs")])
                obs_cos.append(data[subj_idx, t, feature_cols.index("cos_hrs")])

        bl_vals = [baseline[subj_idx, b] for b in range(n_baseline)]

        if len(obs_times) == 0:
            for t in range(seq_len):
                sin_t = data[subj_idx, t, feature_cols.index("sin_hrs")]
                cos_t = data[subj_idx, t, feature_cols.index("cos_hrs")]
                x_t = np.array([1.0, t, sin_t, cos_t] + bl_vals)
                lmm_predictions[j, t, col_idx] = x_t @ beta_hat
            continue

        obs_times_arr = np.array(obs_times, dtype=float)
        obs_y_arr = np.array(obs_y)
        n_obs = len(obs_times)

        X_obs = np.column_stack([
            np.ones(n_obs), obs_times_arr,
            np.array(obs_sin), np.array(obs_cos),
            np.tile(bl_vals, (n_obs, 1)),
        ])
        Z_obs = np.column_stack([np.ones(n_obs), obs_times_arr])

        r = obs_y_arr - X_obs @ beta_hat
        V = Z_obs @ D @ Z_obs.T + sigma2_e * np.eye(n_obs) + 1e-6 * np.eye(n_obs)
        u_hat = D @ Z_obs.T @ np.linalg.solve(V, r)

        for t in range(seq_len):
            sin_t = data[subj_idx, t, feature_cols.index("sin_hrs")]
            cos_t = data[subj_idx, t, feature_cols.index("cos_hrs")]
            x_t = np.array([1.0, t, sin_t, cos_t] + bl_vals)
            z_t = np.array([1.0, t])
            lmm_predictions[j, t, col_idx] = x_t @ beta_hat + z_t @ u_hat

    # Clip bounded predictions
    lmm_predictions[:, :, col_idx] = np.clip(lmm_predictions[:, :, col_idx], 0, 1)

# ── 10. Model comparison ─────────────────────────────────────────────────────

lmm_future = lmm_predictions[:, landmark_t:, :]

print(f"\n{'Variable':<10s}  {'':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'Corr':>8s}")
print("-" * 50)

for col_idx, vname in enumerate(outcome_cols):
    a = future_actual[:, :, col_idx].ravel()

    # VAE
    p_vae = future_pred[:, :, col_idx].ravel()
    mae_v = np.mean(np.abs(a - p_vae))
    rmse_v = np.sqrt(np.mean((a - p_vae) ** 2))
    corr_v = np.corrcoef(a[~np.isnan(a)], p_vae[~np.isnan(a)])[0, 1] if np.nanstd(a) > 0 else float("nan")

    # LMM
    p_lmm = lmm_future[:, :, col_idx].ravel()
    mae_l = np.mean(np.abs(a - p_lmm))
    rmse_l = np.sqrt(np.mean((a - p_lmm) ** 2))
    corr_l = np.corrcoef(a[~np.isnan(a)], p_lmm[~np.isnan(a)])[0, 1] if np.nanstd(a) > 0 else float("nan")

    print(f"{vname:<10s}  {'VAE':>8s}  {mae_v:8.4f}  {rmse_v:8.4f}  {corr_v:8.4f}")
    print(f"{'':10s}  {'LMM':>8s}  {mae_l:8.4f}  {rmse_l:8.4f}  {corr_l:8.4f}")

print("\nDone.")
