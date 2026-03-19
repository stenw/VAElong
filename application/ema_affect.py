"""
EMA Affect Modelling — Proof of Concept
========================================
Model EM_PA, EM_NA (bounded [0, 1]) and EM_BO (binary 0/1) from ecological
momentary assessment data using the vaelong VAE framework.

Features:
  - EM_PA, EM_NA: bounded outcome variables
  - EM_BO: binary outcome variable
  - sin_hrs, cos_hrs: continuous time-varying features
  - AGE, SEX_1, SEX_2, SEX_3: baseline covariates

Models:
  - Dense VAE with EM imputation + hyperparameter tuning
  - Linear Mixed Model (benchmark)
"""

import warnings
import itertools
import copy

import scipy.sparse as sp

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, log_loss
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
outcome_cols = ["EM_PA", "EM_NA", "EM_BO"]
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
    VariableSpec(name="EM_BO",    var_type="binary"),
    VariableSpec(name="sin_hrs",  var_type="continuous"),
    VariableSpec(name="cos_hrs",  var_type="continuous"),
])

print(f"\nBounded indices:    {var_config.bounded_indices}")
print(f"Binary indices:     {var_config.binary_indices}")
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

# ── 5. Hyperparameter tuning ─────────────────────────────────────────────────

hp_grid = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

hp_combos = list(itertools.product(hp_grid["learning_rate"], hp_grid["weight_decay"]))
print(f"Tuning over {len(hp_combos)} hyperparameter combinations...")

best_val_loss = float("inf")
best_hp = None
tuning_results = []

for lr, wd in hp_combos:
    torch.manual_seed(42)
    np.random.seed(42)

    m = LongitudinalVAE(
        input_dim=var_config.n_features,
        hidden_dim=64,
        latent_dim=16,
        seq_len=seq_len,
        n_baseline=n_baseline,
        var_config=var_config,
    )
    t = VAETrainer(m, learning_rate=lr, beta=0.5, var_config=var_config,
                   weight_decay=wd)

    h = t.fit(
        train_loader, val_loader=val_loader, epochs=200, verbose=False,
        use_em_imputation=True, em_iterations=2, patience=20,
    )

    final_val = min(h["val_loss"])
    tuning_results.append({"lr": lr, "weight_decay": wd, "best_val_loss": final_val})
    print(f"  lr={lr:.0e}, wd={wd:.0e}  ->  best val loss = {final_val:.4f}")

    if final_val < best_val_loss:
        best_val_loss = final_val
        best_hp = {"learning_rate": lr, "weight_decay": wd}

tuning_df = pd.DataFrame(tuning_results)
print(f"\nBest hyperparameters: lr={best_hp['learning_rate']:.0e}, "
      f"weight_decay={best_hp['weight_decay']:.0e} "
      f"(val loss = {best_val_loss:.4f})")

# ── 5b. Retrain with best hyperparameters ────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

model = LongitudinalVAE(
    input_dim=var_config.n_features,
    hidden_dim=64,
    latent_dim=16,
    seq_len=seq_len,
    n_baseline=n_baseline,
    var_config=var_config,
)

trainer = VAETrainer(model, learning_rate=best_hp["learning_rate"], beta=0.5,
                     var_config=var_config, weight_decay=best_hp["weight_decay"])

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
ax.set_title("Dense VAE — Training and Validation Loss (best HP)")
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

# Build mask for validation subjects (original mask, not normalised)
val_mask = mask[val_indices, :, :]           # (n_val, seq_len, n_features)
future_mask = val_mask[:, landmark_t:, :]    # (n_val, future_len, n_features)

# ── 8. Plot landmark predictions for 3 individuals ───────────────────────────

rng = np.random.default_rng(123)
chosen = sorted(rng.choice(len(val_indices), size=3, replace=False))
time_axis = np.arange(seq_len)

n_outcomes = len(outcome_cols)
fig, axes = plt.subplots(3, n_outcomes, figsize=(4 * n_outcomes, 9), sharex=True)

for row, c in enumerate(chosen):
    for col, vname in enumerate(outcome_cols):
        ax = axes[row, col]
        vidx = feature_cols.index(vname)

        actual_vals = all_actual[c, :, vidx]
        pred_vals = all_predicted[c, :, vidx]
        obs_mask_c = val_mask[c, :, vidx].astype(bool)

        # Only plot actual where observed (scatter to avoid joining gaps)
        ax.scatter(time_axis[obs_mask_c], actual_vals[obs_mask_c],
                   c="k", s=6, zorder=3, label="Actual")
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

# ── 9. LMM / GLMM benchmark ──────────────────────────────────────────────────
#
# For continuous/bounded outcomes: Gaussian LMM (random intercept + slope)
# For binary outcomes:             Binomial GLMM (random intercept, logit link)
#
# Both models are trained on ALL time points of training subjects (symmetric
# with the VAE).  Predictions for validation subjects use BLUPs computed from
# observations up to landmark_t only.

from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _glmm_blup(obs_y, X_obs, Z_obs, beta, D_inv, max_iter=25, tol=1e-6):
    """Approximate BLUP for a logistic GLMM via Newton-Raphson / PQL."""
    u = np.zeros(D_inv.shape[0])
    for _ in range(max_iter):
        eta = X_obs @ beta + Z_obs @ u
        mu = _sigmoid(eta)
        W = mu * (1 - mu) + 1e-8          # working weights
        grad = Z_obs.T @ (obs_y - mu) - D_inv @ u
        H = Z_obs.T @ (W[:, None] * Z_obs) + D_inv
        delta = np.linalg.solve(H, grad)
        u = u + delta
        if np.max(np.abs(delta)) < tol:
            break
    return u


train_indices = list(train_ds.indices)
n_val = len(val_indices)

lmm_predictions = np.zeros((n_val, seq_len, n_features))

for col_idx, vname in enumerate(outcome_cols):
    is_binary = (vname == "EM_BO")
    model_label = "GLMM" if is_binary else "LMM"
    print(f"  Fitting {model_label} for {vname}...", end=" ", flush=True)

    # -- assemble training data (ALL time points) --
    rows = []
    for i in train_indices:
        for t in range(seq_len):
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

    if is_binary:
        # ── Binomial GLMM (random intercept, logit link) ────────────────
        subjects_in_train = sorted(df_train["subject"].unique())
        n_train_subj = len(subjects_in_train)
        subj_map = {s: i for i, s in enumerate(subjects_in_train)}
        n_rows = len(df_train)

        # Fixed-effects design matrix (with intercept)
        exog = df_train[["time", "sin_hrs", "cos_hrs"] + baseline_cols].values
        exog = np.column_stack([np.ones(n_rows), exog])
        fe_names = ["Intercept", "time", "sin_hrs", "cos_hrs"] + baseline_cols

        # Random intercept per subject (sparse)
        exog_vc = sp.lil_matrix((n_rows, n_train_subj))
        subj_arr = df_train["subject"].values
        for r_idx in range(n_rows):
            exog_vc[r_idx, subj_map[subj_arr[r_idx]]] = 1.0
        exog_vc = exog_vc.tocsc()
        ident = np.zeros(n_train_subj, dtype=int)

        endog = df_train["y"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glmm = BinomialBayesMixedGLM(
                endog, exog, exog_vc, ident, vcp_p=1.0,
                fep_names=fe_names,
            )
            glmm_result = glmm.fit_vb()

        beta_hat = glmm_result.fe_mean
        sigma_u = np.exp(glmm_result.vcp_mean[0])
        D_glmm = np.array([[sigma_u ** 2]])
        D_inv_glmm = np.array([[1.0 / (sigma_u ** 2 + 1e-12)]])
        print(f"done (sigma_u={sigma_u:.4f})")

        # Predict for each validation subject
        for j, subj_idx in enumerate(val_indices):
            obs_times, obs_y, obs_sin, obs_cos = [], [], [], []
            for t in range(landmark_t):
                if mask[subj_idx, t, col_idx] == 1.0:
                    obs_times.append(t)
                    obs_y.append(data[subj_idx, t, col_idx])
                    obs_sin.append(
                        data[subj_idx, t, feature_cols.index("sin_hrs")])
                    obs_cos.append(
                        data[subj_idx, t, feature_cols.index("cos_hrs")])

            bl_vals = [baseline[subj_idx, b] for b in range(n_baseline)]

            if len(obs_times) > 0:
                n_obs = len(obs_times)
                X_obs = np.column_stack([
                    np.ones(n_obs),
                    np.array(obs_times, dtype=float),
                    np.array(obs_sin), np.array(obs_cos),
                    np.tile(bl_vals, (n_obs, 1)),
                ])
                Z_obs = np.ones((n_obs, 1))  # random intercept only
                obs_y_arr = np.array(obs_y)
                u_hat = _glmm_blup(
                    obs_y_arr, X_obs, Z_obs, beta_hat, D_inv_glmm)
            else:
                u_hat = np.zeros(1)

            for t in range(seq_len):
                sin_t = data[subj_idx, t, feature_cols.index("sin_hrs")]
                cos_t = data[subj_idx, t, feature_cols.index("cos_hrs")]
                x_t = np.array([1.0, t, sin_t, cos_t] + bl_vals)
                eta = x_t @ beta_hat + u_hat[0]
                lmm_predictions[j, t, col_idx] = _sigmoid(eta)

    else:
        # ── Gaussian LMM (random intercept + slope) ─────────────────────
        fixed_formula = ("y ~ time + sin_hrs + cos_hrs + "
                         + " + ".join(baseline_cols))
        re_formula = "1 + time"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(fixed_formula, df_train,
                             groups=df_train["subject"],
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
                    obs_sin.append(
                        data[subj_idx, t, feature_cols.index("sin_hrs")])
                    obs_cos.append(
                        data[subj_idx, t, feature_cols.index("cos_hrs")])

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
            V = (Z_obs @ D @ Z_obs.T + sigma2_e * np.eye(n_obs)
                 + 1e-6 * np.eye(n_obs))
            u_hat = D @ Z_obs.T @ np.linalg.solve(V, r)

            for t in range(seq_len):
                sin_t = data[subj_idx, t, feature_cols.index("sin_hrs")]
                cos_t = data[subj_idx, t, feature_cols.index("cos_hrs")]
                x_t = np.array([1.0, t, sin_t, cos_t] + bl_vals)
                z_t = np.array([1.0, t])
                lmm_predictions[j, t, col_idx] = x_t @ beta_hat + z_t @ u_hat

        # Clip bounded predictions to [0, 1]
        lmm_predictions[:, :, col_idx] = np.clip(
            lmm_predictions[:, :, col_idx], 0, 1)

# ── 10. Model comparison ─────────────────────────────────────────────────────

lmm_future = lmm_predictions[:, landmark_t:, :]

eps_ll = 1e-7  # numerical stability for log-likelihood

print(f"\n{'Variable':<10s}  {'':>8s}  {'RMSE':>8s}  {'Corr':>8s}  {'LogLik':>10s}  {'AUC':>8s}")
print("-" * 60)

for col_idx, vname in enumerate(outcome_cols):
    a = future_actual[:, :, col_idx].ravel()
    valid = future_mask[:, :, col_idx].ravel().astype(bool)  # use real mask
    is_binary = (vname == "EM_BO")

    bench_label = "GLMM" if is_binary else "LMM"
    for label, preds_arr in [("VAE", future_pred), (bench_label, lmm_future)]:
        p = preds_arr[:, :, col_idx].ravel()
        rmse = np.sqrt(np.mean((a[valid] - p[valid]) ** 2))
        corr = np.corrcoef(a[valid], p[valid])[0, 1] if np.nanstd(a[valid]) > 0 else float("nan")

        if is_binary:
            p_clip = np.clip(p[valid], eps_ll, 1 - eps_ll)
            ll = -log_loss(a[valid], p_clip)  # negative log-loss = mean log-lik
            auc = roc_auc_score(a[valid], p[valid])
            ll_str = f"{ll:10.4f}"
            auc_str = f"{auc:8.4f}"
        else:
            # Gaussian log-likelihood (up to a constant)
            sigma = rmse  # use RMSE as plug-in sigma
            ll = -0.5 * np.mean(((a[valid] - p[valid]) / max(sigma, eps_ll)) ** 2) \
                 - np.log(max(sigma, eps_ll)) - 0.5 * np.log(2 * np.pi)
            ll_str = f"{ll:10.4f}"
            auc_str = f"{'--':>8s}"

        var_label = vname if label == "VAE" else ""
        print(f"{var_label:<10s}  {label:>8s}  {rmse:8.4f}  {corr:8.4f}  {ll_str}  {auc_str}")

print(f"\nBest HP: lr={best_hp['learning_rate']:.0e}, weight_decay={best_hp['weight_decay']:.0e}")
print("Done.")
