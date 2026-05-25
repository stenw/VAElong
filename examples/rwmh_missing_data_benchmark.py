"""
Simulation benchmark for missing-data landmark prediction.

Compares:
1. VAE with missing-value RWMH updates
2. VAE with latent-space missing-data updates
3. Seq2Seq RNN
4. Ordinary linear mixed model benchmark

The workflow mirrors the newer EMA application more closely than the legacy
simulation scripts:
- explicit train/validation/test split
- early stopping on validation loss
- evaluation on a held-out test set only
- summary metrics on the future portion after the landmark time
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Subset

from vaelong import (
    LongitudinalDataset,
    LongitudinalVAE,
    VAETrainer,
    VariableConfig,
    VariableSpec,
    create_missing_mask,
    generate_mixed_longitudinal_data,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "rwmh_missing_data_benchmark_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MH_STEPS = 2
MH_ADAPTIVE = True
MH_TARGET_ACCEPT = 0.234
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
MISSING_RATE = 0.15
LANDMARK_KIND = "midpoint"
BINARY_OUTCOMES = {"symptom_present"}


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def safe_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) == 0:
        return float("nan")
    if np.nanstd(actual) == 0 or np.nanstd(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def create_variable_config() -> VariableConfig:
    return VariableConfig(variables=[
        VariableSpec(name="biomarker", var_type="continuous"),
        VariableSpec(name="blood_pressure", var_type="bounded", lower=60.0, upper=200.0),
        VariableSpec(name="symptom_present", var_type="binary"),
        VariableSpec(name="score", var_type="continuous"),
    ])


def make_splits(n_subjects: int, seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_subjects)
    n_train = int(TRAIN_FRACTION * n_subjects)
    n_val = int(VAL_FRACTION * n_subjects)
    return {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train:n_train + n_val]),
        "test": np.sort(indices[n_train + n_val:]),
    }


def compute_landmark_index(seq_len: int) -> int:
    if LANDMARK_KIND == "midpoint":
        return seq_len // 2
    raise ValueError(f"Unsupported landmark kind: {LANDMARK_KIND}")


class Seq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM for direct future prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, n_baseline: int,
                 var_config: VariableConfig):
        super().__init__()
        self.var_config = var_config
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.baseline_proj = nn.Linear(hidden_dim + n_baseline, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def _apply_activations(self, output: torch.Tensor) -> torch.Tensor:
        result = output.clone()
        for idx in self.var_config.binary_indices:
            result[:, :, idx] = torch.sigmoid(result[:, :, idx])
        for idx in self.var_config.bounded_indices:
            result[:, :, idx] = torch.sigmoid(result[:, :, idx])
        return result

    def forward(
        self,
        x_obs: torch.Tensor,
        mask_obs: torch.Tensor,
        baseline: torch.Tensor | None,
        future_target: torch.Tensor | None = None,
        future_len: int = 25,
    ) -> torch.Tensor:
        _, (h_enc, c_enc) = self.encoder(x_obs * mask_obs)
        if baseline is not None and baseline.numel() > 0:
            h_combined = torch.cat([h_enc.squeeze(0), baseline], dim=-1)
            h_dec = self.baseline_proj(h_combined).unsqueeze(0)
        else:
            h_dec = h_enc
        c_dec = c_enc

        predictions = []
        dec_input = x_obs[:, -1:, :]
        for t in range(future_len):
            dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            pred_t = self._apply_activations(self.fc_out(dec_out))
            predictions.append(pred_t)
            if future_target is not None and self.training:
                dec_input = future_target[:, t:t + 1, :]
            else:
                dec_input = pred_t.detach()
        return torch.cat(predictions, dim=1)


def build_simulation_data(
    n_samples: int = 500,
    seq_len: int = 50,
    n_baseline: int = 3,
    missing_rate: float = MISSING_RATE,
    missing_pattern: str = "random",
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VariableConfig]:
    var_config = create_variable_config()
    data, baseline = generate_mixed_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        var_config=var_config,
        n_baseline_features=n_baseline,
        noise_level=0.2,
        random_intercept_sd=2.0,
        seed=seed,
    )
    mask = create_missing_mask(
        data.shape, missing_rate=missing_rate, pattern=missing_pattern, seed=seed
    )
    return data, baseline, mask, var_config


def build_dataset(
    data: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    var_config: VariableConfig,
    train_indices: np.ndarray,
) -> LongitudinalDataset:
    normalized_data = np.array(data * mask, copy=True)
    n_features = var_config.n_features
    mean = torch.zeros(1, 1, n_features, dtype=torch.float32)
    std = torch.ones(1, 1, n_features, dtype=torch.float32)
    bounds_info: dict[int, tuple[float, float]] = {}

    for idx in var_config.continuous_indices:
        train_mask = mask[train_indices, :, idx] == 1.0
        train_values = data[train_indices, :, idx][train_mask]
        if train_values.size > 0:
            m = float(train_values.mean())
            s = float(train_values.std(ddof=0))
            if s == 0.0:
                s = 1.0
            mean[0, 0, idx] = m
            std[0, 0, idx] = s
            normalized_data[:, :, idx] = (
                ((data[:, :, idx] - m) / s) * mask[:, :, idx]
            )

    for idx in var_config.bounded_indices:
        lo, hi = var_config.get_bounds()[idx]
        bounds_info[idx] = (lo, hi)
        normalized_data[:, :, idx] = (
            ((data[:, :, idx] - lo) / (hi - lo)) * mask[:, :, idx]
        )
        if var_config.bounded_eps > 0:
            eps = var_config.bounded_eps
            normalized_data[:, :, idx] = (
                np.clip(normalized_data[:, :, idx], eps, 1 - eps) * mask[:, :, idx]
            )

    dataset = LongitudinalDataset(
        normalized_data,
        mask=mask,
        var_config=var_config,
        baseline_covariates=baseline,
        normalize=False,
    )
    dataset.mean = mean
    dataset.std = std
    dataset.bounds_info = bounds_info
    return dataset


def tune_and_train_vae(
    dataset: LongitudinalDataset,
    splits: dict[str, np.ndarray],
    seq_len: int,
    n_baseline: int,
    var_config: VariableConfig,
    imputation_method: str,
    model_label: str,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[LongitudinalVAE, VAETrainer, dict[str, list[float]], dict[str, float], pd.DataFrame]:
    train_loader = DataLoader(
        Subset(dataset, splits["train"].tolist()), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, splits["val"].tolist()), batch_size=32, shuffle=False
    )

    hp_grid = [
        {"learning_rate": 5e-4, "weight_decay": 0.0},
        {"learning_rate": 1e-3, "weight_decay": 0.0},
        {"learning_rate": 5e-4, "weight_decay": 1e-4},
        {"learning_rate": 1e-3, "weight_decay": 1e-4},
    ]
    best_val_loss = float("inf")
    best_hp = hp_grid[0]
    tuning_rows = []

    if verbose:
        print(f"\n--- Tuning {model_label} ---")
    for hp in hp_grid:
        set_seed(seed)
        model = LongitudinalVAE(
            input_dim=var_config.n_features,
            hidden_dim=64,
            latent_dim=16,
            encoder_type="lstm",
            n_baseline=n_baseline,
            var_config=var_config,
        )
        trainer = VAETrainer(
            model,
            learning_rate=hp["learning_rate"],
            beta=0.5,
            var_config=var_config,
            weight_decay=hp["weight_decay"],
        )
        history = trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=200,
            verbose=False,
            use_em_imputation=True,
            em_iterations=2,
            patience=20,
            imputation_method=imputation_method,
            mh_steps=MH_STEPS,
            mh_adaptive=MH_ADAPTIVE,
            mh_target_accept=MH_TARGET_ACCEPT,
        )
        best_candidate_val = float(min(history["val_loss"]))
        tuning_rows.append(
            {
                "Model": model_label,
                "imputation_method": imputation_method,
                "learning_rate": hp["learning_rate"],
                "weight_decay": hp["weight_decay"],
                "best_val_loss": best_candidate_val,
            }
        )
        if verbose:
            print(
                f"  lr={hp['learning_rate']:.0e}, wd={hp['weight_decay']:.0e} "
                f"-> best val loss = {best_candidate_val:.4f}"
            )
        if best_candidate_val < best_val_loss:
            best_val_loss = best_candidate_val
            best_hp = hp

    if verbose:
        print(
            f"Best hyperparameters for {model_label}: lr={best_hp['learning_rate']:.0e}, "
            f"weight_decay={best_hp['weight_decay']:.0e} "
            f"(val loss = {best_val_loss:.4f})"
        )

    set_seed(seed)
    best_model = LongitudinalVAE(
        input_dim=var_config.n_features,
        hidden_dim=64,
        latent_dim=16,
        encoder_type="lstm",
        n_baseline=n_baseline,
        var_config=var_config,
    )
    best_trainer = VAETrainer(
        best_model,
        learning_rate=best_hp["learning_rate"],
        beta=0.5,
        var_config=var_config,
        weight_decay=best_hp["weight_decay"],
    )
    history = best_trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=200,
        verbose=verbose,
        use_em_imputation=True,
        em_iterations=2,
        patience=20,
        imputation_method=imputation_method,
        mh_steps=MH_STEPS,
        mh_adaptive=MH_ADAPTIVE,
        mh_target_accept=MH_TARGET_ACCEPT,
    )
    return best_model, best_trainer, history, best_hp, pd.DataFrame(tuning_rows)


def save_training_curve(
    history: dict[str, list[float]],
    output_dir: Path,
    model_label: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(model_label)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def predict_vae(
    model: LongitudinalVAE,
    dataset: LongitudinalDataset,
    indices: np.ndarray,
    seq_len: int,
    landmark_t: int,
) -> np.ndarray:
    predictions = []
    for idx in indices:
        xi, mi, _, bi, _, _ = dataset[int(idx)]
        pred_i = model.predict_from_landmark(
            xi[:landmark_t].unsqueeze(0),
            mi[:landmark_t].unsqueeze(0),
            total_seq_len=seq_len,
            baseline=bi.unsqueeze(0) if bi.numel() > 0 else None,
        )
        predictions.append(dataset.inverse_transform(pred_i).detach())
    return torch.cat(predictions, dim=0).numpy()


def train_seq2seq(
    dataset: LongitudinalDataset,
    splits: dict[str, np.ndarray],
    var_config: VariableConfig,
    landmark_t: int,
    n_baseline: int,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[Seq2SeqLSTM, dict[str, float], pd.DataFrame]:
    future_len = dataset.data.shape[1] - landmark_t

    def collect(indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        x_obs, mask_obs, future, future_mask, baselines = [], [], [], [], []
        for idx in indices:
            xi, mi, _, bi, _, _ = dataset[int(idx)]
            x_obs.append(xi[:landmark_t])
            mask_obs.append(mi[:landmark_t])
            future.append(xi[landmark_t:])
            future_mask.append(mi[landmark_t:])
            baselines.append(bi)
        return (
            torch.stack(x_obs),
            torch.stack(mask_obs),
            torch.stack(future),
            torch.stack(future_mask),
            torch.stack(baselines),
        )

    train_x_obs, train_mask_obs, train_future, train_future_mask, train_bl = collect(splits["train"])
    val_x_obs, val_mask_obs, val_future, val_future_mask, val_bl = collect(splits["val"])

    def fit_candidate(hidden_dim: int, learning_rate: float, log_progress: bool = False):
        set_seed(seed)
        model = Seq2SeqLSTM(var_config.n_features, hidden_dim, n_baseline, var_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0

        for epoch in range(200):
            model.train()
            perm = torch.randperm(len(train_x_obs))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_x_obs), 32):
                batch_idx = perm[start:start + 32]
                bx = train_x_obs[batch_idx]
                bm = train_mask_obs[batch_idx]
                bf = train_future[batch_idx]
                bfm = train_future_mask[batch_idx]
                bb = train_bl[batch_idx]

                pred = model(bx, bm, bb, future_target=bf, future_len=future_len)
                loss = compute_seq2seq_loss(pred, bf, bfm, var_config)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            model.eval()
            with torch.no_grad():
                val_pred = model(
                    val_x_obs, val_mask_obs, val_bl,
                    future_target=None, future_len=future_len,
                )
                val_loss = float(
                    compute_seq2seq_loss(val_pred, val_future, val_future_mask, var_config).item()
                )

            if log_progress and (epoch + 1) % 25 == 0:
                print(
                    f"  Epoch [{epoch + 1:3d}/200] "
                    f"train={epoch_loss / max(n_batches, 1):.4f} val={val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if log_progress:
                        print(
                            f"  Early stopping at epoch {epoch + 1} "
                            f"(best val loss = {best_val_loss:.4f})"
                        )
                    break

        model.load_state_dict(best_state)
        return model, best_val_loss

    hp_grid = [
        {"hidden_dim": 32, "learning_rate": 5e-4},
        {"hidden_dim": 32, "learning_rate": 1e-3},
        {"hidden_dim": 64, "learning_rate": 5e-4},
        {"hidden_dim": 64, "learning_rate": 1e-3},
    ]
    best_hp = hp_grid[0]
    best_val_loss = float("inf")
    best_model = None
    tuning_rows = []

    if verbose:
        print("\n--- Tuning Seq2Seq RNN benchmark ---")
    for hp in hp_grid:
        candidate_model, candidate_val_loss = fit_candidate(
            hidden_dim=hp["hidden_dim"],
            learning_rate=hp["learning_rate"],
            log_progress=False,
        )
        tuning_rows.append(
            {
                "hidden_dim": hp["hidden_dim"],
                "learning_rate": hp["learning_rate"],
                "best_val_loss": candidate_val_loss,
            }
        )
        if verbose:
            print(
                f"  hidden_dim={hp['hidden_dim']}, lr={hp['learning_rate']:.0e} "
                f"-> best val loss = {candidate_val_loss:.4f}"
            )
        if candidate_val_loss < best_val_loss:
            best_val_loss = candidate_val_loss
            best_hp = hp
            best_model = candidate_model

    if verbose:
        print(
            f"Best RNN hyperparameters: hidden_dim={best_hp['hidden_dim']}, "
            f"lr={best_hp['learning_rate']:.0e} (val loss = {best_val_loss:.4f})"
        )
        print("\n--- Refitting Seq2Seq RNN benchmark with best settings ---")

    best_model, _ = fit_candidate(
        hidden_dim=best_hp["hidden_dim"],
        learning_rate=best_hp["learning_rate"],
        log_progress=verbose,
    )
    return best_model, best_hp, pd.DataFrame(tuning_rows)


def compute_seq2seq_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    var_config: VariableConfig,
) -> torch.Tensor:
    loss = predicted.new_tensor(0.0)
    for cidx in var_config.continuous_indices:
        diff2 = (predicted[:, :, cidx] - target[:, :, cidx]) ** 2
        m = target_mask[:, :, cidx]
        if float(m.sum()) > 0:
            loss = loss + (diff2 * m).sum() / m.sum()

    for cidx in var_config.binary_indices + var_config.bounded_indices:
        p_clamped = predicted[:, :, cidx].clamp(1e-7, 1 - 1e-7)
        tgt = target[:, :, cidx]
        bce = -(tgt * torch.log(p_clamped) + (1 - tgt) * torch.log(1 - p_clamped))
        m = target_mask[:, :, cidx]
        if float(m.sum()) > 0:
            loss = loss + (bce * m).sum() / m.sum()
    return loss


def predict_seq2seq(
    model: Seq2SeqLSTM,
    dataset: LongitudinalDataset,
    indices: np.ndarray,
    seq_len: int,
    landmark_t: int,
) -> np.ndarray:
    future_len = seq_len - landmark_t
    predictions = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            xi, mi, _, bi, _, _ = dataset[int(idx)]
            pred_future = model(
                xi[:landmark_t].unsqueeze(0),
                mi[:landmark_t].unsqueeze(0),
                bi.unsqueeze(0),
                future_target=None,
                future_len=future_len,
            )
            full_pred = torch.cat([xi[:landmark_t], pred_future.squeeze(0)], dim=0)
            predictions.append(dataset.inverse_transform(full_pred.unsqueeze(0)).detach())
    return torch.cat(predictions, dim=0).numpy()


def predict_lmm(
    data: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    var_config: VariableConfig,
    landmark_t: int,
    verbose: bool = True,
) -> np.ndarray:
    seq_len = data.shape[1]
    n_test = len(test_indices)
    n_baseline = baseline.shape[1]
    predictions = np.zeros((n_test, seq_len, var_config.n_features), dtype=np.float32)

    if verbose:
        print("\n--- Fitting ordinary mixed-model benchmark ---")
    for col, var_spec in enumerate(var_config.variables):
        if verbose:
            print(f"  Fitting mixed model for {var_spec.name}...", end=" ", flush=True)
        rows = []
        for subject_idx in train_indices:
            for t in range(landmark_t):
                if mask[int(subject_idx), t, col] == 1.0:
                    row = {
                        "subject": int(subject_idx),
                        "time": t,
                        "y": float(data[int(subject_idx), t, col]),
                    }
                    for b in range(n_baseline):
                        row[f"bl_{b}"] = float(baseline[int(subject_idx), b])
                    rows.append(row)
        df_train = pd.DataFrame(rows)

        fixed_formula = "y ~ time + " + " + ".join(f"bl_{b}" for b in range(n_baseline))
        re_formula = "1 + time"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(
                fixed_formula, df_train, groups=df_train["subject"], re_formula=re_formula
            )
            fitted = md.fit(reml=True, method="lbfgs")

        beta_hat = np.asarray(fitted.fe_params, dtype=float)
        D = np.asarray(fitted.cov_re, dtype=float)
        sigma2_e = float(fitted.scale)
        if verbose:
            print(f"done (sigma^2={sigma2_e:.4f})")

        for j, subject_idx in enumerate(test_indices):
            obs_times = []
            obs_y = []
            for t in range(landmark_t):
                if mask[int(subject_idx), t, col] == 1.0:
                    obs_times.append(float(t))
                    obs_y.append(float(data[int(subject_idx), t, col]))

            bl_vals = [float(baseline[int(subject_idx), b]) for b in range(n_baseline)]
            if len(obs_times) == 0:
                for t in range(seq_len):
                    x_t = np.array([1.0, float(t)] + bl_vals)
                    predictions[j, t, col] = x_t @ beta_hat
                continue

            obs_times = np.asarray(obs_times)
            obs_y = np.asarray(obs_y)
            X_obs = np.column_stack([np.ones(len(obs_times)), obs_times, np.tile(bl_vals, (len(obs_times), 1))])
            Z_obs = np.column_stack([np.ones(len(obs_times)), obs_times])
            residual = obs_y - X_obs @ beta_hat
            V = Z_obs @ D @ Z_obs.T + sigma2_e * np.eye(len(obs_times)) + 1e-6 * np.eye(len(obs_times))
            u_hat = D @ Z_obs.T @ np.linalg.solve(V, residual)

            for t in range(seq_len):
                x_t = np.array([1.0, float(t)] + bl_vals)
                z_t = np.array([1.0, float(t)])
                predictions[j, t, col] = x_t @ beta_hat + z_t @ u_hat

        if var_spec.var_type == "binary":
            predictions[:, :, col] = np.clip(predictions[:, :, col], 0.0, 1.0)
        elif var_spec.var_type == "bounded":
            predictions[:, :, col] = np.clip(predictions[:, :, col], var_spec.lower, var_spec.upper)

    return predictions


def evaluate_predictions(
    actual: np.ndarray,
    predicted_by_model: dict[str, np.ndarray],
    future_mask: np.ndarray,
    var_config: VariableConfig,
    landmark_t: int,
) -> pd.DataFrame:
    eps_ll = 1e-7
    rows = []
    future_actual = actual[:, landmark_t:, :]
    future_mask = future_mask[:, landmark_t:, :]

    for model_name, full_pred in predicted_by_model.items():
        future_pred = full_pred[:, landmark_t:, :]
        for col_idx, var_spec in enumerate(var_config.variables):
            a = future_actual[:, :, col_idx].ravel()
            p = future_pred[:, :, col_idx].ravel()
            valid = future_mask[:, :, col_idx].ravel().astype(bool)
            a = a[valid]
            p = p[valid]
            if len(a) == 0:
                continue

            rmse = float(np.sqrt(np.mean((a - p) ** 2)))
            corr = safe_correlation(a, p)
            row = {
                "Model": model_name,
                "Variable": var_spec.name,
                "RMSE": rmse,
                "Corr": corr,
                "VarType": var_spec.var_type,
            }

            if var_spec.var_type == "binary":
                p_clip = np.clip(p, eps_ll, 1 - eps_ll)
                row["LogLik"] = float(-log_loss(a, p_clip))
                row["AUC"] = float(roc_auc_score(a, p)) if np.unique(a).size > 1 else float("nan")
            else:
                sigma = max(rmse, eps_ll)
                row["LogLik"] = float(
                    -0.5 * np.mean(((a - p) / sigma) ** 2)
                    - np.log(sigma)
                    - 0.5 * np.log(2 * np.pi)
                )
                row["AUC"] = float("nan")

            rows.append(row)
    return pd.DataFrame(rows)


def save_metric_plot(results_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["RMSE", "Corr", "LogLik"]
    if "AUC" in results_df.columns and results_df["AUC"].notna().any():
        metrics.append("AUC")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        sub = results_df.dropna(subset=[metric])
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="Variable", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.legend(title="")
    fig.suptitle("Missing-Data Simulation Benchmark (Held-out Future Test Performance)", fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(output_dir / "rwmh_missing_data_metrics.png", dpi=150)
    plt.close(fig)


def save_profile_plot(
    actual: np.ndarray,
    predicted_by_model: dict[str, np.ndarray],
    mask: np.ndarray,
    chosen_indices: np.ndarray,
    var_config: VariableConfig,
    landmark_t: int,
    output_dir: Path,
) -> None:
    time_axis = np.arange(actual.shape[1])
    n_subjects = len(chosen_indices)
    n_vars = var_config.n_features
    fig, axes = plt.subplots(
        n_subjects,
        n_vars,
        figsize=(4 * n_vars, 3.4 * n_subjects),
        sharex=True,
    )
    axes = np.atleast_2d(axes)
    colors = {
        "VAE-RWMH": "tab:red",
        "VAE-Latent": "tab:orange",
        "RNN": "tab:green",
        "MixedModel": "tab:blue",
    }
    linestyles = {
        "VAE-RWMH": "-",
        "VAE-Latent": "-.",
        "RNN": "--",
        "MixedModel": "--",
    }

    for row, subject_idx in enumerate(chosen_indices):
        for col, var_spec in enumerate(var_config.variables):
            ax = axes[row, col]
            observed = mask[int(subject_idx), :, col].astype(bool)
            ax.scatter(
                time_axis[observed],
                actual[int(subject_idx), observed, col],
                color="black",
                s=8,
                zorder=3,
                label="Observed",
            )
            for model_name, predicted in predicted_by_model.items():
                ax.plot(
                    time_axis[:landmark_t],
                    predicted[row, :landmark_t, col],
                    color=colors[model_name],
                    linewidth=1.0,
                    alpha=0.6,
                    linestyle=linestyles[model_name],
                )
                ax.plot(
                    time_axis[landmark_t:],
                    predicted[row, landmark_t:, col],
                    color=colors[model_name],
                    linewidth=1.6,
                    linestyle=linestyles[model_name],
                    label=model_name,
                )

            ax.axvspan(landmark_t, actual.shape[1] - 1, alpha=0.08, color="tab:red")
            ax.axvline(landmark_t - 0.5, color="grey", linestyle="--", linewidth=0.8)
            if var_spec.var_type == "binary":
                ax.set_ylim(0.0, 1.0)
            if row == 0:
                ax.set_title(var_spec.name)
            if col == 0:
                ax.set_ylabel(f"Subject {int(subject_idx)}")
            if row == n_subjects - 1:
                ax.set_xlabel("Time step")

    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Observed vs fitted/predicted profiles (landmark at t = {landmark_t})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "rwmh_missing_data_profiles.png", dpi=150)
    plt.close(fig)


def run_benchmark(
    show_plots: bool = False,
    seed: int = SEED,
    n_samples: int = 500,
    seq_len: int = 50,
    n_baseline: int = 3,
    missing_rate: float = MISSING_RATE,
    missing_pattern: str = "random",
    output_dir: Path | None = None,
    save_artifacts: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    set_seed(seed)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    data, baseline, mask, var_config = build_simulation_data(
        n_samples=n_samples,
        seq_len=seq_len,
        n_baseline=n_baseline,
        missing_rate=missing_rate,
        missing_pattern=missing_pattern,
        seed=seed,
    )
    splits = make_splits(n_samples, seed=seed)
    dataset = build_dataset(data, baseline, mask, var_config, train_indices=splits["train"])
    landmark_t = compute_landmark_index(seq_len)

    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"Baseline shape: {baseline.shape}")
        print(f"Missing data: {(1 - mask.mean()) * 100:.1f}%")
        print(
            f"Split sizes: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

    vae_best_hp_by_method: dict[str, dict[str, float]] = {}
    vae_predicted_by_model: dict[str, np.ndarray] = {}
    vae_tuning_frames: list[pd.DataFrame] = []
    vae_curve_files = {
        "rwmh": "rwmh_vae_training_curve.png",
        "latent": "latent_vae_training_curve.png",
    }
    vae_labels = {
        "rwmh": "VAE-RWMH",
        "latent": "VAE-Latent",
    }
    for method in ("rwmh", "latent"):
        vae_model, _, history, best_hp, tuning_df = tune_and_train_vae(
            dataset,
            splits,
            seq_len,
            n_baseline,
            var_config,
            imputation_method=method,
            model_label=vae_labels[method],
            seed=seed,
            verbose=verbose,
        )
        vae_best_hp_by_method[method] = best_hp
        vae_tuning_frames.append(tuning_df)
        vae_predicted_by_model[vae_labels[method]] = predict_vae(
            vae_model, dataset, splits["test"], seq_len, landmark_t
        )
        if save_artifacts:
            save_training_curve(
                history,
                output_dir,
                vae_labels[method],
                vae_curve_files[method],
            )
    tuning_df = pd.concat(vae_tuning_frames, ignore_index=True)

    rnn_model, rnn_best_hp, rnn_tuning_df = train_seq2seq(
        dataset, splits, var_config, landmark_t, n_baseline, seed=seed, verbose=verbose
    )
    lmm_pred_test = predict_lmm(
        data,
        baseline,
        mask,
        train_indices=splits["train"],
        test_indices=splits["test"],
        var_config=var_config,
        landmark_t=landmark_t,
        verbose=verbose,
    )

    rnn_pred_test = predict_seq2seq(rnn_model, dataset, splits["test"], seq_len, landmark_t)

    predicted_by_model = {
        **vae_predicted_by_model,
        "RNN": rnn_pred_test,
        "MixedModel": lmm_pred_test,
    }
    results_df = evaluate_predictions(
        actual=data[splits["test"]],
        predicted_by_model=predicted_by_model,
        future_mask=mask[splits["test"]],
        var_config=var_config,
        landmark_t=landmark_t,
    )
    if save_artifacts:
        results_df.to_csv(output_dir / "rwmh_missing_data_results.csv", index=False)
        tuning_df.to_csv(output_dir / "rwmh_missing_data_tuning.csv", index=False)
        rnn_tuning_df.to_csv(output_dir / "rwmh_missing_data_rnn_tuning.csv", index=False)
        save_metric_plot(results_df, output_dir)

    rng = np.random.default_rng(123)
    chosen_local = np.sort(
        rng.choice(len(splits["test"]), size=min(3, len(splits["test"])), replace=False)
    )
    chosen_global = splits["test"][chosen_local]
    if save_artifacts:
        save_profile_plot(
            actual=data,
            predicted_by_model={
                name: values[chosen_local]
                for name, values in predicted_by_model.items()
            },
            mask=mask,
            chosen_indices=chosen_global,
            var_config=var_config,
            landmark_t=landmark_t,
            output_dir=output_dir,
        )

    if verbose:
        print("\n--- Held-out future test metrics ---")
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print(f"\nBest VAE-RWMH hyperparameters: {vae_best_hp_by_method['rwmh']}")
        print(f"Best VAE-Latent hyperparameters: {vae_best_hp_by_method['latent']}")
        print(f"Best RNN hyperparameters: {rnn_best_hp}")
        if save_artifacts:
            print(f"Outputs saved under: {output_dir}")

    if show_plots:
        for plot_name in [
            "rwmh_vae_training_curve.png",
            "latent_vae_training_curve.png",
            "rwmh_missing_data_profiles.png",
            "rwmh_missing_data_metrics.png",
        ]:
            img = plt.imread(OUTPUT_DIR / plot_name)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(plot_name)
            plt.show()

    return {
        "results_df": results_df,
        "tuning_df": tuning_df,
        "best_hp": vae_best_hp_by_method["rwmh"],
        "best_hp_by_method": vae_best_hp_by_method,
        "rnn_best_hp": rnn_best_hp,
        "output_dir": output_dir,
        "seed": seed,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
    }


def main() -> None:
    run_benchmark(show_plots=False)


if __name__ == "__main__":
    main()
