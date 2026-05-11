"""
Generic YAML-driven application runner for VAElong.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from .app_config import ApplicationConfig
from .config import VariableConfig
from .data import LongitudinalDataset
from .model import LongitudinalVAE
from .trainer import VAETrainer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def infer_format(path: Path, format_hint: Optional[str]) -> str:
    if format_hint:
        return format_hint.lower()
    if path.suffix.lower() == ".parquet":
        return "parquet"
    if path.suffix.lower() == ".csv":
        return "csv"
    raise ValueError(f"Could not infer file format from path: {path}")


def load_dataframe(config: ApplicationConfig, data_path_override: Optional[str]) -> pd.DataFrame:
    data_path = config.resolve_data_path(data_path_override)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    data_format = infer_format(data_path, config.data.format)
    if data_format == "parquet":
        df = pd.read_parquet(data_path).copy()
    elif data_format == "csv":
        df = pd.read_csv(data_path).copy()
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    return df


def apply_transforms(df: pd.DataFrame, config: ApplicationConfig) -> pd.DataFrame:
    out = df.copy()
    for transform in config.transforms:
        kind = transform.type
        params = transform.params

        if kind == "time_fraction_of_day":
            source = params.get("source", config.data.time_col)
            output = params.get("output", "time_of_day")
            timestamp = pd.to_datetime(out[source])
            seconds = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
            out[output] = seconds / 86400.0
        elif kind == "sincos":
            source = params["source"]
            sin_name = params["sin_name"]
            cos_name = params["cos_name"]
            period = float(params.get("period", 1.0))
            radians = 2.0 * math.pi * out[source] / period
            out[sin_name] = np.sin(radians)
            out[cos_name] = np.cos(radians)
        elif kind == "binary_threshold":
            source = params["source"]
            output = params["output"]
            threshold = float(params["threshold"])
            operator = params.get("operator", "gt")
            if operator == "gt":
                values = out[source] > threshold
            elif operator == "ge":
                values = out[source] >= threshold
            else:
                raise ValueError(f"Unsupported binary_threshold operator: {operator}")
            out[output] = np.where(out[source].notna(), values.astype(np.float32), np.nan)
        elif kind == "rename_column":
            source = params["source"]
            output = params["output"]
            out = out.rename(columns={source: output})
        else:
            raise ValueError(f"Unsupported transform type: {kind}")

    return out


def validate_required_columns(df: pd.DataFrame, config: ApplicationConfig) -> None:
    required = {
        config.data.subject_col,
        config.data.time_col,
        *config.data.sort_by,
        *config.data.feature_cols,
        *config.data.baseline_cols,
    }
    if config.data.subject_label_col is not None:
        required.add(config.data.subject_label_col)

    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def filter_equal_length_sequences(
    df: pd.DataFrame,
    subject_col: str,
    strict: bool,
) -> tuple[pd.DataFrame, pd.Series, int]:
    counts = df.groupby(subject_col).size().sort_index()
    if counts.empty:
        raise ValueError("No subject trajectories found in the input data.")

    if counts.nunique() == 1:
        return df, counts, int(counts.iloc[0])

    modal_len = int(counts.mode().iloc[0])
    if strict:
        raise ValueError(
            "Subjects do not share a common sequence length. "
            f"Observed counts: {counts.value_counts().sort_index().to_dict()}"
        )

    keep_ids = counts[counts == modal_len].index
    filtered = df[df[subject_col].isin(keep_ids)].copy()
    return filtered, counts.loc[keep_ids], modal_len


def build_subject_arrays(
    df: pd.DataFrame,
    config: ApplicationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    label_col = config.data.subject_label_col or config.data.subject_col
    patient_keys = (
        df[[config.data.subject_col, label_col]]
        .drop_duplicates()
        .sort_values(config.data.subject_col)
        .reset_index(drop=True)
        .rename(columns={label_col: "subject_label", config.data.subject_col: "subject_key"})
    )

    groups = list(df.groupby(config.data.subject_col, sort=True))
    n_subjects = len(groups)
    seq_len = len(groups[0][1])
    feature_cols = config.data.feature_cols
    baseline_cols = config.data.baseline_cols

    data = np.zeros((n_subjects, seq_len, len(feature_cols)), dtype=np.float32)
    mask = np.ones((n_subjects, seq_len, len(feature_cols)), dtype=np.float32)
    baseline = np.zeros((n_subjects, len(baseline_cols)), dtype=np.float32)

    observed_feature_cols = set(config.data.resolved_observed_feature_cols)

    for i, (_, grp) in enumerate(groups):
        grp = grp.sort_values(config.data.time_col).reset_index(drop=True)
        for j, col in enumerate(feature_cols):
            values = grp[col].to_numpy(dtype=np.float32, copy=True)
            data[i, :, j] = np.nan_to_num(values, nan=0.0)
            mask[i, :, j] = (~np.isnan(values)).astype(np.float32)
            if col not in observed_feature_cols:
                mask[i, :, j] = 1.0

        if baseline_cols:
            baseline[i] = grp[baseline_cols].iloc[0].to_numpy(dtype=np.float32, copy=True)

    return data, mask, baseline, patient_keys


def make_splits(
    n_subjects: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_subjects)
    n_train = int(n_subjects * train_fraction)
    n_val = int(n_subjects * val_fraction)
    n_test = n_subjects - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split fractions produce an empty train/validation/test partition.")

    return {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train:n_train + n_val]),
        "test": np.sort(indices[n_train + n_val:]),
    }


def build_model(
    input_dim: int,
    seq_len: int,
    n_baseline: int,
    var_config: VariableConfig,
    config: ApplicationConfig,
    hidden_dim: Optional[int] = None,
    latent_dim: Optional[int] = None,
) -> LongitudinalVAE:
    return LongitudinalVAE(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim if hidden_dim is None else hidden_dim,
        latent_dim=config.model.latent_dim if latent_dim is None else latent_dim,
        encoder_type=config.model.encoder_type,
        seq_len=seq_len if config.model.encoder_type == "dense" else None,
        n_baseline=n_baseline,
        var_config=var_config,
    )


def run_small_hyperparameter_search(
    config: ApplicationConfig,
    input_dim: int,
    seq_len: int,
    n_baseline: int,
    var_config: VariableConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[dict[str, float], pd.DataFrame]:
    full_space = list(
        itertools.product(
            config.tuning.learning_rates,
            config.tuning.weight_decays,
            config.tuning.betas,
            config.tuning.hidden_dims,
            config.tuning.latent_dims,
        )
    )
    rng = np.random.default_rng(config.split.seed)
    if config.tuning.random_samples <= 0:
        raise ValueError("tuning.random_samples must be positive.")

    if len(full_space) <= config.tuning.random_samples:
        search_space = full_space
    else:
        chosen_idx = rng.choice(len(full_space), size=config.tuning.random_samples, replace=False)
        search_space = [full_space[int(i)] for i in np.sort(chosen_idx)]

    print(
        "Running hyperparameter search over "
        f"{len(search_space)} randomly selected combinations "
        f"(from {len(full_space)} possible)..."
    )

    best_val_loss = float("inf")
    best_params: dict[str, float] | None = None
    rows = []

    for learning_rate, weight_decay, beta, hidden_dim, latent_dim in search_space:
        set_seed(config.split.seed)
        model = build_model(
            input_dim=input_dim,
            seq_len=seq_len,
            n_baseline=n_baseline,
            var_config=var_config,
            config=config,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        trainer = VAETrainer(
            model,
            learning_rate=learning_rate,
            beta=beta,
            device=config.training.device,
            var_config=var_config,
            weight_decay=weight_decay,
        )
        history = trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
            verbose=False,
            use_em_imputation=config.training.use_em_imputation,
            em_iterations=config.training.em_iterations,
            patience=config.training.patience,
        )
        candidate_val_loss = float(min(history["val_loss"]))
        rows.append(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "beta": beta,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "best_val_loss": candidate_val_loss,
            }
        )
        print(
            "  "
            f"lr={learning_rate:.0e}, wd={weight_decay:.0e}, beta={beta:.2f}, "
            f"hidden={hidden_dim}, latent={latent_dim} -> best val loss = {candidate_val_loss:.4f}"
        )

        if candidate_val_loss < best_val_loss:
            best_val_loss = candidate_val_loss
            best_params = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "beta": beta,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
            }

    if best_params is None:
        raise RuntimeError("Hyperparameter search did not produce any candidate results.")

    print(
        "Best hyperparameters: "
        f"lr={best_params['learning_rate']:.0e}, "
        f"wd={best_params['weight_decay']:.0e}, "
        f"beta={best_params['beta']:.2f}, "
        f"hidden={best_params['hidden_dim']}, "
        f"latent={best_params['latent_dim']} "
        f"(best val loss = {best_val_loss:.4f})"
    )
    return best_params, pd.DataFrame(rows)


def deterministic_landmark_profile(
    model: LongitudinalVAE,
    x_observed: torch.Tensor,
    mask_observed: torch.Tensor,
    total_seq_len: int,
    baseline: Optional[torch.Tensor],
) -> torch.Tensor:
    return model.predict_from_landmark(
        x_observed,
        mask_observed,
        total_seq_len=total_seq_len,
        baseline=baseline,
    ).cpu()


def compute_landmark_index(config: ApplicationConfig, seq_len: int) -> int:
    if config.landmark.kind == "midpoint":
        return seq_len // 2
    raise ValueError(f"Unsupported landmark kind: {config.landmark.kind}")


def evaluate_landmark_predictions(
    model: LongitudinalVAE,
    dataset: LongitudinalDataset,
    subject_indices: np.ndarray,
    original_mask: np.ndarray,
    seq_len: int,
    landmark_t: int,
    feature_cols: list[str],
    outcome_cols: list[str],
    patient_keys: pd.DataFrame,
    var_config: VariableConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_rows = []

    for subject_idx in subject_indices:
        x, m, _, baseline = dataset[int(subject_idx)]
        x_obs = x[:landmark_t].unsqueeze(0).to(device)
        m_obs = m[:landmark_t].unsqueeze(0).to(device)
        baseline_arg = baseline.unsqueeze(0).to(device) if baseline.numel() > 0 else None

        predicted = model.predict_from_landmark(
            x_obs,
            m_obs,
            total_seq_len=seq_len,
            baseline=baseline_arg,
        ).cpu()

        actual_denorm = dataset.inverse_transform(x.unsqueeze(0)).cpu().numpy()[0]
        pred_denorm = dataset.inverse_transform(predicted).cpu().numpy()[0]
        subject_mask = original_mask[int(subject_idx)]
        subject_label = str(patient_keys.iloc[int(subject_idx)]["subject_label"])
        subject_key = patient_keys.iloc[int(subject_idx)]["subject_key"]

        for feature_idx, feature_name in enumerate(feature_cols):
            for time_idx in range(landmark_t, seq_len):
                rows.append(
                    {
                        "subject_index": int(subject_idx),
                        "subject_key": subject_key,
                        "subject_label": subject_label,
                        "time_index": int(time_idx),
                        "variable": feature_name,
                        "actual": float(actual_denorm[time_idx, feature_idx]),
                        "predicted": float(pred_denorm[time_idx, feature_idx]),
                        "observed": int(subject_mask[time_idx, feature_idx]),
                    }
                )

    predictions_df = pd.DataFrame(rows)

    for variable in outcome_cols:
        feature_idx = feature_cols.index(variable)
        valid = original_mask[subject_indices, landmark_t:, feature_idx].astype(bool).ravel()
        actual = predictions_df.loc[predictions_df["variable"] == variable, "actual"].to_numpy()
        predicted = predictions_df.loc[predictions_df["variable"] == variable, "predicted"].to_numpy()
        actual = actual[valid]
        predicted = predicted[valid]
        if actual.size == 0:
            continue

        spec = next(v for v in var_config.variables if v.name == variable)
        row = {"variable": variable, "observed_points": int(actual.size), "var_type": spec.var_type}
        if spec.var_type == "binary":
            p_clip = np.clip(predicted, 1e-7, 1 - 1e-7)
            row["brier"] = float(np.mean((actual - predicted) ** 2))
            row["log_loss"] = float(log_loss(actual, p_clip))
            row["auc"] = float(roc_auc_score(actual, predicted)) if np.unique(actual).size > 1 else float("nan")
        else:
            row["mae"] = float(mean_absolute_error(actual, predicted))
            row["rmse"] = float(np.sqrt(mean_squared_error(actual, predicted)))
            row["r2"] = float(r2_score(actual, predicted))
        metric_rows.append(row)

    return predictions_df, pd.DataFrame(metric_rows)


def resolve_plot_subjects(
    patient_keys: pd.DataFrame,
    plot_ids: Optional[list[str]],
    test_subject_indices: np.ndarray,
    plot_count: int,
    seed: int,
) -> list[tuple[str, int]]:
    if plot_ids:
        chosen = []
        missing = []
        for requested_id in plot_ids:
            subject_idx = resolve_subject_index(patient_keys, requested_id)
            if subject_idx is None:
                missing.append(requested_id)
            else:
                chosen.append((requested_id, subject_idx))
        if missing:
            print(f"Warning: requested plot ids not found: {missing}")
        if not chosen:
            raise ValueError(f"None of the requested plot ids were found: {plot_ids}")
        return chosen

    if len(test_subject_indices) == 0:
        return []
    rng = np.random.default_rng(seed)
    sampled = np.sort(
        rng.choice(test_subject_indices, size=min(plot_count, len(test_subject_indices)), replace=False)
    )
    return [
        (str(patient_keys.iloc[int(subject_idx)]["subject_label"]), int(subject_idx))
        for subject_idx in sampled
    ]


def resolve_subject_index(patient_keys: pd.DataFrame, requested_id: str) -> Optional[int]:
    requested_str = str(requested_id)
    matches = patient_keys.index[
        (patient_keys["subject_label"].astype(str) == requested_str)
        | (patient_keys["subject_key"].astype(str) == requested_str)
    ]
    if len(matches) == 0:
        return None
    return int(matches[0])


def save_training_curve(history: dict[str, list[float]], output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.get("train_loss", []), label="Train")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=150)
    plt.close(fig)


def save_profile_plot(
    model: LongitudinalVAE,
    dataset: LongitudinalDataset,
    patient_keys: pd.DataFrame,
    mask: np.ndarray,
    feature_cols: list[str],
    outcome_cols: list[str],
    landmark_t: int,
    plot_subjects: list[tuple[str, int]],
    split_lookup: dict[int, str],
    output_dir: Path,
    device: torch.device,
    var_config: VariableConfig,
) -> None:
    if not plot_subjects:
        return

    seq_len = dataset.data.shape[1]
    time_index = np.arange(seq_len)
    n_outcomes = len(outcome_cols)
    fig, axes = plt.subplots(
        len(plot_subjects), n_outcomes,
        figsize=(4 * n_outcomes, 3 * len(plot_subjects)),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    binary_outcomes = {v.name for v in var_config.variables if v.var_type == "binary"}

    for row, (requested_id, subject_idx) in enumerate(plot_subjects):
        x, m, _, baseline = dataset[int(subject_idx)]
        x_obs = x[:landmark_t].unsqueeze(0).to(device)
        m_obs = m[:landmark_t].unsqueeze(0).to(device)
        baseline_arg = baseline.unsqueeze(0).to(device) if baseline.numel() > 0 else None
        profile = deterministic_landmark_profile(model, x_obs, m_obs, seq_len, baseline_arg)
        actual = dataset.inverse_transform(x.unsqueeze(0)).cpu().numpy()[0]
        profile = dataset.inverse_transform(profile).cpu().numpy()[0]
        subject_split = split_lookup.get(int(subject_idx), "unknown")

        for col, outcome_name in enumerate(outcome_cols):
            ax = axes[row, col]
            feature_idx = feature_cols.index(outcome_name)
            observed = mask[int(subject_idx), :, feature_idx].astype(bool)

            ax.scatter(
                time_index[observed],
                actual[observed, feature_idx],
                color="black",
                s=12,
                zorder=3,
                label="Observed",
            )
            ax.plot(
                time_index[:landmark_t],
                profile[:landmark_t, feature_idx],
                color="tab:blue",
                linewidth=1.2,
                label="Fitted from pre-landmark",
            )
            ax.plot(
                time_index[landmark_t:],
                profile[landmark_t:, feature_idx],
                color="tab:red",
                linewidth=1.5,
                label="Forecast after landmark",
            )
            ax.scatter(
                time_index[~observed],
                actual[~observed, feature_idx],
                color="tab:red",
                marker="x",
                s=20,
                label="Missing",
            )
            ax.axvspan(landmark_t, seq_len - 1, alpha=0.08, color="tab:red")
            ax.axvline(landmark_t - 0.5, color="tab:green", linestyle="--", linewidth=1.2, label="Landmark")
            if outcome_name in binary_outcomes:
                ax.set_ylim(0.0, 1.0)
            if row == 0:
                ax.set_title(outcome_name)
            if col == 0:
                subject_label = patient_keys.iloc[int(subject_idx)]["subject_label"]
                ax.set_ylabel(f"{subject_label}\n({subject_split})")
            if row == len(plot_subjects) - 1:
                ax.set_xlabel("Time step")

    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Landmark Profiles for {[sid for sid, _ in plot_subjects]} (t < {landmark_t} observed)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "landmark_prediction_examples.png", dpi=150)
    plt.close(fig)


def run_application(
    config: ApplicationConfig,
    data_path_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    plot_ids_override: Optional[list[str]] = None,
) -> None:
    set_seed(config.split.seed)

    df = load_dataframe(config, data_path_override)
    df = apply_transforms(df, config)
    validate_required_columns(df, config)
    df = df.sort_values(config.data.sort_by).reset_index(drop=True)

    variable_names = [v.name for v in config.variables.variables]
    if variable_names != config.data.feature_cols:
        raise ValueError(
            "variables.specs must match data.outcome_cols + data.time_varying_cols in order. "
            f"Expected {config.data.feature_cols}, got {variable_names}."
        )

    original_subject_count = int(df[config.data.subject_col].nunique())
    df, counts, seq_len = filter_equal_length_sequences(
        df,
        subject_col=config.data.subject_col,
        strict=config.data.strict_seq_len,
    )

    data, mask, baseline, patient_keys = build_subject_arrays(df, config)
    dataset = LongitudinalDataset(
        data,
        mask=mask,
        var_config=config.variables,
        baseline_covariates=baseline if baseline.shape[1] > 0 else None,
        normalize=True,
    )

    splits = make_splits(
        n_subjects=len(patient_keys),
        train_fraction=config.split.train_fraction,
        val_fraction=config.split.val_fraction,
        seed=config.split.seed,
    )
    split_lookup = {
        int(idx): split_name
        for split_name, idx_values in splits.items()
        for idx in idx_values
    }

    train_loader = DataLoader(Subset(dataset, splits["train"].tolist()), batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, splits["val"].tolist()), batch_size=config.training.batch_size, shuffle=False)

    selected_hyperparameters = {
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "beta": config.training.beta,
        "hidden_dim": config.model.hidden_dim,
        "latent_dim": config.model.latent_dim,
    }
    search_results_df = None
    if config.tuning.enabled:
        selected_hyperparameters, search_results_df = run_small_hyperparameter_search(
            config=config,
            input_dim=len(config.data.feature_cols),
            seq_len=seq_len,
            n_baseline=baseline.shape[1],
            var_config=config.variables,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    set_seed(config.split.seed)
    model = build_model(
        input_dim=len(config.data.feature_cols),
        seq_len=seq_len,
        n_baseline=baseline.shape[1],
        var_config=config.variables,
        config=config,
        hidden_dim=int(selected_hyperparameters["hidden_dim"]),
        latent_dim=int(selected_hyperparameters["latent_dim"]),
    )
    trainer = VAETrainer(
        model,
        learning_rate=selected_hyperparameters["learning_rate"],
        beta=selected_hyperparameters["beta"],
        device=config.training.device,
        var_config=config.variables,
        weight_decay=selected_hyperparameters["weight_decay"],
    )
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        verbose=True,
        use_em_imputation=config.training.use_em_imputation,
        em_iterations=config.training.em_iterations,
        patience=config.training.patience,
    )

    landmark_t = compute_landmark_index(config, seq_len)
    predictions_df, metrics_df = evaluate_landmark_predictions(
        model=model,
        dataset=dataset,
        subject_indices=splits["test"],
        original_mask=mask,
        seq_len=seq_len,
        landmark_t=landmark_t,
        feature_cols=config.data.feature_cols,
        outcome_cols=config.data.outcome_cols,
        patient_keys=patient_keys,
        var_config=config.variables,
        device=trainer.device,
    )

    output_dir = config.resolve_output_dir(output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_curve(history, output_dir, title=f"{config.name} Training")
    if search_results_df is not None:
        search_results_df.to_csv(output_dir / "hyperparameter_search_results.csv", index=False)

    plot_subjects = resolve_plot_subjects(
        patient_keys=patient_keys,
        plot_ids=plot_ids_override if plot_ids_override is not None else config.plot.ids,
        test_subject_indices=splits["test"],
        plot_count=config.plot.count,
        seed=config.split.seed,
    )
    save_profile_plot(
        model=model,
        dataset=dataset,
        patient_keys=patient_keys,
        mask=mask,
        feature_cols=config.data.feature_cols,
        outcome_cols=config.data.outcome_cols,
        landmark_t=landmark_t,
        plot_subjects=plot_subjects,
        split_lookup=split_lookup,
        output_dir=output_dir,
        device=trainer.device,
        var_config=config.variables,
    )

    split_rows = []
    for split_name, idx_values in splits.items():
        frame = patient_keys.iloc[idx_values].copy()
        frame["split"] = split_name
        frame["seq_len"] = seq_len
        split_rows.append(frame)
    split_assignments = pd.concat(split_rows, ignore_index=True)

    history_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history.get("val_loss", [np.nan] * len(history["train_loss"])),
        }
    )

    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": len(config.data.feature_cols),
                "hidden_dim": int(selected_hyperparameters["hidden_dim"]),
                "latent_dim": int(selected_hyperparameters["latent_dim"]),
                "encoder_type": config.model.encoder_type,
                "seq_len": seq_len if config.model.encoder_type == "dense" else None,
                "n_baseline": int(baseline.shape[1]),
                "feature_cols": config.data.feature_cols,
                "selected_hyperparameters": selected_hyperparameters,
            },
            "landmark_index": landmark_t,
        },
        model_path,
    )

    metadata = {
        "config": config.to_metadata(),
        "data_path": str(config.resolve_data_path(data_path_override)),
        "output_dir": str(output_dir),
        "n_subjects_total": original_subject_count,
        "n_subjects_modelled": int(len(patient_keys)),
        "n_subjects_excluded_for_seq_len": int(original_subject_count - len(patient_keys)),
        "sequence_length": int(seq_len),
        "landmark_index": int(landmark_t),
        "split_sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "selected_hyperparameters": selected_hyperparameters,
        "plot_subjects": [subject_id for subject_id, _ in plot_subjects],
    }

    history_df.to_csv(output_dir / "training_history.csv", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    predictions_df.to_csv(output_dir / "future_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Application '{config.name}' finished.")
    print(f"Subjects modelled: {len(patient_keys)}")
    print(f"Sequence length: {seq_len}")
    print(f"Landmark index: {landmark_t}")
    print(f"Outputs written to: {output_dir}")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
