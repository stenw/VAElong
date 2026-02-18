"""
Mixed-type longitudinal data example.

Demonstrates:
1. Defining mixed variable types (continuous, binary, bounded)
2. Generating synthetic mixed-type data with baseline covariates
3. Training with missing data
4. Landmark prediction (predicting future from partial observations)
5. Plotting actual vs predicted outcomes for individual subjects
6. Benchmark comparison: VAE vs GLMM vs Seq2Seq LSTM
"""

import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from torch.utils.data import DataLoader

from vaelong import (
    VariableConfig, VariableSpec,
    LongitudinalVAE, CNNLongitudinalVAE,
    VAETrainer, LongitudinalDataset,
    generate_mixed_longitudinal_data, create_missing_mask,
)


def main():
    # ---- 1. Define variable types ----
    var_config = VariableConfig(variables=[
        VariableSpec(name='biomarker', var_type='continuous'),
        VariableSpec(name='blood_pressure', var_type='bounded', lower=60.0, upper=200.0),
        VariableSpec(name='symptom_present', var_type='binary'),
        VariableSpec(name='score', var_type='continuous'),
    ])

    print(f"Variable config: {var_config.n_features} features")
    print(f"  Continuous indices: {var_config.continuous_indices}")
    print(f"  Binary indices:     {var_config.binary_indices}")
    print(f"  Bounded indices:    {var_config.bounded_indices}")

    # ---- 2. Generate mixed-type data with baselines ----
    n_samples = 500
    seq_len = 50
    n_baseline = 3

    data, baseline = generate_mixed_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        var_config=var_config,
        n_baseline_features=n_baseline,
        noise_level=0.2,
        random_intercept_sd=2.0,
        seed=42,
    )

    print(f"\nData shape: {data.shape}")
    print(f"Baseline shape: {baseline.shape}")
    print(f"Binary variable unique values: {np.unique(data[:, :, 2])}")
    print(f"Bounded variable range: [{data[:, :, 1].min():.1f}, {data[:, :, 1].max():.1f}]")

    # ---- 3. Add missing data ----
    mask = create_missing_mask(data.shape, missing_rate=0.15, pattern='random', seed=42)
    data_masked = data * mask
    missing_pct = (1 - mask.mean()) * 100
    print(f"\nMissing data: {missing_pct:.1f}%")

    # ---- 4. Create dataset and dataloader ----
    dataset = LongitudinalDataset(
        data_masked, mask=mask, var_config=var_config,
        baseline_covariates=baseline, normalize=True,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ---- 5. Create and train LSTM model ----
    print("\n--- Training LSTM model ---")
    model = LongitudinalVAE(
        input_dim=var_config.n_features,
        hidden_dim=64,
        latent_dim=16,
        n_baseline=n_baseline,
        var_config=var_config,
    )

    trainer = VAETrainer(
        model, learning_rate=1e-3, beta=0.5, var_config=var_config,
    )

    history = trainer.fit(
        train_loader, val_loader=val_loader, epochs=200, verbose=True,
        use_em_imputation=True, em_iterations=2, patience=20,
    )

    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")

    # ---- 6. Landmark prediction ----
    print("\n--- Landmark prediction ---")
    landmark_t = seq_len // 2  # observe first half, predict all

    # Pick 3 random individuals from the validation set
    rng = np.random.default_rng(123)
    val_indices = val_dataset.indices
    chosen = sorted(rng.choice(len(val_indices), size=3, replace=False))
    sample_idx = [val_indices[c] for c in chosen]

    x_full = torch.stack([dataset[i][0] for i in sample_idx])
    mask_full = torch.stack([dataset[i][1] for i in sample_idx])
    baseline_sample = torch.stack([dataset[i][3] for i in sample_idx])

    # Create observed portion (first half)
    x_observed = x_full[:, :landmark_t, :]
    mask_observed = mask_full[:, :landmark_t, :]

    # Predict full trajectory
    predicted = model.predict_from_landmark(
        x_observed, mask_observed, total_seq_len=seq_len,
        baseline=baseline_sample,
    )

    print(f"Observed shape:  {x_observed.shape}")
    print(f"Predicted shape: {predicted.shape}")

    # ---- 7. Plot actual vs predicted for 3 individuals ----
    print("\n--- Plotting landmark prediction ---")

    # Inverse-transform to original scale
    actual_orig = dataset.inverse_transform(x_full).detach().numpy()
    pred_orig = dataset.inverse_transform(predicted).detach().numpy()

    var_names = [v.name for v in var_config.variables]
    n_vars = var_config.n_features
    n_individuals = 3
    time_axis = np.arange(seq_len)

    fig, axes = plt.subplots(
        n_individuals, n_vars,
        figsize=(4 * n_vars, 3.5 * n_individuals),
        sharex=True,
    )

    for row in range(n_individuals):
        for col in range(n_vars):
            ax = axes[row, col]

            actual_vals = actual_orig[row, :, col]
            pred_vals = pred_orig[row, :, col]

            # Actual trajectory
            ax.plot(time_axis, actual_vals, 'k-', linewidth=1.2, label='Actual')

            # Predicted: observed part (blue, faded) and future part (red)
            ax.plot(time_axis[:landmark_t], pred_vals[:landmark_t],
                    'b-', linewidth=1, alpha=0.5)
            ax.plot(time_axis[landmark_t:], pred_vals[landmark_t:],
                    'r-', linewidth=1.5, label='Predicted')

            # Shade the future region
            ax.axvspan(landmark_t, seq_len - 1, alpha=0.08, color='red')
            ax.axvline(landmark_t, color='grey', linestyle='--', linewidth=0.8)

            if row == 0:
                ax.set_title(var_names[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f'Individual {row + 1}', fontsize=10)
            if row == n_individuals - 1:
                ax.set_xlabel('Time')

    axes[0, -1].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Landmark Prediction (observed up to t = {landmark_t})',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('landmark_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved landmark_prediction.png")

    # ---- 8. Prediction evaluation on validation set ----
    print("\n--- Prediction evaluation (validation set) ---")

    # Evaluate landmark prediction accuracy across all validation subjects
    all_actual = []
    all_predicted = []

    for idx in val_dataset.indices:
        xi = dataset[idx][0].unsqueeze(0)        # (1, seq_len, n_features)
        mi = dataset[idx][1].unsqueeze(0)
        bi = dataset[idx][3].unsqueeze(0)

        xi_obs = xi[:, :landmark_t, :]
        mi_obs = mi[:, :landmark_t, :]

        pred_i = model.predict_from_landmark(
            xi_obs, mi_obs, total_seq_len=seq_len, baseline=bi,
        )

        all_actual.append(dataset.inverse_transform(xi).detach())
        all_predicted.append(dataset.inverse_transform(pred_i).detach())

    all_actual = torch.cat(all_actual, dim=0).numpy()       # (n_val, seq_len, n_features)
    all_predicted = torch.cat(all_predicted, dim=0).numpy()

    # Compute per-variable metrics on the *future* (unobserved) portion only
    future_actual = all_actual[:, landmark_t:, :]
    future_pred = all_predicted[:, landmark_t:, :]

    print(f"{'Variable':<20s}  {'MAE':>8s}  {'RMSE':>8s}  {'Corr':>8s}")
    print("-" * 50)
    for col, v in enumerate(var_config.variables):
        a = future_actual[:, :, col].ravel()
        p = future_pred[:, :, col].ravel()
        mae = np.mean(np.abs(a - p))
        rmse = np.sqrt(np.mean((a - p) ** 2))
        corr = np.corrcoef(a, p)[0, 1] if np.std(a) > 0 else float('nan')
        print(f"{v.name:<20s}  {mae:8.4f}  {rmse:8.4f}  {corr:8.4f}")

    # ---- 9. GLMM benchmark ----
    print("\n--- GLMM benchmark (linear mixed model per variable) ---")

    train_indices = list(train_dataset.indices)
    val_indices_list = list(val_dataset.indices)
    n_val = len(val_indices_list)

    lmm_predictions = np.zeros((n_val, seq_len, var_config.n_features))

    for col, var_spec in enumerate(var_config.variables):
        print(f"  Fitting LMM for {var_spec.name}...", end=" ", flush=True)

        # Build long-format training data (observed portion only)
        rows = []
        for i in train_indices:
            for t in range(landmark_t):
                if mask[i, t, col] == 1.0:
                    row = {'subject': int(i), 'time': t, 'y': float(data[i, t, col])}
                    for b in range(n_baseline):
                        row[f'bl_{b}'] = float(baseline[i, b])
                    rows.append(row)
        df_train = pd.DataFrame(rows)

        # Fit LMM: y ~ time + baselines, random intercept + slope per subject
        fixed_formula = 'y ~ time + ' + ' + '.join(f'bl_{b}' for b in range(n_baseline))
        re_formula = '1 + time'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            md = smf.mixedlm(fixed_formula, df_train, groups=df_train['subject'],
                             re_formula=re_formula)
            mdf = md.fit(reml=True, method='lbfgs')

        beta_hat = np.array(mdf.fe_params)
        D = np.array(mdf.cov_re)
        sigma2_e = mdf.scale
        print(f"done (var_e={sigma2_e:.4f})")

        # Predict for each validation subject via BLUP
        for j, subj_idx in enumerate(val_indices_list):
            obs_times, obs_y = [], []
            for t in range(landmark_t):
                if mask[subj_idx, t, col] == 1.0:
                    obs_times.append(t)
                    obs_y.append(data[subj_idx, t, col])

            bl_vals = [baseline[subj_idx, b] for b in range(n_baseline)]

            if len(obs_times) == 0:
                # No observed data — use population average
                for t in range(seq_len):
                    x_t = np.array([1.0, t] + bl_vals)
                    lmm_predictions[j, t, col] = x_t @ beta_hat
                continue

            obs_times = np.array(obs_times, dtype=float)
            obs_y = np.array(obs_y)
            n_obs = len(obs_times)

            X_obs = np.column_stack([np.ones(n_obs), obs_times,
                                     np.tile(bl_vals, (n_obs, 1))])
            Z_obs = np.column_stack([np.ones(n_obs), obs_times])

            r = obs_y - X_obs @ beta_hat
            V = Z_obs @ D @ Z_obs.T + sigma2_e * np.eye(n_obs) + 1e-6 * np.eye(n_obs)
            u_hat = D @ Z_obs.T @ np.linalg.solve(V, r)

            for t in range(seq_len):
                x_t = np.array([1.0, t] + bl_vals)
                z_t = np.array([1.0, t])
                lmm_predictions[j, t, col] = x_t @ beta_hat + z_t @ u_hat

        # Post-processing per variable type
        if var_spec.var_type == 'binary':
            lmm_predictions[:, :, col] = np.round(
                np.clip(lmm_predictions[:, :, col], 0, 1))
        elif var_spec.var_type == 'bounded':
            lmm_predictions[:, :, col] = np.clip(
                lmm_predictions[:, :, col], var_spec.lower, var_spec.upper)

    # ---- 10. Seq2Seq LSTM benchmark ----
    print("\n--- Seq2Seq LSTM benchmark ---")

    class Seq2SeqLSTM(nn.Module):
        """Encoder–decoder LSTM for direct sequence prediction."""

        def __init__(self, input_dim, hidden_dim, n_baseline, var_config):
            super().__init__()
            self.var_config = var_config
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.baseline_proj = nn.Linear(hidden_dim + n_baseline, hidden_dim)
            self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, input_dim)

        def _apply_activations(self, output):
            result = output.clone()
            for idx in self.var_config.binary_indices:
                result[:, :, idx] = torch.sigmoid(result[:, :, idx])
            for idx in self.var_config.bounded_indices:
                result[:, :, idx] = torch.sigmoid(result[:, :, idx])
            return result

        def forward(self, x_obs, mask_obs, bl, future_target=None, future_len=25):
            batch_size = x_obs.size(0)
            _, (h_enc, c_enc) = self.encoder(x_obs * mask_obs)

            if bl is not None:
                h_combined = torch.cat([h_enc.squeeze(0), bl], dim=-1)
                h_dec = self.baseline_proj(h_combined).unsqueeze(0)
            else:
                h_dec = h_enc
            c_dec = c_enc

            predictions = []
            dec_input = x_obs[:, -1:, :]

            for t in range(future_len):
                dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
                pred_t = self.fc_out(dec_out)
                pred_t = self._apply_activations(pred_t)
                predictions.append(pred_t)
                if future_target is not None and self.training:
                    dec_input = future_target[:, t:t + 1, :]
                else:
                    dec_input = pred_t.detach()

            return torch.cat(predictions, dim=1)

    # Prepare training tensors
    future_len = seq_len - landmark_t
    train_x_obs, train_mask_obs, train_future = [], [], []
    train_future_mask, train_bl = [], []

    for idx in train_indices:
        xi, mi, _, bi = dataset[idx]
        train_x_obs.append(xi[:landmark_t])
        train_mask_obs.append(mi[:landmark_t])
        train_future.append(xi[landmark_t:])
        train_future_mask.append(mi[landmark_t:])
        train_bl.append(bi)

    train_x_obs = torch.stack(train_x_obs)
    train_mask_obs = torch.stack(train_mask_obs)
    train_future = torch.stack(train_future)
    train_future_mask = torch.stack(train_future_mask)
    train_bl = torch.stack(train_bl)

    # Train Seq2Seq
    seq2seq = Seq2SeqLSTM(var_config.n_features, 64, n_baseline, var_config)
    opt_s2s = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)

    for epoch in range(200):
        seq2seq.train()
        perm = torch.randperm(len(train_x_obs))
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(train_x_obs), 32):
            batch_idx = perm[start:start + 32]
            bx = train_x_obs[batch_idx]
            bm = train_mask_obs[batch_idx]
            bf = train_future[batch_idx]
            bfm = train_future_mask[batch_idx]
            bb = train_bl[batch_idx]

            pred = seq2seq(bx, bm, bb, future_target=bf, future_len=future_len)

            loss = torch.tensor(0.0)
            for cidx in var_config.continuous_indices:
                diff2 = (pred[:, :, cidx] - bf[:, :, cidx]) ** 2
                m = bfm[:, :, cidx]
                n_obs_m = m.sum()
                if n_obs_m > 0:
                    loss = loss + (diff2 * m).sum() / n_obs_m

            for cidx in var_config.binary_indices + var_config.bounded_indices:
                p_clamped = pred[:, :, cidx].clamp(1e-7, 1 - 1e-7)
                tgt = bf[:, :, cidx]
                bce = -(tgt * torch.log(p_clamped) + (1 - tgt) * torch.log(1 - p_clamped))
                m = bfm[:, :, cidx]
                n_obs_m = m.sum()
                if n_obs_m > 0:
                    loss = loss + (bce * m).sum() / n_obs_m

            opt_s2s.zero_grad()
            loss.backward()
            opt_s2s.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch + 1}/200] Loss: {epoch_loss / n_batches:.4f}")

    # Predict on validation set
    seq2seq.eval()
    s2s_all_predicted = []

    with torch.no_grad():
        for idx in val_indices_list:
            xi, mi, _, bi = dataset[idx]
            xi_obs = xi[:landmark_t].unsqueeze(0)
            mi_obs = mi[:landmark_t].unsqueeze(0)

            pred_future = seq2seq(xi_obs, mi_obs, bi.unsqueeze(0),
                                  future_target=None, future_len=future_len)

            full_pred = torch.cat([xi[:landmark_t], pred_future.squeeze(0)], dim=0)
            s2s_all_predicted.append(
                dataset.inverse_transform(full_pred.unsqueeze(0)).detach())

    s2s_all_predicted = torch.cat(s2s_all_predicted, dim=0).numpy()

    # ---- 11. Model comparison ----
    print("\n--- Model Comparison (future portion, t=25..49) ---")

    def compute_metrics(actual, predicted):
        metrics = []
        for col in range(actual.shape[-1]):
            a = actual[:, :, col].ravel()
            p = predicted[:, :, col].ravel()
            mae = np.mean(np.abs(a - p))
            rmse = np.sqrt(np.mean((a - p) ** 2))
            corr = np.corrcoef(a, p)[0, 1] if np.std(a) > 0 else float('nan')
            metrics.append((mae, rmse, corr))
        return metrics

    lmm_future = lmm_predictions[:, landmark_t:, :]
    s2s_future = s2s_all_predicted[:, landmark_t:, :]

    vae_m = compute_metrics(future_actual, future_pred)
    lmm_m = compute_metrics(future_actual, lmm_future)
    s2s_m = compute_metrics(future_actual, s2s_future)

    header = (f"{'Variable':<20s}  {'VAE':>8s} {'LMM':>8s} {'RNN':>8s}"
              f"  |  {'VAE':>8s} {'LMM':>8s} {'RNN':>8s}"
              f"  |  {'VAE':>8s} {'LMM':>8s} {'RNN':>8s}")
    metric_header = (f"{'':20s}  {'--- MAE ---':>26s}"
                     f"  |  {'--- RMSE ---':>26s}"
                     f"  |  {'--- Corr ---':>26s}")
    print(metric_header)
    print(header)
    print("-" * len(header))

    for col, v in enumerate(var_config.variables):
        vm, lm, sm = vae_m[col], lmm_m[col], s2s_m[col]
        print(f"{v.name:<20s}  {vm[0]:8.4f} {lm[0]:8.4f} {sm[0]:8.4f}"
              f"  |  {vm[1]:8.4f} {lm[1]:8.4f} {sm[1]:8.4f}"
              f"  |  {vm[2]:8.4f} {lm[2]:8.4f} {sm[2]:8.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
