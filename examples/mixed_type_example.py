"""
Mixed-type longitudinal data example.

Demonstrates:
1. Defining mixed variable types (continuous, binary, bounded)
2. Generating synthetic mixed-type data with baseline covariates
3. Training with missing data
4. Landmark prediction (predicting future from partial observations)
5. Plotting actual vs predicted outcomes for individual subjects
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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
        model, learning_rate=1e-3, beta=1.0, var_config=var_config,
    )

    history = trainer.fit(
        train_loader, val_loader=val_loader, epochs=50, verbose=True,
        use_em_imputation=True, em_iterations=2,
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
    plt.show()
    print("Saved landmark_prediction.png")

    # ---- 8. Generate new samples ----
    print("\n--- Generating new samples ---")
    new_baseline = torch.randn(10, n_baseline)
    samples = model.sample(num_samples=10, seq_len=seq_len, baseline=new_baseline)
    print(f"Generated samples shape: {samples.shape}")

    # Inverse transform to original scale
    samples_original = dataset.inverse_transform(samples)
    print(f"Biomarker range (original scale): "
          f"[{samples_original[:, :, 0].min():.2f}, {samples_original[:, :, 0].max():.2f}]")
    print(f"Blood pressure range (original scale): "
          f"[{samples_original[:, :, 1].min():.1f}, {samples_original[:, :, 1].max():.1f}]")

    print("\nDone!")


if __name__ == '__main__':
    main()
