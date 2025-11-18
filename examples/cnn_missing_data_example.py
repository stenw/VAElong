"""
Example: CNN-based VAE with Missing Data Handling

Demonstrates:
1. Creating synthetic longitudinal data with missing values
2. Training a CNN-based VAE with missing data
3. Using EM-like imputation during training
4. Imputing missing values after training
5. Evaluating reconstruction quality
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vaelong import CNNLongitudinalVAE, VAETrainer, LongitudinalDataset
from vaelong.data import generate_synthetic_longitudinal_data, create_missing_mask


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    print("Generating synthetic longitudinal data...")
    n_samples = 1000
    seq_len = 64  # Using power of 2 for CNN
    n_features = 5

    data = generate_synthetic_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        n_features=n_features,
        noise_level=0.1,
        seed=42
    )

    # Create missing data mask (20% missing with random pattern)
    print("Creating missing data mask...")
    mask = create_missing_mask(
        data.shape,
        missing_rate=0.2,
        pattern='random',
        seed=42
    )

    # Keep original complete data for comparison
    original_data = data.copy()

    # Apply mask to data (zero out missing values)
    data_with_missing = data * mask

    print(f"Data shape: {data.shape}")
    print(f"Missing data rate: {1 - mask.mean():.2%}")

    # Split into train and validation
    train_size = int(0.8 * n_samples)
    train_data = data_with_missing[:train_size]
    train_mask = mask[:train_size]
    val_data = data_with_missing[train_size:]
    val_mask = mask[train_size:]

    # Create datasets and dataloaders
    print("\nCreating datasets...")
    train_dataset = LongitudinalDataset(train_data, mask=train_mask, normalize=True)
    val_dataset = LongitudinalDataset(val_data, mask=val_mask, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create CNN-based VAE model
    print("\nCreating CNN-based VAE model...")
    model = CNNLongitudinalVAE(
        input_dim=n_features,
        seq_len=seq_len,
        latent_dim=16,
        hidden_channels=[32, 64, 128],
        kernel_size=3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = VAETrainer(
        model=model,
        learning_rate=1e-3,
        beta=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Train without EM imputation (baseline)
    print("\n" + "="*60)
    print("Training WITHOUT EM imputation (baseline)...")
    print("="*60)
    history_baseline = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        verbose=True,
        use_em_imputation=False
    )

    # Reset model for fair comparison
    model_em = CNNLongitudinalVAE(
        input_dim=n_features,
        seq_len=seq_len,
        latent_dim=16,
        hidden_channels=[32, 64, 128],
        kernel_size=3
    )

    trainer_em = VAETrainer(
        model=model_em,
        learning_rate=1e-3,
        beta=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Train with EM imputation
    print("\n" + "="*60)
    print("Training WITH EM imputation...")
    print("="*60)
    history_em = trainer_em.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        verbose=True,
        use_em_imputation=True,
        em_iterations=3
    )

    # Evaluate imputation quality on a few samples
    print("\n" + "="*60)
    print("Evaluating imputation quality...")
    print("="*60)

    model.eval()
    model_em.eval()

    # Get a few validation samples
    n_eval_samples = 5
    eval_data = val_data[:n_eval_samples]
    eval_mask = val_mask[:n_eval_samples]
    eval_original = original_data[train_size:train_size+n_eval_samples]

    # Convert to tensors
    eval_data_tensor = torch.FloatTensor(eval_data).to(trainer.device)
    eval_mask_tensor = torch.FloatTensor(eval_mask).to(trainer.device)

    # Normalize using training statistics
    eval_data_norm = (eval_data_tensor - train_dataset.mean.to(trainer.device)) / train_dataset.std.to(trainer.device)

    # Impute missing values using both models
    with torch.no_grad():
        # Baseline model
        imputed_baseline = model.impute_missing(eval_data_norm, eval_mask_tensor, num_iterations=5)
        imputed_baseline = imputed_baseline.cpu() * train_dataset.std + train_dataset.mean

        # EM-trained model
        imputed_em = model_em.impute_missing(eval_data_norm, eval_mask_tensor, num_iterations=5)
        imputed_em = imputed_em.cpu() * train_dataset.std + train_dataset.mean

    # Calculate reconstruction error on missing values only
    missing_mask_inv = 1 - eval_mask_tensor.cpu()

    if missing_mask_inv.sum() > 0:
        error_baseline = ((imputed_baseline - torch.FloatTensor(eval_original)) ** 2 * missing_mask_inv).sum() / missing_mask_inv.sum()
        error_em = ((imputed_em - torch.FloatTensor(eval_original)) ** 2 * missing_mask_inv).sum() / missing_mask_inv.sum()

        print(f"\nMSE on missing values:")
        print(f"  Baseline model: {error_baseline:.4f}")
        print(f"  EM-trained model: {error_em:.4f}")
        print(f"  Improvement: {(error_baseline - error_em) / error_baseline * 100:.2f}%")

    # Visualize results
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot training history
    axes[0, 0].plot(history_baseline['train_loss'], label='Baseline')
    axes[0, 0].plot(history_em['train_loss'], label='EM-trained')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history_baseline['val_loss'], label='Baseline')
    axes[0, 1].plot(history_em['val_loss'], label='EM-trained')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot example imputation for first sample, first feature
    sample_idx = 0
    feature_idx = 0

    time_steps = np.arange(seq_len)
    original_vals = eval_original[sample_idx, :, feature_idx]
    observed_vals = eval_data[sample_idx, :, feature_idx]
    mask_vals = eval_mask[sample_idx, :, feature_idx]
    imputed_baseline_vals = imputed_baseline[sample_idx, :, feature_idx].numpy()
    imputed_em_vals = imputed_em[sample_idx, :, feature_idx].numpy()

    # Plot original data
    axes[1, 0].plot(time_steps, original_vals, 'k-', label='Original (complete)', linewidth=2)
    axes[1, 0].scatter(time_steps[mask_vals == 1], observed_vals[mask_vals == 1],
                      c='green', s=30, label='Observed', zorder=3)
    axes[1, 0].scatter(time_steps[mask_vals == 0], original_vals[mask_vals == 0],
                      c='red', marker='x', s=50, label='Missing (true)', zorder=3)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title(f'Original Data (Sample {sample_idx}, Feature {feature_idx})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot baseline imputation
    axes[1, 1].plot(time_steps, original_vals, 'k--', alpha=0.5, label='Original')
    axes[1, 1].scatter(time_steps[mask_vals == 1], observed_vals[mask_vals == 1],
                      c='green', s=30, label='Observed')
    axes[1, 1].plot(time_steps, imputed_baseline_vals, 'b-', label='Imputed (baseline)', linewidth=2)
    axes[1, 1].scatter(time_steps[mask_vals == 0], imputed_baseline_vals[mask_vals == 0],
                      c='blue', s=50, label='Imputed values', zorder=3)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Baseline Model Imputation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot EM imputation
    axes[2, 0].plot(time_steps, original_vals, 'k--', alpha=0.5, label='Original')
    axes[2, 0].scatter(time_steps[mask_vals == 1], observed_vals[mask_vals == 1],
                      c='green', s=30, label='Observed')
    axes[2, 0].plot(time_steps, imputed_em_vals, 'r-', label='Imputed (EM-trained)', linewidth=2)
    axes[2, 0].scatter(time_steps[mask_vals == 0], imputed_em_vals[mask_vals == 0],
                      c='red', s=50, label='Imputed values', zorder=3)
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_title('EM-trained Model Imputation')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Plot comparison of imputation errors
    error_baseline_per_point = np.abs(imputed_baseline_vals - original_vals)
    error_em_per_point = np.abs(imputed_em_vals - original_vals)

    axes[2, 1].plot(time_steps, error_baseline_per_point, 'b-', label='Baseline error', linewidth=2)
    axes[2, 1].plot(time_steps, error_em_per_point, 'r-', label='EM-trained error', linewidth=2)
    axes[2, 1].scatter(time_steps[mask_vals == 0], error_baseline_per_point[mask_vals == 0],
                      c='blue', s=50, zorder=3)
    axes[2, 1].scatter(time_steps[mask_vals == 0], error_em_per_point[mask_vals == 0],
                      c='red', s=50, zorder=3)
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Absolute Error')
    axes[2, 1].set_title('Imputation Error Comparison')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('cnn_missing_data_results.png', dpi=150)
    print("\nResults saved to 'cnn_missing_data_results.png'")

    # Generate new samples
    print("\nGenerating new samples from learned distribution...")
    samples = model_em.sample(num_samples=10, device=trainer_em.device)
    samples = samples.cpu() * train_dataset.std + train_dataset.mean

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics - Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
