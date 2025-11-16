"""
Example usage of Longitudinal VAE.

This script demonstrates how to:
1. Generate synthetic longitudinal data
2. Create and train a VAE model
3. Visualize results
"""

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from vaelong import LongitudinalVAE, VAETrainer, LongitudinalDataset
from vaelong.data import generate_synthetic_longitudinal_data


def plot_reconstruction(original, reconstructed, n_samples=3):
    """Plot original vs reconstructed sequences."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Plot each feature
        time = np.arange(original.shape[1])
        for j in range(original.shape[2]):
            ax.plot(time, original[i, :, j], 'o-', label=f'Original Feature {j+1}', alpha=0.6)
            ax.plot(time, reconstructed[i, :, j], 's--', label=f'Reconstructed Feature {j+1}', alpha=0.6)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {i+1}: Original vs Reconstructed')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latent_space(vae, dataset, device):
    """Plot latent space representation (for 2D latent space)."""
    vae.eval()
    
    all_z = []
    with torch.no_grad():
        for data, _ in DataLoader(dataset, batch_size=32):
            data = data.to(device)
            mu, _ = vae.encode(data)
            all_z.append(mu.cpu().numpy())
    
    all_z = np.concatenate(all_z, axis=0)
    
    if all_z.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(all_z[:, 0], all_z[:, 1], alpha=0.5)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # For higher dimensional latent space, plot first 2 dimensions
        plt.figure(figsize=(8, 6))
        plt.scatter(all_z[:, 0], all_z[:, 1], alpha=0.5)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation (First 2 Dimensions)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot total loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot reconstruction and KLD losses
    axes[1].plot(history['train_recon'], label='Train Recon Loss')
    axes[1].plot(history['train_kld'], label='Train KLD Loss')
    if history['val_recon']:
        axes[1].plot(history['val_recon'], label='Val Recon Loss')
        axes[1].plot(history['val_kld'], label='Val KLD Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction and KLD Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main example workflow."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    n_samples = 1000
    seq_len = 50
    n_features = 5
    latent_dim = 10
    hidden_dim = 64
    batch_size = 32
    epochs = 50
    
    print("=" * 60)
    print("Longitudinal VAE Example")
    print("=" * 60)
    
    # Generate synthetic data
    print(f"\n1. Generating synthetic longitudinal data...")
    print(f"   - Samples: {n_samples}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Features per timestep: {n_features}")
    
    data = generate_synthetic_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        n_features=n_features,
        noise_level=0.1,
        seed=42
    )
    
    # Create dataset
    dataset = LongitudinalDataset(data, normalize=True)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   - Training samples: {train_size}")
    print(f"   - Validation samples: {val_size}")
    
    # Create model
    print(f"\n2. Creating VAE model...")
    print(f"   - Input dimension: {n_features}")
    print(f"   - Hidden dimension: {hidden_dim}")
    print(f"   - Latent dimension: {latent_dim}")
    
    model = LongitudinalVAE(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=1,
        use_gru=False
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    
    trainer = VAETrainer(model, learning_rate=1e-3, beta=1.0, device=device)
    
    # Train model
    print(f"\n3. Training model for {epochs} epochs...")
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=True
    )
    
    # Evaluate reconstruction
    print("\n4. Evaluating model...")
    model.eval()
    
    # Get some test samples
    test_data, _ = next(iter(val_loader))
    test_data = test_data.to(device)
    
    with torch.no_grad():
        recon_data, _, _ = model(test_data)
    
    # Convert to numpy for plotting
    test_data_np = test_data.cpu().numpy()
    recon_data_np = recon_data.cpu().numpy()
    
    # Denormalize for visualization
    test_data_np = dataset.inverse_transform(torch.tensor(test_data_np)).numpy()
    recon_data_np = dataset.inverse_transform(torch.tensor(recon_data_np)).numpy()
    
    # Generate samples
    print("\n5. Generating new samples...")
    generated_samples = model.sample(num_samples=5, seq_len=seq_len, device=device)
    generated_samples_np = dataset.inverse_transform(generated_samples).cpu().numpy()
    
    # Plot results
    print("\n6. Creating visualizations...")
    
    # Plot reconstruction
    fig1 = plot_reconstruction(test_data_np, recon_data_np, n_samples=3)
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    print("   - Saved reconstruction plot to 'reconstruction.png'")
    
    # Plot training history
    fig2 = plot_training_history(history)
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("   - Saved training history to 'training_history.png'")
    
    # Plot latent space
    plot_latent_space(model, val_dataset, device)
    plt.savefig('latent_space.png', dpi=150, bbox_inches='tight')
    print("   - Saved latent space plot to 'latent_space.png'")
    
    # Save model
    print("\n7. Saving model...")
    trainer.save_model('vae_model.pth')
    print("   - Model saved to 'vae_model.pth'")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
