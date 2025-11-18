"""
Example: CNN-based VAE with Missing Data Handling in Julia

Demonstrates:
1. Creating synthetic longitudinal data with missing values
2. Training a CNN-based VAE with missing data
3. Using EM-like imputation during training
4. Imputing missing values after training
"""

using VAElong
using Random
using Statistics
using Printf

function main()
    # Set random seed for reproducibility
    Random.seed!(42)

    # Generate synthetic data
    println("Generating synthetic longitudinal data...")
    n_samples = 1000
    seq_len = 64  # Using power of 2 for CNN
    n_features = 5

    data = generate_synthetic_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        n_features=n_features,
        noise_level=0.1f0,
        seed=42
    )

    # Create missing data mask (20% missing with random pattern)
    println("Creating missing data mask...")
    mask = create_missing_mask(
        (n_samples, seq_len, n_features),
        missing_rate=0.2f0,
        pattern="random",
        seed=42
    )

    # Keep original complete data for comparison
    original_data = copy(data)

    # Apply mask to data
    data_with_missing = data .* mask

    @printf("Data shape: (%d, %d, %d)\n", size(data)...)
    @printf("Missing data rate: %.2f%%\n", (1 - mean(mask)) * 100)

    # Split into train and validation
    train_size = Int(floor(0.8 * n_samples))
    train_data = data_with_missing[1:train_size, :, :]
    train_mask = mask[1:train_size, :, :]
    val_data = data_with_missing[train_size+1:end, :, :]
    val_mask = mask[train_size+1:end, :, :]

    # Create datasets
    println("\nCreating datasets...")
    train_dataset = LongitudinalDataset(train_data, mask=train_mask, normalize=true)
    val_dataset = LongitudinalDataset(val_data, mask=val_mask, normalize=true)

    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size=32, shuffle=true)
    val_loader = create_data_loader(val_dataset, batch_size=32, shuffle=false)

    # Create CNN-based VAE model
    println("\nCreating CNN-based VAE model...")
    model = CNNLongitudinalVAE(
        n_features,
        seq_len,
        latent_dim=16,
        hidden_channels=[32, 64, 128],
        kernel_size=3
    )

    # Count parameters
    n_params = sum(length, Flux.params(model))
    @printf("Model parameters: %d\n", n_params)

    # Create trainer
    trainer = VAETrainer(
        model,
        learning_rate=1e-3f0,
        β=1.0f0,
        device=cpu  # Use gpu if CUDA is available
    )

    # Train with EM imputation
    println("\n" * "="^60)
    println("Training WITH EM imputation...")
    println("="^60)
    history = fit!(
        trainer,
        train_loader,
        val_loader=val_loader,
        epochs=50,
        verbose=true,
        use_em_imputation=true,
        em_iterations=3
    )

    # Evaluate imputation quality on a few samples
    println("\n" * "="^60)
    println("Evaluating imputation quality...")
    println("="^60)

    n_eval_samples = 5
    eval_data = val_data[1:n_eval_samples, :, :]
    eval_mask = val_mask[1:n_eval_samples, :, :]
    eval_original = original_data[train_size+1:train_size+n_eval_samples, :, :]

    # Convert to Flux format (n_features, seq_len, n_samples)
    eval_data_t = permutedims(eval_data, (3, 2, 1))
    eval_mask_t = permutedims(eval_mask, (3, 2, 1))

    # Normalize using training statistics
    eval_data_norm = (eval_data_t .- train_dataset.mean) ./ train_dataset.std

    # Impute missing values
    imputed = impute_missing(model, eval_data_norm, eval_mask_t, num_iterations=5)
    imputed = imputed .* train_dataset.std .+ train_dataset.mean

    # Convert back to original format
    imputed_original_format = permutedims(imputed, (3, 2, 1))

    # Calculate error on missing values only
    missing_mask_inv = 1 .- eval_mask
    n_missing = sum(missing_mask_inv)

    if n_missing > 0
        error = sum((imputed_original_format .- eval_original) .^ 2 .* missing_mask_inv) / n_missing
        @printf("\nMSE on missing values: %.4f\n", error)
    end

    # Generate new samples
    println("\nGenerating new samples from learned distribution...")
    samples = sample(model, 10)
    samples = samples .* train_dataset.std .+ train_dataset.mean

    @printf("Generated samples shape: (%d, %d, %d)\n", size(samples)...)
    @printf("Sample statistics - Mean: %.3f, Std: %.3f\n", mean(samples), std(samples))

    println("\nDone!")
end

# Run the example
main()
