"""
Example: Mixed-Type VAE with Baseline Covariates and Landmark Prediction in Julia

Demonstrates:
1. Defining variable types (continuous, binary, bounded)
2. Generating mixed-type synthetic data with baseline covariates
3. Training a CNN-based VAE with mixed loss (Gaussian NLL + BCE)
4. Landmark prediction from partial observations
5. Prediction evaluation (MAE, RMSE, correlation)
"""

using VAElong
using Random
using Statistics
using Printf

function main()
    Random.seed!(42)

    # ================================================================
    # 1. Define variable types
    # ================================================================
    println("Setting up variable configuration...")
    var_config = VariableConfig([
        VariableSpec("heart_rate", "continuous"),       # continuous measurement
        VariableSpec("blood_pressure", "continuous"),   # continuous measurement
        VariableSpec("medication_taken", "binary"),     # binary indicator
        VariableSpec("symptom_present", "binary"),      # binary indicator
        VariableSpec("pain_score", "bounded"; lower=0.0f0, upper=10.0f0),  # bounded score
    ])

    nf = n_features(var_config)
    println("  Variables: $nf")
    println("  Continuous: $(length(continuous_indices(var_config)))")
    println("  Binary: $(length(binary_indices(var_config)))")
    println("  Bounded: $(length(bounded_indices(var_config)))")

    # ================================================================
    # 2. Generate mixed-type synthetic data with baselines
    # ================================================================
    println("\nGenerating mixed-type longitudinal data...")
    n_samples = 500
    seq_len = 64  # Power of 2 for CNN
    n_baseline = 3

    data, baseline = generate_mixed_longitudinal_data(
        n_samples=n_samples,
        seq_len=seq_len,
        var_config=var_config,
        n_baseline_features=n_baseline,
        noise_level=0.1f0,
        random_intercept_sd=2.0f0,
        seed=42
    )

    @printf("Data shape: (%d, %d, %d)\n", size(data)...)
    @printf("Baseline shape: (%d, %d)\n", size(baseline)...)

    # Create missing data mask (20% missing)
    mask = create_missing_mask(
        (n_samples, seq_len, nf),
        missing_rate=0.2f0,
        pattern="random",
        seed=42
    )
    @printf("Missing data rate: %.1f%%\n", (1 - mean(mask)) * 100)

    # ================================================================
    # 3. Split and create datasets
    # ================================================================
    train_size = Int(floor(0.8 * n_samples))

    train_dataset = LongitudinalDataset(
        data[1:train_size, :, :];
        mask=mask[1:train_size, :, :],
        normalize=true,
        baseline_covariates=baseline[1:train_size, :],
        var_config=var_config
    )

    val_dataset = LongitudinalDataset(
        data[train_size+1:end, :, :];
        mask=mask[train_size+1:end, :, :],
        normalize=true,
        baseline_covariates=baseline[train_size+1:end, :],
        var_config=var_config
    )

    train_loader = create_data_loader(train_dataset, batch_size=32, shuffle=true)
    val_loader = create_data_loader(val_dataset, batch_size=32, shuffle=false)

    @printf("Train batches: %d, Val batches: %d\n", length(train_loader), length(val_loader))

    # ================================================================
    # 4. Create CNN-based VAE with baselines
    # ================================================================
    println("\nCreating CNN-based VAE with baseline conditioning...")
    model = CNNLongitudinalVAE(
        nf, seq_len;
        latent_dim=16,
        hidden_channels=[32, 64],
        kernel_size=3,
        n_baseline=n_baseline,
        var_config=var_config
    )

    n_params = sum(length, Flux.params(model))
    @printf("Model parameters: %d\n", n_params)

    # ================================================================
    # 5. Train with mixed loss and EM imputation
    # ================================================================
    trainer = VAETrainer(
        model;
        learning_rate=1e-3f0,
        β=0.5f0,
        device=cpu,
        var_config=var_config
    )

    println("\n" * "="^60)
    println("Training with mixed-type loss and EM imputation...")
    println("="^60)
    history = fit!(
        trainer,
        train_loader;
        val_loader=val_loader,
        epochs=100,
        verbose=true,
        use_em_imputation=true,
        em_iterations=3
    )

    # ================================================================
    # 6. Landmark prediction
    # ================================================================
    println("\n" * "="^60)
    println("Landmark prediction...")
    println("="^60)

    # Take a few validation samples
    n_eval = 5
    eval_data, eval_mask, _, eval_baseline = val_dataset[1:n_eval]

    # Use first half as observed (landmark at seq_len/2)
    landmark = seq_len ÷ 2
    observed_data = copy(eval_data)
    observed_data[:, landmark+1:end, :] .= 0.0f0
    observed_mask = copy(eval_mask)
    observed_mask[:, landmark+1:end, :] .= 0.0f0

    predicted = predict_from_landmark(model, observed_data, observed_mask;
                                       baseline=eval_baseline)

    @printf("Observed data shape: (%d, %d, %d)\n", size(observed_data)...)
    @printf("Predicted trajectory shape: (%d, %d, %d)\n", size(predicted)...)

    # ================================================================
    # 7. Prediction evaluation on validation set
    # ================================================================
    println("\n" * "="^60)
    println("Prediction evaluation (future portion)...")
    println("="^60)

    n_val = length(val_dataset)
    all_eval_data, all_eval_mask, _, all_eval_baseline = val_dataset[1:n_val]

    # Observe first half, predict full trajectory
    obs_data = copy(all_eval_data)
    obs_data[:, landmark+1:end, :] .= 0.0f0
    obs_mask = copy(all_eval_mask)
    obs_mask[:, landmark+1:end, :] .= 0.0f0

    all_predicted = predict_from_landmark(model, obs_data, obs_mask;
                                           baseline=all_eval_baseline)

    # Inverse transform both
    actual_orig = inverse_transform(val_dataset, all_eval_data)
    pred_orig = inverse_transform(val_dataset, all_predicted)

    # Metrics on future (unobserved) portion only
    # Data is in Flux format: (n_features, seq_len, n_samples)
    @printf("%-20s  %8s  %8s  %8s\n", "Variable", "MAE", "RMSE", "Corr")
    println("-"^50)
    for (j, vs) in enumerate(var_config.variables)
        a = vec(actual_orig[j, landmark+1:end, :])
        p = vec(pred_orig[j, landmark+1:end, :])
        mae_val = mean(abs.(a .- p))
        rmse_val = sqrt(mean((a .- p) .^ 2))
        corr_val = std(a) > 0 ? cor(a, p) : NaN
        @printf("%-20s  %8.4f  %8.4f  %8.4f\n", vs.name, mae_val, rmse_val, corr_val)
    end

    println("\nDone!")
end

# Run the example
main()
