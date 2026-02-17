"""
Training utilities for VAE.
Supports mixed variable types and baseline covariates.
"""

using Flux
using Statistics
using Printf

"""
    VAETrainer

Trainer for Longitudinal VAE models.

# Fields
- `model`: VAE model instance
- `optimizer`: Flux optimizer
- `β::Float32`: Weight for KL divergence term
- `device`: Device to train on (cpu or gpu)
- `var_config::Union{VariableConfig,Nothing}`: Variable type configuration
- `train_losses::Vector{Float32}`: Training loss history
- `val_losses::Vector{Float32}`: Validation loss history
"""
mutable struct VAETrainer
    model
    optimizer
    β::Float32
    device::Function
    var_config::Union{VariableConfig,Nothing}
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
end

"""
    VAETrainer(model; learning_rate=1e-3, β=1.0f0, device=cpu, var_config=nothing)

Create a VAE trainer.

# Arguments
- `model`: VAE model instance
- `learning_rate::Float32=1e-3f0`: Learning rate for optimizer
- `β::Float32=1.0f0`: Weight for KL divergence term
- `device::Function=cpu`: Device function (cpu or gpu)
- `var_config::Union{VariableConfig,Nothing}=nothing`: Variable type configuration
"""
function VAETrainer(model; learning_rate::Float32=1e-3f0, β::Float32=1.0f0,
                    device::Function=cpu, var_config::Union{VariableConfig,Nothing}=nothing)
    # Move model to device
    model_device = device(model)

    # Create optimizer
    optimizer = Flux.Adam(learning_rate)

    VAETrainer(model_device, optimizer, β, device, var_config, Float32[], Float32[])
end


"""
    _get_baseline_arg(baseline, device)

Return baseline tensor moved to device, or nothing if no baseline features.
"""
function _get_baseline_arg(baseline, device::Function)
    if size(baseline, 1) > 0
        return device(baseline)
    end
    return nothing
end


"""
    train_epoch!(trainer, data_loader; use_em_imputation=false, em_iterations=3)

Train for one epoch.

# Arguments
- `trainer::VAETrainer`: Trainer instance
- `data_loader`: Iterator over training batches (4-tuples)
- `use_em_imputation::Bool=false`: Whether to use EM-like imputation
- `em_iterations::Int=3`: Number of EM iterations per batch

# Returns
- `(avg_loss, avg_recon, avg_kld)`: Average losses
"""
function train_epoch!(trainer::VAETrainer, data_loader; use_em_imputation::Bool=false,
                     em_iterations::Int=3)
    total_loss = 0.0f0
    total_recon = 0.0f0
    total_kld = 0.0f0
    n_batches = 0

    for (batch_data, batch_mask, _, batch_baseline) in data_loader
        # Move to device
        batch_data = trainer.device(batch_data)
        batch_mask = trainer.device(batch_mask)
        baseline_arg = _get_baseline_arg(batch_baseline, trainer.device)

        # Check if there's missing data
        has_missing = sum(batch_mask) < length(batch_mask)

        if use_em_imputation && has_missing
            # EM-like approach
            for em_iter in 1:em_iterations
                # E-step: Impute missing values
                if em_iter > 1
                    recon_batch, μ_temp, logσ²_temp = trainer.model(batch_data;
                        mask=batch_mask, baseline=baseline_arg)
                    # Type-aware imputation
                    imputed = copy(recon_batch)
                    if !isnothing(trainer.var_config)
                        for idx in binary_indices(trainer.var_config)
                            imputed = _set_feature_slice(imputed, idx,
                                Float32.(imputed[idx, :, :] .> 0.5f0))
                        end
                        for idx in bounded_indices(trainer.var_config)
                            imputed = _set_feature_slice(imputed, idx,
                                clamp.(imputed[idx, :, :], 0.0f0, 1.0f0))
                        end
                    end
                    batch_data = batch_mask .* batch_data .+ (1 .- batch_mask) .* imputed
                end

                # M-step: Update model parameters
                loss, recon_loss, kld_loss = train_step!(trainer, batch_data, batch_mask;
                                                          baseline=baseline_arg)
            end
        else
            # Standard training
            mask_arg = has_missing ? batch_mask : nothing
            loss, recon_loss, kld_loss = train_step!(trainer, batch_data, mask_arg;
                                                      baseline=baseline_arg)
        end

        total_loss += loss
        total_recon += recon_loss
        total_kld += kld_loss
        n_batches += 1
    end

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kld = total_kld / n_batches

    return avg_loss, avg_recon, avg_kld
end


"""Single training step."""
function train_step!(trainer::VAETrainer, batch_data, mask; baseline=nothing)
    # Get model parameters
    params = Flux.params(trainer.model)

    # Compute loss and gradients
    loss, recon_loss, kld_loss, grads = Flux.withgradient(params) do
        recon_batch, μ, logσ² = trainer.model(batch_data; mask=mask, baseline=baseline)
        loss, recon_loss, kld_loss = mixed_vae_loss(recon_batch, batch_data, μ, logσ²;
                                                     β=trainer.β, mask=mask,
                                                     var_config=trainer.var_config)
        return loss, recon_loss, kld_loss
    end

    # Update parameters
    Flux.update!(trainer.optimizer, params, grads[1])

    return loss[1], recon_loss[1], kld_loss[1]
end


"""
    validate(trainer, data_loader)

Validate the model.

# Arguments
- `trainer::VAETrainer`: Trainer instance
- `data_loader`: Iterator over validation batches (4-tuples)

# Returns
- `(avg_loss, avg_recon, avg_kld)`: Average validation losses
"""
function validate(trainer::VAETrainer, data_loader)
    total_loss = 0.0f0
    total_recon = 0.0f0
    total_kld = 0.0f0
    n_batches = 0

    for (batch_data, batch_mask, _, batch_baseline) in data_loader
        # Move to device
        batch_data = trainer.device(batch_data)
        batch_mask = trainer.device(batch_mask)
        baseline_arg = _get_baseline_arg(batch_baseline, trainer.device)

        # Check if there's missing data
        has_missing = sum(batch_mask) < length(batch_mask)
        mask_arg = has_missing ? batch_mask : nothing

        # Forward pass
        recon_batch, μ, logσ² = trainer.model(batch_data; mask=mask_arg, baseline=baseline_arg)

        # Compute loss
        loss, recon_loss, kld_loss = mixed_vae_loss(recon_batch, batch_data, μ, logσ²;
                                                     β=trainer.β, mask=mask_arg,
                                                     var_config=trainer.var_config)

        total_loss += loss
        total_recon += recon_loss
        total_kld += kld_loss
        n_batches += 1
    end

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kld = total_kld / n_batches

    return avg_loss, avg_recon, avg_kld
end


"""
    fit!(trainer, train_loader; val_loader=nothing, epochs=100,
         verbose=true, use_em_imputation=false, em_iterations=3)

Train the model.

# Arguments
- `trainer::VAETrainer`: Trainer instance
- `train_loader`: Training data iterator (4-tuples)
- `val_loader=nothing`: Optional validation data iterator
- `epochs::Int=100`: Number of epochs to train
- `verbose::Bool=true`: Whether to print progress
- `use_em_imputation::Bool=false`: Whether to use EM-like imputation
- `em_iterations::Int=3`: Number of EM iterations per batch

# Returns
- `history::Dict`: Training history
"""
function fit!(trainer::VAETrainer, train_loader; val_loader=nothing, epochs::Int=100,
             verbose::Bool=true, use_em_imputation::Bool=false, em_iterations::Int=3)
    history = Dict(
        "train_loss" => Float32[],
        "train_recon" => Float32[],
        "train_kld" => Float32[],
        "val_loss" => Float32[],
        "val_recon" => Float32[],
        "val_kld" => Float32[]
    )

    for epoch in 1:epochs
        # Train
        train_loss, train_recon, train_kld = train_epoch!(trainer, train_loader;
                                                          use_em_imputation=use_em_imputation,
                                                          em_iterations=em_iterations)
        push!(history["train_loss"], train_loss)
        push!(history["train_recon"], train_recon)
        push!(history["train_kld"], train_kld)

        # Validate
        if !isnothing(val_loader)
            val_loss, val_recon, val_kld = validate(trainer, val_loader)
            push!(history["val_loss"], val_loss)
            push!(history["val_recon"], val_recon)
            push!(history["val_kld"], val_kld)
        end

        # Print progress
        if verbose && epoch % 10 == 0
            msg = @sprintf("Epoch [%d/%d] Train Loss: %.4f (Recon: %.4f, KLD: %.4f)",
                          epoch, epochs, train_loss, train_recon, train_kld)
            if !isnothing(val_loader)
                msg *= @sprintf(" | Val Loss: %.4f", val_loss)
            end
            println(msg)
        end
    end

    return history
end


"""Simple data iterator for batching. Returns 4-tuples: (data, mask, lengths, baseline)."""
function create_data_loader(dataset::LongitudinalDataset; batch_size::Int=32, shuffle::Bool=false)
    n_samples = length(dataset)
    indices = shuffle ? randperm(n_samples) : collect(1:n_samples)

    batches = []
    for i in 1:batch_size:n_samples
        batch_indices = indices[i:min(i+batch_size-1, n_samples)]
        batch_data, batch_mask, batch_lengths, batch_baseline = dataset[batch_indices]
        push!(batches, (batch_data, batch_mask, batch_lengths, batch_baseline))
    end

    return batches
end
