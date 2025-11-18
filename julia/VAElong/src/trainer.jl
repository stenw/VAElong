"""
Training utilities for VAE.
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
"""
mutable struct VAETrainer
    model
    optimizer
    β::Float32
    device::Function
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
end

"""
    VAETrainer(model; learning_rate=1e-3, β=1.0f0, device=cpu)

Create a VAE trainer.

# Arguments
- `model`: VAE model instance
- `learning_rate::Float32=1e-3f0`: Learning rate for optimizer
- `β::Float32=1.0f0`: Weight for KL divergence term
- `device::Function=cpu`: Device function (cpu or gpu)
"""
function VAETrainer(model; learning_rate::Float32=1e-3f0, β::Float32=1.0f0, device::Function=cpu)
    # Move model to device
    model_device = device(model)

    # Create optimizer
    optimizer = Flux.Adam(learning_rate)

    VAETrainer(model_device, optimizer, β, device, Float32[], Float32[])
end


"""
    train_epoch!(trainer, data_loader; use_em_imputation=false, em_iterations=3)

Train for one epoch.

# Arguments
- `trainer::VAETrainer`: Trainer instance
- `data_loader`: Iterator over training batches
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

    for (batch_data, batch_mask, _) in data_loader
        # Move to device
        batch_data = trainer.device(batch_data)
        batch_mask = trainer.device(batch_mask)

        # Check if there's missing data
        has_missing = sum(batch_mask) < length(batch_mask)

        if use_em_imputation && has_missing
            # EM-like approach
            for em_iter in 1:em_iterations
                # E-step: Impute missing values
                if em_iter > 1
                    recon_batch, μ_temp, logσ²_temp = trainer.model(batch_data, batch_mask)
                    batch_data = batch_mask .* batch_data .+ (1 .- batch_mask) .* recon_batch
                end

                # M-step: Update model parameters
                loss, recon_loss, kld_loss = train_step!(trainer, batch_data, batch_mask)
                
                # Accumulate losses from each EM iteration
                total_loss += loss
                total_recon += recon_loss
                total_kld += kld_loss
                n_batches += 1
            end
        else
            # Standard training
            mask_arg = has_missing ? batch_mask : nothing
            loss, recon_loss, kld_loss = train_step!(trainer, batch_data, mask_arg)
            
            # Accumulate losses
            total_loss += loss
            total_recon += recon_loss
            total_kld += kld_loss
            n_batches += 1
        end
    end

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kld = total_kld / n_batches

    return avg_loss, avg_recon, avg_kld
end


"""Single training step."""
function train_step!(trainer::VAETrainer, batch_data, mask)
    # Get model parameters
    params = Flux.params(trainer.model)

    # Compute loss and gradients
    loss, recon_loss, kld_loss, grads = Flux.withgradient(params) do
        recon_batch, μ, logσ² = trainer.model(batch_data, mask)
        loss, recon_loss, kld_loss = vae_loss(recon_batch, batch_data, μ, logσ²;
                                              β=trainer.β, mask=mask)
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
- `data_loader`: Iterator over validation batches

# Returns
- `(avg_loss, avg_recon, avg_kld)`: Average validation losses
"""
function validate(trainer::VAETrainer, data_loader)
    total_loss = 0.0f0
    total_recon = 0.0f0
    total_kld = 0.0f0
    n_batches = 0

    for (batch_data, batch_mask, _) in data_loader
        # Move to device
        batch_data = trainer.device(batch_data)
        batch_mask = trainer.device(batch_mask)

        # Check if there's missing data
        has_missing = sum(batch_mask) < length(batch_mask)
        mask_arg = has_missing ? batch_mask : nothing

        # Forward pass
        recon_batch, μ, logσ² = trainer.model(batch_data, mask_arg)

        # Compute loss
        loss, recon_loss, kld_loss = vae_loss(recon_batch, batch_data, μ, logσ²;
                                              β=trainer.β, mask=mask_arg)

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
- `train_loader`: Training data iterator
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


"""Simple data iterator for batching."""
function create_data_loader(dataset::LongitudinalDataset; batch_size::Int=32, shuffle::Bool=false)
    n_samples = length(dataset)
    indices = shuffle ? randperm(n_samples) : collect(1:n_samples)

    batches = []
    for i in 1:batch_size:n_samples
        batch_indices = indices[i:min(i+batch_size-1, n_samples)]
        batch_data, batch_mask, batch_lengths = dataset[batch_indices]
        push!(batches, (batch_data, batch_mask, batch_lengths))
    end

    return batches
end
