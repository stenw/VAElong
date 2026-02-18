"""
VAE model implementations for longitudinal data.
Supports mixed variable types (continuous, binary, bounded) and baseline covariates.
"""

using Flux
using Statistics
using Random

"""
    LongitudinalVAE

LSTM/GRU-based Variational Autoencoder for longitudinal data.
Supports mixed variable types and baseline covariates (CVAE).

# Arguments
- `input_dim::Int`: Dimension of input features at each time step
- `hidden_dim::Int=64`: Dimension of RNN hidden state
- `latent_dim::Int=20`: Dimension of latent space
- `num_layers::Int=1`: Number of RNN layers
- `use_gru::Bool=false`: Use GRU instead of LSTM
- `n_baseline::Int=0`: Number of baseline covariate features
- `var_config::Union{VariableConfig,Nothing}=nothing`: Variable type configuration
"""
struct LongitudinalVAE
    encoder_rnn
    fc_mu
    fc_logvar
    fc_latent
    decoder_rnn
    fc_output
    log_noise_var    # Learned observation log-variance for continuous variables
    input_dim::Int
    hidden_dim::Int
    latent_dim::Int
    n_baseline::Int
    var_config::Union{VariableConfig,Nothing}
end

Flux.@functor LongitudinalVAE

function LongitudinalVAE(input_dim::Int; hidden_dim::Int=64, latent_dim::Int=20,
                         num_layers::Int=1, use_gru::Bool=false,
                         n_baseline::Int=0, var_config::Union{VariableConfig,Nothing}=nothing)
    # Encoder
    encoder_rnn = use_gru ? GRU(input_dim => hidden_dim) : LSTM(input_dim => hidden_dim)
    fc_mu = Dense(hidden_dim + n_baseline, latent_dim)
    fc_logvar = Dense(hidden_dim + n_baseline, latent_dim)

    # Decoder
    fc_latent = Dense(latent_dim + n_baseline, hidden_dim, relu)
    decoder_rnn = use_gru ? GRU(hidden_dim => hidden_dim) : LSTM(hidden_dim => hidden_dim)
    fc_output = Dense(hidden_dim, input_dim)

    # Learned observation log-variance for continuous variables
    n_cont = !isnothing(var_config) ? length(continuous_indices(var_config)) : input_dim
    log_noise_var = zeros(Float32, n_cont)

    LongitudinalVAE(encoder_rnn, fc_mu, fc_logvar, fc_latent, decoder_rnn, fc_output,
                   log_noise_var, input_dim, hidden_dim, latent_dim, n_baseline, var_config)
end

"""Encode input sequence to latent distribution parameters."""
function encode(m::LongitudinalVAE, x; mask=nothing, baseline=nothing)
    # x: (input_dim, seq_len, batch_size)
    if !isnothing(mask)
        x = x .* mask
    end

    h = m.encoder_rnn(x)[end]  # Get final hidden state
    # h: (hidden_dim, batch_size) — last time step output

    # Concatenate baseline covariates
    if !isnothing(baseline) && m.n_baseline > 0
        h = vcat(h, baseline)
    end

    μ = m.fc_mu(h)
    logσ² = m.fc_logvar(h)
    return μ, logσ²
end

"""Reparameterization trick."""
function reparameterize(μ, logσ²)
    σ = exp.(0.5f0 .* logσ²)
    ε = randn(Float32, size(σ))
    return μ .+ σ .* ε
end

"""Decode latent representation to output sequence."""
function decode(m::LongitudinalVAE, z, seq_len; baseline=nothing)
    batch_size = size(z, 2)

    # Concatenate baseline covariates to latent
    if !isnothing(baseline) && m.n_baseline > 0
        z_cond = vcat(z, baseline)
    else
        z_cond = z
    end

    h = m.fc_latent(z_cond)

    # Repeat for each time step
    h_repeated = repeat(h, 1, seq_len)
    h_reshaped = reshape(h_repeated, m.hidden_dim, seq_len, batch_size)

    # Pass through RNN
    rnn_out = m.decoder_rnn(h_reshaped)

    # Generate output
    output = m.fc_output(rnn_out)

    # Apply type-specific activations
    output = _apply_output_activations(output, m.var_config)

    return output
end

"""Forward pass through VAE."""
function (m::LongitudinalVAE)(x; mask=nothing, baseline=nothing)
    seq_len = size(x, 2)

    # Encode
    μ, logσ² = encode(m, x; mask=mask, baseline=baseline)

    # Reparameterize
    z = reparameterize(μ, logσ²)

    # Decode
    recon_x = decode(m, z, seq_len; baseline=baseline)

    return recon_x, μ, logσ²
end

"""
    predict_from_landmark(m::LongitudinalVAE, x_observed, mask_observed, total_seq_len;
                          baseline=nothing)

Landmark prediction: encode observed data, decode the full sequence.
Given data observed up to a landmark time point, predict the full
trajectory including future time steps.

# Arguments
- `m::LongitudinalVAE`: Model
- `x_observed`: (input_dim, observed_len, batch) data observed so far
- `mask_observed`: (input_dim, observed_len, batch) mask for observed data
- `total_seq_len::Int`: Total sequence length to predict
- `baseline`: Optional (n_baseline, batch) baseline covariates

# Returns
- `predicted`: (input_dim, total_seq_len, batch) full predicted trajectory
"""
function predict_from_landmark(m::LongitudinalVAE, x_observed, mask_observed, total_seq_len::Int;
                                baseline=nothing)
    μ, _ = encode(m, x_observed; mask=mask_observed, baseline=baseline)
    # Use mean for deterministic prediction
    predicted = decode(m, μ, total_seq_len; baseline=baseline)
    return predicted
end


"""
    CNNLongitudinalVAE

CNN-based Variational Autoencoder for longitudinal data with missing data handling.
Supports mixed variable types and baseline covariates (CVAE).

# Arguments
- `input_dim::Int`: Dimension of input features at each time step
- `seq_len::Int`: Expected sequence length
- `latent_dim::Int=20`: Dimension of latent space
- `hidden_channels::Vector{Int}=[32, 64, 128]`: Channel sizes for encoder convolutions
- `kernel_size::Int=3`: Kernel size for convolutions
- `n_baseline::Int=0`: Number of baseline covariate features
- `var_config::Union{VariableConfig,Nothing}=nothing`: Variable type configuration
"""
struct CNNLongitudinalVAE
    encoder
    fc_mu
    fc_logvar
    fc_decode
    decoder
    log_noise_var    # Learned observation log-variance for continuous variables
    input_dim::Int
    seq_len::Int
    latent_dim::Int
    encoded_size::Int
    encoded_channels::Int
    encoded_length::Int
    n_baseline::Int
    var_config::Union{VariableConfig,Nothing}
end

Flux.@functor CNNLongitudinalVAE

function CNNLongitudinalVAE(input_dim::Int, seq_len::Int; latent_dim::Int=20,
                            hidden_channels::Vector{Int}=[32, 64, 128], kernel_size::Int=3,
                            n_baseline::Int=0, var_config::Union{VariableConfig,Nothing}=nothing)
    # Build encoder
    encoder_layers = []
    in_channels = input_dim

    for out_channels in hidden_channels
        push!(encoder_layers, Conv((kernel_size,), in_channels => out_channels,
                                   stride=2, pad=kernel_size÷2))
        push!(encoder_layers, BatchNorm(out_channels))
        push!(encoder_layers, x -> relu.(x))
        in_channels = out_channels
    end

    encoder = Chain(encoder_layers...)

    # Calculate encoded size
    dummy_input = zeros(Float32, input_dim, seq_len, 1)
    dummy_output = encoder(dummy_input)
    encoded_size = prod(size(dummy_output)[1:2])
    encoded_channels = size(dummy_output, 1)
    encoded_length = size(dummy_output, 2)

    # Latent layers (input includes baseline covariates)
    fc_mu = Dense(encoded_size + n_baseline, latent_dim)
    fc_logvar = Dense(encoded_size + n_baseline, latent_dim)
    fc_decode = Dense(latent_dim + n_baseline, encoded_size, relu)

    # Build decoder (reverse of encoder)
    decoder_layers = []
    channels = reverse(hidden_channels)

    for (i, out_channels) in enumerate([channels[2:end]; input_dim])
        in_channels = channels[i]
        push!(decoder_layers, ConvTranspose((kernel_size,), in_channels => out_channels,
                                           stride=2, pad=kernel_size÷2))
        if out_channels != input_dim
            push!(decoder_layers, BatchNorm(out_channels))
            push!(decoder_layers, x -> relu.(x))
        end
    end

    decoder = Chain(decoder_layers...)

    # Learned observation log-variance for continuous variables
    n_cont = !isnothing(var_config) ? length(continuous_indices(var_config)) : input_dim
    log_noise_var = zeros(Float32, n_cont)

    CNNLongitudinalVAE(encoder, fc_mu, fc_logvar, fc_decode, decoder,
                      log_noise_var, input_dim, seq_len, latent_dim, encoded_size,
                      encoded_channels, encoded_length, n_baseline, var_config)
end

"""Encode input sequence to latent distribution parameters."""
function encode(m::CNNLongitudinalVAE, x; mask=nothing, baseline=nothing)
    # x: (input_dim, seq_len, batch_size)
    batch_size = size(x, 3)

    # Apply mask if provided
    if !isnothing(mask)
        x = x .* mask
    end

    # Encode
    h = m.encoder(x)
    h_flat = reshape(h, :, batch_size)

    # Concatenate baseline covariates
    if !isnothing(baseline) && m.n_baseline > 0
        h_flat = vcat(h_flat, baseline)
    end

    # Get latent parameters
    μ = m.fc_mu(h_flat)
    logσ² = m.fc_logvar(h_flat)

    return μ, logσ²
end

"""Decode latent representation to output sequence."""
function decode(m::CNNLongitudinalVAE, z; baseline=nothing)
    batch_size = size(z, 2)

    # Concatenate baseline covariates to latent
    if !isnothing(baseline) && m.n_baseline > 0
        z_cond = vcat(z, baseline)
    else
        z_cond = z
    end

    # Map to encoded size
    h = m.fc_decode(z_cond)

    # Reshape to encoded dimensions
    h_reshaped = reshape(h, m.encoded_channels, m.encoded_length, batch_size)

    # Decode
    output = m.decoder(h_reshaped)

    # Crop or pad to match original sequence length
    if size(output, 2) != m.seq_len
        if size(output, 2) > m.seq_len
            output = output[:, 1:m.seq_len, :]
        else
            padding = m.seq_len - size(output, 2)
            output = cat(output, zeros(Float32, m.input_dim, padding, batch_size), dims=2)
        end
    end

    # Apply type-specific activations
    output = _apply_output_activations(output, m.var_config)

    return output
end

"""Forward pass through VAE."""
function (m::CNNLongitudinalVAE)(x; mask=nothing, baseline=nothing)
    # Encode
    μ, logσ² = encode(m, x; mask=mask, baseline=baseline)

    # Reparameterize
    z = reparameterize(μ, logσ²)

    # Decode
    recon_x = decode(m, z; baseline=baseline)

    return recon_x, μ, logσ²
end

"""Generate samples from the learned distribution."""
function sample(m::CNNLongitudinalVAE, num_samples::Int; baseline=nothing)
    z = randn(Float32, m.latent_dim, num_samples)
    return decode(m, z; baseline=baseline)
end

"""
    predict_from_landmark(m::CNNLongitudinalVAE, x_observed, mask_observed; baseline=nothing)

Landmark prediction for CNN model. x_observed should be padded to seq_len with zeros,
and mask_observed should indicate which time steps are observed.

# Arguments
- `m::CNNLongitudinalVAE`: Model
- `x_observed`: (input_dim, seq_len, batch) data with future values zeroed out
- `mask_observed`: (input_dim, seq_len, batch) mask (1 for observed, 0 for future)
- `baseline`: Optional (n_baseline, batch) baseline covariates

# Returns
- `predicted`: (input_dim, seq_len, batch) full predicted trajectory
"""
function predict_from_landmark(m::CNNLongitudinalVAE, x_observed, mask_observed; baseline=nothing)
    μ, _ = encode(m, x_observed; mask=mask_observed, baseline=baseline)
    # Use mean for deterministic prediction
    predicted = decode(m, μ; baseline=baseline)
    return predicted
end

"""Impute missing values using iterative EM-like approach."""
function impute_missing(m::CNNLongitudinalVAE, x, mask; num_iterations::Int=5,
                         noise_scale::Float32=0.1f0, baseline=nothing)
    imputed = copy(x)

    for iteration in 1:num_iterations
        # E-step: Generate predictions for missing values
        recon_x, μ, logσ² = m(imputed; mask=mask, baseline=baseline)

        # Add small noise for uncertainty
        noise = randn(Float32, size(recon_x)) .* noise_scale
        sampled_recon = recon_x .+ noise

        # Type-aware post-processing of imputed values
        if !isnothing(m.var_config)
            for idx in binary_indices(m.var_config)
                sampled_recon_slice = sampled_recon[idx, :, :]
                sampled_recon = _set_feature_slice(sampled_recon, idx,
                    Float32.(sampled_recon_slice .> 0.5f0))
            end
            for idx in bounded_indices(m.var_config)
                sampled_recon_slice = sampled_recon[idx, :, :]
                sampled_recon = _set_feature_slice(sampled_recon, idx,
                    clamp.(sampled_recon_slice, 0.0f0, 1.0f0))
            end
        end

        # Update missing values with sampled predictions
        imputed = mask .* x .+ (1 .- mask) .* sampled_recon
    end

    return imputed
end

"""Helper to set a feature slice in a 3D array (needed for Flux compatibility)."""
function _set_feature_slice(arr::AbstractArray{Float32,3}, idx::Int, values)
    result = copy(arr)
    result[idx, :, :] = values
    return result
end


# ============================================================
# Shared helpers
# ============================================================

"""
    _apply_output_activations(output, var_config)

Apply per-variable-type activations to the decoder output.
- Binary variables: sigmoid
- Bounded variables: sigmoid (data pre-normalized to [0,1])
- Continuous variables: identity (raw output)

Data is in Flux format: (n_features, seq_len, batch)
"""
function _apply_output_activations(output, var_config::Nothing)
    return output  # all continuous, raw output
end

function _apply_output_activations(output, var_config::VariableConfig)
    result = output
    for idx in binary_indices(var_config)
        result = _set_feature_slice(result, idx, Flux.sigmoid.(result[idx, :, :]))
    end
    for idx in bounded_indices(var_config)
        result = _set_feature_slice(result, idx, Flux.sigmoid.(result[idx, :, :]))
    end
    return result
end


# ============================================================
# Loss functions
# ============================================================

"""
    _masked_sum(values, mask)

Sum values where mask=1, normalized by observation count.
"""
function _masked_sum(values, mask)
    n_observed = sum(mask)
    if n_observed > 0
        return sum(values .* mask) / n_observed * length(mask)
    end
    return 0.0f0
end

"""
    vae_loss

VAE loss = Reconstruction loss + KL divergence

# Arguments
- `recon_x`: Reconstructed data
- `x`: Original data
- `μ`: Mean of latent distribution
- `logσ²`: Log variance of latent distribution
- `β::Float32=1.0f0`: Weight for KL divergence term
- `mask`: Optional binary mask for missing data
"""
function vae_loss(recon_x, x, μ, logσ²; β::Float32=1.0f0, mask=nothing)
    # Reconstruction loss
    if !isnothing(mask)
        # Only compute on observed values
        diff = (recon_x .- x) .^ 2
        recon_loss = _masked_sum(diff, mask)
    else
        recon_loss = sum((recon_x .- x) .^ 2)
    end

    # KL divergence
    kld_loss = -0.5f0 * sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²))

    # Total loss
    loss = recon_loss + β * kld_loss

    return loss, recon_loss, kld_loss
end

"""
    mixed_vae_loss(recon_x, x, μ, logσ²; β=1.0f0, mask=nothing, var_config=nothing,
                   log_noise_var=nothing)

VAE loss supporting mixed variable types with learned observation noise.

Computes proper negative log-likelihoods so that all variable types
are on a comparable scale:
- Continuous: Gaussian NLL with learned per-variable variance
  0.5 * (log σ² + (x - μ)² / σ²) — automatically down-weights noisy variables
- Binary: BCE (Bernoulli NLL)
- Bounded: BCE on [0,1]-normalised data

Falls back to pure MSE if var_config is Nothing (backward compatible).

Data is in Flux format: (n_features, seq_len, batch)

# Arguments
- `recon_x`: Reconstructed data
- `x`: Original data
- `μ`: Mean of latent distribution
- `logσ²`: Log variance of latent distribution
- `β::Float32=1.0f0`: Weight for KL divergence term
- `mask`: Optional binary mask for missing data
- `var_config`: Optional VariableConfig for mixed types
- `log_noise_var`: Optional learned log-variance for continuous variables,
  shape (n_continuous,). If nothing, falls back to MSE (σ²=1).
"""
function mixed_vae_loss(recon_x, x, μ, logσ²; β::Float32=1.0f0, mask=nothing,
                         var_config::Union{VariableConfig,Nothing}=nothing,
                         log_noise_var=nothing)
    if isnothing(var_config)
        return vae_loss(recon_x, x, μ, logσ²; β=β, mask=mask)
    end

    recon_loss = 0.0f0

    # Continuous variables: heteroscedastic Gaussian NLL
    cont_idx = continuous_indices(var_config)
    if !isempty(cont_idx)
        cont_recon = recon_x[cont_idx, :, :]
        cont_x = x[cont_idx, :, :]

        if !isnothing(log_noise_var)
            # Proper Gaussian NLL: 0.5 * (log σ² + (x - μ)² / σ²)
            # log_noise_var shape: (n_continuous,) → reshape to (n_cont, 1, 1)
            lnv = reshape(log_noise_var, :, 1, 1)
            nll = 0.5f0 .* (lnv .+ (cont_recon .- cont_x) .^ 2 ./ exp.(lnv))
        else
            # Fallback: MSE (equivalent to σ²=1, dropping constant)
            nll = (cont_recon .- cont_x) .^ 2
        end

        if !isnothing(mask)
            cont_mask = mask[cont_idx, :, :]
            recon_loss = recon_loss + _masked_sum(nll, cont_mask)
        else
            recon_loss = recon_loss + sum(nll)
        end
    end

    # Binary variables: BCE
    bin_idx = binary_indices(var_config)
    if !isempty(bin_idx)
        bin_recon = clamp.(recon_x[bin_idx, :, :], 1.0f-7, 1.0f0 - 1.0f-7)
        bin_x = x[bin_idx, :, :]
        bce = -(bin_x .* log.(bin_recon) .+ (1 .- bin_x) .* log.(1 .- bin_recon))
        if !isnothing(mask)
            bin_mask = mask[bin_idx, :, :]
            recon_loss = recon_loss + _masked_sum(bce, bin_mask)
        else
            recon_loss = recon_loss + sum(bce)
        end
    end

    # Bounded variables: BCE (data in [0,1], output sigmoided)
    bnd_idx = bounded_indices(var_config)
    if !isempty(bnd_idx)
        bnd_recon = clamp.(recon_x[bnd_idx, :, :], 1.0f-7, 1.0f0 - 1.0f-7)
        bnd_x = clamp.(x[bnd_idx, :, :], 0.0f0, 1.0f0)
        bce = -(bnd_x .* log.(bnd_recon) .+ (1 .- bnd_x) .* log.(1 .- bnd_recon))
        if !isnothing(mask)
            bnd_mask = mask[bnd_idx, :, :]
            recon_loss = recon_loss + _masked_sum(bce, bnd_mask)
        else
            recon_loss = recon_loss + sum(bce)
        end
    end

    # KL divergence (unchanged)
    kld_loss = -0.5f0 * sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²))

    loss = recon_loss + β * kld_loss
    return loss, recon_loss, kld_loss
end
