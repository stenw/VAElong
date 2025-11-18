"""
VAE model implementations for longitudinal data.
"""

using Flux
using Statistics
using Random

"""
    LongitudinalVAE

LSTM/GRU-based Variational Autoencoder for longitudinal data.

# Arguments
- `input_dim::Int`: Dimension of input features at each time step
- `hidden_dim::Int=64`: Dimension of RNN hidden state
- `latent_dim::Int=20`: Dimension of latent space
- `num_layers::Int=1`: Number of RNN layers
- `use_gru::Bool=false`: Use GRU instead of LSTM
"""
struct LongitudinalVAE
    encoder_rnn
    fc_mu
    fc_logvar
    fc_latent
    decoder_rnn
    fc_output
    input_dim::Int
    hidden_dim::Int
    latent_dim::Int
end

Flux.@functor LongitudinalVAE

function LongitudinalVAE(input_dim::Int; hidden_dim::Int=64, latent_dim::Int=20,
                         num_layers::Int=1, use_gru::Bool=false)
    # Encoder
    encoder_rnn = use_gru ? GRU(input_dim => hidden_dim) : LSTM(input_dim => hidden_dim)
    fc_mu = Dense(hidden_dim, latent_dim)
    fc_logvar = Dense(hidden_dim, latent_dim)

    # Decoder
    fc_latent = Dense(latent_dim, hidden_dim, relu)
    decoder_rnn = use_gru ? GRU(hidden_dim => hidden_dim) : LSTM(hidden_dim => hidden_dim)
    fc_output = Dense(hidden_dim, input_dim)

    LongitudinalVAE(encoder_rnn, fc_mu, fc_logvar, fc_latent, decoder_rnn, fc_output,
                   input_dim, hidden_dim, latent_dim)
end

"""Encode input sequence to latent distribution parameters."""
function encode(m::LongitudinalVAE, x)
    # x: (input_dim, seq_len, batch_size)
    h = m.encoder_rnn(x)[end]  # Get final hidden state
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
function decode(m::LongitudinalVAE, z, seq_len)
    batch_size = size(z, 2)
    h = m.fc_latent(z)

    # Repeat for each time step
    h_repeated = repeat(h, 1, seq_len)
    h_reshaped = reshape(h_repeated, m.hidden_dim, seq_len, batch_size)

    # Pass through RNN
    rnn_out = m.decoder_rnn(h_reshaped)

    # Generate output
    output = m.fc_output(rnn_out)
    return output
end

"""Forward pass through VAE."""
function (m::LongitudinalVAE)(x, mask=nothing)
    seq_len = size(x, 2)

    # Apply mask if provided
    if !isnothing(mask)
        x = x .* mask
    end

    # Encode
    μ, logσ² = encode(m, x)

    # Reparameterize
    z = reparameterize(μ, logσ²)

    # Decode
    recon_x = decode(m, z, seq_len)

    return recon_x, μ, logσ²
end


"""
    CNNLongitudinalVAE

CNN-based Variational Autoencoder for longitudinal data with missing data handling.

# Arguments
- `input_dim::Int`: Dimension of input features at each time step
- `seq_len::Int`: Expected sequence length
- `latent_dim::Int=20`: Dimension of latent space
- `hidden_channels::Vector{Int}=[32, 64, 128]`: Channel sizes for encoder convolutions
- `kernel_size::Int=3`: Kernel size for convolutions
"""
struct CNNLongitudinalVAE
    encoder
    fc_mu
    fc_logvar
    fc_decode
    decoder
    input_dim::Int
    seq_len::Int
    latent_dim::Int
    encoded_size::Int
    encoded_channels::Int
    encoded_length::Int
end

Flux.@functor CNNLongitudinalVAE

function CNNLongitudinalVAE(input_dim::Int, seq_len::Int; latent_dim::Int=20,
                            hidden_channels::Vector{Int}=[32, 64, 128], kernel_size::Int=3)
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

    # Latent layers
    fc_mu = Dense(encoded_size, latent_dim)
    fc_logvar = Dense(encoded_size, latent_dim)
    fc_decode = Dense(latent_dim, encoded_size, relu)

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

    CNNLongitudinalVAE(encoder, fc_mu, fc_logvar, fc_decode, decoder,
                      input_dim, seq_len, latent_dim, encoded_size,
                      encoded_channels, encoded_length)
end

"""Encode input sequence to latent distribution parameters."""
function encode(m::CNNLongitudinalVAE, x, mask=nothing)
    # x: (input_dim, seq_len, batch_size)
    batch_size = size(x, 3)

    # Apply mask if provided
    if !isnothing(mask)
        x = x .* mask
    end

    # Encode
    h = m.encoder(x)
    h_flat = reshape(h, :, batch_size)

    # Get latent parameters
    μ = m.fc_mu(h_flat)
    logσ² = m.fc_logvar(h_flat)

    return μ, logσ²
end

"""Decode latent representation to output sequence."""
function decode(m::CNNLongitudinalVAE, z)
    batch_size = size(z, 2)

    # Map to encoded size
    h = m.fc_decode(z)

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
            output = vcat(output, zeros(Float32, m.input_dim, padding, batch_size))
        end
    end

    return output
end

"""Forward pass through VAE."""
function (m::CNNLongitudinalVAE)(x, mask=nothing)
    # Encode
    μ, logσ² = encode(m, x, mask)

    # Reparameterize
    z = reparameterize(μ, logσ²)

    # Decode
    recon_x = decode(m, z)

    return recon_x, μ, logσ²
end

"""Generate samples from the learned distribution."""
function sample(m::CNNLongitudinalVAE, num_samples::Int)
    z = randn(Float32, m.latent_dim, num_samples)
    return decode(m, z)
end

"""Impute missing values using iterative EM-like approach."""
function impute_missing(m::CNNLongitudinalVAE, x, mask; num_iterations::Int=5)
    imputed = copy(x)

    for iteration in 1:num_iterations
        # E-step: Generate predictions for missing values
        recon_x, μ, logσ² = m(imputed, mask)

        # Sample from the reconstruction
        σ = exp.(0.5f0 .* logσ²)
        noise = randn(Float32, size(recon_x)) .* reshape(σ, 1, 1, :) .* 0.1f0
        sampled_recon = recon_x .+ noise

        # Update missing values with sampled predictions
        imputed = mask .* x .+ (1 .- mask) .* sampled_recon
    end

    return imputed
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
        recon_loss = sum(diff .* mask)
        n_observed = sum(mask)
        if n_observed > 0
            recon_loss = recon_loss / n_observed * length(mask)
        end
    else
        recon_loss = sum((recon_x .- x) .^ 2)
    end

    # KL divergence
    kld_loss = -0.5f0 * sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²))

    # Total loss
    loss = recon_loss + β * kld_loss

    return loss, recon_loss, kld_loss
end
