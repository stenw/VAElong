"""
Data utilities for longitudinal measurements.
"""

using Statistics
using Random

"""
    LongitudinalDataset

Dataset structure for longitudinal measurements with missing data support.

# Fields
- `data::Array{Float32,3}`: Data of shape (n_features, seq_len, n_samples)
- `mask::Array{Float32,3}`: Binary mask (1=observed, 0=missing)
- `lengths::Vector{Int}`: Sequence lengths
- `mean::Union{Array{Float32},Nothing}`: Mean for normalization
- `std::Union{Array{Float32},Nothing}`: Standard deviation for normalization
"""
struct LongitudinalDataset
    data::Array{Float32,3}
    mask::Array{Float32,3}
    lengths::Vector{Int}
    mean::Union{Array{Float32},Nothing}
    std::Union{Array{Float32},Nothing}
end

"""
    LongitudinalDataset(data; mask=nothing, normalize=true)

Create a longitudinal dataset.

# Arguments
- `data::Array{Float32,3}`: Data of shape (n_samples, seq_len, n_features)
- `mask::Union{Array{Float32,3},Nothing}=nothing`: Optional binary mask
- `normalize::Bool=true`: Whether to normalize the data
"""
function LongitudinalDataset(data::Array{Float32,3}; mask::Union{Array{Float32,3},Nothing}=nothing,
                             normalize::Bool=true)
    n_samples, seq_len, n_features = size(data)

    # Transpose to (n_features, seq_len, n_samples) for Flux compatibility
    data_t = permutedims(data, (3, 2, 1))

    # Handle mask
    if isnothing(mask)
        mask_t = ones(Float32, size(data_t))
    else
        mask_t = permutedims(mask, (3, 2, 1))
    end

    # Lengths (all sequences assumed same length for now)
    lengths = fill(seq_len, n_samples)

    # Normalization
    if normalize
        # Compute statistics only on observed values
        observed_data = data_t .* mask_t
        n_observed = sum(mask_t, dims=(2, 3))
        n_observed[n_observed .== 0] .= 1.0f0

        data_mean = sum(observed_data, dims=(2, 3)) ./ n_observed
        data_std = sqrt.(sum((observed_data .- data_mean .* mask_t) .^ 2, dims=(2, 3)) ./ n_observed)
        data_std[data_std .== 0] .= 1.0f0

        # Normalize
        data_t = (data_t .- data_mean) ./ data_std
    else
        data_mean = nothing
        data_std = nothing
    end

    return LongitudinalDataset(data_t, mask_t, lengths, data_mean, data_std)
end

"""Get batch of data."""
function Base.getindex(dataset::LongitudinalDataset, idxs)
    return dataset.data[:, :, idxs], dataset.mask[:, :, idxs], dataset.lengths[idxs]
end

"""Get dataset size."""
Base.length(dataset::LongitudinalDataset) = size(dataset.data, 3)

"""Inverse transform to denormalize data."""
function inverse_transform(dataset::LongitudinalDataset, data)
    if !isnothing(dataset.mean) && !isnothing(dataset.std)
        return data .* dataset.std .+ dataset.mean
    end
    return data
end


"""
    generate_synthetic_longitudinal_data(; n_samples=1000, seq_len=50,
                                         n_features=5, noise_level=0.1, seed=nothing)

Generate synthetic longitudinal data for testing.

# Arguments
- `n_samples::Int=1000`: Number of samples to generate
- `seq_len::Int=50`: Length of each sequence
- `n_features::Int=5`: Number of features per time step
- `noise_level::Float32=0.1f0`: Amount of noise to add
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility

# Returns
- `data::Array{Float32,3}`: Data of shape (n_samples, seq_len, n_features)
"""
function generate_synthetic_longitudinal_data(; n_samples::Int=1000, seq_len::Int=50,
                                              n_features::Int=5, noise_level::Float32=0.1f0,
                                              seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    data = zeros(Float32, n_samples, seq_len, n_features)

    for i in 1:n_samples
        # Generate temporal patterns
        t = range(0, 4π, length=seq_len)

        for j in 1:n_features
            # Combine trend, seasonality, and noise
            trend = randn(Float32) .* collect(t) ./ (4π)
            seasonality = sin.(t .+ rand(Float32) * 2π) .* rand(Float32)
            noise = randn(Float32, seq_len) .* noise_level

            data[i, :, j] = trend .+ seasonality .+ noise
        end
    end

    return data
end


"""
    create_missing_mask(data_shape; missing_rate=0.2, pattern="random", seed=nothing)

Create a binary mask for missing data.

# Arguments
- `data_shape::Tuple{Int,Int,Int}`: Shape of the data (n_samples, seq_len, n_features)
- `missing_rate::Float32=0.2f0`: Proportion of values to mark as missing
- `pattern::String="random"`: Missing data pattern - "random", "block", or "monotone"
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility

# Returns
- `mask::Array{Float32,3}`: Binary mask where 1=observed, 0=missing
"""
function create_missing_mask(data_shape::Tuple{Int,Int,Int}; missing_rate::Float32=0.2f0,
                            pattern::String="random", seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_samples, seq_len, n_features = data_shape
    mask = ones(Float32, data_shape)

    if pattern == "random"
        # Random missing values
        missing_indices = rand(Float32, data_shape) .< missing_rate
        mask[missing_indices] .= 0.0f0

    elseif pattern == "block"
        # Contiguous blocks of missing values in time
        for i in 1:n_samples
            for j in 1:n_features
                n_blocks = max(1, Int(floor(missing_rate * seq_len / 5)))
                for _ in 1:n_blocks
                    start_idx = rand(1:seq_len)
                    length_block = rand(1:max(2, Int(floor(seq_len * 0.2))))
                    end_idx = min(start_idx + length_block, seq_len)
                    mask[i, start_idx:end_idx, j] .= 0.0f0
                end
            end
        end

    elseif pattern == "monotone"
        # Monotone missingness pattern
        for i in 1:n_samples
            for j in 1:n_features
                if rand(Float32) < missing_rate
                    dropout_point = rand(1:seq_len)
                    mask[i, dropout_point:end, j] .= 0.0f0
                end
            end
        end

    else
        error("Unknown pattern: $pattern. Use 'random', 'block', or 'monotone'")
    end

    return mask
end
