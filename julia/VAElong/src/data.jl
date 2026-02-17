"""
Data utilities for longitudinal measurements.
"""

using Statistics
using Random

"""
    LongitudinalDataset

Dataset structure for longitudinal measurements with missing data support.
Supports mixed variable types (continuous, binary, bounded) and baseline covariates.

# Fields
- `data::Array{Float32,3}`: Data of shape (n_features, seq_len, n_samples)
- `mask::Array{Float32,3}`: Binary mask (1=observed, 0=missing)
- `lengths::Vector{Int}`: Sequence lengths
- `baseline::Array{Float32,2}`: Baseline covariates of shape (n_baseline, n_samples)
- `var_config::VariableConfig`: Variable type configuration
- `mean::Union{Array{Float32},Nothing}`: Mean for normalization (continuous only)
- `std::Union{Array{Float32},Nothing}`: Standard deviation for normalization (continuous only)
- `bounds_info::Dict{Int,Tuple{Float32,Float32}}`: Bounds for bounded variables
"""
struct LongitudinalDataset
    data::Array{Float32,3}
    mask::Array{Float32,3}
    lengths::Vector{Int}
    baseline::Array{Float32,2}
    var_config::VariableConfig
    mean::Union{Array{Float32},Nothing}
    std::Union{Array{Float32},Nothing}
    bounds_info::Dict{Int,Tuple{Float32,Float32}}
end

"""
    LongitudinalDataset(data; mask=nothing, normalize=true,
                         baseline_covariates=nothing, var_config=nothing)

Create a longitudinal dataset with mixed-type support.

# Arguments
- `data::Array{Float32,3}`: Data of shape (n_samples, seq_len, n_features)
- `mask::Union{Array{Float32,3},Nothing}=nothing`: Optional binary mask
- `normalize::Bool=true`: Whether to normalize the data
- `baseline_covariates::Union{Array{Float32,2},Nothing}=nothing`: Optional (n_samples, n_baseline) covariates
- `var_config::Union{VariableConfig,Nothing}=nothing`: Variable type config (default: all continuous)
"""
function LongitudinalDataset(data::Array{Float32,3};
                             mask::Union{Array{Float32,3},Nothing}=nothing,
                             normalize::Bool=true,
                             baseline_covariates::Union{Array{Float32,2},Nothing}=nothing,
                             var_config::Union{VariableConfig,Nothing}=nothing)
    n_samples, seq_len, n_features = size(data)

    # Transpose to (n_features, seq_len, n_samples) for Flux compatibility
    data_t = permutedims(copy(data), (3, 2, 1))

    # Handle mask
    if isnothing(mask)
        mask_t = ones(Float32, size(data_t))
    else
        mask_t = permutedims(mask, (3, 2, 1))
    end

    # Lengths (all sequences assumed same length for now)
    lengths = fill(seq_len, n_samples)

    # Variable config (default: all continuous for backward compatibility)
    if isnothing(var_config)
        vc = all_continuous(n_features)
    else
        vc = var_config
    end

    # Baseline covariates: transpose to (n_baseline, n_samples)
    if isnothing(baseline_covariates)
        baseline_t = zeros(Float32, 0, n_samples)
    else
        baseline_t = permutedims(baseline_covariates, (2, 1))
    end

    # Normalization
    if normalize
        data_t, data_mean, data_std, bounds_info = _normalize_by_type(data_t, mask_t, vc)
    else
        data_mean = nothing
        data_std = nothing
        bounds_info = Dict{Int,Tuple{Float32,Float32}}()
    end

    return LongitudinalDataset(data_t, mask_t, lengths, baseline_t, vc, data_mean, data_std, bounds_info)
end

"""
    _normalize_by_type(data, mask, var_config)

Type-aware normalization:
- Continuous: z-score using observed values only
- Bounded: affine transform to [0, 1] using known bounds
- Binary: no normalization
"""
function _normalize_by_type(data::Array{Float32,3}, mask::Array{Float32,3}, vc::VariableConfig)
    # data shape: (n_features, seq_len, n_samples)
    n_features = size(data, 1)
    data_mean = zeros(Float32, n_features, 1, 1)
    data_std = ones(Float32, n_features, 1, 1)
    bounds_info = Dict{Int,Tuple{Float32,Float32}}()

    # Continuous: z-score using observed values
    for idx in continuous_indices(vc)
        observed = data[idx, :, :] .* mask[idx, :, :]
        n_obs = sum(mask[idx, :, :])
        if n_obs > 0
            m = sum(observed) / n_obs
            s = sqrt(sum((observed .- m .* mask[idx, :, :]) .^ 2) / n_obs)
            if s == 0
                s = 1.0f0
            end
            data_mean[idx, 1, 1] = m
            data_std[idx, 1, 1] = s
            data[idx, :, :] = ((data[idx, :, :] .- m) ./ s) .* mask[idx, :, :]
        end
    end

    # Bounded: affine transform to [0, 1]
    bds = get_bounds(vc)
    for idx in bounded_indices(vc)
        lo, hi = bds[idx]
        bounds_info[idx] = (lo, hi)
        data[idx, :, :] = ((data[idx, :, :] .- lo) ./ (hi - lo)) .* mask[idx, :, :]
    end

    # Binary: no normalization needed

    return data, data_mean, data_std, bounds_info
end

"""Get batch of data. Returns (data, mask, lengths, baseline) 4-tuple."""
function Base.getindex(dataset::LongitudinalDataset, idxs)
    baseline = size(dataset.baseline, 1) > 0 ? dataset.baseline[:, idxs] : dataset.baseline[:, idxs[1:min(end,length(idxs))]]
    return dataset.data[:, :, idxs], dataset.mask[:, :, idxs], dataset.lengths[idxs], baseline
end

"""Get dataset size."""
Base.length(dataset::LongitudinalDataset) = size(dataset.data, 3)

"""
    inverse_transform(dataset, data)

Type-aware inverse transformation.

# Arguments
- `dataset::LongitudinalDataset`: Dataset with normalization statistics
- `data`: Normalized data (n_features, seq_len, ...) in Flux format

# Returns
- Denormalized data
"""
function inverse_transform(dataset::LongitudinalDataset, data)
    result = copy(data)

    if !isnothing(dataset.mean) && !isnothing(dataset.std)
        # Continuous: reverse z-score
        for idx in continuous_indices(dataset.var_config)
            result[idx, :, :] = result[idx, :, :] .* dataset.std[idx, 1, 1] .+ dataset.mean[idx, 1, 1]
        end
    end

    if !isempty(dataset.bounds_info)
        # Bounded: reverse affine from [0,1] to [lower, upper]
        for (idx, (lo, hi)) in dataset.bounds_info
            result[idx, :, :] = result[idx, :, :] .* (hi - lo) .+ lo
        end
    end

    # Binary: no inverse needed
    return result
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
    generate_mixed_longitudinal_data(; n_samples=1000, seq_len=50, var_config=nothing,
                                      n_baseline_features=0, noise_level=0.1, seed=nothing)

Generate synthetic longitudinal data with mixed variable types.

# Arguments
- `n_samples::Int=1000`: Number of samples
- `seq_len::Int=50`: Sequence length
- `var_config::Union{VariableConfig,Nothing}=nothing`: Variable type config
  (default: 2 continuous, 2 binary, 1 bounded)
- `n_baseline_features::Int=0`: Number of baseline covariates
- `noise_level::Float32=0.1f0`: Noise level
- `seed::Union{Int,Nothing}=nothing`: Random seed

# Returns
- `data::Array{Float32,3}`: Data of shape (n_samples, seq_len, n_features)
- `baseline::Union{Array{Float32,2},Nothing}`: Baseline covariates (n_samples, n_baseline) or nothing
"""
function generate_mixed_longitudinal_data(; n_samples::Int=1000, seq_len::Int=50,
                                           var_config::Union{VariableConfig,Nothing}=nothing,
                                           n_baseline_features::Int=0,
                                           noise_level::Float32=0.1f0,
                                           seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    if isnothing(var_config)
        var_config = VariableConfig([
            VariableSpec("continuous_1", "continuous"),
            VariableSpec("continuous_2", "continuous"),
            VariableSpec("binary_1", "binary"),
            VariableSpec("binary_2", "binary"),
            VariableSpec("bounded_1", "bounded"; lower=0.0f0, upper=1.0f0),
        ])
    end

    nf = n_features(var_config)
    data = zeros(Float32, n_samples, seq_len, nf)

    for i in 1:n_samples
        t = collect(range(0, 4π, length=seq_len))

        for (j, vs) in enumerate(var_config.variables)
            # Generate a latent smooth trajectory
            trend = randn(Float32) .* t ./ Float32(4π)
            seasonality = sin.(t .+ rand(Float32) * Float32(2π)) .* rand(Float32)
            noise = randn(Float32, seq_len) .* noise_level
            latent = trend .+ seasonality .+ noise

            if vs.var_type == "continuous"
                data[i, :, j] = latent

            elseif vs.var_type == "binary"
                # Sigmoid of latent, then threshold at 0.5
                prob = 1.0f0 ./ (1.0f0 .+ exp.(-latent))
                data[i, :, j] = Float32.(rand(Float32, seq_len) .< prob)

            elseif vs.var_type == "bounded"
                # Sigmoid to [0, 1], then scale to [lower, upper]
                sig = 1.0f0 ./ (1.0f0 .+ exp.(-latent))
                data[i, :, j] = sig .* (vs.upper - vs.lower) .+ vs.lower
            end
        end
    end

    # Generate baseline covariates
    baseline = nothing
    if n_baseline_features > 0
        baseline = zeros(Float32, n_samples, n_baseline_features)
        for j in 1:n_baseline_features
            if j % 2 == 1  # odd indices (1-based)
                # Continuous baseline
                baseline[:, j] = randn(Float32, n_samples)
            else
                # Binary baseline
                baseline[:, j] = Float32.(rand(Float32, n_samples) .> 0.5f0)
            end
        end
    end

    return data, baseline
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
