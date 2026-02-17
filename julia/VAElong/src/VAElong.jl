"""
VAElong - Variational Autoencoder for Longitudinal Measurements

A Julia package for training Variational Autoencoders on longitudinal/time-series data
with support for missing data handling, mixed variable types (continuous, binary, bounded),
baseline covariates, and landmark prediction.
"""
module VAElong

using Flux
using Statistics
using Random
using LinearAlgebra

# Include source files (config must come first, it defines types used by others)
include("config.jl")
include("model.jl")
include("data.jl")
include("trainer.jl")

# Export config types and functions
export VariableSpec, VariableConfig
export n_features, continuous_indices, binary_indices, bounded_indices
export get_bounds, all_continuous

# Export model types
export LongitudinalVAE, CNNLongitudinalVAE

# Export model functions
export encode, decode, reparameterize, sample, impute_missing
export predict_from_landmark
export vae_loss, mixed_vae_loss

# Export data types and functions
export LongitudinalDataset, generate_synthetic_longitudinal_data
export generate_mixed_longitudinal_data, create_missing_mask
export inverse_transform

# Export trainer types and functions
export VAETrainer, train_epoch!, validate, fit!, create_data_loader

end # module
