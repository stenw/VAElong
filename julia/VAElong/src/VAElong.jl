"""
VAElong - Variational Autoencoder for Longitudinal Measurements

A Julia package for training Variational Autoencoders on longitudinal/time-series data
with support for missing data handling.
"""
module VAElong

using Flux
using Statistics
using Random
using LinearAlgebra

# Include source files
include("model.jl")
include("data.jl")
include("trainer.jl")

# Export model types
export LongitudinalVAE, CNNLongitudinalVAE

# Export model functions
export encode, decode, reparameterize, sample, impute_missing, vae_loss

# Export data types and functions
export LongitudinalDataset, generate_synthetic_longitudinal_data, create_missing_mask
export inverse_transform

# Export trainer types and functions
export VAETrainer, train_epoch!, validate, fit!, create_data_loader

end # module
