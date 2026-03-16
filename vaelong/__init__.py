"""
VAElong - Variational Autoencoder for Longitudinal Measurements
"""

from .config import VariableConfig, VariableSpec
from .model import (
    LongitudinalVAE, CNNLongitudinalVAE,
    TPCNNLongitudinalVAE, TransformerLongitudinalVAE,
    vae_loss_function, mixed_vae_loss_function,
)
from .trainer import VAETrainer
from .data import LongitudinalDataset, create_missing_mask, generate_mixed_longitudinal_data

__version__ = '0.2.0'
__all__ = [
    'VariableConfig', 'VariableSpec',
    'LongitudinalVAE', 'CNNLongitudinalVAE',
    'TPCNNLongitudinalVAE', 'TransformerLongitudinalVAE',
    'vae_loss_function', 'mixed_vae_loss_function',
    'VAETrainer',
    'LongitudinalDataset', 'create_missing_mask',
    'generate_mixed_longitudinal_data',
]
