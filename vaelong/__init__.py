"""
VAElong - Variational Autoencoder for Longitudinal Measurements
"""

from .config import VariableConfig, VariableSpec
from .model import (
    LongitudinalVAE, CNNLongitudinalVAE,
    TPCNNLongitudinalVAE, TransformerLongitudinalVAE,
    vae_loss_function, mixed_vae_loss_function,
)
from .joint_model import JointLongitudinalSurvivalVAE
from .joint_trainer import JointVAETrainer
from .trainer import VAETrainer
from .data import (
    LongitudinalDataset,
    JointLongitudinalSurvivalDataset,
    align_time_varying_covariates_to_grid,
    build_joint_dataset_inputs,
    create_missing_mask,
    generate_mixed_longitudinal_data,
    generate_synthetic_joint_longitudinal_survival_data,
    split_joint_tables_by_fold,
)

__version__ = '0.2.0'
__all__ = [
    'VariableConfig', 'VariableSpec',
    'LongitudinalVAE', 'CNNLongitudinalVAE',
    'TPCNNLongitudinalVAE', 'TransformerLongitudinalVAE',
    'JointLongitudinalSurvivalVAE',
    'vae_loss_function', 'mixed_vae_loss_function',
    'VAETrainer', 'JointVAETrainer',
    'LongitudinalDataset', 'JointLongitudinalSurvivalDataset',
    'align_time_varying_covariates_to_grid',
    'build_joint_dataset_inputs',
    'create_missing_mask',
    'generate_mixed_longitudinal_data',
    'generate_synthetic_joint_longitudinal_survival_data',
    'split_joint_tables_by_fold',
]
