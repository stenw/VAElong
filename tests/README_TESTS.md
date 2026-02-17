# Unit Tests Summary

This document summarizes the unit tests for VAElong and their coverage.

## Test Files

### 1. `test_model.py` — LSTM/GRU Model Tests

**TestLongitudinalVAE** (10 tests)
- Model initialization
- Encoder output shapes
- Encoder with mask
- Reparameterization trick
- Decoder output shapes
- Forward pass shapes
- Forward pass with mask
- Sampling from model
- GRU mode functionality
- Multi-layer RNN support

**TestVAELoss** (7 tests)
- Loss computation
- Loss positivity
- Beta parameter effects
- Perfect reconstruction (zero loss)
- Loss with missing data mask
- Loss with all values missing
- Mask shape validation

### 2. `test_cnn_model.py` — CNN Model and Missing Data Tests

**TestCNNLongitudinalVAE** (8 tests)
- CNN model creation
- Encoder with and without mask
- Decoder output shapes
- Forward pass with and without mask
- Sampling from CNN model
- Missing data imputation

**TestMissingDataUtilities** (4 tests)
- Random, block, and monotone mask creation
- VAE loss with mask

### 3. `test_data.py` — Data Utilities Tests

**TestLongitudinalDataset** (7 tests)
- Dataset initialization
- Data normalization
- Getting items from dataset (4-tuple)
- Variable length sequences
- Inverse transformation
- Dataset with missing data mask
- Normalization with missing data

**TestSyntheticDataGeneration** (4 tests)
- Generated data shape
- Deterministic generation with seed
- Different seeds produce different data
- Noise level affects variability

**TestMissingMaskCreation** (9 tests)
- Random mask shape, values, and missing rate accuracy
- Block and monotone mask creation
- Deterministic mask generation
- Invalid pattern error handling
- Zero and high missing rate edge cases

### 4. `test_trainer.py` — Training Pipeline Tests

**TestVAETrainer** (11 tests)
- Trainer initialization
- Training for one epoch
- Validation
- Fitting with and without validation
- Save and load model
- Beta parameter
- Training with missing data
- Training and fitting with EM imputation

**TestCNNVAETrainer** (3 tests)
- CNN train epoch, fitting, and missing data with EM imputation

### 5. `test_mixed_types.py` — Mixed-Type, Baseline, and Landmark Tests

**TestVariableConfig** (6 tests)
- VariableSpec creation and validation
- Invalid type and bounds errors
- Index properties (continuous, binary, bounded)
- `all_continuous` factory
- `get_bounds` method

**TestMixedSyntheticData** (4 tests)
- Default and custom config generation
- Binary values are 0/1, bounded values within bounds
- Baseline covariate generation
- Deterministic generation

**TestMixedDataset** (5 tests)
- Dataset with var_config and baseline covariates
- Type-aware normalization (z-score continuous, affine bounded, binary unchanged)
- Type-aware inverse transform
- Dataset with mask and mixed types

**TestMixedLossFunction** (3 tests)
- Fallback to standard loss when var_config is None
- Mixed loss computes with all types
- Mixed loss with missing data mask

**TestMixedTypeModel** (6 tests)
- LSTM and CNN models with var_config (output activations)
- LSTM and CNN models with baseline conditioning
- Backward compatibility without var_config

**TestLandmarkPrediction** (3 tests)
- LSTM landmark prediction (different observed vs total length)
- LSTM landmark with baseline
- CNN landmark prediction

**TestMixedTrainerIntegration** (3 tests)
- Full training pipeline with mixed types and baselines
- Training with missing data and mixed types + EM imputation
- CNN training pipeline with mixed types

## Total Test Count

- **Total**: 92 tests
- **Model tests** (`test_model.py`): 17
- **CNN model tests** (`test_cnn_model.py`): 12
- **Data tests** (`test_data.py`): 20
- **Trainer tests** (`test_trainer.py`): 14
- **Mixed-type tests** (`test_mixed_types.py`): 30

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests.test_mixed_types

# Run a specific test class
python -m unittest tests.test_mixed_types.TestLandmarkPrediction

# Run a specific test method
python -m unittest tests.test_mixed_types.TestLandmarkPrediction.test_lstm_landmark_prediction
```
