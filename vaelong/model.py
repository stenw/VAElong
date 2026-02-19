"""
Variational Autoencoder model for longitudinal data.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongitudinalVAE(nn.Module):
    """
    Variational Autoencoder for longitudinal (time-series) data.

    Uses LSTM/GRU layers to handle sequential nature of longitudinal measurements.
    Supports mixed variable types (continuous, binary, bounded) and baseline covariates.

    Args:
        input_dim (int): Dimension of input features at each time step
        hidden_dim (int): Dimension of LSTM hidden state
        latent_dim (int): Dimension of latent space
        num_layers (int): Number of LSTM layers (default: 1)
        use_gru (bool): Use GRU instead of LSTM (default: False)
        n_baseline (int): Number of baseline covariate features (default: 0)
        var_config (VariableConfig): Variable type configuration (default: None, all continuous)
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=20, num_layers=1,
                 use_gru=False, n_baseline=0, var_config=None):
        super(LongitudinalVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.n_baseline = n_baseline
        self.var_config = var_config

        # Encoder: LSTM/GRU + linear layers for mean and log variance
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # mu/logvar input size includes baseline covariates
        self.fc_mu = nn.Linear(hidden_dim + n_baseline, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + n_baseline, latent_dim)

        # Decoder: linear layer + LSTM/GRU with sinusoidal time embeddings
        self.time_emb_dim = min(hidden_dim, 16)
        self.fc_latent = nn.Linear(latent_dim + n_baseline, hidden_dim)
        self.decoder_rnn = rnn_class(
            hidden_dim + self.time_emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)

        # Learned observation log-variance for continuous variables.
        # One parameter per continuous feature, learned during training.
        # This puts continuous NLL on the same scale as binary/bounded BCE.
        n_cont = len(var_config.continuous_indices) if var_config is not None else input_dim
        self.log_noise_var = nn.Parameter(torch.zeros(n_cont))

    def encode(self, x, mask=None, baseline=None):
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask (batch_size, seq_len, input_dim)
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        if mask is not None:
            x = x * mask

        # Pass through RNN and take the last hidden state
        _, hidden = self.encoder_rnn(x)

        # Handle LSTM vs GRU output
        if self.use_gru:
            h = hidden[-1]  # Take last layer's hidden state
        else:
            h = hidden[0][-1]  # Take last layer's hidden state (h, not c)

        # Concatenate baseline covariates
        if baseline is not None and self.n_baseline > 0:
            h = torch.cat([h, baseline], dim=-1)

        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            z: Sample from latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def _sinusoidal_embedding(self, seq_len, device):
        """Fixed sinusoidal positional encoding (Transformer-style).

        Returns:
            emb: (1, seq_len, time_emb_dim) positional embeddings
        """
        d = self.time_emb_dim
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d)
        )
        emb = torch.zeros(seq_len, d, device=device)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term[:d // 2])
        return emb.unsqueeze(0)  # (1, seq_len, d)

    def decode(self, z, seq_len, baseline=None):
        """
        Decode latent representation to output sequence.

        Args:
            z: Latent representation (batch_size, latent_dim)
            seq_len: Length of output sequence
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            output: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)

        # Concatenate baseline covariates to latent
        if baseline is not None and self.n_baseline > 0:
            z_cond = torch.cat([z, baseline], dim=-1)
        else:
            z_cond = z

        # Transform latent to hidden dimension
        h = self.fc_latent(z_cond)
        h = torch.relu(h)

        # Repeat for each time step and concatenate sinusoidal time embeddings
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)
        time_emb = self._sinusoidal_embedding(seq_len, h.device)
        time_emb = time_emb.expand(batch_size, -1, -1)
        decoder_input = torch.cat([h_repeated, time_emb], dim=-1)

        # Pass through RNN
        rnn_out, _ = self.decoder_rnn(decoder_input)

        # Generate output
        output = self.fc_output(rnn_out)

        # Apply type-specific activations
        output = self._apply_output_activations(output)

        return output

    def _apply_output_activations(self, output):
        """Apply per-variable-type activations to the decoder output."""
        if self.var_config is None:
            return output  # all continuous, raw output

        for idx in self.var_config.binary_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])

        for idx in self.var_config.bounded_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])

        return output

    def forward(self, x, mask=None, baseline=None):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask for missing data
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            recon_x: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        seq_len = x.size(1)

        # Encode
        mu, logvar = self.encode(x, mask, baseline)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decode(z, seq_len, baseline)

        return recon_x, mu, logvar

    def sample(self, num_samples, seq_len, device='cpu', baseline=None):
        """
        Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences to generate
            device: Device to generate samples on
            baseline: Optional baseline covariates (num_samples, n_baseline)

        Returns:
            samples: Generated samples (num_samples, seq_len, input_dim)
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Decode
            samples = self.decode(z, seq_len, baseline)

        return samples

    def predict_from_landmark(self, x_observed, mask_observed, total_seq_len,
                               baseline=None):
        """
        Landmark prediction: encode observed data, decode the full sequence.

        Given data observed up to a landmark time point, predict the full
        trajectory including future time steps.

        Args:
            x_observed: (batch, observed_len, input_dim) data observed so far
            mask_observed: (batch, observed_len, input_dim) mask for observed data
            total_seq_len: int, total sequence length to predict
            baseline: optional (batch, n_baseline) baseline covariates

        Returns:
            predicted: (batch, total_seq_len, input_dim) full predicted trajectory
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x_observed, mask_observed, baseline)
            # Use mean for deterministic prediction
            predicted = self.decode(mu, total_seq_len, baseline)
        return predicted


class CNNLongitudinalVAE(nn.Module):
    """
    CNN-based Variational Autoencoder for longitudinal (time-series) data with missing data handling.

    Uses 1D convolutional layers to process sequential data and supports missing value imputation
    through an EM-like approach. Supports mixed variable types and baseline covariates.

    Args:
        input_dim (int): Dimension of input features at each time step
        seq_len (int): Expected sequence length
        latent_dim (int): Dimension of latent space
        hidden_channels (list): List of channel sizes for encoder convolutions (default: [32, 64, 128])
        kernel_size (int): Kernel size for convolutions (default: 3)
        n_baseline (int): Number of baseline covariate features (default: 0)
        var_config (VariableConfig): Variable type configuration (default: None, all continuous)
    """

    def __init__(self, input_dim, seq_len, latent_dim=20, hidden_channels=None, kernel_size=3,
                 n_baseline=0, var_config=None):
        super(CNNLongitudinalVAE, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.n_baseline = n_baseline
        self.var_config = var_config

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.hidden_channels = hidden_channels

        # Calculate the size after convolutions
        self._build_encoder()
        self._build_decoder()

        # Learned observation log-variance for continuous variables
        n_cont = len(var_config.continuous_indices) if var_config is not None else input_dim
        self.log_noise_var = nn.Parameter(torch.zeros(n_cont))

    def _build_encoder(self):
        """Build the encoder convolutional layers."""
        layers = []
        in_channels = self.input_dim

        for out_channels in self.hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size,
                         stride=2, padding=self.kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, self.seq_len)
            dummy_output = self.encoder(dummy_input)
            self.encoded_size = dummy_output.numel()

        # Latent space layers (input includes baseline covariates)
        self.fc_mu = nn.Linear(self.encoded_size + self.n_baseline, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size + self.n_baseline, self.latent_dim)

    def _build_decoder(self):
        """Build the decoder deconvolutional layers."""
        # Map from latent (+ baseline) to encoded size
        self.fc_decode = nn.Linear(self.latent_dim + self.n_baseline, self.encoded_size)

        # Calculate the shape before decoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, self.seq_len)
            dummy_encoded = self.encoder(dummy_input)
            self.encoded_channels = dummy_encoded.shape[1]
            self.encoded_length = dummy_encoded.shape[2]

        # Decoder layers (reverse of encoder)
        layers = []
        channels = list(reversed(self.hidden_channels))

        for i, out_channels in enumerate(channels[1:] + [self.input_dim]):
            in_channels = channels[i]
            # Use ConvTranspose1d for upsampling
            layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel_size,
                                 stride=2, padding=self.kernel_size//2, output_padding=1),
                nn.BatchNorm1d(out_channels) if out_channels != self.input_dim else nn.Identity(),
                nn.ReLU(inplace=True) if out_channels != self.input_dim else nn.Identity()
            ])

        self.decoder = nn.Sequential(*layers)

    def encode(self, x, mask=None, baseline=None):
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask of shape (batch_size, seq_len, input_dim)
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        batch_size = x.size(0)

        # Apply mask if provided (zero out missing values)
        if mask is not None:
            x = x * mask

        # Reshape for Conv1d: (batch, features, time)
        x = x.transpose(1, 2)

        # Encode
        h = self.encoder(x)
        h = h.view(batch_size, -1)

        # Concatenate baseline covariates
        if baseline is not None and self.n_baseline > 0:
            h = torch.cat([h, baseline], dim=-1)

        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            z: Sample from latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, baseline=None):
        """
        Decode latent representation to output sequence.

        Args:
            z: Latent representation (batch_size, latent_dim)
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            output: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)

        # Concatenate baseline covariates to latent
        if baseline is not None and self.n_baseline > 0:
            z_cond = torch.cat([z, baseline], dim=-1)
        else:
            z_cond = z

        # Map to encoded size
        h = self.fc_decode(z_cond)
        h = torch.relu(h)

        # Reshape to encoded dimensions
        h = h.view(batch_size, self.encoded_channels, self.encoded_length)

        # Decode
        output = self.decoder(h)

        # Crop or pad to match original sequence length
        if output.size(2) != self.seq_len:
            if output.size(2) > self.seq_len:
                output = output[:, :, :self.seq_len]
            else:
                padding = self.seq_len - output.size(2)
                output = F.pad(output, (0, padding))

        # Reshape back: (batch, time, features)
        output = output.transpose(1, 2)

        # Apply type-specific activations
        output = self._apply_output_activations(output)

        return output

    def _apply_output_activations(self, output):
        """Apply per-variable-type activations to the decoder output."""
        if self.var_config is None:
            return output  # all continuous, raw output

        for idx in self.var_config.binary_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])

        for idx in self.var_config.bounded_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])

        return output

    def forward(self, x, mask=None, baseline=None):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask for missing data
            baseline: Optional baseline covariates (batch_size, n_baseline)

        Returns:
            recon_x: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        mu, logvar = self.encode(x, mask, baseline)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decode(z, baseline)

        return recon_x, mu, logvar

    def sample(self, num_samples, device='cpu', baseline=None):
        """
        Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            baseline: Optional baseline covariates (num_samples, n_baseline)

        Returns:
            samples: Generated samples (num_samples, seq_len, input_dim)
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Decode
            samples = self.decode(z, baseline)

        return samples

    def predict_from_landmark(self, x_observed, mask_observed, baseline=None):
        """
        Landmark prediction: encode observed data, decode the full sequence.

        For CNN model, x_observed should be padded to seq_len with zeros,
        and mask_observed should indicate which time steps are observed.

        Args:
            x_observed: (batch, seq_len, input_dim) data with future values zeroed out
            mask_observed: (batch, seq_len, input_dim) mask (1 for observed, 0 for future)
            baseline: optional (batch, n_baseline) baseline covariates

        Returns:
            predicted: (batch, seq_len, input_dim) full predicted trajectory
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x_observed, mask_observed, baseline)
            # Use mean for deterministic prediction
            predicted = self.decode(mu, baseline)
        return predicted

    def impute_missing(self, x, mask, num_iterations=5, baseline=None, noise_scale=0.1):
        """
        Impute missing values using iterative EM-like approach.

        Args:
            x: Input with missing values (batch_size, seq_len, input_dim)
            mask: Binary mask (1=observed, 0=missing)
            num_iterations: Number of EM iterations
            baseline: Optional baseline covariates (batch_size, n_baseline)
            noise_scale: Scaling factor for noise during imputation (default: 0.1)

        Returns:
            imputed: Data with imputed values
        """
        self.eval()
        imputed = x.clone()

        with torch.no_grad():
            for iteration in range(num_iterations):
                # E-step: Generate predictions for missing values
                recon_x, mu, logvar = self.forward(imputed, mask, baseline)

                # Add small noise for uncertainty
                noise = torch.randn_like(recon_x) * noise_scale
                sampled_recon = recon_x + noise

                # Type-aware post-processing of imputed values
                if self.var_config is not None:
                    for idx in self.var_config.binary_indices:
                        sampled_recon[:, :, idx] = (sampled_recon[:, :, idx] > 0.5).float()
                    for idx in self.var_config.bounded_indices:
                        sampled_recon[:, :, idx] = sampled_recon[:, :, idx].clamp(0, 1)

                # Update missing values with sampled predictions
                imputed = mask * x + (1 - mask) * sampled_recon

        return imputed


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, mask=None):
    """
    VAE loss = Reconstruction loss + KL divergence

    Supports missing data through optional masking.

    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (default: 1.0, can be < 1 for beta-VAE)
        mask: Optional binary mask for missing data (1=observed, 0=missing)

    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss
    """
    # Apply mask if provided
    if mask is not None:
        # Only compute reconstruction loss on observed values
        diff = (recon_x - x) ** 2
        recon_loss = (diff * mask).sum()
        # Normalize by number of observed values
        n_observed = mask.sum()
        if n_observed > 0:
            recon_loss = recon_loss / n_observed * mask.numel()
    else:
        # Standard reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + beta * kld_loss

    return loss, recon_loss, kld_loss


def _masked_sum(values, mask):
    """Sum values where mask=1, normalized by observation count."""
    n_observed = mask.sum()
    if n_observed > 0:
        return (values * mask).sum() / n_observed * mask.numel()
    return torch.tensor(0.0, device=values.device)


def mixed_vae_loss_function(recon_x, x, mu, logvar, beta=1.0, mask=None,
                            var_config=None, log_noise_var=None):
    """
    VAE loss supporting mixed variable types with learned observation noise.

    Computes proper negative log-likelihoods so that all variable types
    are on a comparable scale:
    - Continuous: Gaussian NLL with learned per-variable variance
      0.5 * (log σ² + (x - μ)² / σ²)   — automatically down-weights noisy variables
    - Binary: BCE (Bernoulli NLL)
    - Bounded: BCE on [0,1]-normalised data (Beta-like NLL)

    Falls back to pure MSE if var_config is None (backward compatible).

    Args:
        recon_x: Reconstructed data (batch, seq_len, n_features)
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
        mask: Optional binary mask for missing data (1=observed, 0=missing)
        var_config: Optional VariableConfig for mixed types
        log_noise_var: Optional learned log-variance for continuous variables,
            shape (n_continuous,). If None, falls back to MSE (σ²=1).

    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss
    """
    if var_config is None:
        return vae_loss_function(recon_x, x, mu, logvar, beta, mask)

    recon_loss = torch.tensor(0.0, device=recon_x.device)

    # Continuous variables: heteroscedastic Gaussian NLL
    cont_idx = var_config.continuous_indices
    if cont_idx:
        cont_recon = recon_x[:, :, cont_idx]
        cont_x = x[:, :, cont_idx]

        if log_noise_var is not None:
            # Proper Gaussian NLL: 0.5 * (log σ² + (x - μ)² / σ²)
            # Clamp to [-4, 2] ≈ [σ²=0.018, σ²=7.4] to prevent collapse.
            # On normalized data σ² should stay near 1 (log_noise_var ≈ 0).
            lnv = log_noise_var.clamp(-4.0, 2.0).view(1, 1, -1)
            nll = 0.5 * (lnv + (cont_recon - cont_x) ** 2 / lnv.exp())
        else:
            # Fallback: MSE (equivalent to σ²=1, dropping constant)
            nll = (cont_recon - cont_x) ** 2

        if mask is not None:
            cont_mask = mask[:, :, cont_idx]
            recon_loss = recon_loss + _masked_sum(nll, cont_mask)
        else:
            recon_loss = recon_loss + nll.sum()

        # L2 penalty on log_noise_var to anchor near σ²=1 and prevent drift
        if log_noise_var is not None:
            recon_loss = recon_loss + 10.0 * (log_noise_var ** 2).sum()

    # Binary variables: BCE (Bernoulli NLL)
    bin_idx = var_config.binary_indices
    if bin_idx:
        bin_recon = recon_x[:, :, bin_idx].clamp(1e-7, 1 - 1e-7)
        bin_x = x[:, :, bin_idx]
        if mask is not None:
            bin_mask = mask[:, :, bin_idx]
            bce = F.binary_cross_entropy(bin_recon, bin_x, reduction='none')
            recon_loss = recon_loss + _masked_sum(bce, bin_mask)
        else:
            recon_loss = recon_loss + F.binary_cross_entropy(bin_recon, bin_x, reduction='sum')

    # Bounded variables: BCE on [0,1]-normalised data
    bnd_idx = var_config.bounded_indices
    if bnd_idx:
        bnd_recon = recon_x[:, :, bnd_idx].clamp(1e-7, 1 - 1e-7)
        bnd_x = x[:, :, bnd_idx].clamp(0, 1)
        if mask is not None:
            bnd_mask = mask[:, :, bnd_idx]
            bce = F.binary_cross_entropy(bnd_recon, bnd_x, reduction='none')
            recon_loss = recon_loss + _masked_sum(bce, bnd_mask)
        else:
            recon_loss = recon_loss + F.binary_cross_entropy(bnd_recon, bnd_x, reduction='sum')

    # KL divergence (unchanged)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + beta * kld_loss
    return loss, recon_loss, kld_loss


# ---------------------------------------------------------------------------
# Time-Parameterized Convolution helpers
# ---------------------------------------------------------------------------

class TPConv1d(nn.Module):
    """1-D convolution whose kernel is generated by an MLP from time offsets.

    For regularly-spaced data the default offsets ``[-K//2, …, K//2]`` are used.
    For irregular spacing, pass custom ``offsets`` to :meth:`forward`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Convolution stride (default 2).
        padding: Padding (default ``kernel_size // 2``).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2

        # MLP that maps offsets → kernel weights
        kernel_total = out_channels * in_channels * kernel_size
        self.kernel_mlp = nn.Sequential(
            nn.Linear(kernel_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, kernel_total),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Default offsets for regular spacing (centred around 0)
        offsets = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer('default_offsets', offsets)

    def forward(self, x, offsets=None):
        """
        Args:
            x: (batch, in_channels, length)
            offsets: Optional (kernel_size,) time offsets for irregular spacing.
        Returns:
            (batch, out_channels, new_length)
        """
        if offsets is None:
            offsets = self.default_offsets
        kernel_flat = self.kernel_mlp(offsets.unsqueeze(0))  # (1, total)
        kernel = kernel_flat.view(self.out_channels, self.in_channels, self.kernel_size)
        return F.conv1d(x, kernel, bias=self.bias, stride=self.stride, padding=self.padding)


class TPConvTranspose1d(nn.Module):
    """Transposed 1-D convolution with MLP-generated kernel (decoder counterpart).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
        stride: Stride (default 2).
        padding: Padding (default ``kernel_size // 2``).
        output_padding: Output padding (default 1).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
                 padding=None, output_padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.output_padding = output_padding

        kernel_total = in_channels * out_channels * kernel_size
        self.kernel_mlp = nn.Sequential(
            nn.Linear(kernel_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, kernel_total),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        offsets = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer('default_offsets', offsets)

    def forward(self, x, offsets=None):
        """
        Args:
            x: (batch, in_channels, length)
            offsets: Optional (kernel_size,) time offsets.
        Returns:
            (batch, out_channels, new_length)
        """
        if offsets is None:
            offsets = self.default_offsets
        kernel_flat = self.kernel_mlp(offsets.unsqueeze(0))
        kernel = kernel_flat.view(self.in_channels, self.out_channels, self.kernel_size)
        return F.conv_transpose1d(
            x, kernel, bias=self.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
        )


# ---------------------------------------------------------------------------
# TPCNN VAE
# ---------------------------------------------------------------------------

class TPCNNLongitudinalVAE(nn.Module):
    """
    Time-Parameterized CNN VAE for longitudinal data.

    Convolution kernels are generated on-the-fly by small MLPs that take
    relative time offsets as input, enabling the model to handle irregularly-
    sampled time series.  For regularly-spaced data the default integer offsets
    are used and the model reduces to a standard CNN with weight-sharing via
    the MLP.

    Constructor signature is identical to :class:`CNNLongitudinalVAE` (drop-in
    replacement).

    Args:
        input_dim (int): Features per time step.
        seq_len (int): Fixed sequence length.
        latent_dim (int): Latent space dimension (default 20).
        hidden_channels (list): Channel sizes for encoder (default [32, 64, 128]).
        kernel_size (int): Kernel size (default 3).
        n_baseline (int): Baseline covariate features (default 0).
        var_config (VariableConfig): Variable type configuration (default None).
    """

    def __init__(self, input_dim, seq_len, latent_dim=20, hidden_channels=None,
                 kernel_size=3, n_baseline=0, var_config=None):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.n_baseline = n_baseline
        self.var_config = var_config

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.hidden_channels = hidden_channels

        self._build_encoder()
        self._build_decoder()

        n_cont = len(var_config.continuous_indices) if var_config is not None else input_dim
        self.log_noise_var = nn.Parameter(torch.zeros(n_cont))

    # -- build helpers -------------------------------------------------------

    def _build_encoder(self):
        layers = nn.ModuleList()
        bn_layers = nn.ModuleList()
        in_ch = self.input_dim
        for out_ch in self.hidden_channels:
            layers.append(TPConv1d(in_ch, out_ch, self.kernel_size, stride=2))
            bn_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.encoder_tp_layers = layers
        self.encoder_bn_layers = bn_layers

        # Compute encoded size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, self.seq_len)
            for tp, bn in zip(self.encoder_tp_layers, self.encoder_bn_layers):
                dummy = F.relu(bn(tp(dummy)))
            self.encoded_channels = dummy.shape[1]
            self.encoded_length = dummy.shape[2]
            self.encoded_size = dummy.numel()

        self.fc_mu = nn.Linear(self.encoded_size + self.n_baseline, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size + self.n_baseline, self.latent_dim)

    def _build_decoder(self):
        self.fc_decode = nn.Linear(self.latent_dim + self.n_baseline, self.encoded_size)

        layers = nn.ModuleList()
        bn_layers = nn.ModuleList()
        channels = list(reversed(self.hidden_channels))

        for i, out_ch in enumerate(channels[1:] + [self.input_dim]):
            in_ch = channels[i]
            is_last = (out_ch == self.input_dim)
            layers.append(TPConvTranspose1d(
                in_ch, out_ch, self.kernel_size, stride=2,
                output_padding=1,
            ))
            bn_layers.append(nn.Identity() if is_last else nn.BatchNorm1d(out_ch))
        self.decoder_tp_layers = layers
        self.decoder_bn_layers = bn_layers

    # -- core methods --------------------------------------------------------

    def encode(self, x, mask=None, baseline=None):
        batch_size = x.size(0)
        if mask is not None:
            x = x * mask
        # (batch, seq, feat) -> (batch, feat, seq)
        h = x.transpose(1, 2)
        for tp, bn in zip(self.encoder_tp_layers, self.encoder_bn_layers):
            h = F.relu(bn(tp(h)))
        h = h.view(batch_size, -1)
        if baseline is not None and self.n_baseline > 0:
            h = torch.cat([h, baseline], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, baseline=None):
        batch_size = z.size(0)
        if baseline is not None and self.n_baseline > 0:
            z = torch.cat([z, baseline], dim=-1)
        h = torch.relu(self.fc_decode(z))
        h = h.view(batch_size, self.encoded_channels, self.encoded_length)
        for i, (tp, bn) in enumerate(zip(self.decoder_tp_layers, self.decoder_bn_layers)):
            h = tp(h)
            is_last = (i == len(self.decoder_tp_layers) - 1)
            if not is_last:
                h = F.relu(bn(h))
        # Crop / pad to seq_len
        if h.size(2) > self.seq_len:
            h = h[:, :, :self.seq_len]
        elif h.size(2) < self.seq_len:
            h = F.pad(h, (0, self.seq_len - h.size(2)))
        output = h.transpose(1, 2)  # (batch, seq_len, input_dim)
        return self._apply_output_activations(output)

    def _apply_output_activations(self, output):
        if self.var_config is None:
            return output
        for idx in self.var_config.binary_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])
        for idx in self.var_config.bounded_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])
        return output

    def forward(self, x, mask=None, baseline=None):
        mu, logvar = self.encode(x, mask, baseline)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, baseline)
        return recon_x, mu, logvar

    def sample(self, num_samples, device='cpu', baseline=None):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            return self.decode(z, baseline)

    def predict_from_landmark(self, x_observed, mask_observed, baseline=None):
        """Landmark prediction (CNN-style: input pre-padded to seq_len)."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x_observed, mask_observed, baseline)
            return self.decode(mu, baseline)


# ---------------------------------------------------------------------------
# Transformer VAE
# ---------------------------------------------------------------------------

class TransformerLongitudinalVAE(nn.Module):
    """
    Transformer-based VAE for longitudinal data.

    Uses an encoder-only Transformer (bidirectional self-attention) for both
    the VAE encoder and decoder pathways.  The encoder pools the attended
    sequence into a latent distribution; the decoder broadcasts the sampled
    latent across all time positions and refines via self-attention.

    Args:
        input_dim (int): Features per time step.
        seq_len (int): Fixed sequence length.
        latent_dim (int): Latent space dimension (default 20).
        d_model (int): Transformer hidden dimension (default 64).
        nhead (int): Number of attention heads (default 4).
        num_layers (int): Transformer layers for encoder & decoder (default 2).
        dim_feedforward (int): Feed-forward hidden size (default 128).
        dropout (float): Dropout rate (default 0.1).
        n_baseline (int): Baseline covariate features (default 0).
        var_config (VariableConfig): Variable type configuration (default None).
    """

    def __init__(self, input_dim, seq_len, latent_dim=20, d_model=64,
                 nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, n_baseline=0, var_config=None):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.n_baseline = n_baseline
        self.var_config = var_config

        # --- Encoder pathway ---
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.fc_mu = nn.Linear(d_model + n_baseline, latent_dim)
        self.fc_logvar = nn.Linear(d_model + n_baseline, latent_dim)

        # --- Decoder pathway ---
        self.fc_latent = nn.Linear(latent_dim + n_baseline, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers,
        )
        self.fc_output = nn.Linear(d_model, input_dim)

        # Learned observation noise variance
        n_cont = len(var_config.continuous_indices) if var_config is not None else input_dim
        self.log_noise_var = nn.Parameter(torch.zeros(n_cont))

    # -- helpers -------------------------------------------------------------

    def _sinusoidal_embedding(self, seq_len, device):
        """Fixed sinusoidal positional encoding of size d_model."""
        d = self.d_model
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d)
        )
        emb = torch.zeros(seq_len, d, device=device)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term[:d // 2])
        return emb.unsqueeze(0)  # (1, seq_len, d_model)

    def _sequence_mask(self, mask):
        """Derive per-time-step mask from per-feature mask.

        Args:
            mask: (batch, seq_len, n_features) with 1=observed, 0=missing.
        Returns:
            key_padding_mask: (batch, seq_len) bool with True=ignore (PyTorch convention).
        """
        # A time step is "observed" if any feature is observed
        observed = mask.any(dim=-1)  # (batch, seq_len)
        return ~observed  # True = ignore

    def _apply_output_activations(self, output):
        if self.var_config is None:
            return output
        for idx in self.var_config.binary_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])
        for idx in self.var_config.bounded_indices:
            output = output.clone()
            output[:, :, idx] = torch.sigmoid(output[:, :, idx])
        return output

    # -- core methods --------------------------------------------------------

    def encode(self, x, mask=None, baseline=None):
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional (batch, seq_len, input_dim) binary mask.
            baseline: Optional (batch, n_baseline).

        Returns:
            mu, logvar: each (batch, latent_dim)
        """
        if mask is not None:
            x = x * mask

        # Project and add positional encoding
        h = self.input_projection(x)  # (batch, seq, d_model)
        pe = self._sinusoidal_embedding(x.size(1), x.device)
        h = h + pe

        # Build attention padding mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = self._sequence_mask(mask)

        h = self.encoder_transformer(h, src_key_padding_mask=key_padding_mask)

        # Masked mean pooling over time
        if mask is not None:
            observed = mask.any(dim=-1, keepdim=True).float()  # (batch, seq, 1)
            h_sum = (h * observed).sum(dim=1)
            n_obs = observed.sum(dim=1).clamp(min=1)
            h_pooled = h_sum / n_obs  # (batch, d_model)
        else:
            h_pooled = h.mean(dim=1)

        if baseline is not None and self.n_baseline > 0:
            h_pooled = torch.cat([h_pooled, baseline], dim=-1)

        return self.fc_mu(h_pooled), self.fc_logvar(h_pooled)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, baseline=None):
        """
        Decode latent to output sequence.

        Args:
            z: (batch, latent_dim)
            baseline: Optional (batch, n_baseline).

        Returns:
            output: (batch, seq_len, input_dim)
        """
        if baseline is not None and self.n_baseline > 0:
            z = torch.cat([z, baseline], dim=-1)

        h = torch.relu(self.fc_latent(z))  # (batch, d_model)
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, d_model)

        pe = self._sinusoidal_embedding(self.seq_len, z.device)
        h = h + pe

        h = self.decoder_transformer(h)
        output = self.fc_output(h)  # (batch, seq_len, input_dim)
        return self._apply_output_activations(output)

    def forward(self, x, mask=None, baseline=None):
        mu, logvar = self.encode(x, mask, baseline)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, baseline)
        return recon_x, mu, logvar

    def sample(self, num_samples, device='cpu', baseline=None):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            return self.decode(z, baseline)

    def predict_from_landmark(self, x_observed, mask_observed, baseline=None):
        """Landmark prediction (CNN-style: input pre-padded to seq_len)."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x_observed, mask_observed, baseline)
            return self.decode(mu, baseline)
