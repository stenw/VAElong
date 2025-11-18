"""
Variational Autoencoder model for longitudinal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LongitudinalVAE(nn.Module):
    """
    Variational Autoencoder for longitudinal (time-series) data.
    
    Uses LSTM/GRU layers to handle sequential nature of longitudinal measurements.
    
    Args:
        input_dim (int): Dimension of input features at each time step
        hidden_dim (int): Dimension of LSTM hidden state
        latent_dim (int): Dimension of latent space
        num_layers (int): Number of LSTM layers (default: 1)
        use_gru (bool): Use GRU instead of LSTM (default: False)
    """
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=20, num_layers=1, use_gru=False):
        super(LongitudinalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_gru = use_gru
        
        # Encoder: LSTM/GRU + linear layers for mean and log variance
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=False
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: linear layer + LSTM/GRU
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = rnn_class(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Pass through RNN and take the last hidden state
        _, hidden = self.encoder_rnn(x)
        
        # Handle LSTM vs GRU output
        if self.use_gru:
            h = hidden[-1]  # Take last layer's hidden state
        else:
            h = hidden[0][-1]  # Take last layer's hidden state (h, not c)
        
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
    
    def decode(self, z, seq_len):
        """
        Decode latent representation to output sequence.
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            seq_len: Length of output sequence
            
        Returns:
            output: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)
        
        # Transform latent to hidden dimension
        h = self.fc_latent(z)
        h = torch.relu(h)
        
        # Repeat for each time step
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Pass through RNN
        rnn_out, _ = self.decoder_rnn(h_repeated)
        
        # Generate output
        output = self.fc_output(rnn_out)
        
        return output
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            recon_x: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        seq_len = x.size(1)
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decode(z, seq_len)
        
        return recon_x, mu, logvar
    
    def sample(self, num_samples, seq_len, device='cpu'):
        """
        Generate samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples (num_samples, seq_len, input_dim)
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Decode
            samples = self.decode(z, seq_len)
            
        return samples


class CNNLongitudinalVAE(nn.Module):
    """
    CNN-based Variational Autoencoder for longitudinal (time-series) data with missing data handling.

    Uses 1D convolutional layers to process sequential data and supports missing value imputation
    through an EM-like approach.

    Args:
        input_dim (int): Dimension of input features at each time step
        seq_len (int): Expected sequence length
        latent_dim (int): Dimension of latent space
        hidden_channels (list): List of channel sizes for encoder convolutions (default: [32, 64, 128])
        kernel_size (int): Kernel size for convolutions (default: 3)
    """

    def __init__(self, input_dim, seq_len, latent_dim=20, hidden_channels=None, kernel_size=3):
        super(CNNLongitudinalVAE, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.hidden_channels = hidden_channels

        # Calculate the size after convolutions
        self._build_encoder()
        self._build_decoder()

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

        # Latent space layers
        self.fc_mu = nn.Linear(self.encoded_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size, self.latent_dim)

    def _build_decoder(self):
        """Build the decoder deconvolutional layers."""
        # Map from latent to encoded size
        self.fc_decode = nn.Linear(self.latent_dim, self.encoded_size)

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

    def encode(self, x, mask=None):
        """
        Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask of shape (batch_size, seq_len, input_dim)
                  where 1 indicates observed values and 0 indicates missing values

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

    def decode(self, z):
        """
        Decode latent representation to output sequence.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            output: Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)

        # Map to encoded size
        h = self.fc_decode(z)
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

        return output

    def forward(self, x, mask=None):
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional binary mask for missing data

        Returns:
            recon_x: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        mu, logvar = self.encode(x, mask)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decode(z)

        return recon_x, mu, logvar

    def sample(self, num_samples, device='cpu'):
        """
        Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            samples: Generated samples (num_samples, seq_len, input_dim)
        """
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Decode
            samples = self.decode(z)

        return samples

    def impute_missing(self, x, mask, num_iterations=5):
        """
        Impute missing values using iterative EM-like approach.

        Alternates between:
        1. Estimating NN parameters given current data
        2. Generating predictions for missing values and sampling from them

        Args:
            x: Input with missing values (batch_size, seq_len, input_dim)
            mask: Binary mask (1=observed, 0=missing)
            num_iterations: Number of EM iterations

        Returns:
            imputed: Data with imputed values
        """
        self.eval()
        imputed = x.clone()

        with torch.no_grad():
            for iteration in range(num_iterations):
                # E-step: Generate predictions for missing values
                recon_x, mu, logvar = self.forward(imputed, mask)

                # Sample from the reconstruction (add noise for uncertainty)
                std = torch.exp(0.5 * logvar)
                noise = torch.randn_like(recon_x) * std.unsqueeze(1).unsqueeze(2) * 0.1
                sampled_recon = recon_x + noise

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
