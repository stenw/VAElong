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


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (default: 1.0, can be < 1 for beta-VAE)
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kld_loss
    
    return loss, recon_loss, kld_loss
