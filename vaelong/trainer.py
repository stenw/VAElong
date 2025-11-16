"""
Training utilities for VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .model import vae_loss_function


class VAETrainer:
    """
    Trainer class for Longitudinal VAE.
    
    Args:
        model: LongitudinalVAE model instance
        learning_rate: Learning rate for optimizer (default: 1e-3)
        beta: Weight for KL divergence term (default: 1.0)
        device: Device to train on (default: 'cuda' if available else 'cpu')
    """
    
    def __init__(self, model, learning_rate=1e-3, beta=1.0, device=None):
        self.model = model
        self.beta = beta
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_recon_loss: Average reconstruction loss
            avg_kld_loss: Average KL divergence loss
        """
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(batch_data)
            
            # Compute loss
            loss, recon_loss, kld_loss = vae_loss_function(
                recon_batch, batch_data, mu, logvar, self.beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        
        return avg_loss, avg_recon, avg_kld
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss: Average validation loss
            avg_recon_loss: Average reconstruction loss
            avg_kld_loss: Average KL divergence loss
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                recon_batch, mu, logvar = self.model(batch_data)
                
                # Compute loss
                loss, recon_loss, kld_loss = vae_loss_function(
                    recon_batch, batch_data, mu, logvar, self.beta
                )
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        
        return avg_loss, avg_recon, avg_kld
    
    def fit(self, train_loader, val_loader=None, epochs=100, verbose=True):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            history: Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kld': [],
            'val_loss': [],
            'val_recon': [],
            'val_kld': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss, train_recon, train_kld = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_kld'].append(train_kld)
            
            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kld = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_recon'].append(val_recon)
                history['val_kld'].append(val_kld)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f})'
                if val_loader is not None:
                    msg += f' | Val Loss: {val_loss:.4f}'
                print(msg)
        
        return history
    
    def save_model(self, path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
