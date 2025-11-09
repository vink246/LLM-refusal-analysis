"""
Sparse Autoencoder (SAE) Training for Circuit Discovery
Based on the methodology from "Sparse Feature Circuits" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
from tqdm import tqdm
import json


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with Top-K sparsity for guaranteed sparse features
    This ensures sparsity regardless of training dynamics
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 0.01, k_percent: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff
        self.k_percent = k_percent
        self.k = max(1, int(hidden_dim * k_percent))  # Number of features to keep active
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        # Decoder  
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize weights for better sparsity
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to promote sparsity"""
        with torch.no_grad():
            # Initialize encoder with smaller weights to prevent saturation
            nn.init.xavier_uniform_(self.encoder.weight, gain=0.1)
            nn.init.zeros_(self.encoder.bias)
            
            # Initialize decoder as transpose of encoder (tied weights approach)
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            nn.init.zeros_(self.decoder.bias)
    
    def top_k_sparse_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity - only keep top k activations per sample"""
        # Get the top-k indices
        topk_vals, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # Create sparse activation tensor
        sparse_activation = torch.zeros_like(x)
        sparse_activation.scatter_(-1, topk_indices, topk_vals)
        
        return sparse_activation
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with guaranteed top-k sparsity
        
        Returns:
            reconstructed: Reconstructed input
            features: Sparse feature activations (guaranteed sparse)
            loss: Reconstruction loss (sparsity enforced structurally)
        """
        # Encode
        pre_activation = self.encoder(x)
        
        # Apply ReLU then top-k sparsity for guaranteed sparsity
        relu_features = F.relu(pre_activation)
        features = self.top_k_sparse_activation(relu_features)
        
        # Decode
        reconstructed = self.decoder(features)
        
        # Loss is just reconstruction (sparsity is enforced structurally)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        return reconstructed, features, reconstruction_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features"""
        pre_activation = self.encoder(x)
        relu_features = F.relu(pre_activation)
        return self.top_k_sparse_activation(relu_features)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstruction"""
        return self.decoder(features)


class ActivationDataset(Dataset):
    """Dataset for training SAEs on collected activations"""
    
    def __init__(self, activation_files: List[str], layer: str, max_samples: int = 10000):
        self.activations = []
        
        print(f"Loading activations for layer {layer}...")
        for file_path in tqdm(activation_files):
            if os.path.exists(file_path):
                try:
                    data = torch.load(file_path)
                    if layer in data:
                        layer_activations = data[layer].float()  # Ensure float32 dtype
                        # Flatten spatial dimensions if needed
                        if layer_activations.dim() > 2:
                            layer_activations = layer_activations.view(layer_activations.size(0), -1)
                        self.activations.append(layer_activations)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if self.activations:
            self.activations = torch.cat(self.activations, dim=0)
            # Limit samples if needed
            if len(self.activations) > max_samples:
                indices = torch.randperm(len(self.activations))[:max_samples]
                self.activations = self.activations[indices]
            print(f"Loaded {len(self.activations)} activation samples")
        else:
            raise ValueError(f"No activations found for layer {layer}")
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


class SAETrainer:
    """Trainer for sparse autoencoders"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 sparsity_coeff: float = 0.01,
                 k_percent: float = 0.05,
                 lr: float = 1e-3,
                 device: str = "cuda"):
        
        self.device = device
        self.sae = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff, k_percent).to(device)
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train(self, dataloader: DataLoader, epochs: int = 100) -> Dict[str, List[float]]:
        """Train the SAE"""
        
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'l0_norm': []  # Average number of active features
        }
        
        self.sae.train()
        
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_sparsity_loss = 0.0
            epoch_l0_norm = 0.0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(self.device).float()  # Ensure float32 dtype
                
                self.optimizer.zero_grad()
                
                reconstructed, features, reconstruction_loss = self.sae(batch)
                
                # The loss is now just reconstruction loss (sparsity enforced structurally)
                total_loss = reconstruction_loss
                
                # Compute additional metrics
                with torch.no_grad():
                    # Sparsity is enforced by top-k, so this is just for logging
                    sparsity_loss = torch.tensor(0.0)  # No explicit sparsity loss
                    l0_norm = (features > 0).float().sum(dim=-1).mean()
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                epoch_sparsity_loss += sparsity_loss.item()
                epoch_l0_norm += l0_norm.item()
                num_batches += 1
            
            # Average over batches
            epoch_total_loss /= num_batches
            epoch_recon_loss /= num_batches  
            epoch_sparsity_loss /= num_batches
            epoch_l0_norm /= num_batches
            
            history['total_loss'].append(epoch_total_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['sparsity_loss'].append(epoch_sparsity_loss)
            history['l0_norm'].append(epoch_l0_norm)
            
            self.lr_scheduler.step(epoch_total_loss)
            
            print(f"Epoch {epoch+1}: Total Loss: {epoch_total_loss:.4f}, "
                  f"Recon: {epoch_recon_loss:.4f}, Sparsity: {epoch_sparsity_loss:.4f}, "
                  f"L0: {epoch_l0_norm:.1f}")
            
            # Early stopping check
            if epoch_total_loss < 1e-4:  # Convergence threshold
                print("Convergence reached, stopping early")
                break
        
        return history
    
    def save(self, filepath: str):
        """Save the trained SAE"""
        torch.save({
            'model_state_dict': self.sae.state_dict(),
            'input_dim': self.sae.input_dim,
            'hidden_dim': self.sae.hidden_dim,
            'sparsity_coeff': self.sae.sparsity_coeff,
            'k_percent': self.sae.k_percent
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cuda"):
        """Load a trained SAE"""
        checkpoint = torch.load(filepath, map_location=device)
        trainer = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            sparsity_coeff=checkpoint['sparsity_coeff'],
            k_percent=checkpoint.get('k_percent', 0.05),  # Default to 5% for backwards compatibility
            device=device
        )
        trainer.sae.load_state_dict(checkpoint['model_state_dict'])
        return trainer


class SAEManager:
    """Manager for training and using multiple SAEs across layers"""
    
    def __init__(self, sae_dir: str = "results/saes"):
        self.sae_dir = Path(sae_dir)
        self.sae_dir.mkdir(parents=True, exist_ok=True)
        self.trained_saes = {}
    
    def train_saes_for_model(self, 
                           model_name: str,
                           activation_files: List[str],
                           layers: List[str],
                           sae_hidden_dim: int = 8192,  # 8x expansion as in paper
                           max_samples: int = 100000,
                           batch_size: int = 512,
                           epochs: int = 100,
                           sparsity_coeff: float = 0.01):
        """Train SAEs for all layers of a model"""
        
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = self.sae_dir / safe_model_name
        model_sae_dir.mkdir(exist_ok=True)
        
        for layer in layers:
            print(f"\n=== Training SAE for {model_name} - {layer} ===")
            
            # Check if SAE already exists
            sae_path = model_sae_dir / f"{layer}_sae.pt"
            if sae_path.exists():
                print(f"SAE already exists at {sae_path}, skipping...")
                continue
            
            try:
                # Create dataset for this layer
                dataset = ActivationDataset(activation_files, layer, max_samples)
                
                if len(dataset) == 0:
                    print(f"No activations found for layer {layer}, skipping...")
                    continue
                
                input_dim = dataset.activations.shape[1]
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Train SAE with Top-K sparsity
                trainer = SAETrainer(
                    input_dim=input_dim,
                    hidden_dim=sae_hidden_dim,
                    sparsity_coeff=sparsity_coeff,  # Use configurable sparsity coefficient
                    k_percent=0.05,  # 5% sparsity (guaranteed)
                    lr=1e-3,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                history = trainer.train(dataloader, epochs)
                
                # Save SAE and training history
                trainer.save(sae_path)
                with open(model_sae_dir / f"{layer}_training_history.json", 'w') as f:
                    json.dump(history, f, indent=2)
                
                print(f"SAE trained and saved to {sae_path}")
                
            except Exception as e:
                print(f"Error training SAE for layer {layer}: {e}")
                continue
    
    def load_saes_for_model(self, model_name: str, layers: List[str]) -> Dict[str, SAETrainer]:
        """Load trained SAEs for a model"""
        
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = self.sae_dir / safe_model_name
        
        saes = {}
        for layer in layers:
            sae_path = model_sae_dir / f"{layer}_sae.pt"
            if sae_path.exists():
                try:
                    saes[layer] = SAETrainer.load(sae_path)
                    print(f"Loaded SAE for layer {layer}")
                except Exception as e:
                    print(f"Error loading SAE for layer {layer}: {e}")
            else:
                print(f"SAE not found for layer {layer}: {sae_path}")
        
        return saes
    
    def encode_activations(self, activations: Dict[str, torch.Tensor], 
                         saes: Dict[str, SAETrainer]) -> Dict[str, torch.Tensor]:
        """Encode activations using trained SAEs"""
        
        encoded = {}
        for layer, activation in activations.items():
            if layer in saes:
                sae = saes[layer]
                
                # Get the device and dtype of the SAE model
                sae_device = next(sae.sae.parameters()).device
                sae_dtype = next(sae.sae.parameters()).dtype
                
                # Move activation to the same device and dtype as SAE
                activation = activation.to(device=sae_device, dtype=sae_dtype)
                
                # Flatten if needed
                original_shape = activation.shape
                if activation.dim() > 2:
                    activation = activation.view(activation.size(0), -1)
                
                with torch.no_grad():
                    features = sae.sae.encode(activation)
                
                # Reshape to (batch_size, seq_len, hidden_dim) if original was 3D
                if len(original_shape) == 3:
                    features = features.view(original_shape[0], original_shape[1], -1)
                
                encoded[layer] = features
        
        return encoded


def get_activation_files(result_dir: str, model_name: str, categories: List[str]) -> List[str]:
    """Get all activation files for a model and categories"""
    
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    activation_files = []
    activation_dir = Path(result_dir) / "activations"
    
    for category in categories:
        file_path = activation_dir / f"{safe_model_name}_{category}_activations.pt"
        if file_path.exists():
            activation_files.append(str(file_path))
    
    return activation_files