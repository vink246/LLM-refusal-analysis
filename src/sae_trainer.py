"""
Sparse Autoencoder (SAE) Training for Circuit Discovery
Based on the methodology from "Sparse Feature Circuits" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
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


def _load_activations_with_metadata(activation_files: List[str], 
                                     refusal_label_files: List[str],
                                     layer: str) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Load activations with metadata (category, is_toxic) for balanced sampling.
    
    Args:
        activation_files: List of activation file paths
        refusal_label_files: List of refusal label file paths (corresponding to activation_files)
        layer: Layer name to extract from activations
    
    Returns:
        activations: Concatenated activations tensor
        metadata: List of metadata dicts with keys: 'category', 'is_toxic', 'activation_index'
    """
    all_activations = []
    all_metadata = []
    
    for act_file, refusal_file in zip(activation_files, refusal_label_files):
        if not os.path.exists(act_file):
            continue
            
        try:
            # Load activations
            data = torch.load(act_file, map_location='cpu')
            if layer not in data:
                continue
                
            layer_activations = data[layer].float()
            # Flatten spatial dimensions if needed
            if layer_activations.dim() > 2:
                layer_activations = layer_activations.view(layer_activations.size(0), -1)
            
            # Load refusal labels
            refusal_labels = []
            if os.path.exists(refusal_file):
                with open(refusal_file, 'r') as f:
                    refusal_labels = json.load(f)
            else:
                # If no refusal labels, assume all safe (refusal_label=False)
                refusal_labels = [False] * len(layer_activations)
                print(f"Warning: No refusal labels found for {act_file}, assuming all safe")
            
            # Match activations to labels by index
            num_samples = min(len(layer_activations), len(refusal_labels))
            layer_activations = layer_activations[:num_samples]
            refusal_labels = refusal_labels[:num_samples]
            
            # Extract category from filename (format: {model}_{category}_activations.pt)
            category = Path(act_file).stem.replace('_activations', '').split('_', 1)[-1]
            
            # Create metadata for each sample
            start_idx = len(all_activations)
            for i, (activation, is_toxic) in enumerate(zip(layer_activations, refusal_labels)):
                all_metadata.append({
                    'category': category,
                    'is_toxic': bool(is_toxic),
                    'activation_index': start_idx + i
                })
            
            all_activations.append(layer_activations)
            
        except Exception as e:
            print(f"Error loading {act_file}: {e}")
            continue
    
    if not all_activations:
        raise ValueError(f"No activations found for layer {layer}")
    
    activations = torch.cat(all_activations, dim=0)
    return activations, all_metadata


class ActivationDataset(Dataset):
    """Dataset for training SAEs on collected activations with balanced sampling"""
    
    def __init__(self, 
                 activation_files: List[str], 
                 layer: str, 
                 max_samples: int = 10000,
                 balance_strategy: str = "none",
                 refusal_label_files: Optional[List[str]] = None,
                 samples_per_category: Optional[int] = None,
                 safe_toxic_ratio: float = 0.5):
        """
        Initialize dataset with optional balanced sampling.
        
        Args:
            activation_files: List of activation file paths
            layer: Layer name to extract
            max_samples: Maximum total samples
            balance_strategy: "none", "stratified", or "weighted"
            refusal_label_files: List of refusal label file paths (corresponding to activation_files)
            samples_per_category: Target samples per category after balancing
            safe_toxic_ratio: Target ratio of safe to toxic (0.5 = equal split)
        """
        self.activations = []
        self.metadata = []
        
        print(f"Loading activations for layer {layer}...")
        
        # Load activations with metadata if balancing is requested
        if balance_strategy != "none" and refusal_label_files:
            try:
                activations, metadata = _load_activations_with_metadata(
                    activation_files, refusal_label_files, layer
                )
                self.activations = activations
                self.metadata = metadata
                
                # Apply balanced sampling
                if balance_strategy == "stratified":
                    self._apply_stratified_sampling(samples_per_category, safe_toxic_ratio)
                elif balance_strategy == "weighted":
                    self._apply_weighted_sampling(samples_per_category, safe_toxic_ratio)
                    
            except Exception as e:
                print(f"Error in balanced loading: {e}, falling back to simple loading")
                balance_strategy = "none"
        
        # Fallback to simple loading if balancing failed or not requested
        if balance_strategy == "none" or not self.activations:
            self.activations = []
            for file_path in tqdm(activation_files):
                if os.path.exists(file_path):
                    try:
                        data = torch.load(file_path, map_location='cpu')
                        if layer in data:
                            layer_activations = data[layer].float()
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
        
        if len(self.activations) == 0:
            raise ValueError(f"No activations found for layer {layer}")
        
        print(f"Loaded {len(self.activations)} activation samples")
        if self.metadata:
            safe_count = sum(1 for m in self.metadata if not m['is_toxic'])
            toxic_count = sum(1 for m in self.metadata if m['is_toxic'])
            print(f"  Safe: {safe_count}, Toxic: {toxic_count}")
    
    def _apply_stratified_sampling(self, samples_per_category: Optional[int], safe_toxic_ratio: float):
        """Apply stratified sampling to ensure balanced representation"""
        if not self.metadata:
            return
        
        # Group by (category, is_toxic)
        groups = {}
        for i, meta in enumerate(self.metadata):
            key = (meta['category'], meta['is_toxic'])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Calculate samples per group
        if samples_per_category:
            # Equal samples per category, then split by safe/toxic
            samples_per_group = int(samples_per_category * safe_toxic_ratio)
            toxic_samples_per_group = samples_per_category - samples_per_group
        else:
            # Use minimum available samples to ensure balance
            min_samples = min(len(indices) for indices in groups.values())
            samples_per_group = int(min_samples * safe_toxic_ratio)
            toxic_samples_per_group = min_samples - samples_per_group
        
        # Sample from each group
        selected_indices = []
        for (category, is_toxic), indices in groups.items():
            if is_toxic:
                n_samples = min(toxic_samples_per_group, len(indices))
            else:
                n_samples = min(samples_per_group, len(indices))
            
            if n_samples > 0:
                selected = np.random.choice(indices, size=n_samples, replace=False)
                selected_indices.extend(selected.tolist())
        
        # Shuffle and apply
        np.random.shuffle(selected_indices)
        self.activations = self.activations[selected_indices]
        self.metadata = [self.metadata[i] for i in selected_indices]
    
    def _apply_weighted_sampling(self, samples_per_category: Optional[int], safe_toxic_ratio: float):
        """Apply weighted sampling based on inverse frequency"""
        if not self.metadata:
            return
        
        # Count samples per group
        group_counts = {}
        for meta in self.metadata:
            key = (meta['category'], meta['is_toxic'])
            group_counts[key] = group_counts.get(key, 0) + 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.metadata)
        weights = []
        for meta in self.metadata:
            key = (meta['category'], meta['is_toxic'])
            # Weight inversely proportional to group size
            weight = total_samples / (group_counts[key] * len(group_counts))
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample with replacement if needed, or without if we have enough
        n_samples = samples_per_category * len(set(m['category'] for m in self.metadata)) if samples_per_category else len(self.metadata)
        n_samples = min(n_samples, len(self.metadata))
        
        selected_indices = np.random.choice(
            len(self.metadata), 
            size=n_samples, 
            replace=(n_samples > len(self.metadata)),
            p=weights
        )
        
        self.activations = self.activations[selected_indices]
        self.metadata = [self.metadata[i] for i in selected_indices]
    
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
                           sparsity_coeff: float = 0.01,
                           balance_strategy: str = "none",
                           refusal_label_dir: Optional[str] = None,
                           samples_per_category: Optional[int] = None,
                           safe_toxic_ratio: float = 0.5):
        """
        Train SAEs for all layers of a model with optional balanced sampling.
        
        Args:
            model_name: Name of the model
            activation_files: List of activation file paths
            layers: List of layer names to train SAEs for
            sae_hidden_dim: Hidden dimension for SAE
            max_samples: Maximum samples per layer
            batch_size: Batch size for training
            epochs: Number of training epochs
            sparsity_coeff: Sparsity coefficient
            balance_strategy: "none", "stratified", or "weighted"
            refusal_label_dir: Directory containing refusal label files
            samples_per_category: Target samples per category after balancing
            safe_toxic_ratio: Target ratio of safe to toxic (0.5 = equal split)
        """
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = self.sae_dir / safe_model_name
        model_sae_dir.mkdir(exist_ok=True)
        
        # Get refusal label files if balancing is requested
        refusal_label_files = None
        if balance_strategy != "none" and refusal_label_dir:
            refusal_label_files = []
            refusal_label_path = Path(refusal_label_dir)
            for act_file in activation_files:
                # Extract category from activation filename
                act_filename = Path(act_file).stem
                # Format: {safe_model_name}_{category}_activations.pt
                category = act_filename.replace('_activations', '').replace(f"{safe_model_name}_", "")
                refusal_file = refusal_label_path / f"{safe_model_name}_{category}_refusal.json"
                refusal_label_files.append(str(refusal_file))
        
        for layer in layers:
            print(f"\n=== Training SAE for {model_name} - {layer} ===")
            
            # Check if SAE already exists
            sae_path = model_sae_dir / f"{layer}_sae.pt"
            if sae_path.exists():
                print(f"SAE already exists at {sae_path}, skipping...")
                continue
            
            try:
                # Create dataset for this layer with optional balancing
                dataset = ActivationDataset(
                    activation_files=activation_files,
                    layer=layer,
                    max_samples=max_samples,
                    balance_strategy=balance_strategy,
                    refusal_label_files=refusal_label_files,
                    samples_per_category=samples_per_category,
                    safe_toxic_ratio=safe_toxic_ratio
                )
                
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
                import traceback
                traceback.print_exc()
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