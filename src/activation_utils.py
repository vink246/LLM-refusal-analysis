import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import json
import pickle

class ActivationManager:
    def __init__(self, output_dir: str = "results/activations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_activations_batch(self, activations: Dict[str, torch.Tensor], filename: str):
        """Save batch activations in OR-Bench compatible format"""
        filepath = self.output_dir / f"{filename}.pt"
        
        # Convert to CPU and ensure proper format
        processed_activations = {}
        for layer_name, tensor in activations.items():
            if tensor is not None:
                processed_activations[layer_name] = tensor.cpu()
        
        torch.save(processed_activations, filepath)
    
    def load_activations(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load saved activations"""
        filepath = self.output_dir / f"{filename}.pt"
        return torch.load(filepath)
    
    def merge_activations(self, activation_batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge multiple activation batches along batch dimension"""
        merged = {}
        
        for batch in activation_batches:
            for layer_name, tensor in batch.items():
                if layer_name not in merged:
                    merged[layer_name] = []
                merged[layer_name].append(tensor)
        
        # Concatenate along batch dimension
        for layer_name in merged:
            merged[layer_name] = torch.cat(merged[layer_name], dim=0)
        
        return merged
    
    def extract_patterns_from_batch(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract patterns from batch activations (OR-Bench format)"""
        patterns = []
        
        for layer_name, tensor in activations.items():
            if tensor is not None:
                # Average over sequence dimension if present
                if tensor.dim() > 2:
                    tensor = tensor.mean(dim=1)  # Average over tokens
                
                # Flatten and weight
                flattened = tensor.flatten()
                
                # Apply layer-specific weights
                weight = self._get_layer_weight(layer_name)
                patterns.append(flattened * weight)
        
        return torch.cat(patterns) if patterns else torch.tensor([])
    
    def _get_layer_weight(self, layer_name: str) -> float:
        """Get weight for different layer types"""
        if layer_name.startswith('residuals_'):
            return 0.3
        elif layer_name.startswith('mlp_'):
            return 0.7
        elif layer_name.startswith('attention_'):
            return 0.2
        else:
            return 0.5