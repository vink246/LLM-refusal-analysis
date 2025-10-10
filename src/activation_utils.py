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
    
    def save_activations(self, activations: Dict[str, Any], filename: str):
        """Save activations to file"""
        filepath = self.output_dir / f"{filename}.pkl"
        
        # Convert any torch tensors to numpy for serialization
        processed_activations = self._process_activations_for_saving(activations)
        
        with open(filepath, 'wb') as f:
            pickle.dump(processed_activations, f)
    
    def load_activations(self, filename: str) -> Dict[str, Any]:
        """Load activations from file"""
        filepath = self.output_dir / f"{filename}.pkl"
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _process_activations_for_saving(self, activations: Dict[str, Any]) -> Dict[str, Any]:
        """Convert torch tensors to numpy for serialization"""
        processed = {}
        for key, value in activations.items():
            if isinstance(value, dict):
                processed[key] = self._process_activations_for_saving(value)
            elif isinstance(value, torch.Tensor):
                processed[key] = {
                    'numpy_array': value.numpy(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                processed[key] = value
        return processed
    
    def extract_activation_pattern(self, activations: Dict, component_weights: Dict = None) -> torch.Tensor:
        """Extract flattened activation pattern from collected activations"""
        if component_weights is None:
            component_weights = {
                'mlp_output': 1.0,
                'ffn_gate_proj': 0.3,
                'ffn_up_proj': 0.3,
                'ffn_down_proj': 0.4,
                'residual_input': 0.2,
                'residual_output': 0.2
            }
        
        patterns = []
        
        for layer_key in sorted(activations.keys(), key=lambda x: int(x.split('_')[1])):
            layer_acts = activations[layer_key]
            
            for component, weight in component_weights.items():
                if component in layer_acts:
                    act_data = layer_acts[component]
                    if isinstance(act_data, dict) and 'numpy_array' in act_data:
                        # Load from saved format
                        act_tensor = torch.from_numpy(act_data['numpy_array'])
                    else:
                        act_tensor = act_data
                    
                    # Take mean across sequence dimension, then flatten
                    act_mean = act_tensor.mean(dim=1)  # Average over tokens
                    flattened = act_mean.flatten() * weight
                    patterns.append(flattened)
        
        return torch.cat(patterns) if patterns else torch.tensor([])
    
    def aggregate_activations_by_category(self, results: List[Dict]) -> Dict[str, torch.Tensor]:
        """Aggregate activations across multiple examples"""
        aggregated = {}
        
        for result in results:
            pattern = self.extract_activation_pattern(result['activations'])
            if pattern.numel() > 0:
                if 'patterns' not in aggregated:
                    aggregated['patterns'] = []
                aggregated['patterns'].append(pattern)
        
        if aggregated.get('patterns'):
            aggregated['mean_pattern'] = torch.stack(aggregated['patterns']).mean(dim=0)
            aggregated['std_pattern'] = torch.stack(aggregated['patterns']).std(dim=0)
        
        return aggregated