"""
Main analysis script for discovering refusal circuits across categories
"""

import os
import yaml
import torch
from tqdm import tqdm
from typing import Dict, List, Any
import json

from data_utils import load_all_categories
from models import load_model_with_hooks
from activation_utils import ActivationManager
from circuit_utils import CircuitDiscoverer, CircuitConfig, SparseFeatureCircuit, CircuitVisualizer, compare_circuits_across_categories

class RefusalCircuitAnalyzer:
    """Main class for analyzing refusal circuits across categories"""
    
    def _discover_category_circuit(self, model, model_name: str, category: str, 
                                 samples: List[Dict]) -> SparseFeatureCircuit:
        """Discover circuit for a specific category using PRE-COMPUTED activations"""
        
        circuit_config = CircuitConfig(
            node_threshold=self.config.get('node_threshold', 0.1),
            edge_threshold=self.config.get('edge_threshold', 0.01),
            aggregation_method=self.config.get('aggregation_method', 'none'),
            attribution_method=self.config.get('attribution_method', 'stats')  # Default to stats for pre-computed
        )
        
        discoverer = CircuitDiscoverer(model, self.activation_manager, circuit_config)
        
        # Extract refusal labels
        refusal_labels = [s['refusal_label'] for s in samples]
        
        # ✅ CORRECT: Use pre-computed activation file
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        activation_file = os.path.join(
            self.config['result_dir'], 
            'activations', 
            f"{safe_model_name}_{category}_activations.pt"
        )
        
        if not os.path.exists(activation_file):
            print(f"Warning: Activation file not found: {activation_file}")
            print("Falling back to on-the-fly activation collection...")
            # Fallback to original method
            prompts = [s['prompt'] for s in samples]
            return self._discover_category_circuit_onthefly(model, prompts, refusal_labels, category)
        
        print(f"Using pre-computed activations from: {activation_file}")
        circuit = discoverer.discover_circuit_from_saved_activations(
            activation_file=activation_file,
            refusal_labels=refusal_labels,
            metric_type="refusal"
        )
        
        return circuit
    
    def _discover_category_circuit_onthefly(self, model, prompts: List[str],
                                          refusal_labels: List[bool], category: str) -> SparseFeatureCircuit:
        """Fallback method to collect activations on-the-fly if pre-computed ones aren't available"""
        # This would use the original method that runs the model
        # But we should avoid this since we already collected activations
        pass


def main():
    """Main function to run refusal circuit analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover refusal circuits across categories")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    analyzer = RefusalCircuitAnalyzer(args.config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()