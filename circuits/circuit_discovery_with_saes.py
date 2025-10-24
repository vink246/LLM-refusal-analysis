"""
Circuit discovery using trained sparse autoencoders
Integrated with the methodology from the sparse feature circuits paper
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

from circuit_utils import CircuitDiscoverer, CircuitConfig, SparseFeatureCircuit
from sae_trainer import SAEManager, get_activation_files


class SAECircuitDiscoverer(CircuitDiscoverer):
    """Circuit discoverer that uses trained sparse autoencoders"""
    
    def __init__(self, model, activation_manager, sae_manager: SAEManager, config: CircuitConfig):
        super().__init__(model, activation_manager, config)
        self.sae_manager = sae_manager
        self.trained_saes = {}
    
    def load_saes(self, model_name: str, layers: List[str]):
        """Load trained SAEs for the model"""
        self.trained_saes = self.sae_manager.load_saes_for_model(model_name, layers)
    
    def discover_circuit_with_saes(self,
                                 activation_file: str,
                                 refusal_labels: List[bool],
                                 model_name: str,
                                 metric_type: str = "refusal") -> SparseFeatureCircuit:
        """
        Discover circuit using trained SAE features
        """
        
        # Load pre-computed activations
        print(f"Loading activations from: {activation_file}")
        raw_activations = torch.load(activation_file)
        
        # Encode activations using SAEs
        print("Encoding activations with SAEs...")
        encoded_activations = self.sae_manager.encode_activations(raw_activations, self.trained_saes)
        
        circuit = SparseFeatureCircuit()
        
        # Step 1: Compute feature importances
        feature_importances = self._compute_sae_feature_importances(
            encoded_activations, refusal_labels, metric_type
        )
        
        # Step 2: Identify important nodes (SAE features)
        important_nodes = self._identify_important_sae_nodes(feature_importances)
        
        # Step 3: Compute edge importances between SAE features
        edge_importances = self._compute_sae_edge_importances(
            encoded_activations, refusal_labels, important_nodes, metric_type
        )
        
        # Step 4: Build circuit graph
        circuit = self._build_sae_circuit_graph(important_nodes, edge_importances, model_name)
        
        return circuit
    
    def _compute_sae_feature_importances(self,
                                       encoded_activations: Dict[str, torch.Tensor],
                                       refusal_labels: List[bool],
                                       metric_type: str) -> Dict[str, torch.Tensor]:
        """Compute importance of SAE features"""
        
        refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.float32)
        importances = {}
        
        for layer, features in encoded_activations.items():
            # features shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
            if features.dim() == 3:
                # Average over sequence dimension
                features = features.mean(dim=1)
            
            # Compute correlation with refusal labels
            batch_size, hidden_dim = features.shape
            layer_importance = torch.zeros(hidden_dim)
            
            for feature_idx in range(hidden_dim):
                feature_vals = features[:, feature_idx]
                
                # Handle constant features
                if feature_vals.std() == 0:
                    correlation = 0.0
                else:
                    # Compute correlation coefficient
                    correlation_matrix = torch.corrcoef(torch.stack([
                        feature_vals, refusal_labels_tensor
                    ]))
                    correlation = correlation_matrix[0, 1] if not torch.isnan(correlation_matrix[0, 1]) else 0.0
                
                layer_importance[feature_idx] = torch.abs(correlation)
            
            importances[layer] = layer_importance
        
        return importances
    
    def _identify_important_sae_nodes(self, feature_importances: Dict[str, torch.Tensor]) -> Dict[str, List[Tuple]]:
        """Identify important SAE feature nodes"""
        
        important_nodes = {}
        
        for layer, importance_tensor in feature_importances.items():
            # Apply threshold
            threshold = self.config.node_threshold * importance_tensor.max()
            important_indices = torch.where(importance_tensor > threshold)[0]
            
            important_nodes[layer] = [
                (idx.item(), importance_tensor[idx].item())
                for idx in important_indices
            ]
            
            print(f"Layer {layer}: {len(important_nodes[layer])} important features")
        
        return important_nodes
    
    def _compute_sae_edge_importances(self,
                                    encoded_activations: Dict[str, torch.Tensor],
                                    refusal_labels: List[bool],
                                    important_nodes: Dict[str, List[Tuple]],
                                    metric_type: str) -> Dict[Tuple[str, str], float]:
        """Compute importance of edges between SAE features"""
        
        # This is a simplified version - in practice, you'd want to compute
        # causal relationships between features across layers
        
        edge_importances = {}
        layers = list(important_nodes.keys())
        
        # For demo purposes, create random edges
        # In real implementation, you'd compute actual causal relationships
        for i, layer_i in enumerate(layers):
            for j, layer_j in enumerate(layers):
                if i < j:  # Only consider connections from earlier to later layers
                    for node_i in important_nodes[layer_i][:10]:  # Limit for demo
                        for node_j in important_nodes[layer_j][:10]:
                            edge_key = (f"{layer_i}_feature_{node_i[0]}", f"{layer_j}_feature_{node_j[0]}")
                            # Use product of importances as proxy for edge importance
                            edge_importance = node_i[1] * node_j[1] * 0.1
                            edge_importances[edge_key] = edge_importance
        
        return edge_importances
    
    def _build_sae_circuit_graph(self,
                               important_nodes: Dict[str, List[Tuple]],
                               edge_importances: Dict[Tuple[str, str], float],
                               model_name: str) -> SparseFeatureCircuit:
        """Build circuit graph from SAE features"""
        
        circuit = SparseFeatureCircuit()
        
        # Add SAE feature nodes
        for layer, nodes in important_nodes.items():
            for feature_idx, importance in nodes:
                circuit.add_node(
                    feature_id=str(feature_idx),
                    layer=int(''.join(filter(str.isdigit, layer)) or 0),  # Extract layer number
                    position=0,  # Would need token positions for more granularity
                    importance=importance,
                    feature_type="sae_feature"
                )
        
        # Add edges between SAE features
        for (source, target), importance in edge_importances.items():
            if abs(importance) > self.config.edge_threshold:
                circuit.add_edge(source, target, importance)
        
        return circuit


def main():
    """Main function to run SAE training and circuit discovery"""
    import argparse
    import yaml
    from refusal_circuit_analyzer import RefusalCircuitAnalyzer
    
    parser = argparse.ArgumentParser(description="Train SAEs and discover circuits")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--train_saes", action="store_true", help="Train SAEs before circuit discovery")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize SAE manager
    sae_manager = SAEManager(config.get('sae_dir', 'results/saes'))
    
    # Train SAEs if requested
    if args.train_saes:
        print("=== Training Sparse Autoencoders ===")
        for model_name in config['models']:
            print(f"\nTraining SAEs for {model_name}...")
            
            # Get all activation files for this model
            activation_files = get_activation_files(
                config['result_dir'], 
                model_name, 
                config['categories']
            )
            
            if not activation_files:
                print(f"No activation files found for {model_name}")
                continue
            
            # Train SAEs for all layers
            sae_manager.train_saes_for_model(
                model_name=model_name,
                activation_files=activation_files,
                layers=config['activation_layers'],
                sae_hidden_dim=config.get('sae_hidden_dim', 8192),
                max_samples=config.get('sae_max_samples', 100000),
                batch_size=config.get('sae_batch_size', 512),
                epochs=config.get('sae_epochs', 100)
            )
    
    # Now run circuit discovery with SAEs
    print("\n=== Discovering Circuits with SAEs ===")
    
    # We need to modify the RefusalCircuitAnalyzer to use SAEs
    # For now, let's create a simple analysis loop
    
    for model_name in config['models']:
        print(f"\nAnalyzing circuits for {model_name}...")
        
        # Load trained SAEs
        saes = sae_manager.load_saes_for_model(model_name, config['activation_layers'])
        if not saes:
            print(f"No trained SAEs found for {model_name}, skipping...")
            continue
        
        # Initialize circuit discoverer with SAEs
        circuit_config = CircuitConfig(
            node_threshold=config.get('node_threshold', 0.1),
            edge_threshold=config.get('edge_threshold', 0.01),
            aggregation_method=config.get('aggregation_method', 'none'),
            attribution_method=config.get('attribution_method', 'stats')
        )
        
        # We'd need the model object here - for now, we'll use a placeholder
        # In practice, you'd load the model as in your original code
        model_obj = None  # This would be your actual model
        
        discoverer = SAECircuitDiscoverer(model_obj, None, sae_manager, circuit_config)
        discoverer.load_saes(model_name, config['activation_layers'])
        
        # Analyze each category
        for category in config['categories']:
            print(f"  Discovering circuit for {category}...")
            
            # This is where you'd load your actual data
            # For now, this is a placeholder structure
            
            # You'd need to:
            # 1. Load the activation file for this model and category
            # 2. Load the refusal labels
            # 3. Run circuit discovery
            
            activation_file = f"{config['result_dir']}/activations/{model_name.replace('/', '-')}_{category}_activations.pt"
            
            if not os.path.exists(activation_file):
                print(f"    Activation file not found: {activation_file}")
                continue
            
            # Placeholder for refusal labels - you'd load these from your data
            refusal_labels = [True, False] * 100  # This should be your actual labels
            
            try:
                circuit = discoverer.discover_circuit_with_saes(
                    activation_file=activation_file,
                    refusal_labels=refusal_labels,
                    model_name=model_name
                )
                
                # Save circuit
                safe_model_name = model_name.replace('/', '-')
                circuit_dir = Path(config['result_dir']) / "sae_circuits"
                circuit_dir.mkdir(exist_ok=True)
                
                circuit_file = circuit_dir / f"{safe_model_name}_{category}_sae_circuit.json"
                circuit_data = {
                    'nodes': circuit.nodes,
                    'edges': {f"{src}_{tgt}": data for (src, tgt), data in circuit.edges.items()},
                    'node_importances': circuit.node_importances,
                    'edge_importances': circuit.edge_importances
                }
                
                with open(circuit_file, 'w') as f:
                    json.dump(circuit_data, f, indent=2)
                
                print(f"    Circuit saved to {circuit_file}")
                
            except Exception as e:
                print(f"    Error discovering circuit: {e}")
                continue


if __name__ == "__main__":
    main()