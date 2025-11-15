"""
Circuit discovery using trained sparse autoencoders
Integrated with the methodology from the sparse feature circuits paper
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
from pathlib import Path
import json

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, current_dir)

from circuits.circuits_utils import CircuitDiscoverer, CircuitConfig, SparseFeatureCircuit
from src.sae_trainer import SAEManager, get_activation_files


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
        
        importances = {}
        
        for layer, features in encoded_activations.items():
            # features shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
            if features.dim() == 3:
                # Average over sequence dimension
                features = features.mean(dim=1)
            
            # Move refusal labels to the same device as features
            refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.float32, device=features.device)
            
            # Compute correlation with refusal labels
            batch_size, hidden_dim = features.shape
            layer_importance = torch.zeros(hidden_dim, device=features.device)
            
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
                
                # Convert correlation to tensor if it's a scalar, then take absolute value
                if isinstance(correlation, float):
                    layer_importance[feature_idx] = abs(correlation)
                else:
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
        """Compute importance of edges between SAE features using correlation analysis"""
        
        edge_importances = {}
        layers = sorted(important_nodes.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.float32)
        
        # Get feature activations for important nodes
        node_activations = {}
        for layer, nodes in important_nodes.items():
            if layer not in encoded_activations:
                continue
            features = encoded_activations[layer]
            if features.dim() == 3:
                features = features.mean(dim=1)  # Average over sequence
            
            node_activations[layer] = {}
            for feature_idx, importance in nodes:
                node_activations[layer][feature_idx] = features[:, feature_idx]
        
        # Compute edge importances between consecutive layers
        for i in range(len(layers) - 1):
            layer_i = layers[i]
            layer_j = layers[i + 1]
            
            if layer_i not in node_activations or layer_j not in node_activations:
                continue
            
            # Compute correlations between features in adjacent layers
            for node_i_idx, (node_i_feat, node_i_imp) in enumerate(important_nodes[layer_i][:20]):  # Limit for efficiency
                if node_i_feat not in node_activations[layer_i]:
                    continue
                    
                feat_i_vals = node_activations[layer_i][node_i_feat]
                
                for node_j_idx, (node_j_feat, node_j_imp) in enumerate(important_nodes[layer_j][:20]):
                    if node_j_feat not in node_activations[layer_j]:
                        continue
                    
                    feat_j_vals = node_activations[layer_j][node_j_feat]
                    
                    # Compute correlation between features
                    if feat_i_vals.std() > 0 and feat_j_vals.std() > 0:
                        correlation = torch.corrcoef(torch.stack([feat_i_vals, feat_j_vals]))[0, 1]
                        if torch.isnan(correlation):
                            correlation = 0.0
                        
                        # Edge importance = correlation * product of node importances
                        # This captures both feature interaction strength and individual importance
                        edge_importance = abs(correlation) * node_i_imp * node_j_imp
                        
                        source_id = f"{layer_i}_feature_{node_i_feat}"
                        target_id = f"{layer_j}_feature_{node_j_feat}"
                        edge_key = (source_id, target_id)
                        edge_importances[edge_key] = edge_importance.item()
        
        return edge_importances
    
    def _build_sae_circuit_graph(self,
                               important_nodes: Dict[str, List[Tuple]],
                               edge_importances: Dict[Tuple[str, str], float],
                               model_name: str) -> SparseFeatureCircuit:
        """Build circuit graph from SAE features"""
        
        circuit = SparseFeatureCircuit()
        
        # Track node IDs for edge creation
        node_id_map = {}  # (layer, feature_idx) -> node_id
        
        # Add SAE feature nodes
        for layer, nodes in important_nodes.items():
            layer_num = int(''.join(filter(str.isdigit, layer)) or 0)
            for feature_idx, importance in nodes:
                node_id = circuit.add_node(
                    feature_id=str(feature_idx),
                    layer=layer_num,
                    position=0,  # Would need token positions for more granularity
                    importance=importance,
                    feature_type="sae_feature"
                )
                node_id_map[(layer, feature_idx)] = node_id
        
        # Add edges between SAE features
        for (source_str, target_str), importance in edge_importances.items():
            if abs(importance) > self.config.edge_threshold:
                # Parse source and target from strings like "residuals_10_feature_123"
                try:
                    # Extract layer and feature from source/target strings
                    source_parts = source_str.split('_feature_')
                    target_parts = target_str.split('_feature_')
                    
                    if len(source_parts) == 2 and len(target_parts) == 2:
                        source_layer = source_parts[0]
                        source_feat = int(source_parts[1])
                        target_layer = target_parts[0]
                        target_feat = int(target_parts[1])
                        
                        source_key = (source_layer, source_feat)
                        target_key = (target_layer, target_feat)
                        
                        if source_key in node_id_map and target_key in node_id_map:
                            circuit.add_edge(
                                node_id_map[source_key], 
                                node_id_map[target_key], 
                                importance
                            )
                except (ValueError, KeyError) as e:
                    # Skip edges that can't be parsed
                    continue
        
        return circuit


# Note: The main() function below is not used - circuit discovery is handled by RefusalCircuitAnalyzer
# This file only contains the SAECircuitDiscoverer class which is imported and used by RefusalCircuitAnalyzer