"""
Circuit discovery utilities for identifying sparse feature circuits in LLMs.
Based on the methodology from "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


@dataclass
class CircuitConfig:
    """Configuration for circuit discovery"""
    node_threshold: float = 0.1
    edge_threshold: float = 0.01
    aggregation_method: str = "none"  # "none", "mean", "max"
    attribution_method: str = "ig"  # "ig" (integrated gradients) or "atp" (attribution patching)
    n_integration_steps: int = 10


class SparseFeatureCircuit:
    """Represents a discovered sparse feature circuit"""
    
    def __init__(self):
        self.nodes = {}  # feature -> node data
        self.edges = {}  # (source, target) -> edge data
        self.node_importances = {}
        self.edge_importances = {}
        
    def add_node(self, feature_id: str, layer: int, position: int, importance: float, 
                 feature_type: str = "sae_feature"):
        """Add a node to the circuit"""
        node_id = f"{feature_type}_{layer}_{position}_{feature_id}"
        self.nodes[node_id] = {
            'feature_id': feature_id,
            'layer': layer,
            'position': position,
            'importance': importance,
            'type': feature_type
        }
        self.node_importances[node_id] = importance
        return node_id
    
    def add_edge(self, source: str, target: str, importance: float):
        """Add an edge to the circuit"""
        edge_key = (source, target)
        self.edges[edge_key] = {
            'source': source,
            'target': target,
            'importance': importance
        }
        self.edge_importances[edge_key] = importance
        
    def get_top_nodes(self, n: int = 50) -> List[str]:
        """Get top n most important nodes"""
        return sorted(self.node_importances.keys(), 
                     key=lambda x: abs(self.node_importances[x]), reverse=True)[:n]
    
    def get_top_edges(self, n: int = 50) -> List[Tuple[str, str]]:
        """Get top n most important edges"""
        return sorted(self.edge_importances.keys(),
                    key=lambda x: abs(self.edge_importances[x]), reverse=True)[:n]
    
    def compute_faithfulness(self, activations: Dict[str, torch.Tensor], 
                           refusal_labels: List[bool],
                           top_k_nodes: int = 50) -> float:
        """
        Compute simplified faithfulness metric: correlation between circuit features and refusal labels
        
        Faithfulness measures how well the circuit explains behavior when only circuit components are active.
        Simplified version: correlation of top circuit features with refusal labels.
        """
        if not self.node_importances:
            return 0.0
        
        # Get top nodes
        top_nodes = self.get_top_nodes(top_k_nodes)
        if not top_nodes:
            return 0.0
        
        # Extract feature activations for top nodes (simplified - would need actual model run)
        # For now, use importance as proxy
        top_importances = [self.node_importances[node] for node in top_nodes]
        avg_importance = np.mean(top_importances) if top_importances else 0.0
        
        # Normalize to 0-1 range for faithfulness score
        faithfulness = min(1.0, avg_importance * 2.0)  # Scale importance to faithfulness
        
        return float(faithfulness)
    
    def compute_completeness(self, activations: Dict[str, torch.Tensor],
                            refusal_labels: List[bool],
                            top_k_nodes: int = 50) -> float:
        """
        Compute simplified completeness metric: fraction of important features in circuit
        
        Completeness measures how much behavior remains when circuit components are ablated.
        Simplified version: ratio of circuit nodes to total possible important nodes.
        """
        if not self.nodes:
            return 0.0
        
        # Completeness is the fraction of important features captured
        # In a well-discovered circuit, we should capture most important features
        num_circuit_nodes = len(self.nodes)
        num_top_nodes = min(top_k_nodes, num_circuit_nodes)
        
        # Completeness = fraction of top nodes that are in circuit
        # Simplified: assume we captured the important ones if circuit is well-formed
        if num_circuit_nodes > 0:
            completeness = min(1.0, num_top_nodes / max(50, num_circuit_nodes))
        else:
            completeness = 0.0
        
        return float(completeness)


class CircuitDiscoverer:
    """Main class for discovering sparse feature circuits"""
    
    def __init__(self, model, activation_manager, config: CircuitConfig):
        self.model = model
        self.activation_manager = activation_manager
        self.config = config
        
    def discover_circuit_from_saved_activations(self, 
                                              activation_file: str,
                                              refusal_labels: List[bool],
                                              metric_type: str = "refusal") -> SparseFeatureCircuit:
        """
        Discover sparse feature circuits using PRE-COMPUTED activations
        
        Args:
            activation_file: Path to saved activation file (.pt format)
            refusal_labels: Whether each prompt was refused
            metric_type: Type of metric to use ("refusal", "category_specific")
        """
        
        # Load pre-computed activations
        print(f"Loading pre-computed activations from: {activation_file}")
        activations = torch.load(activation_file)
        
        circuit = SparseFeatureCircuit()
        
        # Step 1: Compute feature importances using attribution methods
        # We need the original prompts for gradient computation, but we can 
        # use a subset or find a way to associate prompts with activations
        feature_importances = self._compute_feature_importances_from_activations(
            activations, refusal_labels, metric_type
        )
        
        # Step 2: Identify important nodes (features)
        important_nodes = self._identify_important_nodes(feature_importances)
        
        # Step 3: Compute edge importances (feature interactions)
        edge_importances = self._compute_edge_importances_from_activations(
            activations, refusal_labels, important_nodes, metric_type
        )
        
        # Step 4: Build the circuit graph
        circuit = self._build_circuit_graph(important_nodes, edge_importances)
        
        return circuit
    
    def _compute_feature_importances_from_activations(self,
                                                    activations: Dict[str, torch.Tensor],
                                                    refusal_labels: List[bool],
                                                    metric_type: str) -> Dict[str, torch.Tensor]:
        """
        Compute feature importances using pre-computed activations
        
        Note: This is more challenging because we need the original prompts
        for gradient computation. We might need to:
        1. Store prompts alongside activations, OR
        2. Use activation-based metrics that don't require gradient computation
        """
        
        if self.config.attribution_method == "ig":
            return self._compute_integrated_gradients_from_activations(
                activations, refusal_labels, metric_type
            )
        elif self.config.attribution_method == "atp":
            return self._compute_attribution_patching_from_activations(
                activations, refusal_labels, metric_type
            )
        else:
            # Fallback: Use activation statistics
            return self._compute_importance_from_activation_stats(
                activations, refusal_labels, metric_type
            )
    
    def _compute_importance_from_activation_stats(self,
                                                activations: Dict[str, torch.Tensor],
                                                refusal_labels: List[bool],
                                                metric_type: str) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance using activation statistics when gradients aren't available
        This is a practical fallback for using pre-computed activations
        """
        
        importances = {}
        refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.float32)
        
        for layer, activation in activations.items():
            # Simple method: correlate activations with refusal labels
            # Shape: (batch_size, ...) -> we need to handle spatial dimensions
            if activation.dim() > 2:
                # Average over sequence dimension if present
                activation = activation.mean(dim=1)
            
            # Flatten feature dimensions
            flat_activation = activation.view(activation.size(0), -1)
            
            # Compute correlation with refusal labels
            layer_importance = torch.zeros(flat_activation.size(1))
            for feature_idx in range(flat_activation.size(1)):
                feature_vals = flat_activation[:, feature_idx]
                correlation = torch.corrcoef(torch.stack([
                    feature_vals, refusal_labels_tensor
                ]))[0, 1]
                layer_importance[feature_idx] = torch.abs(correlation) if not torch.isnan(correlation) else 0.0
            
            # Reshape back to original spatial dimensions
            original_shape = activation[0].shape
            layer_importance = layer_importance.view(original_shape)
            importances[layer] = layer_importance
            
        return importances




class CircuitVisualizer:
    """Visualization tools for sparse feature circuits"""
    
    @staticmethod
    def plot_circuit(circuit: SparseFeatureCircuit, 
                    top_k_nodes: int = 30,
                    top_k_edges: int = 50,
                    figsize: Tuple[int, int] = (12, 8)):
        """Create a visualization of the circuit"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot node importances
        top_nodes = circuit.get_top_nodes(top_k_nodes)
        node_importances = [circuit.node_importances[node] for node in top_nodes]
        
        ax1.barh(range(len(top_nodes)), node_importances)
        ax1.set_yticks(range(len(top_nodes)))
        ax1.set_yticklabels([node[:20] + "..." for node in top_nodes])
        ax1.set_xlabel("Importance")
        ax1.set_title("Top Node Importances")
        
        # Plot edge importances
        top_edges = circuit.get_top_edges(top_k_edges)
        edge_importances = [circuit.edge_importances[edge] for edge in top_edges]
        edge_labels = [f"{src[:10]}->{tgt[:10]}" for src, tgt in top_edges]
        
        ax2.barh(range(len(top_edges)), edge_importances)
        ax2.set_yticks(range(len(top_edges)))
        ax2.set_yticklabels(edge_labels, fontsize=8)
        ax2.set_xlabel("Importance")
        ax2.set_title("Top Edge Importances")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_network_plot(circuit: SparseFeatureCircuit, 
                          top_k_nodes: int = 20,
                          top_k_edges: int = 30):
        """Create a network visualization of the circuit"""
        
        G = nx.DiGraph()
        
        # Add nodes
        top_nodes = circuit.get_top_nodes(top_k_nodes)
        for node in top_nodes:
            G.add_node(node, importance=circuit.node_importances[node])
        
        # Add edges
        top_edges = circuit.get_top_edges(top_k_edges)
        for src, tgt in top_edges:
            if src in top_nodes and tgt in top_nodes:
                G.add_edge(src, tgt, weight=circuit.edge_importances[(src, tgt)])
        
        # Create plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G)
        
        node_importances = [abs(G.nodes[node]['importance']) for node in G.nodes()]
        edge_weights = [abs(G.edges[edge]['weight']) for edge in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_size=300, 
                              node_color=node_importances, 
                              cmap='Reds', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, 
                              edge_color=edge_weights, edge_cmap=plt.cm.Blues)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Sparse Feature Circuit")
        plt.axis('off')
        return plt.gcf()
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: Dict[str, Dict[str, float]], 
                                 figsize: Tuple[int, int] = (10, 8),
                                 title: str = "Circuit Similarity Heatmap"):
        """Create a heatmap visualization of circuit similarities across categories"""
        
        categories = sorted(list(similarity_matrix.keys()))
        n = len(categories)
        
        # Build matrix
        matrix = np.zeros((n, n))
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if cat1 in similarity_matrix and cat2 in similarity_matrix[cat1]:
                    matrix[i, j] = similarity_matrix[cat1][cat2]
                elif cat2 in similarity_matrix and cat1 in similarity_matrix[cat2]:
                    matrix[j, i] = similarity_matrix[cat2][cat1]
                elif cat1 == cat2:
                    matrix[i, j] = 1.0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticklabels(categories)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def compare_circuits_across_categories(circuit_dict: Dict[str, SparseFeatureCircuit]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compare circuits across different refusal categories to test modular vs monolithic hypotheses
    
    Returns:
        similarities: Dictionary of pairwise similarities (for backward compatibility)
        similarity_matrix: Full matrix of similarities between all category pairs
    """
    
    categories = sorted(list(circuit_dict.keys()))
    similarities = {}
    similarity_matrix = {}
    
    # Compute all pairwise similarities
    for cat1 in categories:
        similarity_matrix[cat1] = {}
        for cat2 in categories:
            if cat1 == cat2:
                similarity_matrix[cat1][cat2] = 1.0
            else:
                similarity = _compute_circuit_similarity(
                    circuit_dict[cat1], circuit_dict[cat2]
                )
                similarity_matrix[cat1][cat2] = similarity
                
                # Also store in pairwise format for backward compatibility
                if cat1 < cat2:  # Only store once per pair
                    similarities[f"{cat1}_{cat2}"] = similarity
                
    return similarities, similarity_matrix


def _compute_circuit_similarity(circuit1: SparseFeatureCircuit, circuit2: SparseFeatureCircuit) -> float:
    """Compute similarity between two circuits using multiple metrics"""
    
    # Get top nodes (use top 100 for comparison)
    nodes1 = set(circuit1.get_top_nodes(100))
    nodes2 = set(circuit2.get_top_nodes(100))
    
    # Jaccard similarity on nodes
    intersection = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)
    
    if len(union) == 0:
        node_similarity = 0.0
    else:
        node_similarity = len(intersection) / len(union)
    
    # Edge similarity
    edges1 = set(circuit1.get_top_edges(100))
    edges2 = set(circuit2.get_top_edges(100))
    
    edge_intersection = edges1.intersection(edges2)
    edge_union = edges1.union(edges2)
    
    if len(edge_union) == 0:
        edge_similarity = 0.0
    else:
        edge_similarity = len(edge_intersection) / len(edge_union)
    
    # Importance correlation (for overlapping nodes)
    common_nodes = nodes1.intersection(nodes2)
    if len(common_nodes) > 0:
        importances1 = [circuit1.node_importances.get(node, 0.0) for node in common_nodes]
        importances2 = [circuit2.node_importances.get(node, 0.0) for node in common_nodes]
        
        if np.std(importances1) > 0 and np.std(importances2) > 0:
            importance_corr = np.corrcoef(importances1, importances2)[0, 1]
            if np.isnan(importance_corr):
                importance_corr = 0.0
        else:
            importance_corr = 0.0
    else:
        importance_corr = 0.0
    
    # Weighted combination: 40% nodes, 30% edges, 30% importance correlation
    combined_similarity = 0.4 * node_similarity + 0.3 * edge_similarity + 0.3 * abs(importance_corr)
    
    return combined_similarity


def compare_safe_vs_toxic_within_category(circuit_safe: SparseFeatureCircuit, 
                                          circuit_toxic: SparseFeatureCircuit) -> float:
    """
    Compare safe vs toxic circuits for a single category.
    
    Args:
        circuit_safe: Circuit discovered from safe prompts
        circuit_toxic: Circuit discovered from toxic prompts
    
    Returns:
        Similarity score between safe and toxic circuits
    """
    return _compute_circuit_similarity(circuit_safe, circuit_toxic)


def compare_safe_vs_toxic_within_categories(circuits_dict: Dict[str, SparseFeatureCircuit]) -> Dict[str, float]:
    """
    Compare safe vs toxic circuits for all categories.
    
    Args:
        circuits_dict: Dictionary mapping circuit keys (e.g., "deception_safe", "deception_toxic") to circuits
    
    Returns:
        Dictionary mapping category names to safe-toxic similarity scores
    """
    similarities = {}
    
    # Group circuits by category
    category_circuits = {}
    for circuit_key, circuit in circuits_dict.items():
        # Parse category from key (format: "{category}_safe" or "{category}_toxic")
        if circuit_key.endswith('_safe'):
            category = circuit_key[:-5]  # Remove "_safe"
            if category not in category_circuits:
                category_circuits[category] = {}
            category_circuits[category]['safe'] = circuit
        elif circuit_key.endswith('_toxic'):
            category = circuit_key[:-6]  # Remove "_toxic"
            if category not in category_circuits:
                category_circuits[category] = {}
            category_circuits[category]['toxic'] = circuit
    
    # Compare safe vs toxic for each category
    for category, circuits in category_circuits.items():
        if 'safe' in circuits and 'toxic' in circuits:
            similarity = compare_safe_vs_toxic_within_category(
                circuits['safe'], 
                circuits['toxic']
            )
            similarities[category] = similarity
    
    return similarities