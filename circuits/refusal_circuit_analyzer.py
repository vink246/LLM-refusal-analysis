"""
Main analysis script for discovering refusal circuits across categories
"""

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, current_dir)

from src.data_utils import load_all_categories
from src.models import load_model_with_hooks
from src.activation_utils import ActivationManager
from circuits.circuits_utils import CircuitDiscoverer, CircuitConfig, SparseFeatureCircuit, CircuitVisualizer, compare_circuits_across_categories, compare_safe_vs_toxic_within_categories
import matplotlib.pyplot as plt
from circuits.circuit_discovery_with_saes import SAECircuitDiscoverer
from circuits.statistical_analysis import assess_modularity_with_statistics
from src.sae_trainer import SAEManager, get_activation_files

class RefusalCircuitAnalyzer:
    """Main class for analyzing refusal circuits across categories"""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config()
        
        self.result_dir = Path(self.config.get('result_dir', 'results'))
        self.activation_manager = ActivationManager(str(self.result_dir / 'activations'))
        self.sae_manager = SAEManager(self.config.get('sae_dir', str(self.result_dir / 'saes')))
        self.circuits = {}  # Store discovered circuits: {model_name: {category: circuit}}
        self.comparison_results = {}  # Store comparison results for final report
    
    def _validate_config(self):
        """Validate configuration file has required fields"""
        required_fields = ['models', 'categories', 'activation_layers']
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValueError(f"Configuration missing required fields: {missing_fields}")
        
        if not isinstance(self.config['models'], list) or len(self.config['models']) == 0:
            raise ValueError("Configuration must specify at least one model")
        
        if not isinstance(self.config['categories'], list) or len(self.config['categories']) == 0:
            raise ValueError("Configuration must specify at least one category")
        
        if not isinstance(self.config['activation_layers'], list) or len(self.config['activation_layers']) == 0:
            raise ValueError("Configuration must specify at least one activation layer")
    
    def run_analysis(self):
        """Main analysis workflow"""
        print("=== Refusal Circuit Analysis ===")
        
        # Step 1: Train or load SAEs
        if self.config.get('train_saes', False):
            print("\n=== Training Sparse Autoencoders ===")
            self._train_saes()
        
        # Step 2: Discover circuits for each model and category
        print("\n=== Discovering Circuits ===")
        for model_name in self.config['models']:
            print(f"\nAnalyzing circuits for {model_name}...")
            self._discover_circuits_for_model(model_name)
        
        # Step 3: Compare circuits across categories
        print("\n=== Comparing Circuits Across Categories ===")
        self._compare_circuits()
        
        # Step 4: Generate visualizations and reports
        print("\n=== Generating Reports ===")
        self._generate_reports()
        
        print("\n=== Analysis Complete ===")
    
    def _train_saes(self):
        """Train SAEs for all models"""
        for model_name in self.config['models']:
            print(f"\nTraining SAEs for {model_name}...")
            
            activation_files = get_activation_files(
                self.config['result_dir'],
                model_name,
                self.config['categories']
            )
            
            if not activation_files:
                print(f"  No activation files found for {model_name}, skipping...")
                continue
            
            # Get SAE training config with balancing parameters
            sae_training_config = self.config.get('sae_training', {})
            balance_strategy = sae_training_config.get('balance_strategy', 'none')
            refusal_label_dir = str(self.result_dir / 'refusal_labels')
            
            self.sae_manager.train_saes_for_model(
                model_name=model_name,
                activation_files=activation_files,
                layers=self.config['activation_layers'],
                sae_hidden_dim=self.config.get('sae_hidden_dim', 8192),
                max_samples=self.config.get('sae_max_samples', 100000),
                batch_size=self.config.get('sae_batch_size', 512),
                epochs=self.config.get('sae_epochs', 100),
                balance_strategy=balance_strategy,
                refusal_label_dir=refusal_label_dir,
                samples_per_category=sae_training_config.get('samples_per_category'),
                safe_toxic_ratio=sae_training_config.get('safe_toxic_ratio', 0.5)
            )
    
    def _get_model_folder_name(self, model_name: str) -> str:
        """Convert model name to organized folder name"""
        model_mapping = {
            "meta-llama/Llama-2-7b-chat-hf": "llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v0.1"
        }
        return model_mapping.get(model_name, model_name.replace('/', '-').replace(' ', '_').lower())
    
    def _discover_circuits_for_model(self, model_name: str):
        """Discover circuits for all categories of a model"""
        # Validate activation files exist
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        activation_dir = self.result_dir / 'activations'
        
        missing_activations = []
        for category in self.config['categories']:
            activation_file = activation_dir / f"{safe_model_name}_{category}_activations.pt"
            if not activation_file.exists():
                missing_activations.append(category)
        
        if missing_activations:
            print(f"  Warning: Missing activation files for categories: {missing_activations}")
            print(f"  Skipping circuit discovery for {model_name}")
            return
        
        # Load trained SAEs
        saes = self.sae_manager.load_saes_for_model(model_name, self.config['activation_layers'])
        if not saes:
            print(f"  No trained SAEs found for {model_name}")
            print(f"  Run with train_saes=true or ensure SAEs are already trained")
            return
        
        # Validate SAEs match expected layers
        missing_layers = [layer for layer in self.config['activation_layers'] if layer not in saes]
        if missing_layers:
            print(f"  Warning: Missing SAEs for layers: {missing_layers}")
            print(f"  Continuing with available SAEs...")
        
        # Initialize circuit discoverer
        circuit_config = CircuitConfig(
            node_threshold=self.config.get('node_threshold', 0.1),
            edge_threshold=self.config.get('edge_threshold', 0.01),
            aggregation_method=self.config.get('aggregation_method', 'none'),
            attribution_method=self.config.get('attribution_method', 'stats')
        )
        
        # We don't need the actual model for SAE-based discovery
        discoverer = SAECircuitDiscoverer(None, self.activation_manager, self.sae_manager, circuit_config)
        discoverer.load_saes(model_name, self.config['activation_layers'])
        
        # Discover circuits for each category
        # For modular vs monolithic question: discover category-level circuits (combine safe+toxic)
        # For safe vs toxic question: discover separate circuits for safe and toxic
        model_circuits = {}
        discover_separate_safe_toxic = self.config.get('discover_separate_safe_toxic', True)
        
        for category in self.config['categories']:
            print(f"  Discovering circuits for {category}...")
            
            safe_model_name = model_name.replace('/', '-').replace(' ', '_')
            activation_file = self.result_dir / 'activations' / f"{safe_model_name}_{category}_activations.pt"
            
            if not activation_file.exists():
                print(f"    Activation file not found: {activation_file}")
                continue
            
            # Load refusal labels
            refusal_file = self.result_dir / 'refusal_labels' / f"{safe_model_name}_{category}_refusal.json"
            if refusal_file.exists():
                with open(refusal_file, 'r') as f:
                    refusal_labels = json.load(f)
            else:
                print(f"    Warning: Refusal labels not found, inferring from activations...")
                activations = torch.load(activation_file, map_location='cpu')
                if activations:
                    first_layer = list(activations.keys())[0]
                    batch_size = activations[first_layer].shape[0]
                    refusal_labels = [False] * batch_size
                else:
                    print(f"    Error: Could not load activations")
                    continue
            
            circuit_dir = self.result_dir / "circuits"
            circuit_dir.mkdir(exist_ok=True)
            
            if discover_separate_safe_toxic:
                # Discover separate circuits for safe and toxic
                try:
                    # Split activations and labels by safe/toxic
                    activations_data = torch.load(activation_file, map_location='cpu')
                    safe_indices = [i for i, label in enumerate(refusal_labels) if not label]
                    toxic_indices = [i for i, label in enumerate(refusal_labels) if label]
                    
                    if len(safe_indices) > 0:
                        # Create safe circuit
                        safe_activations = {}
                        for layer, tensor in activations_data.items():
                            safe_activations[layer] = tensor[safe_indices]
                        
                        # Save temporary activation file for safe
                        safe_activation_file = circuit_dir / f"{safe_model_name}_{category}_safe_activations_temp.pt"
                        torch.save(safe_activations, safe_activation_file)
                        
                        safe_refusal_labels = [False] * len(safe_indices)
                        safe_circuit = discoverer.discover_circuit_with_saes(
                            activation_file=str(safe_activation_file),
                            refusal_labels=safe_refusal_labels,
                            model_name=model_name
                        )
                        
                        # Save safe circuit
                        safe_circuit_key = f"{category}_safe"
                        model_circuits[safe_circuit_key] = safe_circuit
                        safe_circuit_file = circuit_dir / f"{safe_model_name}_{category}_safe_circuit.json"
                        circuit_data = {
                            'model_name': model_name,
                            'category': category,
                            'circuit_type': 'safe',
                            'nodes': {k: v for k, v in safe_circuit.nodes.items()},
                            'edges': {f"{src}_{tgt}": data for (src, tgt), data in safe_circuit.edges.items()},
                            'node_importances': {k: float(v) for k, v in safe_circuit.node_importances.items()},
                            'edge_importances': {f"{src}_{tgt}": float(v) for (src, tgt), v in safe_circuit.edge_importances.items()}
                        }
                        with open(safe_circuit_file, 'w') as f:
                            json.dump(circuit_data, f, indent=2)
                        print(f"    Safe circuit saved: {safe_circuit_file} (Nodes: {len(safe_circuit.nodes)}, Edges: {len(safe_circuit.edges)})")
                        
                        # Clean up temp file
                        safe_activation_file.unlink()
                    
                    if len(toxic_indices) > 0:
                        # Create toxic circuit
                        toxic_activations = {}
                        for layer, tensor in activations_data.items():
                            toxic_activations[layer] = tensor[toxic_indices]
                        
                        # Save temporary activation file for toxic
                        toxic_activation_file = circuit_dir / f"{safe_model_name}_{category}_toxic_activations_temp.pt"
                        torch.save(toxic_activations, toxic_activation_file)
                        
                        toxic_refusal_labels = [True] * len(toxic_indices)
                        toxic_circuit = discoverer.discover_circuit_with_saes(
                            activation_file=str(toxic_activation_file),
                            refusal_labels=toxic_refusal_labels,
                            model_name=model_name
                        )
                        
                        # Save toxic circuit
                        toxic_circuit_key = f"{category}_toxic"
                        model_circuits[toxic_circuit_key] = toxic_circuit
                        toxic_circuit_file = circuit_dir / f"{safe_model_name}_{category}_toxic_circuit.json"
                        circuit_data = {
                            'model_name': model_name,
                            'category': category,
                            'circuit_type': 'toxic',
                            'nodes': {k: v for k, v in toxic_circuit.nodes.items()},
                            'edges': {f"{src}_{tgt}": data for (src, tgt), data in toxic_circuit.edges.items()},
                            'node_importances': {k: float(v) for k, v in toxic_circuit.node_importances.items()},
                            'edge_importances': {f"{src}_{tgt}": float(v) for (src, tgt), v in toxic_circuit.edge_importances.items()}
                        }
                        with open(toxic_circuit_file, 'w') as f:
                            json.dump(circuit_data, f, indent=2)
                        print(f"    Toxic circuit saved: {toxic_circuit_file} (Nodes: {len(toxic_circuit.nodes)}, Edges: {len(toxic_circuit.edges)})")
                        
                        # Clean up temp file
                        toxic_activation_file.unlink()
                    
                    # Also discover category-level circuit (combine safe+toxic) for Type 1 comparison
                    try:
                        category_circuit = discoverer.discover_circuit_with_saes(
                            activation_file=str(activation_file),
                            refusal_labels=refusal_labels,
                            model_name=model_name
                        )
                        category_level_circuits[category] = category_circuit
                    except Exception as e:
                        print(f"    Warning: Could not discover category-level circuit: {e}")
                    
                except Exception as e:
                    print(f"    Error discovering separate circuits: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Discover single category-level circuit (combine safe+toxic)
                try:
                    circuit = discoverer.discover_circuit_with_saes(
                        activation_file=str(activation_file),
                        refusal_labels=refusal_labels,
                        model_name=model_name
                    )
                    
                    model_circuits[category] = circuit
                    
                    # Save circuit
                    circuit_file = circuit_dir / f"{safe_model_name}_{category}_circuit.json"
                    circuit_data = {
                        'model_name': model_name,
                        'category': category,
                        'circuit_type': 'combined',
                        'nodes': {k: v for k, v in circuit.nodes.items()},
                        'edges': {f"{src}_{tgt}": data for (src, tgt), data in circuit.edges.items()},
                        'node_importances': {k: float(v) for k, v in circuit.node_importances.items()},
                        'edge_importances': {f"{src}_{tgt}": float(v) for (src, tgt), v in circuit.edge_importances.items()}
                    }
                    
                    with open(circuit_file, 'w') as f:
                        json.dump(circuit_data, f, indent=2)
                    
                    print(f"    Circuit saved to {circuit_file}")
                    print(f"    Nodes: {len(circuit.nodes)}, Edges: {len(circuit.edges)}")
                    
                except Exception as e:
                    print(f"    Error discovering circuit: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Add category-level circuits for Type 1 comparison
        if discover_separate_safe_toxic and category_level_circuits:
            # Merge category-level circuits into model_circuits for Type 1 comparison
            for category, circuit in category_level_circuits.items():
                # Use category name as key (not category_safe or category_toxic)
                if category not in model_circuits:
                    model_circuits[category] = circuit
        
        self.circuits[model_name] = model_circuits
    
    def _compare_circuits(self):
        """Compare circuits: Type 1 (category-to-category) and Type 2 (safe vs toxic within category)"""
        self.comparison_results = {}  # Store for final report
        
        for model_name, model_circuits in self.circuits.items():
            if len(model_circuits) < 2:
                print(f"  {model_name}: Need at least 2 circuits for comparison")
                continue
            
            print(f"\n  Comparing circuits for {model_name}...")
            
            # Separate circuits by type
            category_circuits = {}  # For category-to-category comparison
            safe_toxic_circuits = {}  # For safe vs toxic comparison
            
            for circuit_key, circuit in model_circuits.items():
                if circuit_key.endswith('_safe') or circuit_key.endswith('_toxic'):
                    safe_toxic_circuits[circuit_key] = circuit
                else:
                    # Category-level circuit (not split by safe/toxic)
                    category_circuits[circuit_key] = circuit
            
            # Type 1: Category-to-Category Comparisons
            if len(category_circuits) >= 2:
                print(f"    Type 1: Category-to-Category Comparisons")
                similarities, similarity_matrix = compare_circuits_across_categories(category_circuits)
            
            # Assess modularity with statistical significance
            similarity_values = list(similarities.values())
            if similarity_values:
                monolithic_threshold = self.config.get('analysis', {}).get('modularity_assessment', {}).get('monolithic_threshold', 0.8)
                partially_modular_threshold = self.config.get('analysis', {}).get('modularity_assessment', {}).get('partially_modular_threshold', 0.5)
                
                modularity_assessment = assess_modularity_with_statistics(
                    similarity_values,
                    monolithic_threshold,
                    partially_modular_threshold
                )
                
                avg_similarity = modularity_assessment['average_similarity']
                assessment = modularity_assessment['assessment']
                confidence = modularity_assessment['assessment_confidence']
                stats_result = modularity_assessment['statistics']
                
                    print(f"      Average similarity: {avg_similarity:.3f}")
                    print(f"      Assessment: {assessment} (confidence: {confidence})")
                if stats_result and stats_result['significant']:
                        print(f"      Statistical significance: p={stats_result['p_value']:.4f} (significant)")
                elif stats_result:
                        print(f"      Statistical significance: p={stats_result['p_value']:.4f} (not significant)")
                    
                    # Store results
                    if model_name not in self.comparison_results:
                        self.comparison_results[model_name] = {}
                    self.comparison_results[model_name]['category_to_category'] = {
                    'similarities': similarities,
                    'similarity_matrix': similarity_matrix,
                    'average_similarity': avg_similarity,
                    'assessment': assessment,
                    'assessment_confidence': confidence,
                    'statistics': stats_result,
                        'categories': list(category_circuits.keys())
                    }
                    
                    # Generate similarity heatmap
                    try:
                        viz_dir = self.result_dir / "visualizations"
                        viz_dir.mkdir(exist_ok=True)
                        model_folder = self._get_model_folder_name(model_name)
                        model_viz_dir = viz_dir / model_folder
                        model_viz_dir.mkdir(exist_ok=True)
                        heatmap_dir = model_viz_dir / "similarity_heatmaps"
                        heatmap_dir.mkdir(exist_ok=True)
                        
                        fig = CircuitVisualizer.create_similarity_heatmap(
                            similarity_matrix,
                            title=f"Category-to-Category Circuit Similarity: {model_name}"
                        )
                        heatmap_path = heatmap_dir / f"{model_folder}_category_to_category_heatmap.png"
                        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f"      Saved category-to-category heatmap: {heatmap_path}")
                    except Exception as e:
                        print(f"      Error creating category-to-category heatmap: {e}")
            
            # Type 2: Safe vs Toxic within Category Comparisons
            if safe_toxic_circuits:
                print(f"    Type 2: Safe vs Toxic within Category Comparisons")
                safe_toxic_similarities = compare_safe_vs_toxic_within_categories(safe_toxic_circuits)
                
                if safe_toxic_similarities:
                    avg_safe_toxic_sim = np.mean(list(safe_toxic_similarities.values()))
                    print(f"      Average safe-toxic similarity: {avg_safe_toxic_sim:.3f}")
                    for category, sim in safe_toxic_similarities.items():
                        print(f"        {category}: {sim:.3f}")
                    
                    # Store results
                    if model_name not in self.comparison_results:
                        self.comparison_results[model_name] = {}
                    self.comparison_results[model_name]['safe_vs_toxic'] = {
                        'similarities': safe_toxic_similarities,
                        'average_similarity': float(avg_safe_toxic_sim),
                        'categories': list(safe_toxic_similarities.keys())
                    }
                    
                    # Generate safe vs toxic heatmap
                    try:
                        viz_dir = self.result_dir / "visualizations"
                        model_folder = self._get_model_folder_name(model_name)
                        model_viz_dir = viz_dir / model_folder
                        heatmap_dir = model_viz_dir / "similarity_heatmaps"
                        heatmap_dir.mkdir(exist_ok=True)
                        
                        # Create a bar plot or heatmap for safe vs toxic similarities
                        fig, ax = plt.subplots(figsize=(10, 6))
                        categories = list(safe_toxic_similarities.keys())
                        similarities = list(safe_toxic_similarities.values())
                        
                        ax.barh(categories, similarities, color='steelblue')
                        ax.set_xlabel('Safe-to-Toxic Circuit Similarity', fontsize=12)
                        ax.set_ylabel('Category', fontsize=12)
                        ax.set_title(f'Safe vs Toxic Circuit Similarity by Category: {model_name}', fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1.0)
                        ax.grid(axis='x', alpha=0.3)
                        
                        # Add value labels
                        for i, (cat, sim) in enumerate(safe_toxic_similarities.items()):
                            ax.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        heatmap_path = heatmap_dir / f"{model_folder}_safe_vs_toxic_heatmap.png"
                        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f"      Saved safe vs toxic heatmap: {heatmap_path}")
                    except Exception as e:
                        print(f"      Error creating safe vs toxic heatmap: {e}")
            
            # Save combined comparison results
                comparison_file = self.result_dir / "circuits" / f"{model_name.replace('/', '-')}_comparison.json"
                def json_serializable(obj):
                    """Convert numpy types and other non-JSON types to JSON-serializable types"""
                    import numpy as np
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [json_serializable(v) for v in obj]
                    else:
                        return obj
                
                comparison_data = {
                    'model_name': model_name,
                'comparison_types': {}
            }
            
            if 'category_to_category' in self.comparison_results.get(model_name, {}):
                cat_data = self.comparison_results[model_name]['category_to_category']
                comparison_data['comparison_types']['category_to_category'] = {
                    'similarities': {k: float(v) for k, v in cat_data['similarities'].items()},
                    'similarity_matrix': {k: {k2: float(v2) for k2, v2 in v.items()} 
                                         for k, v in cat_data['similarity_matrix'].items()},
                    'average_similarity': float(cat_data['average_similarity']),
                    'assessment': cat_data['assessment'],
                    'assessment_confidence': cat_data['assessment_confidence'],
                    'statistics': json_serializable(cat_data['statistics']) if cat_data.get('statistics') else {},
                    'categories': cat_data['categories']
                }
            
            if 'safe_vs_toxic' in self.comparison_results.get(model_name, {}):
                st_data = self.comparison_results[model_name]['safe_vs_toxic']
                comparison_data['comparison_types']['safe_vs_toxic'] = {
                    'similarities': {k: float(v) for k, v in st_data['similarities'].items()},
                    'average_similarity': float(st_data['average_similarity']),
                    'categories': st_data['categories']
                }
                
                with open(comparison_file, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
                
            print(f"    Comparison results saved to {comparison_file}")
    
    def _generate_reports(self):
        """Generate visualization and analysis reports"""
        # Create visualizations for each model
        for model_name, model_circuits in self.circuits.items():
            if not model_circuits:
                continue
            
            print(f"\n  Generating visualizations for {model_name}...")
            viz_dir = self.result_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            safe_model_name = model_name.replace('/', '-').replace(' ', '_')
            
            # Create organized folder structure
            model_folder = self._get_model_folder_name(model_name)
            model_viz_dir = viz_dir / model_folder
            model_viz_dir.mkdir(exist_ok=True)
            circuit_dir = model_viz_dir / "circuit_importance"
            circuit_dir.mkdir(exist_ok=True)
            network_dir = model_viz_dir / "network_diagrams"
            network_dir.mkdir(exist_ok=True)
            
            # Visualize each circuit
            for category, circuit in model_circuits.items():
                try:
                    # Create importance plots
                    fig = CircuitVisualizer.plot_circuit(circuit, top_k_nodes=30, top_k_edges=50)
                    fig_path = circuit_dir / f"{category}_circuit.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Saved importance plot: {fig_path}")
                    
                    # Create network plot with adaptive parameters based on circuit size
                    try:
                        # Adjust visualization parameters based on circuit size
                        num_nodes = len(circuit.nodes)
                        if num_nodes <= 100:
                            # Small circuits: show more detail
                            top_k_nodes, top_k_edges = 20, 30
                        elif num_nodes <= 1000:
                            # Medium circuits: moderate detail
                            top_k_nodes, top_k_edges = 30, 40
                        else:
                            # Large circuits: focus on most important elements
                            top_k_nodes, top_k_edges = 50, 60
                            
                        network_fig = CircuitVisualizer.create_network_plot(
                            circuit, top_k_nodes=top_k_nodes, top_k_edges=top_k_edges
                        )
                        network_path = network_dir / f"{category}_network.png"
                        network_fig.savefig(network_path, dpi=150, bbox_inches='tight')
                        plt.close(network_fig)
                        print(f"    Saved network plot: {network_path} ({num_nodes:,} total nodes, showing top {top_k_nodes})")
                    except Exception as e:
                        print(f"    Warning: Could not create network plot for {category}: {e}")
                    
                except Exception as e:
                    print(f"    Error creating visualization for {category}: {e}")
        
        # Generate final comprehensive report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive final analysis report"""
        print("\n  Generating final analysis report...")
        
        report = {
            'summary': {
                'total_models': len(self.circuits),
                'total_categories': len(set(cat for model_circuits in self.circuits.values() 
                                           for cat in model_circuits.keys())),
                'analysis_date': datetime.now().isoformat()
            },
            'models': {}
        }
        
        # Add circuit statistics for each model
        for model_name, model_circuits in self.circuits.items():
            safe_model_name = model_name.replace('/', '-').replace(' ', '_')
            model_report = {
                'model_name': model_name,
                'categories_analyzed': list(model_circuits.keys()),
                'circuit_statistics': {}
            }
            
            # Circuit statistics per category
            for category, circuit in model_circuits.items():
                # Load activations for faithfulness/completeness computation
                safe_model_name = model_name.replace('/', '-').replace(' ', '_')
                activation_file = self.result_dir / 'activations' / f"{safe_model_name}_{category}_activations.pt"
                
                faithfulness = 0.0
                completeness = 0.0
                
                if activation_file.exists():
                    try:
                        activations = torch.load(activation_file, map_location='cpu')
                        refusal_file = self.result_dir / 'refusal_labels' / f"{safe_model_name}_{category}_refusal.json"
                        if refusal_file.exists():
                            with open(refusal_file, 'r') as f:
                                refusal_labels = json.load(f)
                            
                            faithfulness = circuit.compute_faithfulness(activations, refusal_labels)
                            completeness = circuit.compute_completeness(activations, refusal_labels)
                    except Exception as e:
                        print(f"    Warning: Could not compute faithfulness/completeness for {category}: {e}")
                
                model_report['circuit_statistics'][category] = {
                    'num_nodes': len(circuit.nodes),
                    'num_edges': len(circuit.edges),
                    'top_node_importance': max(circuit.node_importances.values()) if circuit.node_importances else 0.0,
                    'avg_node_importance': sum(circuit.node_importances.values()) / len(circuit.node_importances) if circuit.node_importances else 0.0,
                    'top_edge_importance': max(circuit.edge_importances.values()) if circuit.edge_importances else 0.0,
                    'avg_edge_importance': sum(circuit.edge_importances.values()) / len(circuit.edge_importances) if circuit.edge_importances else 0.0,
                    'faithfulness': faithfulness,
                    'completeness': completeness
                }
            
            # Add comparison results if available
            if hasattr(self, 'comparison_results') and model_name in self.comparison_results:
                comp = self.comparison_results[model_name]
                model_report['comparisons'] = {}
                
                # Category-to-category comparison
                if 'category_to_category' in comp:
                    cat_comp = comp['category_to_category']
                    model_report['comparisons']['category_to_category'] = {
                        'average_similarity': float(cat_comp['average_similarity']),
                        'assessment': cat_comp['assessment'],
                        'assessment_confidence': cat_comp.get('assessment_confidence', 'UNKNOWN'),
                        'pairwise_similarities': {k: float(v) for k, v in cat_comp['similarities'].items()},
                    'statistics': {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in (cat_comp.get('statistics', {}).items() if cat_comp.get('statistics') else {})}
                    }
                
                # Safe vs toxic comparison
                if 'safe_vs_toxic' in comp:
                    st_comp = comp['safe_vs_toxic']
                    model_report['comparisons']['safe_vs_toxic'] = {
                        'average_similarity': float(st_comp['average_similarity']),
                        'category_similarities': {k: float(v) for k, v in st_comp['similarities'].items()},
                        'categories': st_comp['categories']
                }
            
            report['models'][model_name] = model_report
        
        # Save report with JSON serialization helper
        def json_serializable(obj):
            """Convert numpy types and other non-JSON types to JSON-serializable types"""
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serializable(v) for v in obj]
            else:
                return obj
        
        report_file = self.result_dir / "circuit_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(json_serializable(report), f, indent=2)
        
        # Generate text summary
        self._generate_text_summary(report)
        
        print(f"    Final report saved to: {report_file}")
    
    def _generate_text_summary(self, report: Dict):
        """Generate human-readable text summary report"""
        summary_file = self.result_dir / "circuit_analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CIRCUIT ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Models Analyzed: {report['summary']['total_models']}\n")
            f.write(f"Total Categories: {report['summary']['total_categories']}\n\n")
            
            for model_name, model_data in report['models'].items():
                f.write("-" * 80 + "\n")
                f.write(f"Model: {model_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Categories: {', '.join(model_data['categories_analyzed'])}\n\n")
                
                # Circuit statistics
                f.write("Circuit Statistics:\n")
                for category, stats in model_data['circuit_statistics'].items():
                    f.write(f"  {category}:\n")
                    f.write(f"    Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}\n")
                    f.write(f"    Avg Node Importance: {stats['avg_node_importance']:.4f}\n")
                    f.write(f"    Avg Edge Importance: {stats['avg_edge_importance']:.4f}\n")
                
                # Comparison results
                if 'comparisons' in model_data:
                    comparisons = model_data['comparisons']
                    
                    # Type 1: Category-to-Category
                    if 'category_to_category' in comparisons:
                        comp = comparisons['category_to_category']
                        f.write(f"\nType 1: Category-to-Category Comparison Results:\n")
                    f.write(f"  Average Similarity: {comp['average_similarity']:.4f}\n")
                    f.write(f"  Assessment: {comp['assessment']}\n")
                    f.write(f"  Confidence: {comp.get('assessment_confidence', 'UNKNOWN')}\n")
                    if 'statistics' in comp and comp['statistics']:
                        stats = comp['statistics']
                        f.write(f"  Statistical Test:\n")
                        f.write(f"    p-value: {stats.get('p_value', 'N/A'):.4f}\n")
                        f.write(f"    Significant: {stats.get('significant', False)}\n")
                        f.write(f"    95% CI: [{stats.get('confidence_interval_95', (0, 0))[0]:.4f}, {stats.get('confidence_interval_95', (0, 0))[1]:.4f}]\n")
                    f.write(f"  Pairwise Similarities:\n")
                    for pair, sim in comp['pairwise_similarities'].items():
                        f.write(f"    {pair}: {sim:.4f}\n")
                    
                    # Type 2: Safe vs Toxic
                    if 'safe_vs_toxic' in comparisons:
                        comp = comparisons['safe_vs_toxic']
                        f.write(f"\nType 2: Safe vs Toxic within Category Comparison Results:\n")
                        f.write(f"  Average Similarity: {comp['average_similarity']:.4f}\n")
                        f.write(f"  Category Similarities:\n")
                        for category, sim in comp['category_similarities'].items():
                            f.write(f"    {category}: {sim:.4f}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("CONCLUSIONS\n")
            f.write("=" * 80 + "\n")
            
            # Generate conclusions
            for model_name, model_data in report['models'].items():
                if 'comparisons' in model_data:
                    comparisons = model_data['comparisons']
                    
                    f.write(f"\n{model_name}:\n")
                    
                    # Type 1 conclusions
                    if 'category_to_category' in comparisons:
                        comp = comparisons['category_to_category']
                    assessment = comp['assessment']
                    confidence = comp.get('assessment_confidence', 'UNKNOWN')
                    avg_sim = comp['average_similarity']
                    
                        f.write(f"  Type 1: Category-to-Category Analysis\n")
                    if assessment == "MONOLITHIC":
                            f.write(f"    The model shows MONOLITHIC refusal behavior (similarity: {avg_sim:.3f}).\n")
                            f.write(f"    This suggests shared circuits across refusal categories.\n")
                    elif assessment == "MODULAR":
                            f.write(f"    The model shows MODULAR refusal behavior (similarity: {avg_sim:.3f}).\n")
                            f.write(f"    This suggests category-specific circuits for different refusal types.\n")
                        else:
                            f.write(f"    The model shows PARTIALLY MODULAR refusal behavior (similarity: {avg_sim:.3f}).\n")
                            f.write(f"    This suggests a mix of shared and category-specific circuits.\n")
                        
                        if confidence == "HIGH":
                            f.write(f"    This assessment is statistically significant (p < 0.05).\n")
                    else:
                            f.write(f"    Note: This assessment has low statistical confidence.\n")
                    
                    # Type 2 conclusions
                    if 'safe_vs_toxic' in comparisons:
                        comp = comparisons['safe_vs_toxic']
                        avg_sim = comp['average_similarity']
                        
                        f.write(f"  Type 2: Safe vs Toxic within Category Analysis\n")
                        f.write(f"    Average safe-toxic similarity: {avg_sim:.3f}\n")
                        if avg_sim < 0.3:
                            f.write(f"    Low similarity suggests separate circuits for safe vs toxic prompts within categories.\n")
                        elif avg_sim > 0.7:
                            f.write(f"    High similarity suggests shared circuits for safe and toxic prompts within categories.\n")
                    else:
                            f.write(f"    Moderate similarity suggests partially shared circuits for safe and toxic prompts.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"    Text summary saved to: {summary_file}")
    


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