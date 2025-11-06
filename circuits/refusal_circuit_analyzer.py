"""
Main analysis script for discovering refusal circuits across categories
"""

import os
import sys
import yaml
import torch
from tqdm import tqdm
from typing import Dict, List, Any
import json
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, current_dir)

from src.data_utils import load_all_categories
from src.models import load_model_with_hooks
from src.activation_utils import ActivationManager
from circuits.circuits_utils import CircuitDiscoverer, CircuitConfig, SparseFeatureCircuit, CircuitVisualizer, compare_circuits_across_categories
from circuits.circuit_discovery_with_saes import SAECircuitDiscoverer
from src.sae_trainer import SAEManager, get_activation_files

class RefusalCircuitAnalyzer:
    """Main class for analyzing refusal circuits across categories"""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.result_dir = Path(self.config.get('result_dir', 'results'))
        self.activation_manager = ActivationManager(str(self.result_dir / 'activations'))
        self.sae_manager = SAEManager(self.config.get('sae_dir', str(self.result_dir / 'saes')))
        self.circuits = {}  # Store discovered circuits: {model_name: {category: circuit}}
    
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
            
            self.sae_manager.train_saes_for_model(
                model_name=model_name,
                activation_files=activation_files,
                layers=self.config['activation_layers'],
                sae_hidden_dim=self.config.get('sae_hidden_dim', 8192),
                max_samples=self.config.get('sae_max_samples', 100000),
                batch_size=self.config.get('sae_batch_size', 512),
                epochs=self.config.get('sae_epochs', 100)
            )
    
    def _discover_circuits_for_model(self, model_name: str):
        """Discover circuits for all categories of a model"""
        # Load trained SAEs
        saes = self.sae_manager.load_saes_for_model(model_name, self.config['activation_layers'])
        if not saes:
            print(f"  No trained SAEs found for {model_name}")
            print(f"  Run with train_saes=true or ensure SAEs are already trained")
            return
        
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
        
        # Discover circuit for each category
        model_circuits = {}
        for category in self.config['categories']:
            print(f"  Discovering circuit for {category}...")
            
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
            
            try:
                circuit = discoverer.discover_circuit_with_saes(
                    activation_file=str(activation_file),
                    refusal_labels=refusal_labels,
                    model_name=model_name
                )
                
                model_circuits[category] = circuit
                
                # Save circuit
                circuit_dir = self.result_dir / "circuits"
                circuit_dir.mkdir(exist_ok=True)
                circuit_file = circuit_dir / f"{safe_model_name}_{category}_circuit.json"
                
                circuit_data = {
                    'model_name': model_name,
                    'category': category,
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
        
        self.circuits[model_name] = model_circuits
    
    def _compare_circuits(self):
        """Compare circuits across categories to test monolithic vs modular hypothesis"""
        for model_name, model_circuits in self.circuits.items():
            if len(model_circuits) < 2:
                print(f"  {model_name}: Need at least 2 categories for comparison")
                continue
            
            print(f"\n  Comparing circuits for {model_name}...")
            similarities = compare_circuits_across_categories(model_circuits)
            
            # Assess modularity
            similarity_values = list(similarities.values())
            if similarity_values:
                avg_similarity = sum(similarity_values) / len(similarity_values)
                monolithic_threshold = self.config.get('analysis', {}).get('modularity_assessment', {}).get('monolithic_threshold', 0.8)
                partially_modular_threshold = self.config.get('analysis', {}).get('modularity_assessment', {}).get('partially_modular_threshold', 0.5)
                
                if avg_similarity >= monolithic_threshold:
                    assessment = "MONOLITHIC"
                elif avg_similarity >= partially_modular_threshold:
                    assessment = "PARTIALLY MODULAR"
                else:
                    assessment = "MODULAR"
                
                print(f"    Average similarity: {avg_similarity:.3f}")
                print(f"    Assessment: {assessment}")
                
                # Save comparison results
                comparison_file = self.result_dir / "circuits" / f"{model_name.replace('/', '-')}_comparison.json"
                comparison_data = {
                    'model_name': model_name,
                    'similarities': {k: float(v) for k, v in similarities.items()},
                    'average_similarity': float(avg_similarity),
                    'assessment': assessment
                }
                
                with open(comparison_file, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
    
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
            
            # Visualize each circuit
            for category, circuit in model_circuits.items():
                try:
                    fig = CircuitVisualizer.plot_circuit(circuit, top_k_nodes=30, top_k_edges=50)
                    fig_path = viz_dir / f"{safe_model_name}_{category}_circuit.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    print(f"    Saved visualization: {fig_path}")
                except Exception as e:
                    print(f"    Error creating visualization for {category}: {e}")
    
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