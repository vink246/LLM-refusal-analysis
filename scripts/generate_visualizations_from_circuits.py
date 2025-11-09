#!/usr/bin/env python3
"""
Generate visualizations (importance plots, network diagrams, heatmaps) from
saved circuit JSON files in `results/circuits/`.

Usage:
  python scripts/generate_visualizations_from_circuits.py \
      --circuits-dir results/circuits --out-dir results/visualizations

This is a small utility to recreate the same visualizations produced by
`run_circuit_analysis.py` when you already have circuit JSON files saved.
"""
import argparse
import json
from pathlib import Path
from circuits.circuits_utils import SparseFeatureCircuit, CircuitVisualizer
import matplotlib.pyplot as plt

MODEL_FOLDER_MAP = {
    "meta-llama/Llama-2-7b-chat-hf": "llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v0.1"
}


def load_circuit_from_json(circuit_json_path: Path) -> (str, str, SparseFeatureCircuit):
    with open(circuit_json_path, 'r') as f:
        data = json.load(f)

    model_name = data.get('model_name', '')
    category = data.get('category', circuit_json_path.stem)

    circuit = SparseFeatureCircuit()

    # Load node_importances and nodes
    node_importances = data.get('node_importances', {})
    nodes = data.get('nodes', {})
    for node_id, node_meta in nodes.items():
        # ensure internal structures use the same ids
        importance = node_importances.get(node_id, node_meta.get('importance', 0.0))
        circuit.nodes[node_id] = node_meta
        circuit.node_importances[node_id] = float(importance)

    # Load edges and edge_importances
    edge_importances = data.get('edge_importances', {})
    edges = data.get('edges', {})
    # edges in saved format were keyed like "src_tgt": {source, target, importance}
    for key, edge_meta in edges.items():
        src = edge_meta.get('source')
        tgt = edge_meta.get('target')
        importance = edge_importances.get(f"{src}_{tgt}", edge_meta.get('importance', 0.0))
        circuit.edges[(src, tgt)] = edge_meta
        circuit.edge_importances[(src, tgt)] = float(importance)

    return model_name, category, circuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuits-dir', type=str, default='results/circuits')
    parser.add_argument('--out-dir', type=str, default='results/visualizations')
    parser.add_argument('--top-nodes', type=int, default=30)
    parser.add_argument('--top-edges', type=int, default=50)
    args = parser.parse_args()

    circuits_dir = Path(args.circuits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not circuits_dir.exists():
        print(f"Circuits directory not found: {circuits_dir}")
        return

    # Iterate over circuit files
    for circuit_file in sorted(circuits_dir.glob('*_circuit.json')):
        print(f"Loading: {circuit_file}")
        model_name, category, circuit = load_circuit_from_json(circuit_file)

        # Determine model folder name
        model_folder = MODEL_FOLDER_MAP.get(model_name, model_name.replace('/', '-').replace(' ', '_').lower())
        model_viz_dir = out_dir / model_folder
        circuit_dir = model_viz_dir / 'circuit_importance'
        network_dir = model_viz_dir / 'network_diagrams'
        circuit_dir.mkdir(parents=True, exist_ok=True)
        network_dir.mkdir(parents=True, exist_ok=True)

        # Importance plot
        try:
            fig = CircuitVisualizer.plot_circuit(circuit, top_k_nodes=args.top_nodes, top_k_edges=args.top_edges)
            fig_path = circuit_dir / f"{category}_circuit.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved importance plot: {fig_path}")
        except Exception as e:
            print(f"  Error creating importance plot for {category}: {e}")

        # Network plot
        try:
            num_nodes = len(circuit.nodes)
            if num_nodes <= 100:
                top_k_nodes, top_k_edges = 20, 30
            elif num_nodes <= 1000:
                top_k_nodes, top_k_edges = 30, 40
            else:
                top_k_nodes, top_k_edges = 50, 60

            network_fig = CircuitVisualizer.create_network_plot(circuit, top_k_nodes=top_k_nodes, top_k_edges=top_k_edges)
            network_path = network_dir / f"{category}_network.png"
            network_fig.savefig(network_path, dpi=150, bbox_inches='tight')
            plt.close(network_fig)
            print(f"  Saved network plot: {network_path} ({num_nodes} nodes, showing top {top_k_nodes})")
        except Exception as e:
            print(f"  Warning: Could not create network plot for {category}: {e}")

    print('\nDone. Visualizations saved to:', out_dir)


if __name__ == '__main__':
    main()
