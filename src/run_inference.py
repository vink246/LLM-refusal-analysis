'''
PLEACEHOLDER UNTESTED SCRIPT TO RUN INFERENCE 
'''

import os
import sys
import yaml
import torch
from tqdm import tqdm
import json

# Add current directory and src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from data_utils import load_all_categories
from models import load_model_with_hooks

def run_experiment(config_path):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['result_dir'], exist_ok=True)
    activation_dir = os.path.join(cfg['result_dir'], 'activations')
    os.makedirs(activation_dir, exist_ok=True)
    refusal_dir = os.path.join(cfg['result_dir'], 'refusal_labels')
    os.makedirs(refusal_dir, exist_ok=True)

    # Load dataset
    data_by_category = load_all_categories(
        dataset_dir=cfg['dataset_dir'],
        categories=cfg['categories'],
        num_samples_per_category=cfg['num_samples_per_category'],
        shuffle=cfg.get('shuffle', True)
    )

    # Load models
    models = {}
    for model_name in cfg['models']:
        print(f"Loading {model_name} ...")
        models[model_name] = load_model_with_hooks(
            model_name,
            layers=cfg.get('activation_layers', []),
            device=cfg.get('device', 'cuda')
        )

    # Run inference per model per category
    for model_name, model_obj in models.items():
        for category, prompts in data_by_category.items():
            print(f"Running {model_name} on category {category} ...")
            batch_size = cfg.get('batch_size', 4)

            all_refusal_labels = []
            all_activations = []

            for i in tqdm(range(0, len(prompts), batch_size)):
                batch_data = prompts[i:i+batch_size]
                # Extract just the prompt strings for the model
                batch_prompts = [p['prompt'] for p in batch_data]
                # Run inference and collect activations
                outputs, activations = model_obj.run_with_activations(batch_prompts)
                
                # Simple refusal detection via OR-Bench label (assume batch_data includes label)
                batch_labels = [p['refusal_label'] for p in batch_data]
                all_refusal_labels.extend(batch_labels)
                all_activations.append(activations)

            # Save refusal labels (sanitize model name for filename)
            safe_model_name = model_name.replace('/', '-').replace(' ', '_')
            refusal_file = os.path.join(refusal_dir, f"{safe_model_name}_{category}_refusal.json")
            with open(refusal_file, 'w') as f:
                json.dump(all_refusal_labels, f)

            # Save activations
            if cfg.get('save_activations', True):
                activations_file = os.path.join(activation_dir, f"{safe_model_name}_{category}_activations.pt")
                # Concatenate activation dicts along batch dimension
                merged_activations = {}
                for act_dict in all_activations:
                    for layer, tensor in act_dict.items():
                        if layer not in merged_activations:
                            merged_activations[layer] = []
                        merged_activations[layer].append(tensor)
                # Stack along first dimension
                for layer in merged_activations:
                    merged_activations[layer] = torch.cat(merged_activations[layer], dim=0)
                torch.save(merged_activations, activations_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_experiment(args.config)
