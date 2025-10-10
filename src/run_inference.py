'''
PLEACEHOLDER UNTESTED SCRIPT TO RUN INFERENCE 
'''

import os
import yaml
import torch
from tqdm import tqdm
import json
from src.data_utils import load_orbench_dataset # yet to implement
from src.models import load_model_with_hooks # yet to implement

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
    data_by_category = {}
    for cat in cfg['categories']:
        data_by_category[cat] = load_orbench_dataset(
            dataset_dir=cfg['dataset_dir'],
            category=cat,
            num_samples=cfg['num_samples_per_category'],
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
                batch_prompts = prompts[i:i+batch_size]
                # Run inference and collect activations
                outputs, activations = model_obj.run_with_activations(batch_prompts)
                
                # Simple refusal detection via OR-Bench label (assume batch_prompts includes label)
                batch_labels = [p['refusal_label'] for p in batch_prompts]
                all_refusal_labels.extend(batch_labels)
                all_activations.append(activations)

            # Save refusal labels
            refusal_file = os.path.join(refusal_dir, f"{model_name}_{category}_refusal.json")
            with open(refusal_file, 'w') as f:
                json.dump(all_refusal_labels, f)

            # Save activations
            if cfg.get('save_activations', True):
                activations_file = os.path.join(activation_dir, f"{model_name}_{category}_activations.pt")
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
