'''
Placeholder file for loading, sampling, and shufflinf datasets 
'''

import json
import random
from typing import List, Dict, Any
from pathlib import Path

def load_dataset(dataset_dir: str, category: str, num_samples: int = 100, shuffle: bool = True) -> List[Dict[str, Any]]:
    """
    Load OR-Bench dataset for a specific category.
    Assumes dataset structure with JSON files containing 'prompt' and 'refusal_label'
    """
    dataset_path = Path(dataset_dir) / f"{category}.json"
    
    if not dataset_path.exists():
        # Try JSONL format
        dataset_path = Path(dataset_dir) / f"{category}.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    data = []
    if dataset_path.suffix == '.jsonl':
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    # Ensure each item has required fields
    processed_data = []
    for item in data:
        if isinstance(item, dict):
            # Extract prompt and refusal label
            prompt = item.get('prompt', '')
            refusal_label = item.get('refusal_label', False)
            
            # Handle different label formats
            if isinstance(refusal_label, str):
                refusal_label = refusal_label.lower() in ['true', 'yes', '1', 'refusal']
            elif isinstance(refusal_label, int):
                refusal_label = bool(refusal_label)
                
            processed_data.append({
                'prompt': prompt,
                'refusal_label': refusal_label,
                'category': category,
                'original_data': item
            })
    
    if shuffle:
        random.shuffle(processed_data)
    
    if num_samples > 0:
        processed_data = processed_data[:num_samples]
    
    return processed_data

def load_all_categories(dataset_dir: str, categories: List[str], num_samples_per_category: int = 100, shuffle: bool = True) -> Dict[str, List[Dict]]:
    """Load data for all specified categories"""
    data_by_category = {}
    for category in categories:
        try:
            data_by_category[category] = load_dataset(
                dataset_dir=dataset_dir,
                category=category,
                num_samples=num_samples_per_category,
                shuffle=shuffle
            )
            print(f"Loaded {len(data_by_category[category])} samples for category: {category}")
        except Exception as e:
            print(f"Error loading category {category}: {e}")
            data_by_category[category] = []
    
    return data_by_category