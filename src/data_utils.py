'''
Placeholder file for loading, sampling, and shufflinf datasets 
'''

import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Try relative import first, fall back to absolute import
try:
    from .orbench_data_processor import ORBenchDataProcessor
except ImportError:
    from orbench_data_processor import ORBenchDataProcessor

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

def load_orbench_dataset(dataset_dir: str, category: str, num_samples: int = 100, 
                        safe_toxic_ratio: float = 0.5, shuffle: bool = True) -> List[Dict[str, Any]]:
    """
    Load OR-Bench dataset for a specific category using the ORBenchDataProcessor.
    
    Args:
        dataset_dir: Path to OR-Bench dataset directory
        category: Category name to load
        num_samples: Total number of samples to load
        safe_toxic_ratio: Ratio of safe to toxic samples (0.5 = equal split)
        shuffle: Whether to shuffle the data
        
    Returns:
        List of processed samples with prompt, category, and refusal_label fields
    """
    processor = ORBenchDataProcessor(dataset_dir)
    category_data = processor.extract_category_data(
        category=category,
        num_safe_samples=int(num_samples * safe_toxic_ratio),
        num_toxic_samples=int(num_samples * (1 - safe_toxic_ratio)),
        include_hard=False,
        shuffle=shuffle
    )
    return category_data

def load_all_categories(dataset_dir: str, categories: List[str], num_samples_per_category: int = 100, shuffle: bool = True) -> Dict[str, List[Dict]]:
    """Load data for all specified categories"""
    data_by_category = {}
    
    # Check if this is OR-Bench dataset (contains CSV files)
    dataset_path = Path(dataset_dir)
    orbench_files = ['or-bench-80k.csv', 'or-bench-hard-1k.csv', 'or-bench-toxic.csv']
    is_orbench = all((dataset_path / f).exists() for f in orbench_files)
    
    for category in categories:
        try:
            if is_orbench:
                # Use OR-Bench specific loader
                data_by_category[category] = load_orbench_dataset(
                    dataset_dir=dataset_dir,
                    category=category,
                    num_samples=num_samples_per_category,
                    safe_toxic_ratio=0.5,  # Equal split between safe and toxic
                    shuffle=shuffle
                )
            else:
                # Use generic JSON/JSONL loader
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