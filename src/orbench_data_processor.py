"""
OR-Bench Data Processor

This module extracts and organizes OR-Bench dataset data to work with the inference pipeline.
It processes the CSV files and creates structured data with appropriate refusal labels.
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random


class ORBenchDataProcessor:
    """
    Processes OR-Bench datasets to extract and organize data for LLM refusal analysis.
    
    The OR-Bench dataset consists of three main files:
    - or-bench-80k.csv: Large dataset of prompts that should NOT be refused (safe prompts)
    - or-bench-hard-1k.csv: Smaller dataset of borderline prompts that should NOT be refused
    - or-bench-toxic.csv: Dataset of prompts that SHOULD be refused (toxic prompts)
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize the processor with the dataset directory.
        
        Args:
            dataset_dir: Path to the directory containing OR-Bench CSV files
        """
        self.dataset_dir = Path(dataset_dir)
        self.validate_dataset_files()
        
    def validate_dataset_files(self):
        """Validate that all required OR-Bench files exist."""
        required_files = [
            "or-bench-80k.csv",
            "or-bench-hard-1k.csv", 
            "or-bench-toxic.csv"
        ]
        
        for file in required_files:
            file_path = self.dataset_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required OR-Bench file not found: {file_path}")
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from a CSV file in the dataset directory.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = self.dataset_dir / filename
        return pd.read_csv(file_path)
    
    def process_safe_prompts(self, filename: str, dataset_type: str) -> List[Dict[str, Any]]:
        """
        Process prompts that should NOT be refused (safe prompts).
        
        Args:
            filename: Name of the CSV file containing safe prompts
            dataset_type: Type identifier for the dataset (e.g., "safe_80k", "hard_1k")
            
        Returns:
            List of processed prompt dictionaries
        """
        df = self.load_csv_data(filename)
        processed_data = []
        
        for _, row in df.iterrows():
            prompt = str(row['prompt']).strip()
            category = str(row['category']).strip()
            
            # Skip empty prompts
            if not prompt or prompt.lower() == 'nan':
                continue
                
            processed_data.append({
                'prompt': prompt,
                'category': category,
                'refusal_label': False,  # These prompts should NOT be refused
                'dataset_type': dataset_type,
                'original_source': filename
            })
        
        return processed_data
    
    def process_toxic_prompts(self) -> List[Dict[str, Any]]:
        """
        Process prompts that SHOULD be refused (toxic prompts).
        
        Returns:
            List of processed prompt dictionaries
        """
        df = self.load_csv_data("or-bench-toxic.csv")
        processed_data = []
        
        for _, row in df.iterrows():
            prompt = str(row['prompt']).strip()
            category = str(row['category']).strip()
            
            # Skip empty prompts
            if not prompt or prompt.lower() == 'nan':
                continue
                
            processed_data.append({
                'prompt': prompt,
                'category': category,
                'refusal_label': True,  # These prompts SHOULD be refused
                'dataset_type': 'toxic',
                'original_source': 'or-bench-toxic.csv'
            })
        
        return processed_data
    
    def get_available_categories(self) -> Dict[str, List[str]]:
        """
        Get all available categories from each dataset file.
        
        Returns:
            Dictionary mapping dataset type to list of categories
        """
        categories = {}
        
        # Safe prompts from 80k dataset
        df_80k = self.load_csv_data("or-bench-80k.csv")
        categories['safe_80k'] = sorted(df_80k['category'].unique().tolist())
        
        # Hard prompts (borderline safe)
        df_hard = self.load_csv_data("or-bench-hard-1k.csv")
        categories['hard_1k'] = sorted(df_hard['category'].unique().tolist())
        
        # Toxic prompts
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        categories['toxic'] = sorted(df_toxic['category'].unique().tolist())
        
        # All unique categories
        all_categories = set()
        for cat_list in categories.values():
            all_categories.update(cat_list)
        categories['all'] = sorted(list(all_categories))
        
        return categories
    
    def extract_category_data(self, 
                            category: str, 
                            num_safe_samples: int = 50,
                            num_toxic_samples: int = 50,
                            include_hard: bool = True,
                            shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        Extract data for a specific category from all datasets.
        
        Args:
            category: Category name to extract
            num_safe_samples: Number of safe samples per dataset type
            num_toxic_samples: Number of toxic samples
            include_hard: Whether to include hard (borderline) samples
            shuffle: Whether to shuffle the results
            
        Returns:
            List of processed samples for the category
        """
        category_data = []
        
        # Extract safe prompts from 80k dataset
        df_80k = self.load_csv_data("or-bench-80k.csv")
        safe_80k = df_80k[df_80k['category'] == category]
        if len(safe_80k) > 0:
            if shuffle:
                safe_80k = safe_80k.sample(n=min(num_safe_samples, len(safe_80k)), random_state=42)
            else:
                safe_80k = safe_80k.head(num_safe_samples)
            
            for _, row in safe_80k.iterrows():
                category_data.append({
                    'prompt': str(row['prompt']).strip(),
                    'category': category,
                    'refusal_label': False,
                    'dataset_type': 'safe_80k',
                    'original_source': 'or-bench-80k.csv'
                })
        
        # Extract hard prompts if requested
        if include_hard:
            df_hard = self.load_csv_data("or-bench-hard-1k.csv")
            hard_samples = df_hard[df_hard['category'] == category]
            if len(hard_samples) > 0:
                if shuffle:
                    hard_samples = hard_samples.sample(n=min(num_safe_samples, len(hard_samples)), random_state=42)
                else:
                    hard_samples = hard_samples.head(num_safe_samples)
                
                for _, row in hard_samples.iterrows():
                    category_data.append({
                        'prompt': str(row['prompt']).strip(),
                        'category': category,
                        'refusal_label': False,
                        'dataset_type': 'hard_1k',
                        'original_source': 'or-bench-hard-1k.csv'
                    })
        
        # Extract toxic prompts
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        toxic_samples = df_toxic[df_toxic['category'] == category]
        if len(toxic_samples) > 0:
            if shuffle:
                toxic_samples = toxic_samples.sample(n=min(num_toxic_samples, len(toxic_samples)), random_state=42)
            else:
                toxic_samples = toxic_samples.head(num_toxic_samples)
            
            for _, row in toxic_samples.iterrows():
                category_data.append({
                    'prompt': str(row['prompt']).strip(),
                    'category': category,
                    'refusal_label': True,
                    'dataset_type': 'toxic',
                    'original_source': 'or-bench-toxic.csv'
                })
        
        if shuffle:
            random.shuffle(category_data)
        
        return category_data
    
    def create_balanced_dataset(self, 
                              categories: List[str],
                              samples_per_category: int = 100,
                              safe_toxic_ratio: float = 0.5,
                              include_hard: bool = True,
                              shuffle: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a balanced dataset across multiple categories.
        
        Args:
            categories: List of categories to include
            samples_per_category: Total samples per category
            safe_toxic_ratio: Ratio of safe to toxic samples (0.5 = equal split)
            include_hard: Whether to include hard samples
            shuffle: Whether to shuffle results
            
        Returns:
            Dictionary mapping category names to sample lists
        """
        balanced_data = {}
        
        toxic_samples_per_cat = int(samples_per_category * (1 - safe_toxic_ratio))
        safe_samples_per_cat = samples_per_category - toxic_samples_per_cat
        
        # If including hard samples, split safe samples between 80k and hard
        if include_hard:
            safe_80k_samples = safe_samples_per_cat // 2
            hard_samples = safe_samples_per_cat - safe_80k_samples
        else:
            safe_80k_samples = safe_samples_per_cat
            hard_samples = 0
        
        available_categories = self.get_available_categories()['all']
        
        for category in categories:
            if category not in available_categories:
                print(f"Warning: Category '{category}' not found in dataset. Skipping...")
                continue
                
            category_data = self.extract_category_data(
                category=category,
                num_safe_samples=safe_80k_samples,
                num_toxic_samples=toxic_samples_per_cat,
                include_hard=include_hard,
                shuffle=shuffle
            )
            
            balanced_data[category] = category_data
            print(f"Extracted {len(category_data)} samples for category '{category}'")
        
        return balanced_data
    
    def save_processed_data(self, 
                           data: Dict[str, List[Dict[str, Any]]], 
                           output_dir: str,
                           format: str = 'json') -> Dict[str, str]:
        """
        Save processed data to files.
        
        Args:
            data: Dictionary of category data
            output_dir: Output directory path
            format: Output format ('json' or 'jsonl')
            
        Returns:
            Dictionary mapping category names to output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for category, samples in data.items():
            if format == 'json':
                filename = f"{category}.json"
                file_path = output_path / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
            
            elif format == 'jsonl':
                filename = f"{category}.jsonl"
                file_path = output_path / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            saved_files[category] = str(file_path)
            print(f"Saved {len(samples)} samples for '{category}' to {file_path}")
        
        return saved_files
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the OR-Bench datasets.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}
        
        # 80k dataset stats
        df_80k = self.load_csv_data("or-bench-80k.csv")
        stats['or-bench-80k'] = {
            'total_samples': len(df_80k),
            'categories': df_80k['category'].nunique(),
            'category_counts': df_80k['category'].value_counts().to_dict(),
            'refusal_type': 'safe_prompts'
        }
        
        # Hard 1k dataset stats
        df_hard = self.load_csv_data("or-bench-hard-1k.csv")
        stats['or-bench-hard-1k'] = {
            'total_samples': len(df_hard),
            'categories': df_hard['category'].nunique(),
            'category_counts': df_hard['category'].value_counts().to_dict(),
            'refusal_type': 'borderline_prompts'
        }
        
        # Toxic dataset stats
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        stats['or-bench-toxic'] = {
            'total_samples': len(df_toxic),
            'categories': df_toxic['category'].nunique(),
            'category_counts': df_toxic['category'].value_counts().to_dict(),
            'refusal_type': 'toxic_prompts'
        }
        
        # Overall stats
        all_categories = set()
        all_categories.update(df_80k['category'].unique())
        all_categories.update(df_hard['category'].unique())
        all_categories.update(df_toxic['category'].unique())
        
        stats['overall'] = {
            'total_unique_categories': len(all_categories),
            'all_categories': sorted(list(all_categories)),
            'total_samples': len(df_80k) + len(df_hard) + len(df_toxic)
        }
        
        return stats


def main():
    """
    Example usage of the ORBenchDataProcessor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Process OR-Bench dataset for LLM refusal analysis")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                       help="Path to OR-Bench dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--categories", type=str, nargs="+",
                       default=["violence", "deception", "privacy", "illegal_activity"],
                       help="Categories to extract")
    parser.add_argument("--samples_per_category", type=int, default=100,
                       help="Number of samples per category")
    parser.add_argument("--safe_toxic_ratio", type=float, default=0.5,
                       help="Ratio of safe to toxic samples (0.5 = equal split)")
    parser.add_argument("--format", type=str, choices=['json', 'jsonl'], default='json',
                       help="Output format")
    parser.add_argument("--show_stats", action='store_true',
                       help="Show dataset statistics")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ORBenchDataProcessor(args.dataset_dir)
    
    # Show statistics if requested
    if args.show_stats:
        print("OR-Bench Dataset Statistics:")
        print("=" * 50)
        stats = processor.get_dataset_statistics()
        for dataset, stat_info in stats.items():
            print(f"\n{dataset}:")
            for key, value in stat_info.items():
                if key == 'category_counts' and isinstance(value, dict):
                    print(f"  {key}:")
                    for cat, count in sorted(value.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"    {cat}: {count}")
                    if len(value) > 10:
                        print(f"    ... and {len(value) - 10} more categories")
                else:
                    print(f"  {key}: {value}")
    
    # Check available categories
    available_categories = processor.get_available_categories()
    print(f"\nAvailable categories: {available_categories['all']}")
    
    # Filter requested categories to only include available ones
    valid_categories = [cat for cat in args.categories if cat in available_categories['all']]
    if len(valid_categories) != len(args.categories):
        missing = set(args.categories) - set(valid_categories)
        print(f"Warning: Categories not found in dataset: {missing}")
    
    if not valid_categories:
        print("Error: No valid categories found!")
        return
    
    # Create balanced dataset
    print(f"\nProcessing {len(valid_categories)} categories...")
    balanced_data = processor.create_balanced_dataset(
        categories=valid_categories,
        samples_per_category=args.samples_per_category,
        safe_toxic_ratio=args.safe_toxic_ratio,
        include_hard=True,
        shuffle=True
    )
    
    # Save processed data
    saved_files = processor.save_processed_data(
        data=balanced_data,
        output_dir=args.output_dir,
        format=args.format
    )
    
    print(f"\nProcessing complete! Files saved to: {args.output_dir}")
    for category, file_path in saved_files.items():
        print(f"  {category}: {file_path}")


if __name__ == "__main__":
    main()