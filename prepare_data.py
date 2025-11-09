"""
Simple example script showing how to use ORBenchDataProcessor 
to prepare data for the inference pipeline.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.orbench_data_processor import ORBenchDataProcessor

def prepare_orbench_data():
    """
    Prepare OR-Bench data for inference experiments.
    """
    # Configuration
    dataset_dir = "datasets/or-bench"
    output_dir = "processed_orbench_data"
    
    # All categories from OR-bench
    categories = ["violence", "deception", "harassment", "harmful", "hate", "illegal", "privacy", "self-harm", "sexual", "unethical"]
    samples_per_category = 100
    
    print(f"Processing OR-Bench data...")
    print(f"  Input: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Categories: {categories}")
    print(f"  Samples per category: {samples_per_category}")
    
    # Initialize processor
    processor = ORBenchDataProcessor(dataset_dir)
    
    # Create balanced dataset
    balanced_data = processor.create_balanced_dataset(
        categories=categories,
        samples_per_category=samples_per_category,
        safe_toxic_ratio=0.5,  # 50% safe, 50% toxic
        include_hard=True,
        shuffle=True
    )
    
    # Save processed data
    saved_files = processor.save_processed_data(
        data=balanced_data,
        output_dir=output_dir,
        format='json'
    )
    
    print(f"\nData processing complete!")
    print(f"Files saved to {output_dir}:")
    for category, filepath in saved_files.items():
        sample_count = len(balanced_data[category])
        refuse_count = sum(1 for s in balanced_data[category] if s['refusal_label'])
        safe_count = sample_count - refuse_count
        print(f"  {category}: {sample_count} samples ({safe_count} safe, {refuse_count} toxic)")
    
    # Update instructions
    print(f"\nTo use with inference script:")
    print(f"1. Update configurations/orbench_run.yaml:")
    print(f'   dataset_dir: "{output_dir}"')
    print(f"2. Run: python src/run_inference.py --config configurations/orbench_run.yaml")

if __name__ == "__main__":
    prepare_orbench_data()