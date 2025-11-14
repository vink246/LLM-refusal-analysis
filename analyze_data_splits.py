#!/usr/bin/env python3
"""
Analyze data splits across safe, toxic, and hard samples for OR-Bench dataset.
This script helps identify data imbalances that may affect SAE training.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orbench_data_processor import ORBenchDataProcessor


def analyze_data_splits(dataset_dir: str, output_dir: str = "results/data_analysis"):
    """
    Analyze and visualize data splits across safe, toxic, and hard samples
    for all OR-Bench categories.
    """
    print("=" * 80)
    print("OR-Bench Data Split Analysis")
    print("=" * 80)
    
    # Initialize processor
    processor = ORBenchDataProcessor(dataset_dir)
    
    # Get all available categories
    available_categories = processor.get_available_categories()['all']
    print(f"\nFound {len(available_categories)} categories: {', '.join(available_categories)}")
    
    # Load all datasets
    df_80k = processor.load_csv_data("or-bench-80k.csv")
    df_hard = processor.load_csv_data("or-bench-hard-1k.csv")
    df_toxic = processor.load_csv_data("or-bench-toxic.csv")
    
    # Analyze splits per category
    category_stats = {}
    
    for category in available_categories:
        safe_80k = len(df_80k[df_80k['category'] == category])
        hard_1k = len(df_hard[df_hard['category'] == category])
        toxic = len(df_toxic[df_toxic['category'] == category])
        
        total = safe_80k + hard_1k + toxic
        
        category_stats[category] = {
            'safe_80k': safe_80k,
            'hard_1k': hard_1k,
            'toxic': toxic,
            'total': total,
            'safe_ratio': (safe_80k + hard_1k) / total if total > 0 else 0,
            'toxic_ratio': toxic / total if total > 0 else 0
        }
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Category-wise Data Split Summary")
    print("=" * 80)
    print(f"{'Category':<20} {'Safe (80k)':<15} {'Hard (1k)':<15} {'Toxic':<15} {'Total':<10} {'Toxic %':<10}")
    print("-" * 80)
    
    for category in sorted(available_categories):
        stats = category_stats[category]
        toxic_pct = stats['toxic_ratio'] * 100
        print(f"{category:<20} {stats['safe_80k']:<15} {stats['hard_1k']:<15} "
              f"{stats['toxic']:<15} {stats['total']:<10} {toxic_pct:<10.2f}%")
    
    # Overall statistics
    total_safe_80k = len(df_80k)
    total_hard_1k = len(df_hard)
    total_toxic = len(df_toxic)
    total_all = total_safe_80k + total_hard_1k + total_toxic
    
    print("\n" + "=" * 80)
    print("Overall Dataset Statistics")
    print("=" * 80)
    print(f"Safe (80k dataset): {total_safe_80k:,} samples ({total_safe_80k/total_all*100:.2f}%)")
    print(f"Hard (1k dataset):  {total_hard_1k:,} samples ({total_hard_1k/total_all*100:.2f}%)")
    print(f"Toxic dataset:     {total_toxic:,} samples ({total_toxic/total_all*100:.2f}%)")
    print(f"Total:              {total_all:,} samples")
    print(f"\nToxic/Safe Ratio: {total_toxic/(total_safe_80k+total_hard_1k):.4f}")
    
    # Create visualizations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar plot: Counts per category (stacked)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Stacked bar chart
    categories = sorted(available_categories)
    safe_counts = [category_stats[cat]['safe_80k'] for cat in categories]
    hard_counts = [category_stats[cat]['hard_1k'] for cat in categories]
    toxic_counts = [category_stats[cat]['toxic'] for cat in categories]
    
    x_pos = range(len(categories))
    width = 0.6
    
    axes[0, 0].bar(x_pos, safe_counts, width, label='Safe (80k)', color='green', alpha=0.7)
    axes[0, 0].bar(x_pos, hard_counts, width, bottom=safe_counts, label='Hard (1k)', color='orange', alpha=0.7)
    axes[0, 0].bar(x_pos, toxic_counts, width, 
                   bottom=[s+h for s, h in zip(safe_counts, hard_counts)], 
                   label='Toxic', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Category')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Data Split by Category (Stacked)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Toxic ratio per category
    toxic_ratios = [category_stats[cat]['toxic_ratio'] * 100 for cat in categories]
    axes[0, 1].bar(x_pos, toxic_ratios, width, color='red', alpha=0.7)
    axes[0, 1].axhline(y=50, color='black', linestyle='--', label='Balanced (50%)')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Toxic Percentage (%)')
    axes[0, 1].set_title('Toxic Sample Percentage by Category')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Total samples per category
    total_counts = [category_stats[cat]['total'] for cat in categories]
    axes[1, 0].bar(x_pos, total_counts, width, color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Total Samples')
    axes[1, 0].set_title('Total Samples per Category')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Heatmap: Safe vs Toxic counts
    heatmap_data = pd.DataFrame({
        'Safe (80k)': safe_counts,
        'Hard (1k)': hard_counts,
        'Toxic': toxic_counts
    }, index=categories)
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='d', cmap='YlOrRd', 
                ax=axes[1, 1], cbar_kws={'label': 'Sample Count'})
    axes[1, 1].set_title('Sample Count Heatmap by Category')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Dataset Type')
    
    plt.tight_layout()
    plot_path = output_path / "data_split_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization: {plot_path}")
    
    # Save statistics to JSON
    stats_file = output_path / "data_split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'overall': {
                'total_safe_80k': total_safe_80k,
                'total_hard_1k': total_hard_1k,
                'total_toxic': total_toxic,
                'total_all': total_all,
                'toxic_ratio': total_toxic / total_all if total_all > 0 else 0
            },
            'by_category': category_stats
        }, f, indent=2)
    print(f"Saved statistics: {stats_file}")
    
    # Print warnings about imbalances
    print("\n" + "=" * 80)
    print("Data Imbalance Warnings")
    print("=" * 80)
    
    imbalanced_categories = []
    for category, stats in category_stats.items():
        if stats['toxic_ratio'] < 0.1:  # Less than 10% toxic
            imbalanced_categories.append((category, stats['toxic_ratio'] * 100))
            print(f"⚠️  {category}: Only {stats['toxic_ratio']*100:.2f}% toxic samples "
                  f"({stats['toxic']} toxic vs {stats['safe_80k'] + stats['hard_1k']} safe)")
        elif stats['toxic_ratio'] > 0.9:  # More than 90% toxic
            print(f"⚠️  {category}: {stats['toxic_ratio']*100:.2f}% toxic samples "
                  f"({stats['toxic']} toxic vs {stats['safe_80k'] + stats['hard_1k']} safe)")
    
    if imbalanced_categories:
        print(f"\n⚠️  Found {len(imbalanced_categories)} categories with <10% toxic samples.")
        print("   This may significantly impact SAE training quality.")
        print("   Consider:")
        print("   - Using data augmentation for toxic samples")
        print("   - Training separate SAEs for safe vs toxic")
        print("   - Using weighted sampling during SAE training")
    else:
        print("✓ No severe imbalances detected.")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data splits across safe, toxic, and hard samples in OR-Bench"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to OR-Bench dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/data_analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    analyze_data_splits(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()

