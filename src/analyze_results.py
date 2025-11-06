#!/usr/bin/env python3
"""
Analyze inference results and create comprehensive evaluation reports
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

def load_evaluation_data(results_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Load all evaluation data from the results directory"""
    evaluation_dir = os.path.join(results_dir, 'evaluation_results')
    
    if not os.path.exists(evaluation_dir):
        print(f"Evaluation directory not found: {evaluation_dir}")
        return {}
    
    data = defaultdict(lambda: defaultdict(list))
    
    for file_path in Path(evaluation_dir).glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                evaluation_data = json.load(f)
            
            # Extract model name and category from filename
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                model_name = '_'.join(parts[:-1])  # Everything except last part (category)
                category = parts[-1]
                
                data[model_name][category] = evaluation_data
                print(f"Loaded {len(evaluation_data)} samples for {model_name} - {category}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return dict(data)

def compute_metrics(evaluation_data: List[Dict]) -> Dict[str, Any]:
    """Compute evaluation metrics for a set of evaluation data"""
    if not evaluation_data:
        return {}
    
    total_samples = len(evaluation_data)
    correct_detections = 0
    true_refusals = 0
    detected_refusals = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    for sample in evaluation_data:
        true_label = sample['true_refusal_label']
        detected_label = sample['detected_refusal']
        
        if true_label == detected_label:
            correct_detections += 1
        
        if true_label:
            true_refusals += 1
            if detected_label:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if detected_label:
                false_positives += 1
            else:
                true_negatives += 1
        
        if detected_label:
            detected_refusals += 1
    
    # Compute metrics
    accuracy = correct_detections / total_samples if total_samples > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_refusals': true_refusals,
        'detected_refusals': detected_refusals,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }

def create_category_summary(evaluation_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Create a summary for a specific category across all models"""
    category_metrics = {}
    
    for model_name, data in evaluation_data.items():
        metrics = compute_metrics(data)
        category_metrics[model_name] = metrics
    
    # Compute overall category metrics
    all_data = []
    for data in evaluation_data.values():
        all_data.extend(data)
    
    overall_metrics = compute_metrics(all_data)
    
    return {
        'overall_metrics': overall_metrics,
        'model_metrics': category_metrics,
        'total_samples': len(all_data)
    }

def create_model_summary(evaluation_data: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Any]:
    """Create a summary for a specific model across all categories"""
    model_metrics = {}
    
    for category, data in evaluation_data.items():
        metrics = compute_metrics(data)
        model_metrics[category] = metrics
    
    # Compute overall model metrics
    all_data = []
    for data in evaluation_data.values():
        all_data.extend(data)
    
    overall_metrics = compute_metrics(all_data)
    
    return {
        'overall_metrics': overall_metrics,
        'category_metrics': model_metrics,
        'total_samples': len(all_data)
    }

def analyze_results(results_dir: str, output_file: str = None):
    """Analyze all results and create comprehensive reports"""
    print(f"Analyzing results from: {results_dir}")
    
    # Load evaluation data
    evaluation_data = load_evaluation_data(results_dir)
    
    if not evaluation_data:
        print("No evaluation data found!")
        return
    
    # Create analysis results
    analysis_results = {
        'summary': {},
        'category_analysis': {},
        'model_analysis': {},
        'detailed_results': evaluation_data
    }
    
    # Overall summary
    all_data = []
    for model_data in evaluation_data.values():
        for category_data in model_data.values():
            all_data.extend(category_data)
    
    analysis_results['summary'] = {
        'total_samples': len(all_data),
        'total_models': len(evaluation_data),
        'total_categories': len(set(cat for model_data in evaluation_data.values() for cat in model_data.keys())),
        'overall_metrics': compute_metrics(all_data)
    }
    
    # Category analysis
    all_categories = set()
    for model_data in evaluation_data.values():
        all_categories.update(model_data.keys())
    
    for category in all_categories:
        category_data = {}
        for model_name, model_data in evaluation_data.items():
            if category in model_data:
                category_data[model_name] = model_data[category]
        
        analysis_results['category_analysis'][category] = create_category_summary(category_data)
    
    # Model analysis
    for model_name, model_data in evaluation_data.items():
        analysis_results['model_analysis'][model_name] = create_model_summary(model_data)
    
    # Save analysis results
    if output_file is None:
        output_file = os.path.join(results_dir, 'analysis_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis results saved to: {output_file}")
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    summary = analysis_results['summary']
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total models: {summary['total_models']}")
    print(f"Total categories: {summary['total_categories']}")
    
    overall_metrics = summary['overall_metrics']
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"Overall Precision: {overall_metrics['precision']:.3f}")
    print(f"Overall Recall: {overall_metrics['recall']:.3f}")
    print(f"Overall F1 Score: {overall_metrics['f1_score']:.3f}")
    
    # Print category breakdown
    print("\n=== CATEGORY BREAKDOWN ===")
    for category, analysis in analysis_results['category_analysis'].items():
        metrics = analysis['overall_metrics']
        print(f"{category}:")
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
    
    # Print model breakdown
    print("\n=== MODEL BREAKDOWN ===")
    for model_name, analysis in analysis_results['model_analysis'].items():
        metrics = analysis['overall_metrics']
        print(f"{model_name}:")
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Path to results directory")
    parser.add_argument("--output", type=str, help="Output file for analysis results")
    args = parser.parse_args()
    
    analyze_results(args.results_dir, args.output)

