#!/usr/bin/env python3
"""
Run analysis on inference results
"""

import os
import sys
import argparse

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from analyze_results import analyze_results

def main():
    parser = argparse.ArgumentParser(description="Analyze inference results")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Path to results directory")
    parser.add_argument("--output", type=str, help="Output file for analysis results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    analyze_results(args.results_dir, args.output)

if __name__ == "__main__":
    main()

