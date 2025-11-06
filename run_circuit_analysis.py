#!/usr/bin/env python3
"""
Main entry point for circuit analysis pipeline
Runs the complete workflow: SAE training -> Circuit discovery -> Comparison -> Reporting
"""

import os
import sys
import argparse
import yaml

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from circuits.refusal_circuit_analyzer import RefusalCircuitAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description="Run complete circuit analysis pipeline for refusal behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage on PACE ICE:
  # Run inference first (if not already done)
  python src/run_inference.py --config configurations/orbench_run.yaml
  
  # Run circuit analysis with SAE training
  python run_circuit_analysis.py --config configurations/sae_training.yaml
  
  # Run circuit analysis with pre-trained SAEs
  python run_circuit_analysis.py --config configurations/circuit_discovery.yaml
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--train-saes",
        action="store_true",
        help="Force SAE training even if train_saes is false in config"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load config to check if we need to override train_saes
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.train_saes:
        config['train_saes'] = True
    
    # Save modified config temporarily if needed
    if args.train_saes and config.get('train_saes') != args.train_saes:
        import tempfile
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config)
        temp_config.close()
        config_path = temp_config.name
    else:
        config_path = args.config
    
    try:
        # Validate config file exists
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        analyzer = RefusalCircuitAnalyzer(config_path)
        analyzer.run_analysis()
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        print(f"\nResults saved to: {analyzer.result_dir}")
        print(f"  - Circuits: {analyzer.result_dir / 'circuits'}")
        print(f"  - Visualizations: {analyzer.result_dir / 'visualizations'}")
        print(f"  - Reports: {analyzer.result_dir / 'circuit_analysis_report.json'}")
        print(f"  - Summary: {analyzer.result_dir / 'circuit_analysis_summary.txt'}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running circuit analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temp config if created
        if args.train_saes and config_path != args.config:
            os.unlink(config_path)

if __name__ == "__main__":
    main()

