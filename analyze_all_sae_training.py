#!/usr/bin/env python3
"""
Analyze and visualize SAE training curves for both LLaMA-2 and Mistral models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_training_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_all_sae_training():
    """Analyze SAE training histories and create comprehensive visualizations for both models."""
    
    # Base directory
    base_dir = Path("/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/saes")
    
    # Both models with styling
    models = {
        "LLaMA-2-7b-chat-hf": {
            "path": "meta-llama-Llama-2-7b-chat-hf",
            "colors": ["#1f77b4", "#aec7e8", "#ff7f0e"],  # Blues
            "linestyle": "-",
            "marker": "o"
        },
        "Mistral-7B-Instruct-v0.1": {
            "path": "mistralai-Mistral-7B-Instruct-v0.1", 
            "colors": ["#ff7f0e", "#ffbb78", "#2ca02c"],  # Oranges/greens
            "linestyle": "--",
            "marker": "s"
        }
    }
    
    sae_configs = [
        ("residuals_10", "Residual Stream Layer 10"),
        ("mlp_11", "MLP Layer 11"), 
        ("residuals_15", "Residual Stream Layer 15")
    ]
    
    # Load all training histories
    all_histories = {}
    for model_name, model_info in models.items():
        sae_dir = base_dir / model_info["path"]
        if not sae_dir.exists():
            print(f"Warning: No SAE directory found for {model_name}")
            continue
            
        all_histories[model_name] = {}
        
        for config_name, display_name in sae_configs:
            filepath = sae_dir / f"{config_name}_training_history.json"
            if filepath.exists():
                all_histories[model_name][config_name] = {
                    'data': load_training_history(filepath),
                    'display_name': display_name
                }
                print(f"✓ Loaded {model_name} - {display_name}")
            else:
                print(f"✗ Missing {model_name} - {display_name}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1])
    
    # Plot 1: Combined Reconstruction Loss Comparison (Log Scale)
    ax1 = fig.add_subplot(gs[0, 0])
    
    for model_name, model_info in models.items():
        if model_name not in all_histories:
            continue
            
        for i, (config_name, config_info) in enumerate(all_histories[model_name].items()):
            data = config_info['data']
            epochs = range(1, len(data['reconstruction_loss']) + 1)
            
            ax1.semilogy(epochs, data['reconstruction_loss'], 
                        color=model_info["colors"][i], 
                        linestyle=model_info["linestyle"],
                        marker=model_info["marker"],
                        markersize=3,
                        linewidth=2,
                        label=f"{model_name} - {config_info['display_name']}")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss (Log Scale)')
    ax1.set_title('SAE Training Convergence - Both Models')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Loss Comparison Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    
    model_names = []
    layer_names = []
    final_losses = []
    colors = []
    
    for model_name, model_info in models.items():
        if model_name not in all_histories:
            continue
            
        for i, (config_name, config_info) in enumerate(all_histories[model_name].items()):
            final_loss = config_info['data']['reconstruction_loss'][-1]
            
            model_names.append(model_name.split('-')[0])  # LLaMA or Mistral
            layer_names.append(config_info['display_name'])
            final_losses.append(final_loss)
            colors.append(model_info["colors"][i])
    
    x_pos = np.arange(len(final_losses))
    bars = ax2.bar(x_pos, final_losses, color=colors)
    
    ax2.set_xlabel('Model - Layer Configuration')
    ax2.set_ylabel('Final Reconstruction Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{m}\n{l}" for m, l in zip(model_names, layer_names)], 
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, final_losses)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: L0 Norm (Sparsity) Evolution
    ax3 = fig.add_subplot(gs[0, 2])
    
    for model_name, model_info in models.items():
        if model_name not in all_histories:
            continue
            
        for i, (config_name, config_info) in enumerate(all_histories[model_name].items()):
            data = config_info['data']
            epochs = range(1, len(data['l0_norm']) + 1)
            
            ax3.plot(epochs, data['l0_norm'], 
                    color=model_info["colors"][i],
                    linestyle=model_info["linestyle"],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{model_name.split('-')[0]} L{config_name.split('_')[-1]}")
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Active Features (L0 Norm)')
    ax3.set_title('Feature Sparsity Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Individual model detailed plots (second row)
    for col, (model_name, model_info) in enumerate(models.items()):
        if model_name not in all_histories:
            continue
            
        ax = fig.add_subplot(gs[1, col])
        
        for i, (config_name, config_info) in enumerate(all_histories[model_name].items()):
            data = config_info['data']
            epochs = range(1, len(data['reconstruction_loss']) + 1)
            
            ax.semilogy(epochs, data['reconstruction_loss'],
                       color=model_info["colors"][i],
                       linewidth=2,
                       marker=model_info["marker"],
                       markersize=2,
                       label=config_info['display_name'])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss (Log)')
        ax.set_title(f'{model_name} Training Detail')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Training Statistics Summary (bottom)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary statistics table
    summary_text = "SAE TRAINING SUMMARY - BOTH MODELS\n" + "="*80 + "\n\n"
    
    for model_name, model_info in models.items():
        if model_name not in all_histories:
            continue
            
        summary_text += f"{model_name}:\n" + "-"*50 + "\n"
        
        for config_name, config_info in all_histories[model_name].items():
            data = config_info['data']
            display_name = config_info['display_name']
            
            initial_loss = data['reconstruction_loss'][0]
            final_loss = data['reconstruction_loss'][-1]
            min_loss = min(data['reconstruction_loss'])
            total_epochs = len(data['reconstruction_loss'])
            final_sparsity = data['l0_norm'][-1]
            
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            summary_text += f"  {display_name}:\n"
            summary_text += f"    Epochs: {total_epochs} | Final Loss: {final_loss:.6f} | Improvement: {improvement:.1f}%\n"
            summary_text += f"    Min Loss: {min_loss:.6f} | Final Sparsity: {final_sparsity:.1f} features\n\n"
        
        summary_text += "\n"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = "/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/visualizations/comprehensive_sae_analysis.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive SAE analysis saved to: {output_path}")
    
    # Create individual detailed plots for each model
    for model_name in all_histories.keys():
        create_detailed_model_plot(model_name, all_histories[model_name], models[model_name])
    
    plt.show()

def create_detailed_model_plot(model_name, histories, model_info):
    """Create detailed plot for individual model."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Detailed SAE Training Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    sae_names = list(histories.keys())
    
    # Top row: Individual loss curves
    for i, (config_name, config_info) in enumerate(histories.items()):
        ax = axes[0, i]
        data = config_info['data']
        epochs = range(1, len(data['reconstruction_loss']) + 1)
        
        # Linear scale
        ax.plot(epochs, data['reconstruction_loss'], 'b-', linewidth=2, 
                color=model_info["colors"][i], label='Reconstruction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f"{config_info['display_name']} - Linear Scale")
        ax.grid(True, alpha=0.3)
        
        # Add final loss annotation
        final_loss = data['reconstruction_loss'][-1]
        ax.annotate(f'Final: {final_loss:.6f}', 
                   xy=(len(epochs), final_loss), xytext=(len(epochs)*0.7, final_loss*1.5),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, fontweight='bold')
    
    # Bottom row: Log scale and convergence analysis
    for i, (config_name, config_info) in enumerate(histories.items()):
        ax = axes[1, i]
        data = config_info['data']
        epochs = range(1, len(data['reconstruction_loss']) + 1)
        
        # Log scale
        ax.semilogy(epochs, data['reconstruction_loss'], 'b-', linewidth=2,
                   color=model_info["colors"][i], label='Reconstruction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Log Scale)')
        ax.set_title(f"{config_info['display_name']} - Log Scale")
        ax.grid(True, alpha=0.3)
        
        # Add improvement percentage
        initial_loss = data['reconstruction_loss'][0]
        final_loss = data['reconstruction_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        ax.text(0.05, 0.95, f'Improvement: {improvement:.1f}%', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual model plot
    clean_name = model_name.replace('/', '-').replace('_', '-')
    output_path = f"/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/visualizations/detailed_{clean_name}_sae_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed {model_name} analysis saved to: {output_path}")
    
    plt.close()

def print_comprehensive_summary():
    """Print comprehensive training summary for both models."""
    
    base_dir = Path("/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/saes")
    
    models = {
        "LLaMA-2-7b-chat-hf": "meta-llama-Llama-2-7b-chat-hf",
        "Mistral-7B-Instruct-v0.1": "mistralai-Mistral-7B-Instruct-v0.1"
    }
    
    sae_configs = [
        ("residuals_10", "Residual Stream Layer 10"),
        ("mlp_11", "MLP Layer 11"), 
        ("residuals_15", "Residual Stream Layer 15")
    ]
    
    print("\n" + "="*100)
    print("COMPREHENSIVE SAE TRAINING ANALYSIS - BOTH MODELS")
    print("="*100)
    
    for model_name, model_path in models.items():
        sae_dir = base_dir / model_path
        
        print(f"\n🤖 {model_name}")
        print("-" * 80)
        
        if not sae_dir.exists():
            print("   ❌ No SAE directory found")
            continue
        
        for config_name, display_name in sae_configs:
            filepath = sae_dir / f"{config_name}_training_history.json"
            if filepath.exists():
                data = load_training_history(filepath)
                
                initial_loss = data['reconstruction_loss'][0]
                final_loss = data['reconstruction_loss'][-1] 
                min_loss = min(data['reconstruction_loss'])
                min_loss_epoch = data['reconstruction_loss'].index(min_loss) + 1
                total_epochs = len(data['reconstruction_loss'])
                
                initial_sparsity = data['l0_norm'][0]
                final_sparsity = data['l0_norm'][-1]
                
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                
                print(f"\n   📊 {display_name}")
                print(f"      Training Epochs: {total_epochs}")
                print(f"      Initial Loss: {initial_loss:.8f}")
                print(f"      Final Loss:   {final_loss:.8f} ({improvement:+.2f}% improvement)")
                print(f"      Best Loss:    {min_loss:.8f} (epoch {min_loss_epoch})")
                print(f"      Sparsity: {initial_sparsity:.1f} → {final_sparsity:.1f} features")
                
                # Convergence status
                if total_epochs >= 10:
                    recent_losses = data['reconstruction_loss'][-10:]
                    loss_std = np.std(recent_losses)
                    if loss_std < final_loss * 0.01:
                        status = "✓ Well converged"
                    else:
                        status = "⚠ Still improving"
                else:
                    status = "⚠ Short training"
                    
                print(f"      Status: {status}")
            else:
                print(f"   ❌ {display_name}: No training history found")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    print("Analyzing SAE training for both LLaMA-2 and Mistral...")
    print_comprehensive_summary()
    analyze_all_sae_training()
    print("\nComprehensive analysis complete!")