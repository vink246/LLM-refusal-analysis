#!/usr/bin/env python3
"""
Analyze and visualize SAE training curves and loss values.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_sae_training():
    """Analyze SAE training histories and create visualizations for both models."""
    
    # Base directory
    base_dir = Path("/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/saes")
    
    # Both models
    models = {
        "LLaMA-2": {
            "path": "meta-llama-Llama-2-7b-chat-hf",
            "color_prefix": "blues"
        },
        "Mistral": {
            "path": "mistralai-Mistral-7B-Instruct-v0.1", 
            "color_prefix": "oranges"
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
        all_histories[model_name] = {}
        
        for config_name, display_name in sae_configs:
            filepath = sae_dir / f"{config_name}_training_history.json"
            if filepath.exists():
                all_histories[model_name][config_name] = {
                    'data': load_training_history(filepath),
                    'display_name': display_name
                }
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SAE Training Analysis - LLaMA-2-7b-chat-hf', fontsize=16, fontweight='bold')
    
    # Plot 1: Reconstruction Loss Over Time
    ax1 = axes[0, 0]
    for config_name, info in histories.items():
        data = info['data']
        epochs = range(1, len(data['reconstruction_loss']) + 1)
        ax1.plot(epochs, data['reconstruction_loss'], 
                label=info['display_name'], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('Reconstruction Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization of small values
    
    # Plot 2: Final Loss Values Comparison
    ax2 = axes[0, 1]
    final_losses = []
    labels = []
    for config_name, info in histories.items():
        final_loss = info['data']['reconstruction_loss'][-1]
        final_losses.append(final_loss)
        labels.append(info['display_name'])
    
    bars = ax2.bar(range(len(final_losses)), final_losses, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('SAE Configuration')
    ax2.set_ylabel('Final Reconstruction Loss')
    ax2.set_title('Final Reconstruction Loss Comparison')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, final_losses)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: L0 Norm (Sparsity) Over Time
    ax3 = axes[1, 0]
    for config_name, info in histories.items():
        data = info['data']
        epochs = range(1, len(data['l0_norm']) + 1)
        ax3.plot(epochs, data['l0_norm'], 
                label=info['display_name'], linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('L0 Norm (Active Features)')
    ax3.set_title('Feature Sparsity Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training Statistics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis for text display
    
    # Create summary statistics
    summary_text = "SAE Training Summary\n" + "="*30 + "\n\n"
    
    for config_name, info in histories.items():
        data = info['data']
        display_name = info['display_name']
        
        initial_loss = data['reconstruction_loss'][0]
        final_loss = data['reconstruction_loss'][-1]
        min_loss = min(data['reconstruction_loss'])
        total_epochs = len(data['reconstruction_loss'])
        final_sparsity = data['l0_norm'][-1]
        
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        summary_text += f"{display_name}:\n"
        summary_text += f"  • Total Epochs: {total_epochs}\n"
        summary_text += f"  • Initial Loss: {initial_loss:.6f}\n"
        summary_text += f"  • Final Loss: {final_loss:.6f}\n"
        summary_text += f"  • Min Loss: {min_loss:.6f}\n"
        summary_text += f"  • Improvement: {improvement:.1f}%\n"
        summary_text += f"  • Final Sparsity: {final_sparsity:.1f} features\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/visualizations/sae_training_analysis.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved to: {output_path}")
    
    # Create individual detailed plots for each SAE
    for config_name, info in histories.items():
        create_detailed_sae_plot(config_name, info, sae_dir)
    
    plt.show()

def create_detailed_sae_plot(config_name, info, base_dir):
    """Create detailed plot for individual SAE."""
    
    data = info['data']
    display_name = info['display_name']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Detailed Training Analysis: {display_name}', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(data['reconstruction_loss']) + 1)
    
    # Plot 1: Loss with different scales
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['reconstruction_loss'], 'b-', linewidth=2, label='Reconstruction Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Reconstruction Loss (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Loss in log scale
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, data['reconstruction_loss'], 'b-', linewidth=2, label='Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Log Scale)')
    ax2.set_title('Reconstruction Loss (Log Scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: L0 Norm evolution
    ax3 = axes[1, 0]
    ax3.plot(epochs, data['l0_norm'], 'g-', linewidth=2, label='L0 Norm')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Active Features')
    ax3.set_title('Feature Sparsity (L0 Norm)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Convergence analysis
    ax4 = axes[1, 1]
    
    # Calculate loss improvement rate
    if len(data['reconstruction_loss']) > 1:
        loss_diffs = np.diff(data['reconstruction_loss'])
        improvement_rate = [-diff for diff in loss_diffs]  # Positive = improvement
        
        ax4.plot(epochs[1:], improvement_rate, 'r-', linewidth=2, label='Loss Improvement Rate')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Improvement Rate')
        ax4.set_title('Training Convergence Rate')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    # Save individual plot
    output_path = f"/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/visualizations/sae_detailed_{config_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed {display_name} plot saved to: {output_path}")
    
    plt.close()

def print_training_summary():
    """Print detailed training summary to console."""
    
    sae_dir = Path("/home/hice1/ccheah6/scratch/NLP/LLM-refusal-analysis/results/saes/meta-llama-Llama-2-7b-chat-hf")
    
    sae_configs = [
        ("residuals_10", "Residual Stream Layer 10"),
        ("mlp_11", "MLP Layer 11"), 
        ("residuals_15", "Residual Stream Layer 15")
    ]
    
    print("\n" + "="*60)
    print("SAE TRAINING ANALYSIS SUMMARY")
    print("="*60)
    
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
            
            print(f"\n{display_name}")
            print("-" * len(display_name))
            print(f"Training Epochs: {total_epochs}")
            print(f"Initial Reconstruction Loss: {initial_loss:.8f}")
            print(f"Final Reconstruction Loss:   {final_loss:.8f}")
            print(f"Best Reconstruction Loss:    {min_loss:.8f} (epoch {min_loss_epoch})")
            print(f"Total Improvement: {improvement:.2f}%")
            print(f"Initial Sparsity: {initial_sparsity:.1f} active features")
            print(f"Final Sparsity:   {final_sparsity:.1f} active features")
            
            # Check convergence
            if total_epochs >= 10:
                recent_losses = data['reconstruction_loss'][-10:]
                loss_std = np.std(recent_losses)
                if loss_std < final_loss * 0.01:  # Less than 1% variation
                    print("Status: ✓ Well converged")
                else:
                    print("Status: ⚠ Still improving")
            else:
                print("Status: ⚠ Short training run")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Analyzing SAE training histories...")
    print_training_summary()
    analyze_sae_training()
    print("\nAnalysis complete!")