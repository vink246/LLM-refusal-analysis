#!/usr/bin/env python3
"""
Organize visualizations into structured folders by model and visualization type.
"""

import os
import shutil
import re
from pathlib import Path

def organize_visualizations():
    """Reorganize visualization files into model-specific folders with type subfolders."""
    
    # Base paths
    viz_dir = Path("results/visualizations")
    
    if not viz_dir.exists():
        print(f"Visualization directory {viz_dir} does not exist!")
        return
    
    # Create new folder structure
    models = {
        "meta-llama-Llama-2-7b-chat-hf": "llama-2-7b-chat-hf",
        "mistralai-Mistral-7B-Instruct-v0.1": "mistral-7b-instruct-v0.1"
    }
    
    viz_types = {
        "circuit": "circuit_importance",
        "network": "network_diagrams", 
        "similarity_heatmap": "similarity_heatmaps"
    }
    
    # Create directory structure
    for model_key, model_folder in models.items():
        model_path = viz_dir / model_folder
        model_path.mkdir(exist_ok=True)
        
        for viz_type, viz_folder in viz_types.items():
            (model_path / viz_folder).mkdir(exist_ok=True)
    
    # Get all visualization files
    viz_files = list(viz_dir.glob("*.png"))
    
    print(f"Found {len(viz_files)} visualization files to organize")
    
    moved_count = 0
    
    for file_path in viz_files:
        filename = file_path.name
        
        # Skip if already in organized structure
        if file_path.parent != viz_dir:
            continue
            
        print(f"\nProcessing: {filename}")
        
        # Determine model
        model_folder = None
        for model_key, folder_name in models.items():
            if filename.startswith(model_key):
                model_folder = folder_name
                break
        
        if not model_folder:
            print(f"  ⚠️  Unknown model in filename: {filename}")
            continue
        
        # Determine visualization type
        viz_folder = None
        new_filename = filename
        
        if "_similarity_heatmap.png" in filename:
            viz_folder = "similarity_heatmaps"
            # Simplify similarity heatmap name
            new_filename = f"{model_folder}_similarity_heatmap.png"
        elif "_network.png" in filename:
            viz_folder = "network_diagrams"
            # Extract category for network files
            pattern = rf"{list(models.keys())[0] if model_folder == 'llama-2-7b-chat-hf' else list(models.keys())[1]}_(.+)_network\.png"
            match = re.search(pattern, filename)
            if match:
                category = match.group(1)
                new_filename = f"{category}_network.png"
        elif "_circuit.png" in filename:
            viz_folder = "circuit_importance"
            # Extract category for circuit files
            pattern = rf"{list(models.keys())[0] if model_folder == 'llama-2-7b-chat-hf' else list(models.keys())[1]}_(.+)_circuit\.png"
            match = re.search(pattern, filename)
            if match:
                category = match.group(1)
                new_filename = f"{category}_circuit.png"
        
        if not viz_folder:
            print(f"  ⚠️  Unknown visualization type: {filename}")
            continue
        
        # Move file to organized location
        new_path = viz_dir / model_folder / viz_folder / new_filename
        
        try:
            shutil.move(str(file_path), str(new_path))
            print(f"  ✅ Moved to: {model_folder}/{viz_folder}/{new_filename}")
            moved_count += 1
        except Exception as e:
            print(f"  ❌ Error moving file: {e}")
    
    print(f"\n📊 Organization Summary:")
    print(f"  Successfully moved: {moved_count} files")
    
    # Display new structure
    print(f"\n📁 New Structure:")
    for model_key, model_folder in models.items():
        model_path = viz_dir / model_folder
        if model_path.exists():
            print(f"\n  {model_folder}/")
            for viz_type, viz_folder in viz_types.items():
                viz_path = model_path / viz_folder
                if viz_path.exists():
                    files = list(viz_path.glob("*.png"))
                    print(f"    {viz_folder}/ ({len(files)} files)")
                    for file in sorted(files):
                        print(f"      - {file.name}")
    
    print(f"\n✨ Visualization organization complete!")


if __name__ == "__main__":
    organize_visualizations()