# LLM-refusal-analysis
Class project for Georgia Tech's CS 4650: investigating monolith vs modular refusal behavior of LLMs across refusal categories using sparse feature circuits.

## Project Overview

This project analyzes whether LLM refusal behavior is **monolithic** (shared circuits across categories) or **modular** (category-specific circuits) by:
1. Collecting model activations on OR-Bench dataset
2. Training Sparse Autoencoders (SAEs) to decompose activations into interpretable features
3. Discovering sparse feature circuits for each refusal category using SAE-encoded features
4. Comparing circuits across categories with statistical significance testing

**Current Status**: The pipeline supports Llama-2 and Mistral models with a focus on Mistral for circuit discovery due to SAE training stability issues with Llama-2.

## Project Components

### Core Infrastructure

#### 1. **Inference Pipeline** (`src/run_inference.py`)
- Runs inference on LLMs (Llama-2, Mistral) with OR-Bench dataset
- Collects activations from specified model layers
- Detects refusal behavior from model outputs
- Saves activations, refusal labels, and evaluation results

#### 2. **SAE Training** (`src/sae_trainer.py`)
- Trains Sparse Autoencoders on collected activations
- Decomposes activations into sparse, interpretable features
- Supports training separate SAEs for each model layer
- Saves trained SAEs for circuit discovery

#### 3. **Circuit Discovery** (`circuits/`)
- **`circuits_utils.py`**: Core circuit data structures and utilities
  - `SparseFeatureCircuit`: Represents discovered circuits with nodes, edges, and importance scores
  - `CircuitDiscoverer`: Base circuit discovery algorithm 
  - `CircuitVisualizer`: Visualization tools (importance plots, network diagrams, heatmaps)
  - `compare_circuits_across_categories`: Cross-category similarity computation
  - Faithfulness and completeness metrics for circuit evaluation
- **`circuit_discovery_with_saes.py`**: SAE-based circuit discovery
  - `SAECircuitDiscoverer`: Discovers circuits using trained SAE features and correlation analysis
  - Loads pre-computed activations and encodes them with trained SAEs
  - Computes feature importances via correlation with refusal labels
  - Builds circuit graphs with SAE features as nodes
- **`refusal_circuit_analyzer.py`**: Main orchestrator class
  - `RefusalCircuitAnalyzer`: Complete analysis workflow coordinator
  - Manages SAE training via `SAEManager`, circuit discovery, and cross-category comparison
  - Generates comprehensive reports, visualizations, and statistical assessments
  - Configuration validation and organized output directory structure
- **`statistical_analysis.py`**: Statistical hypothesis testing
  - Statistical significance tests for circuit similarities
  - Confidence intervals and p-value computation
  - Modularity assessment with statistical confidence

#### 4. **Data Processing** (`src/`)
- **`data_utils.py`**: Dataset loading utilities
  - Loads OR-Bench dataset with proper safe/toxic ratios
  - Handles multiple categories
- **`orbench_data_processor.py`**: OR-Bench specific processing
- **`models.py`**: Model loading with activation hooks
- **`activation_utils.py`**: Activation management and storage
- **`evaluation.py`**: Refusal detection evaluation
- **`analyze_results.py`**: Analysis of inference results

### Entry Points

- **`src/run_inference.py`**: Run inference and collect activations
- **`run_circuit_analysis.py`**: Complete circuit analysis pipeline
- **`run_analysis.py`**: Analyze inference results (refusal detection metrics)

## Environment setup

### Setting up a Conda Environment in Scratch (PACE-ICE)

#### 1. Load the Anaconda Module
PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
> ```bash
> module avail anaconda
> ```

#### 2. Create a Conda Environment in Scratch

By default, `conda` environments go to your home directory (`~/.conda/envs`),  
which has limited quota. To avoid filling it up, create a custom environment directory in scratch:

```bash
conda env create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env --file llm_refusal_env.yml
```

Alternatively, you could create the environment from scratch without using the yml file:

```bash
# Create the environment explicitly in scratch
conda create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env python=3.11 -y
```

#### 3. Activate the Environment

```bash
conda activate /scratch/<path_to_env_parent_dir>/llm_refusal_env
```

> You **must use the full path** when activating environments created outside your home directory.

#### 4. Install Required Packages

Once activated, install any packages you need, such as PyTorch, OpenCV, etc.

```bash
# Example for PyTorch + utilities
conda install pytorch torchvision torchaudio -c pytorch -y
conda install numpy matplotlib opencv scipy scikit-learn tqdm -y
conda install jupyterlab -y
```

If you created the environment from scratch, use the requirements.txt file:

```bash
pip install -r requirements.txt
```

#### 5. Verify Installation

```bash
python -m pip list
python -c "import torch; print(torch.__version__)"
```

#### 6. (Optional) Export and Reuse the Environment

You can save your environment spec for reproducibility:

```bash
conda env export > llm_refusal_env.yml
```

and recreate it later with:

```bash
conda env create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env --file llm_refusal_env.yml
```

#### 8. Deactivate Environment

```bash
conda deactivate
```

### Dataset Download Instructions
#### OR-Bench
```bash
hf download bench-llm/or-bench --repo-type dataset --local-dir /home/hice1/<gt_username>/scratch/datasets/or-bench
```

### Model Download Instructions

#### 1. Login to Hugging Face
```bash
huggingface-cli login
# This will prompt you for your Hugging Face API token
``` 

#### 2. Set a scratch folder for model downloads
```bash
export HF_HOME=/home/hice1/<gt_username>/scratch/huggingface
```

#### 3. Download LLaMA-2-7B-Chat
```bash
hf download meta-llama/Llama-2-7b-chat-hf --cache-dir $HF_HOME
```
#### 5. Download Mistral-7B-Instruct
```bash
hf download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir $HF_HOME
```

## Usage Instructions (PACE ICE)

### Step 1: Run Inference

First, collect activations from models on the OR-Bench dataset:

```bash
python src/run_inference.py --config configurations/orbench_run.yaml
```

This will:
- Load 100 samples per category (50 safe + 50 toxic)
- Run inference on specified models
- Save activations to `results/activations/`
- Save refusal labels to `results/refusal_labels/`
- Save evaluation results to `results/evaluation_results/`

**Note**: This step requires GPU access. Use a PACE ICE GPU node or submit as a job.

**Current Configuration**: The default `orbench_run.yaml` is configured for 2 categories (violence, deception) and 100 samples per category. To analyze all 10 categories, update the configuration file or use one of the alternative configs like `circuit_discovery.yaml`.

### Step 2: Train Sparse Autoencoders

Train SAEs on the collected activations:

```bash
python run_circuit_analysis.py --config configurations/sae_training.yaml --train-saes
```

Or modify `configurations/sae_training.yaml` to set `train_saes: true`.

This will:
- Load activation files for each model and category combination
- Train separate SAEs for each specified layer using the `SAEManager` class
- Use correlation-based feature importance computation for circuit discovery
- Save trained SAEs with training history to `results/saes/<model_name>/`

**Note**: SAE training is computationally intensive and may require significant GPU memory. Use GPU nodes and consider running as a batch job. The SAE training integrates with the circuit discovery pipeline through the `SAECircuitDiscoverer` class.

**Performance Note**: On high-end GPUs like H200, SAE training may complete very quickly due to the small default dataset (100 samples per category in `orbench_run.yaml`). If training completes in seconds, verify that: (1) activation files were found and loaded, (2) you see progress bars for multiple epochs and batches, and (3) SAE files are saved to `results/saes/<model_name>/`. For more thorough training, increase `sae_max_samples` or use the full 10-category configuration in `circuit_discovery.yaml`.

### Step 3: Discover Circuits

Discover sparse feature circuits for each category:

```bash
python run_circuit_analysis.py --config configurations/circuit_discovery.yaml
```

This will:
- Load trained SAEs
- Discover circuits for each model-category combination
- Save circuits to `results/circuits/`
- Compare circuits across categories with statistical significance testing
- Generate similarity metrics and modularity assessment (monolithic/partially modular/modular)
- Compute faithfulness and completeness metrics for each circuit
- Create similarity heatmaps showing circuit relationships across categories
- Generate circuit visualizations (importance plots and network diagrams) for each category
- Create comprehensive final analysis reports (JSON and text format) with conclusions

### Step 4: Analyze Results

Analyze inference results (refusal detection):

```bash
python run_analysis.py --results-dir results
```

This generates comprehensive evaluation metrics and reports.

## Configuration Files

### `configurations/orbench_run.yaml`
- Dataset path and OR-Bench categories (currently configured for 2 categories: violence, deception)
- Model specifications (Llama-2-7b-chat-hf, Mistral-7B-Instruct-v0.1)
- Activation layer specifications (residuals_10, residuals_15, mlp_11)
- Batch size and device settings

### `configurations/sae_training.yaml`
- SAE training hyperparameters
- Layer specifications
- Training settings (epochs, batch size, etc.)

### `configurations/circuit_discovery.yaml`
- Circuit discovery parameters (node_threshold: 0.1, edge_threshold: 0.01)
- Model selection (currently configured for Mistral only)
- All 10 categories: deception, harassment, harmful, hate, illegal, privacy, self-harm, sexual, unethical, violence
- Visualization and statistical analysis settings

## Output Structure

```
results/
├── activations/              # Model activations (.pt files)
│   └── <safe_model_name>_<category>_activations.pt
├── refusal_labels/           # Ground truth refusal labels (.json)
│   └── <safe_model_name>_<category>_refusal.json
├── model_outputs/            # Model responses (.json)
│   └── <safe_model_name>_<category>_outputs.json
├── evaluation_results/       # Evaluation data (.json)
│   └── <safe_model_name>_<category>_evaluation.json
├── saes/                     # Trained SAEs (.pt files)
│   └── <model_name>/
│       ├── <layer>_sae.pt
│       └── <layer>_training_history.json
├── circuits/                 # Discovered circuits (.json)
│   ├── <safe_model_name>_<category>_circuit.json
│   └── <safe_model_name>_comparison.json
├── visualizations/           # Circuit visualizations (.png)
│   └── <model_folder>/
│       ├── circuit_importance/
│       │   └── <category>_circuit_importance.png
│       ├── network_diagrams/
│       │   └── <category>_network_diagram.png
│       └── similarity_heatmaps/
│           └── <model_folder>_similarity_heatmap.png
├── circuit_analysis_report.json    # Comprehensive circuit analysis (JSON)
├── circuit_analysis_summary.txt    # Human-readable analysis summary
└── analysis_results.json    # Inference evaluation report (from run_analysis.py)
```

## Running on PACE ICE

### Interactive Session (GPU)

```bash
# Request GPU node
qsub -I -l walltime=4:00:00 -l nodes=1:ppn=1:gpus=1 -q gpu

# Load modules and activate environment
module load anaconda3
conda activate /scratch/<path>/llm_refusal_env

# Run inference
python src/run_inference.py --config configurations/orbench_run.yaml
```

### Batch Job Submission

Create a job script `run_inference.sh`:

```bash
#!/bin/bash
#PBS -N llm_inference
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q gpu
#PBS -j oe

module load anaconda3
conda activate /scratch/<path>/llm_refusal_env

cd $PBS_O_WORKDIR
python src/run_inference.py --config configurations/orbench_run.yaml
```

Submit with: `qsub run_inference.sh`

### SAE Training (Long Running)

SAE training can take several hours. Use a long-running job:

```bash
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=1:gpus=1
```

## Research Hypothesis

**Monolithic Hypothesis**: Refusal behavior uses shared circuits across categories
- High circuit similarity (>0.8) between categories
- Same features/nodes active for different refusal types

**Modular Hypothesis**: Refusal behavior uses category-specific circuits
- Low circuit similarity (<0.5) between categories
- Different features/nodes for different refusal types

**Partially Modular**: Mixed behavior
- Moderate similarity (0.5-0.8)
- Some shared features, some category-specific

The analysis automatically assesses which hypothesis is supported based on circuit similarity metrics with statistical significance testing via the `statistical_analysis.py` module. Assessments include confidence levels (HIGH/LOW) based on p-values from t-tests, implemented through the `assess_modularity_with_statistics` function.

### Interpreting Results

After running circuit analysis, check the following outputs:

1. **Similarity Heatmaps** (`results/visualizations/<model>_similarity_heatmap.png`):
   - Visual representation of circuit similarities between categories
   - Green = high similarity (monolithic), Red = low similarity (modular)
   - Diagonal is always 1.0 (self-similarity)

2. **Comparison Results** (`results/circuits/<model>_comparison.json`):
   - Pairwise similarity scores between categories
   - Average similarity across all category pairs
   - Assessment: MONOLITHIC, PARTIALLY MODULAR, or MODULAR
   - Statistical significance (p-values, confidence intervals)
   - Assessment confidence (HIGH/LOW based on statistical tests)

3. **Final Reports**:
   - `circuit_analysis_report.json`: Complete analysis with all metrics
     - Circuit statistics per category (nodes, edges, importance metrics)
     - Faithfulness and completeness scores
     - Statistical test results
   - `circuit_analysis_summary.txt`: Human-readable summary with:
     - Key findings and circuit statistics
     - Comparison results with statistical significance
     - **Conclusions section** with research interpretations

## Troubleshooting

### Import Errors
- Ensure you're running from the project root directory
- Check that all paths in config files are absolute or relative to project root
- Verify Python path includes project directories

### Missing Activations
- Run inference first (`python src/run_inference.py --config configurations/orbench_run.yaml`)
- Check that activation files exist in `results/activations/`
- Verify model names and categories match between inference and circuit discovery

### SAE Training Issues
- Ensure sufficient GPU memory (7B models need ~20GB+)
- Reduce `sae_max_samples` if running out of memory
- Check that activation files are properly formatted and contain expected layers
- Verify `SAEManager` can find activation files in correct directory structure
- If SAE training fails, check that all specified `activation_layers` exist in activation files

### Circuit Discovery Errors
- Ensure SAEs are trained before circuit discovery (check `results/saes/<model_name>/` exists)
- Check that refusal labels match activation batch sizes
- Verify layer names match between SAE training and circuit discovery configurations
- For missing activation files, ensure inference was run first for all categories
- Configuration validation will catch missing required fields (`models`, `categories`, `activation_layers`)
- If getting empty circuits, try lowering `node_threshold` and `edge_threshold` values
- Check error messages for specific file path issues and SAE loading failures

### Statistical Analysis
- Statistical tests require at least 2 similarity values (need 2+ categories)
- Low confidence assessments indicate results are not statistically significant
- Check p-values in comparison JSON files to interpret significance

## Dependencies

See `requirements.txt` and `llm_refusal_env.yml` for full dependency list.

Key dependencies:
- PyTorch (with CUDA support for GPU)
- Transformers (Hugging Face)
- NumPy, Matplotlib, Seaborn
- NetworkX (for circuit network visualizations)
- scipy (for statistical tests)
- scikit-learn
- PyYAML