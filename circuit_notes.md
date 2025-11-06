### Key Concepts from the Paper

#### 1. **Sparse Autoencoders (SAEs)**
- **Purpose**: Decompose model activations into interpretable, monosemantic features
- **Architecture**: Encoder-Decoder with ReLU activation for sparsity
- **Training Objective**: Minimize reconstruction error + L1 sparsity penalty
- **Mathematical Formulation**:
  ```
  x = x̂ + ε(x) = Σ f_i(x)v_i + b + ε(x)
  ```
  Where `f_i(x)` are sparse feature activations and `v_i` are feature directions

#### 2. **Sparse Feature Circuits**
- **Definition**: Causal subgraphs of human-interpretable features that explain specific model behaviors
- **Components**: Nodes (SAE features) and Edges (causal interactions between features)
- **Discovery Method**: Uses integrated gradients and attribution patching to identify causally important features

#### 3. **Circuit Faithfulness & Completeness**
- **Faithfulness**: How well the circuit explains model behavior when only circuit components are active
- **Completeness**: How much behavior remains when circuit components are ablated

#### 4. **SHIFT Method** (Sparse Human-Interpretable Feature Trimming)
- Removes unintended signals from classifiers using human judgement
- Doesn't require disambiguating labeled data
- Based on interpreting and ablating task-irrelevant features

## Project Implementation Files

### Core Infrastructure Files

#### 1. `sae_trainer.py`
**Purpose**: Train sparse autoencoders on collected activations

**Key Components**:
- `SparseAutoencoder`: Neural network architecture with encoder-decoder structure
  - Encoder: Linear layer + ReLU for sparse features
  - Decoder: Linear reconstruction layer
  - L1 sparsity regularization during training
- `ActivationDataset`: Loads and processes pre-collected activations
  - Handles multiple activation files
  - Flattens spatial dimensions when needed
  - Limits samples for efficient training
- `SAETrainer`: Manages training process
  - Adam optimizer with learning rate scheduling
  - Tracks reconstruction and sparsity losses
  - Early stopping based on convergence
- `SAEManager`: Coordinates multiple SAEs across layers
  - Trains separate SAE for each model layer
  - Manages saving/loading of trained SAEs
  - Encodes new activations using trained SAEs

**Key Methods**:
- `train_saes_for_model()`: Train SAEs for all layers of a model
- `encode_activations()`: Convert raw activations to sparse features
- `load_saes_for_model()`: Load pre-trained SAEs for circuit discovery

#### 2. `circuit_utils.py`
**Purpose**: Core circuit discovery and analysis infrastructure

**Key Components**:
- `CircuitConfig`: Configuration for circuit discovery parameters
  - Node and edge thresholds
  - Attribution methods (integrated gradients, attribution patching)
  - Aggregation strategies
- `SparseFeatureCircuit`: Data structure representing discovered circuits
  - Stores nodes (features) and edges (interactions)
  - Tracks importance scores
  - Provides analysis methods
- `CircuitDiscoverer`: Main circuit discovery algorithm
  - Computes feature importances using attribution methods
  - Identifies important nodes and edges
  - Builds circuit graphs
- `CircuitVisualizer`: Creates visualizations of circuits
  - Importance plots for nodes and edges
  - Network diagrams of circuit structure
  - Similarity heatmaps for cross-category comparison

**Key Methods**:
- `discover_circuit()`: Main circuit discovery workflow
- `compute_feature_importances()`: Calculate causal importance of features
- `compare_circuits_across_categories()`: Test modular vs monolithic hypotheses

#### 3. `circuit_discovery_with_saes.py`
**Purpose**: Integrate SAE training with circuit discovery

**Key Components**:
- `SAECircuitDiscoverer`: Extended discoverer that uses SAE features
  - Encodes activations through trained SAEs
  - Computes importances on sparse features
  - Builds circuits from interpretable features
- Main pipeline coordination
  - Trains SAEs if needed
  - Runs circuit discovery across all models and categories
  - Saves results for analysis

**Key Methods**:
- `discover_circuit_with_saes()`: Complete SAE-based circuit discovery
- `_compute_sae_feature_importances()`: Statistical analysis on SAE features
- `_identify_important_sae_nodes()`: Threshold-based node selection

### Configuration Files

#### 4. `refusal_circuit_analyzer.py`
**Purpose**: Main analysis pipeline for refusal circuits

**Key Components**:
- `RefusalCircuitAnalyzer`: Orchestrates complete analysis workflow
  - Loads models and data
  - Discovers circuits for each category
  - Compares circuits across categories
  - Generates visualizations
- Cross-category comparison
  - Computes circuit similarities
  - Creates similarity heatmaps
  - Tests modular vs monolithic hypotheses

**Key Methods**:
- `run_analysis()`: Complete analysis pipeline
- `_compare_circuits()`: Quantitative comparison of circuits
- `_create_similarity_heatmap()`: Visualize cross-category similarities

#### 5. Configuration Files
- `configs/sae_training.yaml`: SAE training parameters
  - Model and dataset specifications
  - SAE architecture parameters (hidden dimensions, sparsity)
  - Training hyperparameters
- `configs/circuit_discovery.yaml`: Circuit discovery settings
  - Node and edge thresholds
  - Attribution methods
  - Output directories

#### 6. Scripts
- `scripts/train_saes_and_discover_circuits.sh`: Complete pipeline execution
  - Coordinates SAE training and circuit discovery
  - Handles configuration passing
  - Manages execution flow

### Workflow Explanation

#### Step 1: SAE Training
- Train separate sparse autoencoders for each model layer
- Use collected activations as training data
- Achieve sparse, interpretable feature representations

#### Step 2: Circuit Discovery 
- Encode activations through trained SAEs
- Compute feature importances using statistical methods
- Identify causally important features and their interactions
- Build sparse feature circuits for each refusal category

#### Step 3: Analysis & Comparison
- Compare circuits across refusal categories (violence vs misinformation)
- Compute similarity metrics to test modular vs monolithic hypotheses
- Generate visualizations for interpretation
