# Analysis and Recommendations for LLM Refusal Circuit Analysis

## Executive Summary

This document provides a comprehensive analysis of the codebase and recommendations for addressing data imbalance issues, improving circuit visualization, and cleaning up the codebase structure.

## 1. Data Imbalance Analysis

### Current Situation

The OR-Bench dataset has a significant imbalance between safe and toxic samples:

- **Safe samples (80k + 1k hard)**: ~81,000 samples
- **Toxic samples**: ~1,000 samples (from `or-bench-toxic.csv`)
- **Ratio**: Approximately 81:1 safe-to-toxic ratio

This severe imbalance can significantly impact SAE training quality because:
1. SAEs trained on imbalanced data will learn features biased toward safe samples
2. Toxic-specific features may be underrepresented in the learned SAE features
3. Circuit discovery for refusal behavior may miss important toxic-specific patterns

### Solution: Data Split Analysis Script

**Created**: `analyze_data_splits.py`

This script:
- Analyzes data distribution across all OR-Bench categories
- Visualizes safe/toxic/hard splits per category
- Identifies categories with severe imbalances
- Generates warnings for categories with <10% toxic samples

**Usage**:
```bash
python analyze_data_splits.py --dataset_dir /path/to/or-bench --output_dir results/data_analysis
```

### Recommendations for Addressing Imbalance

1. **Separate SAE Training** (IMPLEMENTED):
   - Train separate SAEs for safe and toxic samples
   - This ensures both types of patterns are captured
   - Use `separate_safe_toxic: true` in configuration

2. **Weighted Sampling** (RECOMMENDED):
   - During SAE training, use weighted sampling to balance safe and toxic samples
   - This can be added to `SAETrainer` class

3. **Data Augmentation** (FUTURE):
   - Augment toxic samples if needed
   - Use techniques like paraphrasing or prompt variations

4. **Category-Specific Analysis**:
   - Some categories may have sufficient toxic samples
   - Focus analysis on categories with balanced data

## 2. Circuit Visualization Enhancements

### New Visualizations Implemented

#### 2.1 Safe vs Toxic Separate Heatmaps

**Function**: `CircuitVisualizer.create_safe_toxic_heatmap()`

Creates two side-by-side heatmaps:
- **Left**: Similarity matrix for safe circuits across categories
- **Right**: Similarity matrix for toxic circuits across categories

**Purpose**: Shows how similar circuits are within each type (safe or toxic) across different categories.

**Output**: `{model}_safe_toxic_separate_heatmap.png`

#### 2.2 Cross-Refusal Heatmap

**Function**: `CircuitVisualizer.create_cross_refusal_heatmap()`

Creates a heatmap with:
- **Rows**: Safe circuits (by category)
- **Columns**: Toxic circuits (by category)
- **Diagonal**: Same category, comparing safe vs toxic circuits

**Purpose**: 
- The diagonal shows if refusal circuits for a category are the same as safe circuits
- Off-diagonal shows cross-category similarities
- High diagonal values = similar circuits for safe and toxic (monolithic)
- Low diagonal values = different circuits (modular)

**Output**: `{model}_safe_vs_toxic_cross_heatmap.png`

### Implementation Details

1. **Circuit Discovery Enhancement**:
   - Modified `SAECircuitDiscoverer.discover_circuit_with_saes()` to support `separate_safe_toxic` parameter
   - When enabled, splits activations by refusal label and discovers separate circuits
   - Returns tuple: `(safe_circuit, toxic_circuit)`

2. **Analyzer Updates**:
   - `RefusalCircuitAnalyzer` now stores separate safe and toxic circuits
   - Generates new visualizations automatically when `separate_safe_toxic: true` in config
   - Saves circuits separately: `{category}_safe_circuit.json` and `{category}_toxic_circuit.json`

## 3. Codebase Cleanup Recommendations

### 3.1 File Organization

**Current Issues**:
- Some duplicate functionality between files
- Unused placeholder code in `data_utils.py`
- Mixed concerns in some modules

**Recommendations**:

1. **Consolidate Data Loading**:
   - Remove duplicate dataset loading logic
   - Keep `ORBenchDataProcessor` as the primary data processor
   - Simplify `data_utils.py` to use `ORBenchDataProcessor` exclusively

2. **Remove Unused Code**:
   - `analyze_all_sae_training.py` and `analyze_sae_training.py` - check if still needed
   - Placeholder methods in `CircuitDiscoverer` that aren't used
   - Unused imports and dead code

3. **Organize Configuration Files**:
   - Create example configs for different use cases:
     - `configurations/orbench_full.yaml` - all 10 categories
     - `configurations/orbench_balanced.yaml` - balanced safe/toxic
     - `configurations/orbench_separate_circuits.yaml` - with separate safe/toxic

### 3.2 Code Structure Improvements

**Recommended Structure**:
```
src/
  ├── data/              # Data loading and processing
  │   ├── orbench_processor.py
  │   └── data_utils.py
  ├── models/            # Model loading and inference
  │   ├── model_loader.py
  │   └── activation_collector.py
  ├── sae/               # SAE training
  │   └── sae_trainer.py
  └── evaluation/        # Evaluation metrics
      └── refusal_metrics.py

circuits/
  ├── discovery/         # Circuit discovery algorithms
  │   └── sae_discovery.py
  ├── visualization/     # Visualization tools
  │   └── circuit_viz.py
  └── analysis/          # Analysis and comparison
      ├── circuit_analyzer.py
      └── statistical_analysis.py
```

### 3.3 Documentation Improvements

1. **Add Docstrings**: Ensure all public functions have comprehensive docstrings
2. **Type Hints**: Add type hints throughout (partially done)
3. **Configuration Guide**: Document all configuration options
4. **Troubleshooting Guide**: Expand troubleshooting section in README

## 4. Configuration Updates

### New Configuration Option

Add to your YAML config files:

```yaml
# Separate safe and toxic circuits
separate_safe_toxic: true  # Default: false

# This enables:
# - Separate circuit discovery for safe and toxic samples
# - New visualization heatmaps
# - Separate circuit JSON files
```

### Example Configuration

```yaml
# configurations/orbench_separate_circuits.yaml
models:
  - "meta-llama/Llama-2-7b-chat-hf"

categories:
  - "violence"
  - "deception"
  - "privacy"
  # ... other categories

activation_layers:
  - "residuals_10"
  - "residuals_15"
  - "mlp_11"

# NEW: Enable separate safe/toxic circuit discovery
separate_safe_toxic: true

# SAE training settings
sae_hidden_dim: 8192
sae_max_samples: 100000
sae_batch_size: 512
sae_epochs: 100

# Circuit discovery settings
node_threshold: 0.1
edge_threshold: 0.01

result_dir: "results"
sae_dir: "results/saes"
```

## 5. Usage Instructions

### Step 1: Analyze Data Splits

```bash
python analyze_data_splits.py \
  --dataset_dir /path/to/or-bench \
  --output_dir results/data_analysis
```

This will:
- Show category-wise statistics
- Generate visualization plots
- Identify imbalanced categories
- Save statistics to JSON

### Step 2: Run Circuit Analysis with Separate Circuits

```bash
python run_circuit_analysis.py \
  --config configurations/orbench_separate_circuits.yaml
```

This will:
- Train SAEs (if `train_saes: true`)
- Discover separate safe and toxic circuits for each category
- Generate all three types of heatmaps:
  1. Original similarity heatmap (all categories)
  2. Safe vs Toxic separate heatmaps
  3. Safe vs Toxic cross-comparison heatmap

### Step 3: Interpret Results

**Safe vs Toxic Separate Heatmaps**:
- Compare similarity patterns within safe circuits vs within toxic circuits
- High similarity in toxic heatmap = monolithic refusal behavior
- Low similarity = modular refusal behavior

**Cross-Refusal Heatmap**:
- **Diagonal values**: Same category, safe vs toxic
  - High (>0.7): Similar circuits = refusal uses same features as safe
  - Low (<0.3): Different circuits = refusal uses distinct features
- **Off-diagonal**: Cross-category comparisons

## 6. Expected Output Structure

```
results/
├── data_analysis/                    # NEW
│   ├── data_split_analysis.png
│   └── data_split_statistics.json
├── circuits/
│   ├── {model}_{category}_safe_circuit.json    # NEW
│   ├── {model}_{category}_toxic_circuit.json  # NEW
│   └── {model}_comparison.json
└── visualizations/
    └── {model}/
        └── similarity_heatmaps/
            ├── {model}_similarity_heatmap.png              # Original
            ├── {model}_safe_toxic_separate_heatmap.png     # NEW
            └── {model}_safe_vs_toxic_cross_heatmap.png     # NEW
```

## 7. Next Steps

### Immediate Actions

1. ✅ **DONE**: Data split analysis script
2. ✅ **DONE**: Separate safe/toxic circuit discovery
3. ✅ **DONE**: New visualization heatmaps
4. ⏳ **TODO**: Test with actual data
5. ⏳ **TODO**: Update README with new features
6. ⏳ **TODO**: Create example configuration files

### Future Enhancements

1. **Weighted SAE Training**: Implement weighted sampling for balanced training
2. **Category-Specific SAEs**: Train separate SAEs per category if needed
3. **Interactive Visualizations**: Create interactive heatmaps with plotly
4. **Statistical Significance**: Add significance tests for safe vs toxic comparisons
5. **Ablation Studies**: Test impact of data imbalance on circuit quality

## 8. Known Issues and Limitations

1. **Data Imbalance**: OR-Bench has ~81:1 safe-to-toxic ratio
   - **Impact**: May affect SAE training quality
   - **Mitigation**: Use separate safe/toxic circuits (implemented)

2. **Small Toxic Dataset**: Only ~1k toxic samples total
   - **Impact**: May limit statistical power
   - **Mitigation**: Focus on categories with sufficient toxic samples

3. **Circuit Discovery**: Current method uses correlation-based importance
   - **Limitation**: May miss complex interactions
   - **Future**: Consider gradient-based attribution methods

## 9. Conclusion

The implemented changes address the main concerns:

1. ✅ **Data Imbalance Visibility**: `analyze_data_splits.py` shows the problem clearly
2. ✅ **Separate Circuit Discovery**: Enables analysis of safe vs toxic circuits
3. ✅ **Enhanced Visualizations**: New heatmaps provide deeper insights
4. ⏳ **Code Cleanup**: Recommendations provided, needs implementation

The codebase is now better equipped to handle imbalanced data and provide more nuanced analysis of refusal circuits across safe and toxic samples.

