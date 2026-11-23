# Project Summary

## Overview

This project extracts the dimensionality reduction workflow (PCA) from the original Machine Learning Student Project into a standalone assignment and extends it with two additional techniques (Linear Discriminant Analysis and Isomap) and a consistent evaluation framework.

## Module Breakdown

| Module | Purpose | Complexity |
|--------|---------|------------|
| `config/app_config.py` | Configuration management | Low |
| `utils/data_validator.py` | Data validation | Low |
| `utils/plot_manager.py` | Plot handling | Low |
| `utils/logger.py` | Logging setup | Low |
| `data_loaders/dataset_manager.py` | Digits dataset | Low |
| `workflows/dimensionality_reduction_workflow.py` | PCA, LDA, Isomap, evaluation | Medium |
| `main.py` | Orchestrator | Low |
| `examples/run_dimensionality_reduction.py` | Standalone demo | Low |
| `exercises/exercise_01_dimensionality_reduction.py` | Practice problem | Low |

## Design Principles

### 1. Separation of Concerns
- **Config**: Only configuration
- **Utils**: Only utilities
- **Data Loaders**: Only data loading
- **Workflows**: Only the dimensionality reduction workflow

### 2. One Code Path for Every Technique
- `make_reducer()` is the only place techniques are created
- `self.techniques` drives every comparison, table, and plot
- Adding a technique requires changes in these two places only

### 3. Fair Comparison
- Same standardization and the same train/test split for every technique
- Scaler and reducer fitted on training data only
- Technique-specific metrics (explained variance, reconstruction error) reported only for PCA

## Workflow Architecture

```
┌─────────────────┐
│   AppConfig     │
│  (immutable)    │
└────────┬────────┘
         │
         ├─────────────────────────────┐
         │                             │
┌────────▼─────────┐         ┌─────────▼──────────┐
│  DatasetManager  │         │   PlotManager      │
│  - Load digits   │         │   - Save plots     │
│  - Validate      │         │   - Display plots  │
└────────┬─────────┘         └─────────▲──────────┘
         │                             │
         ▼                             │
┌──────────────────────────────────────┴─┐
│   DimensionalityReductionWorkflow      │
│   - run(): PCA 2D + 95% variance       │
│   - split_data()                       │
│   - compare_techniques_2d()            │
│       PCA / LDA / Isomap               │
│       trustworthiness, silhouette,     │
│       K-Means ARI, KNN accuracy        │
│   - evaluate_downstream_classification │
│       2, 3, 5, 9 components            │
└────────────────────────────────────────┘
```

## Technologies and Libraries

- **Python 3.10+**
- **NumPy**: Numerical computing
- **Pandas**: Tabular results
- **Matplotlib**: Visualization
- **Scikit-learn**: Decomposition, manifold learning, metrics

No dependencies were added beyond those of the original project.
