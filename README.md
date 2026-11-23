# Dimensionality Reduction Workflow Assignment 2 Part 2

This production-oriented machine learning project has been designed for intermediate students. It focuses entirely on **dimensionality reduction**, compressing high-dimensional data into a few informative features, using scikit-learn, with extensive documentation and educational comments throughout.

The project is extracted from the dimensionality reduction workflow (PCA) of the original Machine Learning Student Project and extended with **two additional techniques**: Linear Discriminant Analysis (LDA) and Isomap. The three techniques are compared on the same data, with the same preprocessing and the same evaluation. The measured results are in [ASSIGNMENT_REPORT.md](ASSIGNMENT_REPORT.md).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Guide](#module-guide)
- [Techniques](#techniques)
- [Evaluation Methods](#evaluation-methods)
- [Exercise](#exercise)
- [Key Concepts](#key-concepts)
- [Best Practices](#best-practices-demonstrated)
- [Troubleshooting](#troubleshooting)
- [Learning Path](#learning-path)

---

## Overview

I have organized this project as a modular, educational codebase built around a single, complete dimensionality reduction workflow. Each module features:

- Comprehensive documentation with detailed comments explaining the rationale behind design decisions
- Independent executability for focused learning and testing
- Production-ready implementation following software engineering best practices
- Student-friendly structure with a clear example and a practice exercise

### Learning Objectives

Through this project, you will learn:

1. **The Curse of Dimensionality**: Why fewer, better features help visualization, storage, and learning
2. **Three Families of Techniques**: Unsupervised linear (PCA), supervised linear (LDA), and non-linear manifold learning (Isomap)
3. **Explained Variance and Reconstruction Error**: How much information PCA keeps
4. **Fair Evaluation**: Fitting on training data, evaluating on held-out data, and choosing metrics that apply to each technique
5. **Software Engineering**: Modular code design, logging, validation, and configuration management

---

## Project Structure

```
dimensionality-reduction-workflow-assignment-2-part-2/
│
├── README.md                                    # This file
├── ASSIGNMENT_REPORT.md                         # Results, technique choices, conclusions
├── requirements.txt                             # Python dependencies
├── main.py                                      # Runs the full workflow
│
├── config/                                      # Configuration management
│   ├── __init__.py
│   └── app_config.py                            # Centralized configuration
│
├── utils/                                       # Utility modules
│   ├── __init__.py
│   ├── data_validator.py                        # Data validation utilities
│   ├── plot_manager.py                          # Plot handling (save/display)
│   └── logger.py                                # Logging configuration
│
├── data_loaders/                                # Data loading
│   ├── __init__.py
│   └── dataset_manager.py                       # Digits dataset
│
├── workflows/                                   # ML workflow
│   ├── __init__.py
│   └── dimensionality_reduction_workflow.py     # PCA, LDA, Isomap, evaluation
│
├── examples/                                    # Standalone example
│   └── run_dimensionality_reduction.py          # Step-by-step demo
│
├── exercises/                                   # Practice exercise
│   └── exercise_01_dimensionality_reduction.py  # Digit compression challenge
│
├── outputs/                                     # Generated artifacts
│   └── dimensionality_reduction/                # 2D plots and accuracy curves
│
└── visuals/                                     # Visual documentation
    └── (optional: add your own diagrams)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher (the code uses `dict[str, Any]` and `X | Y` type hints)
- pip (Python package manager)

### Setup Steps

1. **Clone the repository and open its root folder**
   ```bash
   git clone <repository-url> dimensionality-reduction-workflow-assignment-2-part-2
   cd dimensionality-reduction-workflow-assignment-2-part-2
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

No dependencies were added beyond those of the original project (NumPy, pandas, Matplotlib, scikit-learn).

---

## Quick Start

### Option 1: Run the Step-by-Step Example (recommended)

```bash
python examples/run_dimensionality_reduction.py
```

This command will:
- Run the original PCA analysis (2D projection and components for 95% variance)
- Compare PCA, LDA, and Isomap in 2D on a held-out test set
- Measure classification accuracy after reduction for 2, 3, 5, and 9 components
- Save all plots to `outputs/dimensionality_reduction/` and print an interpretation

### Option 2: Run the Full Workflow

```bash
python main.py --save-plots --no-show
```

### Option 3: Test Individual Modules

```bash
python -m config.app_config
python -m utils.data_validator
python -m data_loaders.dataset_manager
python -m workflows.dimensionality_reduction_workflow
```

---

## Module Guide

### 1. Configuration, Utilities

`config/app_config.py` and `utils/` are unchanged from the original project. `test_size` and `random_state` from the configuration control the train/test split used for the comparison.

### 2. Data Loaders (`data_loaders/dataset_manager.py`)

| Dataset | Type | Samples | Features | Classes |
|---------|------|---------|----------|---------|
| Digits | 8x8 grayscale images | 1797 | 64 (pixels, 0 to 16) | 10 digits, about 180 each |

The loader now validates the digits data, as the original project's other loaders already did.

### 3. Dimensionality Reduction Workflow (`workflows/dimensionality_reduction_workflow.py`)

| Method | What It Does | Status |
|--------|--------------|--------|
| `run(X, y)` | Standardize, PCA to 2D, plot, components for 95% variance | Original (unchanged) |
| `make_reducer(name, k)` | Create PCA, LDA, or Isomap with k components | New |
| `split_data(X, y)` | Stratified 80/20 split | New |
| `compare_techniques_2d(...)` | 2D embeddings plus trustworthiness, silhouette, K-Means ARI, KNN accuracy, fit time | New |
| `evaluate_downstream_classification(...)` | KNN test accuracy for 2, 3, 5, 9 components; PCA variance and reconstruction error | New |

**Generated Plots** (in `outputs/dimensionality_reduction/`):
- `pca_digits_2d.png`: Original PCA projection of all 1797 images
- `dr_comparison_2d.png`: PCA, LDA, and Isomap 2D embeddings of the test set, side by side
- `dr_downstream_knn_accuracy.png`: KNN accuracy against number of components for each technique

---

## Techniques

| Technique | Supervised | Linear | Can Transform New Data | Inverse Transform | Origin |
|-----------|-----------|--------|------------------------|-------------------|--------|
| PCA | No | Yes | Yes | Yes | Original |
| **LDA** | **Yes** | Yes | Yes | No | **Added** |
| **Isomap** (`n_neighbors=10`) | No | **No** | Yes | No | **Added** |

### Why These Two Techniques Were Added

- **LDA**: PCA keeps the directions of largest variance, but on the digits data the largest variance is not the same as the best separation between digits (PCA's 2D projection keeps only 21.6% of the variance and overlaps most digits). LDA uses the labels to find the directions that best separate the classes. Because it is linear, like PCA, the comparison isolates one question: does supervision produce a better low-dimensional space?
- **Isomap**: Handwritten digits vary smoothly (stroke thickness, slant, shape), which is usually described as data lying on a curved, low-dimensional manifold. Isomap is a non-linear technique that preserves distances along that manifold. Because it is unsupervised, like PCA, the comparison isolates a second question: does a non-linear method capture structure that PCA misses?

Techniques that were considered and not chosen:
- **t-SNE**: Popular for digit visualization, but it cannot transform new data. It could not be fitted on training data and evaluated on the test set like the other techniques, so the comparison would not be fair.
- **NMF**: Requires non-negative input, so it could not use the same standardized data as the other techniques.
- **Truncated SVD and Kernel PCA**: Too close to PCA to add a meaningfully different perspective, and Kernel PCA needs kernel tuning.
- **UMAP**: Would add a new dependency.

---

## Evaluation Methods

Every technique is fitted on the **training set** (1437 images) and evaluated on the **test set** (360 images). Fitting on the training set matters most for LDA. If it were fitted on all the data, its class separation would be measured on the same labels it was trained with.

| Metric | Applies To | What It Measures |
|--------|-----------|------------------|
| Trustworthiness (k=10) | All | Whether 2D neighbours were also neighbours in 64D (structure preservation) |
| Silhouette (true labels) | All | Whether images of the same digit are close together and far from other digits |
| K-Means ARI | All | Whether K-Means (K=10) can recover the digits after reduction (clustering performance) |
| KNN test accuracy | All | How much class information survives reduction (classification performance), for 2, 3, 5, and 9 components |
| Fit time | All | Computational cost |
| Explained variance ratio | PCA only | Share of the variance kept |
| Reconstruction error | PCA only | How well the original pixels can be rebuilt |

Explained variance and reconstruction error are **not forced** on LDA and Isomap. LDA has no inverse transform, and its `explained_variance_ratio_` measures between-class variance, which is a different quantity. Isomap has no inverse transform either, and its `reconstruction_error()` refers to geodesic distances rather than pixels. Comparing these numbers with PCA's would be misleading.

The component grid stops at 9 because LDA can produce at most (number of classes - 1) = 9 components.

---

## Exercise

**File**: `exercises/exercise_01_dimensionality_reduction.py`

**Scenario**: Compress scanned handwritten digits for a postal service while keeping the information needed to recognize them.

**Time**: 30 to 45 minutes

---

## Key Concepts

### Variance Is Not the Same as Usefulness

PCA keeps the directions where the pixels vary most. Those directions do not necessarily separate the digits. This is why a supervised technique (LDA) or a non-linear technique (Isomap) can produce a more useful 2D view than PCA, even though PCA is optimal for its own goal (variance and reconstruction).

### Linear versus Non-Linear

PCA and LDA project the data onto flat planes. Isomap first builds a nearest-neighbour graph and measures distances along it, so it can unfold curved structure.

### Supervised versus Unsupervised

LDA needs labels. When labels are available and the goal is classification, this is a strong advantage. When labels are not available, or the goal is compression or reconstruction, PCA or Isomap must be used.

---

## Best Practices Demonstrated

1. **Fair Comparison**: Same data, same scaling, same split, and same metrics for every technique
2. **No Test-Set Leakage**: Scaler and reducer fitted on training data only
3. **Honest Metrics**: Technique-specific metrics are reported only where they are meaningful
4. **Reproducibility**: `random_state=42`, so results are identical on every run
5. **Modular Code**: Each module has a single, clear responsibility

---

## Troubleshooting

### Import Errors

**Solution**: Run commands from the project root (the folder that contains `main.py`).

### RuntimeWarning When Running `python -m workflows.dimensionality_reduction_workflow`

This warning ("found in sys.modules after import of package") is harmless. It also occurs in the original project, because `workflows/__init__.py` imports the module before `-m` runs it.

### Plots Not Displaying

**Solution**: Use `show_plots=True` in `AppConfig`, or omit `--no-show` when running `main.py`.

---

## Learning Path

### Beginner Path
1. Run `python examples/run_dimensionality_reduction.py` and study the plots
2. Read `ASSIGNMENT_REPORT.md`
3. Complete Exercise 1

### Intermediate Path
1. Change Isomap's `n_neighbors` (5, 10, 30) and observe trustworthiness and accuracy
2. Add a cumulative explained variance plot for PCA

### Advanced Path
1. Add t-SNE as a visualization-only comparison and explain why it cannot join the test-set evaluation
2. Repeat the comparison over several random seeds and report mean and standard deviation

---

## Additional Resources

- [Scikit-learn Decomposition Guide (PCA)](https://scikit-learn.org/stable/modules/decomposition.html)
- [Scikit-learn Manifold Learning Guide (Isomap)](https://scikit-learn.org/stable/modules/manifold.html)
- [Scikit-learn Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)

---

## License

This project is designed for educational purposes. Feel free to use, modify, and share for learning.

## Acknowledgments

- **Scikit-learn team**: For the excellent ML library
- **Original author (Majid)**: For the initial comprehensive ML script
