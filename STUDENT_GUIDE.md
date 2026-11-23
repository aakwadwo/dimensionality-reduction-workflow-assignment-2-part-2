# Student Guide

Welcome to Dimensionality Reduction Workflow Assignment 2 Part 2. This guide provides a structured learning plan for the dimensionality reduction codebase.

## Project Overview

This is a focused, production-style machine learning codebase that includes:

- The original PCA workflow (2D projection and explained variance analysis)
- Two techniques added for this assignment: Linear Discriminant Analysis (LDA) and Isomap
- A fair comparison of all three techniques on a held-out test set
- Supporting modules for configuration, validation, logging, plotting, and data loading
- An executable example script and a practice exercise

## Structured Learning Plan

### Week 1: Understanding the Foundation

**Days 1 through 2: Initial Familiarization**
1. Read QUICKSTART.md
2. Execute the example and the full workflow:
   ```bash
   python examples/run_dimensionality_reduction.py
   python main.py --save-plots --no-show
   ```
3. Review the generated plots in the `outputs/dimensionality_reduction/` directory

**Days 3 through 4: Understanding the Structure**
1. Read the Techniques and Evaluation Methods sections in README.md
2. Execute individual modules:
   ```bash
   python -m config.app_config
   python -m data_loaders.dataset_manager
   ```

**Days 5 through 7: Detailed Code Review**
Read these files in order:
1. `config/app_config.py`: Configuration patterns
2. `utils/data_validator.py`: Defensive programming
3. `utils/plot_manager.py`: Resource management
4. `data_loaders/dataset_manager.py`: Data loading
5. `workflows/dimensionality_reduction_workflow.py`: The complete workflow

**Week 1 Objective**: Understand the overall structure and be able to run everything successfully.

---

### Week 2: Hands-On Practice

**Exercise 1 (Dimensionality Reduction)**
1. Open `exercises/exercise_01_dimensionality_reduction.py`
2. Read the scenario and learning objectives
3. Complete all sections marked with TODO comments
4. Answer the questions at the bottom of the file
5. Compare your results with ASSIGNMENT_REPORT.md

**Week 2 Objective**: Be able to apply, evaluate, and choose between dimensionality reduction techniques.

---

### Week 3: Extension

1. Vary Isomap's `n_neighbors` and observe the effect
2. Add t-SNE for visualization only, and explain why it cannot join the test-set comparison
3. Repeat the comparison with several random seeds

---

## Success Criteria

By the end of this project, you should be able to:

1. Explain explained variance and reconstruction error, and why they apply to PCA only
2. Explain the difference between supervised (LDA) and unsupervised (PCA, Isomap) reduction
3. Explain the difference between linear and non-linear (manifold) reduction
4. Evaluate a reduced representation with trustworthiness, silhouette, clustering, and classification accuracy
5. Choose a technique and number of components for a given goal
