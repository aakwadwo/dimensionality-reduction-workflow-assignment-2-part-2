# Quick Start Guide

This guide will help you get the project running in approximately 5 minutes.

## Step 1: Installation (2 minutes)

```bash
# Navigate to the repository root
cd dimensionality-reduction-workflow-assignment-2-part-2

# Install all required packages
pip install -r requirements.txt
```

## Step 2: Run the Dimensionality Reduction Example (1 minute)

```bash
python examples/run_dimensionality_reduction.py
```

When this completes successfully, you should observe:
- The original PCA results (variance explained and components for 95% variance)
- A 2D comparison table of PCA, LDA, and Isomap on the test set
- A table of KNN accuracy after reduction for 2, 3, 5, and 9 components
- An interpretation naming the best technique for each goal
- Generated plots saved in the `outputs/dimensionality_reduction/` directory

## Step 3: Run the Full Workflow (1 minute)

```bash
python main.py --save-plots --no-show
```

## Step 4: Attempt the Practice Exercise

```bash
python exercises/exercise_01_dimensionality_reduction.py
```

---

## Understanding the Output

```
Technique  ...  Trustworthiness  Silhouette  K-Means ARI  KNN Accuracy (2D)
      PCA  ...           0.8036      0.0251       0.3000             0.5222
      LDA  ...           0.7799      0.1950       0.5180             0.7083
   Isomap  ...           0.8755      0.2252       0.5053             0.8194
```

**Interpretation**:
- In 2D, Isomap keeps the most neighbourhood structure (trustworthiness) and separates the digits best (silhouette, KNN accuracy)
- PCA's 2D projection keeps only about 22% of the variance, so most digits overlap
- With 9 components, LDA reaches 0.975 KNN accuracy, slightly above using all 64 features (0.964)

---

## Recommended Next Steps

1. **Read ASSIGNMENT_REPORT.md** for the results and their interpretation
2. **Study `workflows/dimensionality_reduction_workflow.py`**, beginning with `run`, then `compare_techniques_2d`
3. **Complete the practice exercise**

---

## Troubleshooting Common Issues

### Import Errors

Verify you are in the project root (the folder that contains `main.py`).

### Plots Not Displaying

```python
config = AppConfig(
    save_plots=True,
    show_plots=True,  # Change from False to True
)
```

---

You are now ready to begin exploring the project. The code contains extensive comments to support your learning process.
