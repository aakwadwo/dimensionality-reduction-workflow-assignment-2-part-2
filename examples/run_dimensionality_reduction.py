"""
Standalone Dimensionality Reduction Example
============================================

This script demonstrates PCA (Principal Component Analysis) and compares
it with two additional techniques: LDA (Linear Discriminant Analysis)
and Isomap.
Run this to see dimensionality reduction in action!

Usage:
    python examples/run_dimensionality_reduction.py

What This Script Does:
    1. Loads digits dataset (64 features)
    2. Applies PCA to reduce to 2D
    3. Visualizes results
    4. Determines components needed for 95% variance
    5. Splits data into train/test sets
    6. Compares PCA, LDA and Isomap in 2D on the test set
    7. Compares classification accuracy after reduction (2-9 components)
    8. Summarizes which technique performs best and why

Learning Objectives:
    - Understand dimensionality reduction
    - Apply PCA for visualization
    - Interpret explained variance
    - Balance dimensionality vs information retention
    - Compare linear vs non-linear and supervised vs unsupervised techniques
    - Choose evaluation metrics that fit each technique
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import DimensionalityReductionWorkflow


def main():
    """Run complete dimensionality reduction workflow."""

    # Configure logging
    configure_logging()

    # Create configuration
    config = AppConfig(
        random_state=42,
        test_size=0.20,
        output_dir=Path("outputs/dimensionality_reduction"),
        save_plots=True,
        show_plots=False,
    )

    print("="*70)
    print("DIMENSIONALITY REDUCTION WORKFLOW DEMO")
    print("="*70)
    print()

    # Load data
    print("Step 1: Loading Data...")
    dm = DatasetManager()
    X, y = dm.load_digits_data()
    print(f"  ✓ Loaded {len(X)} digit images")
    print(f"  ✓ Original dimensions: {X.shape[1]} features (8×8 pixels)")
    print(f"  ✓ Classes: {len(np.unique(y))} digits (0-9)")
    print()

    # Run workflow
    print("Step 2: Applying PCA...")
    workflow = DimensionalityReductionWorkflow(config)
    pca, X_2d, components_95 = workflow.run(X, y)
    print()

    # Display results
    variance_2d = pca.explained_variance_ratio_.sum()

    print("="*70)
    print("PCA RESULTS")
    print("="*70)
    print()
    print(f"Original Dimensions:       {X.shape[1]} features")
    print(f"Reduced Dimensions:        {X_2d.shape[1]} features (for visualization)")
    print()
    print(f"Variance Explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"Variance Explained by PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"Total (2 components):      {variance_2d:.1%}")
    print()
    print(f"Components for 95% Variance: {components_95}")
    print()

    # Compare techniques
    print("="*70)
    print("COMPARING PCA, LDA AND ISOMAP")
    print("="*70)
    print()
    print("Step 3: Splitting Data (all techniques fitted on training data only)...")
    X_train, X_test, y_train, y_test = workflow.split_data(X, y)
    print(f"  ✓ Training set: {len(X_train)} samples")
    print(f"  ✓ Test set: {len(X_test)} samples")
    print()

    print("Step 4: Reducing to 2D and Evaluating on the Test Set...")
    results_2d, _ = workflow.compare_techniques_2d(X_train, X_test, y_train, y_test)
    print()
    print(results_2d.round(4).to_string(index=False))
    print()

    print("Step 5: Classification Accuracy After Reduction (KNN, k=5)...")
    downstream = workflow.evaluate_downstream_classification(
        X_train, X_test, y_train, y_test
    )
    baseline_accuracy = downstream.attrs["baseline_accuracy"]
    print()
    print(downstream.round(4).to_string(index=False))
    print(f"\n  Reference - KNN on all {X.shape[1]} features: {baseline_accuracy:.4f}")
    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    print(f"2D Visualization:")
    print(f"  → We reduced from {X.shape[1]} to {X_2d.shape[1]} dimensions")
    print(f"  → PCA retained {variance_2d:.1%} of the original variance")
    print(f"  → Lost {(1-variance_2d)*100:.1f}% of the variance (acceptable for visualization)")
    print()
    print(f"95% Variance Threshold:")
    print(f"  → Need {components_95} components (instead of {X.shape[1]})")
    print(f"  → That's a {(1 - components_95/X.shape[1])*100:.0f}% reduction!")
    print(f"  → Retains 95% of the variance with {components_95/X.shape[1]:.1%} of features")
    print()

    metrics_2d = results_2d.set_index("Technique")
    best_2d = metrics_2d["KNN Accuracy (2D)"].idxmax()
    best_trust = metrics_2d["Trustworthiness"].idxmax()
    best_sil = metrics_2d["Silhouette"].idxmax()
    best_ari = metrics_2d["K-Means ARI"].idxmax()

    last = downstream.iloc[-1]
    best_last = max(workflow.techniques, key=lambda t: last[f"{t} KNN Accuracy"])

    print("Technique Comparison:")
    print(f"  → Best 2D class separation (KNN accuracy):   {best_2d}")
    print(f"  → Best 2D silhouette (true digit labels):    {best_sil}")
    print(f"  → Best 2D K-Means recovery of digits (ARI):  {best_ari}")
    print(f"  → Best neighbourhood preservation (trust.):  {best_trust}")
    print(f"  → Best KNN accuracy at {int(last['Components'])} components:       "
          f"{best_last} ({last[f'{best_last} KNN Accuracy']:.4f} vs "
          f"{baseline_accuracy:.4f} with all {X.shape[1]} features)")
    print()
    print("Per number of components (best technique):")
    for _, row in downstream.iterrows():
        winner = max(workflow.techniques, key=lambda t: row[f"{t} KNN Accuracy"])
        print(f"  {int(row['Components'])} components: {winner:7s} "
              f"({row[f'{winner} KNN Accuracy']:.4f})   "
              f"PCA variance retained = {row['PCA Variance Retained']:.1%}")
    print()
    print("Benefits:")
    print("  ✓ Faster training (fewer features to process)")
    print("  ✓ Less storage (fewer numbers to store)")
    print("  ✓ Noise reduction (removed less important variations)")
    print("  ✓ Visualization (can plot high-dimensional data)")
    print()
    print(f"Plots saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
