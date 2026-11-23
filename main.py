"""
Main Orchestrator - Run the Dimensionality Reduction Workflow
==============================================================

This script runs the complete dimensionality reduction workflow:
1. PCA to 2D with explained variance analysis (original workflow)
2. Components needed for 95% variance
3. 2D comparison of PCA, LDA and Isomap on a held-out test set
4. Classification accuracy after reduction (2, 3, 5 and 9 components)

Usage:
    python main.py

    # Save plots without displaying them:
    python main.py --save-plots --no-show

    # Custom output directory:
    python main.py --output-dir my_results --save-plots

This is useful for:
- Getting a complete overview of the dimensionality reduction techniques
- Comparing different dimensionality reduction approaches
- Generating a full set of results and visualizations
"""

import argparse
import logging
from pathlib import Path

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging
from workflows import DimensionalityReductionWorkflow


class MachineLearningApplication:
    """
    Orchestrates all machine learning workflows.

    This class coordinates the execution of multiple ML workflows,
    ensuring proper setup and error handling.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the application.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create managers and workflows
        self.datasets = DatasetManager()
        self.dimensionality_reduction = DimensionalityReductionWorkflow(config)

    def run_all(self) -> None:
        """
        Run all ML workflows in sequence.

        Catches and logs errors without stopping the entire application.
        This ensures that if one workflow fails, others still run.
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        workflows = [
            ("Dimensionality Reduction", self._run_dimensionality_reduction),
        ]

        print("="*70)
        print("DIMENSIONALITY REDUCTION WORKFLOW ORCHESTRATOR")
        print("="*70)
        print()

        for name, workflow_func in workflows:
            try:
                print(f"\n{'='*70}")
                print(f"Running {name} Workflow...")
                print('='*70)
                workflow_func()
                print(f"\n✓ {name} workflow completed successfully!")
            except Exception as e:
                self.logger.exception(f"{name} workflow failed: {e}")
                print(f"\n✗ {name} workflow failed: {e}")

        print("\n" + "="*70)
        print("DIMENSIONALITY REDUCTION WORKFLOW COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.config.output_dir}")

    def _run_dimensionality_reduction(self) -> None:
        """Run dimensionality reduction workflow."""
        X, y = self.datasets.load_digits_data()
        pca, X_2d, components_95 = self.dimensionality_reduction.run(X, y)

        variance_2d = pca.explained_variance_ratio_.sum()

        print(f"\nDimensions: {X.shape[1]} → 2")
        print(f"Variance retained: {variance_2d:.1%}")
        print(f"Components for 95% variance: {components_95}")

        # Compare PCA, LDA and Isomap (fitted on training data only)
        X_train, X_test, y_train, y_test = self.dimensionality_reduction.split_data(X, y)

        results_2d, _ = self.dimensionality_reduction.compare_techniques_2d(
            X_train, X_test, y_train, y_test
        )
        print("\n2D Comparison (test set):")
        print(results_2d.round(4).to_string(index=False))

        downstream = self.dimensionality_reduction.evaluate_downstream_classification(
            X_train, X_test, y_train, y_test
        )
        print("\nKNN Test Accuracy After Reduction:")
        print(downstream.round(4).to_string(index=False))
        print(f"No reduction ({X.shape[1]} features): "
              f"{downstream.attrs['baseline_accuracy']:.4f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the dimensionality reduction workflow demonstrations."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated artifacts (default: outputs/)",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save generated figures to output directory",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Configure logging
    configure_logging()

    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = AppConfig(
        random_state=42,
        test_size=0.20,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        show_plots=not args.no_show,
    )

    # Run application
    app = MachineLearningApplication(config)
    app.run_all()


if __name__ == "__main__":
    main()
