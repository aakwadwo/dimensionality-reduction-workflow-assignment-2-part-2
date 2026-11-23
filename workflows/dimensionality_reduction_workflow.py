"""
Dimensionality Reduction Workflow Module
=========================================

Purpose:
    Reduce high-dimensional data to fewer dimensions while preserving information.
    Essential for visualization, noise reduction, and computational efficiency.

The Curse of Dimensionality:
    As dimensions increase:
    - Distance becomes less meaningful
    - Data becomes sparse
    - Computation becomes expensive
    - Visualization becomes impossible (we can only see 2-3D!)

Why Dimensionality Reduction?
    1. Visualization: Plot high-D data in 2D/3D
    2. Noise reduction: Keep signal, remove noise
    3. Feature extraction: Find important patterns
    4. Computational efficiency: Faster training
    5. Storage: Less data to store

Techniques Covered:
    Original workflow:
    1. PCA (Principal Component Analysis)
        - Finds directions of maximum variance
        - Projects data onto these directions
        - Keeps most important information
        - Linear method (assumes linear relationships)
        - Unsupervised (never looks at the labels)
    
    Added in Assignment 2 Part 2:
    2. LDA (Linear Discriminant Analysis)
        - Finds directions that best SEPARATE the classes
        - Supervised: uses the digit labels during fitting
        - Linear, like PCA, so it isolates one question:
          "does using labels give a better low-dimensional space?"
        - At most (n_classes - 1) = 9 components for 10 digits
    3. Isomap (Isometric Mapping)
        - Non-linear "manifold learning" technique
        - Builds a nearest-neighbour graph, measures distances ALONG the
          graph (geodesic distances), then embeds those distances
        - Unsupervised, like PCA, so it isolates a different question:
          "does a non-linear method capture structure PCA misses?"
        - Unlike t-SNE, Isomap can transform NEW data after fitting,
          so it can be fitted on training data and evaluated on a
          held-out test set exactly like PCA and LDA
    
    Together: unsupervised-linear (PCA), supervised-linear (LDA),
    unsupervised-non-linear (Isomap).

How the Techniques Are Compared (fairly):
    - Same data, same standardization, same train/test split
    - Every technique is fitted on the TRAINING set only and
      evaluated on the held-out TEST set (important for LDA, which
      would otherwise have "seen" the labels it is judged on)
    - Metrics that apply to ALL techniques:
        * Trustworthiness: are neighbours in the reduced space also
          neighbours in the original 64-D space? (structure preservation)
        * Silhouette score of the true digit labels (class separation)
        * K-Means ARI: can K-Means recover the digits after reduction?
        * KNN accuracy: classification performance after reduction,
          for several numbers of components
        * Fit time
    - Metrics that apply to PCA ONLY (not forced on the others):
        * Explained variance ratio
        * Reconstruction error (LDA and Isomap have no inverse transform)

Key Concepts:
    - Principal Components: New features that are combinations of original features
    - Explained Variance: How much information each component captures
    - Dimensionality: Number of features (e.g., 64 → 2)
    - Feature extraction vs. selection

Real-World Applications:
    - Image compression (reduce pixels while keeping quality)
    - Face recognition (eigenfaces)
    - Genomics (visualize gene expression patterns)
    - Recommender systems (reduce user-item matrix)
    - Text analysis (reduce word dimensions)
"""

import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config.app_config import AppConfig
from utils.data_validator import DataValidator
from utils.plot_manager import PlotManager


class DimensionalityReductionWorkflow:
    """
    Perform PCA, LDA and Isomap dimensionality reduction with visualization.
    
    This class implements:
    - Data standardization (required for PCA)
    - PCA for 2D visualization
    - Variance analysis
    - Determining components needed for target variance
    - (Assignment 2 Part 2) A fair comparison of PCA, LDA and Isomap:
      2D embeddings, structure/separation metrics, and classification
      accuracy after reduction for several numbers of components
    
    Attributes:
        config (AppConfig): Application configuration
        logger (Logger): For tracking progress
        plots (PlotManager): For handling visualizations
    """
    
    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the dimensionality reduction workflow.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plots = PlotManager(config)
        
        # Techniques compared in this assignment (original + 2 added)
        self.techniques: list[str] = ["PCA", "LDA", "Isomap"]
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[PCA, np.ndarray, int]:
        """
        Perform PCA dimensionality reduction and analysis.
        
        PCA (Principal Component Analysis):
            1. Standardize features (mean=0, std=1)
            2. Compute covariance matrix
            3. Find eigenvectors (principal components)
            4. Project data onto top components
        
        What are Principal Components?
            - Linear combinations of original features
            - Ordered by variance explained (PC1 explains most, PC2 second-most, etc.)
            - Orthogonal (perpendicular) to each other
            
            Example for 2D → 1D:
            Original: [height, weight]
            PC1 might be: 0.7*height + 0.7*weight (size)
            PC2 might be: 0.7*height - 0.7*weight (shape)
        
        Why Standardize?
            Features with large values dominate PCA without standardization
            Example: Income ($50,000) vs. Age (30) - income would dominate!
            Standardization puts all features on equal footing
        
        Workflow Steps:
            1. Validate data
            2. Standardize features
            3. Apply PCA to reduce to 2D (for visualization)
            4. Calculate variance explained
            5. Determine components needed for 95% variance
            6. Visualize results
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels for coloring visualization (n_samples,)
        
        Returns:
            pca_2d: Fitted PCA model (2 components)
            X_2d: Transformed data (n_samples, 2)
            components_95: Number of components for 95% variance
        
        Example:
            Input: 64 features (8×8 pixel images)
            Output: 2 features (for visualization)
            Variance: Maybe 30% of original information
            Components for 95%: Maybe 20 components
        
        Interpretation:
            If 2 components retain 80% variance:
            → 2D plot captures most of the data structure
            
            If 2 components retain only 20% variance:
            → 2D plot is oversimplified, but still useful for exploration
        
        Raises:
            ValueError: If data is invalid or lengths don't match
        """
        # Validate input data
        DataValidator.validate_features(X)
        DataValidator.validate_target(y)
        
        # Ensure matching lengths
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of rows. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        # Step 1: Standardize features
        # This is CRUCIAL for PCA!
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.logger.info(
            f"Standardized {X.shape[1]} features to mean=0, std=1"
        )
        
        # Step 2: Reduce to 2D for visualization
        pca_2d = PCA(n_components=2)  # Keep only 2 components
        X_2d = pca_2d.fit_transform(X_scaled)
        
        # Calculate variance retained by 2 components
        variance_2d = float(pca_2d.explained_variance_ratio_.sum())
        
        self.logger.info(
            f"2D PCA retains {variance_2d:.1%} of original variance"
        )
        
        # Step 3: Visualize 2D projection
        self._plot_pca_2d(X_2d, y)
        
        # Step 4: Find components needed for 95% variance
        # Fit PCA with all possible components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Calculate cumulative variance explained
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find first index where cumulative variance >= 0.95
        components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
        
        self.logger.info(
            f"Need {components_95} components to retain 95% of variance"
        )
        self.logger.info(
            f"Dimensionality reduction: {X.shape[1]} → {components_95} "
            f"({components_95/X.shape[1]:.1%} of original)"
        )
        
        return pca_2d, X_2d, components_95
    
    def _plot_pca_2d(
        self,
        X_2d: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Visualize 2D PCA projection colored by labels.
        
        This plot helps us:
        1. See if classes are separable in 2D
        2. Understand data structure
        3. Identify patterns or outliers
        4. Evaluate if PCA is useful
        
        Good Signs:
            - Classes are separated (different colors form distinct groups)
            - Clear structure visible
            - High variance explained by 2 components
        
        Bad Signs:
            - Classes completely mixed (colors overlapping everywhere)
            - No clear structure
            - Low variance explained (need more components)
        
        Args:
            X_2d: 2D projected data (n_samples, 2)
            y: Labels for coloring (n_samples,)
        
        Axes:
            - PC1 (x-axis): First principal component (most variance)
            - PC2 (y-axis): Second principal component (second-most variance)
        """
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot colored by class
        scatter = plt.scatter(
            X_2d[:, 0],  # First principal component
            X_2d[:, 1],  # Second principal component
            c=y,
            cmap='tab10',  # Colormap with 10 distinct colors
            s=30,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.3,
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, label="Class Label")
        cbar.set_label("Class Label", fontsize=11)
        
        # Labels and formatting
        plt.xlabel("First Principal Component (PC1)", fontsize=12)
        plt.ylabel("Second Principal Component (PC2)", fontsize=12)
        plt.title(
            "PCA: Dimensionality Reduction to 2D",
            fontsize=14,
            fontweight='bold'
        )
        plt.grid(True, alpha=0.3)
        
        # Add explanation
        explanation = (
            "Each point is a sample, colored by its class.\n"
            "PC1 captures the most variance,\n"
            "PC2 captures the second-most variance."
        )
        plt.text(
            0.02, 0.98,
            explanation,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        self.plots.finish("pca_digits_2d.png")


    # ------------------------------------------------------------------
    # Assignment 2 Part 2: comparing PCA, LDA and Isomap
    # ------------------------------------------------------------------
    
    # Neighbourhood size used by Isomap and by the trustworthiness metric
    N_NEIGHBORS: int = 10
    
    def make_reducer(self, name: str, n_components: int) -> Any:
        """
        Create a dimensionality reduction technique by name.
        
        Args:
            name: "PCA", "LDA" or "Isomap"
            n_components: Number of dimensions to reduce to
        
        Returns:
            Unfitted scikit-learn transformer
        
        Settings:
            - PCA: Default settings (exactly as in run())
            - LDA: Default "svd" solver; n_components <= n_classes - 1
            - Isomap: n_neighbors=10. Too few neighbours breaks the graph
              into pieces; too many "short-circuits" the manifold and
              makes Isomap behave like PCA.
        
        Raises:
            ValueError: If the technique name is unknown
        """
        if name == "PCA":
            return PCA(n_components=n_components)
        if name == "LDA":
            return LinearDiscriminantAnalysis(n_components=n_components)
        if name == "Isomap":
            return Isomap(n_neighbors=self.N_NEIGHBORS, n_components=n_components)
        raise ValueError(
            f"Unknown technique '{name}'. Choose one of: {self.techniques}"
        )
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratified train/test split (same settings as the classification
        workflow of the original project: test_size and random_state from
        the configuration).
        
        Why split for dimensionality reduction?
            LDA is supervised. If we fitted it on ALL the data and then
            measured class separation on that same data, it would be
            graded on the labels it was trained with. Fitting every
            technique on the training set and measuring on the test set
            keeps the comparison fair.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        DataValidator.validate_features(X)
        DataValidator.validate_target(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        
        self.logger.info(
            f"Data split: {len(X_train)} training, {len(X_test)} testing samples"
        )
        
        return X_train, X_test, y_train, y_test
    
    def compare_techniques_2d(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Reduce to 2D with every technique and compare the results.
        
        Process (identical for every technique):
            1. StandardScaler fitted on the training set
            2. Technique fitted on the scaled training set
               (LDA also receives y_train; PCA and Isomap ignore labels)
            3. Test set transformed with the fitted scaler + technique
            4. Metrics computed on the 2D TEST embedding
        
        Metrics (all applicable to every technique):
            - Trustworthiness (0 to 1, higher is better):
                Of each point's 10 nearest neighbours in 2D, how many were
                also close in the original 64-D space? Low values mean the
                2D plot shows "fake" neighbours.
            - Silhouette (-1 to 1, higher is better):
                Using the TRUE digit labels: are images of the same digit
                close together and far from other digits?
            - K-Means ARI (about 0 = random, 1 = perfect):
                Run K-Means (K=10) in the 2D space and compare the clusters
                with the true digits (Adjusted Rand Index).
            - KNN Accuracy (2D):
                Train a 5-nearest-neighbour classifier on the 2D training
                embedding and test it on the 2D test embedding.
            - Fit Time (seconds)
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Labels
        
        Returns:
            results: DataFrame with one row per technique
            embeddings: 2D test embedding for each technique
        """
        # Step 1: Standardize (fit on training data only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = []
        embeddings: dict[str, np.ndarray] = {}
        
        for name in self.techniques:
            reducer = self.make_reducer(name, n_components=2)
            
            # Step 2: Fit on training data (time it)
            start = time.perf_counter()
            train_2d = reducer.fit_transform(X_train_scaled, y_train)
            fit_seconds = time.perf_counter() - start
            
            # Step 3: Transform the unseen test data
            test_2d = reducer.transform(X_test_scaled)
            embeddings[name] = test_2d
            
            # Step 4: Metrics on the test embedding
            trust = trustworthiness(
                X_test_scaled, test_2d, n_neighbors=self.N_NEIGHBORS
            )
            silhouette = silhouette_score(test_2d, y_test)
            
            kmeans_labels = KMeans(
                n_clusters=len(np.unique(y_train)),
                n_init=10,
                random_state=self.config.random_state,
            ).fit_predict(test_2d)
            ari = adjusted_rand_score(y_test, kmeans_labels)
            
            knn = KNeighborsClassifier().fit(train_2d, y_train)
            knn_accuracy = knn.score(test_2d, y_test)
            
            results.append({
                "Technique": name,
                "Supervised": "Yes" if name == "LDA" else "No",
                "Linear": "No" if name == "Isomap" else "Yes",
                "Trustworthiness": trust,
                "Silhouette": silhouette,
                "K-Means ARI": ari,
                "KNN Accuracy (2D)": knn_accuracy,
                "Fit Time (s)": fit_seconds,
            })
            
            self.logger.info(
                f"{name:7s} | 2D trustworthiness: {trust:.3f} | "
                f"silhouette: {silhouette:.3f} | KNN accuracy: {knn_accuracy:.3f}"
            )
        
        self._plot_2d_comparison(embeddings, y_test, pd.DataFrame(results))
        
        return pd.DataFrame(results), embeddings
    
    def _plot_2d_comparison(
        self,
        embeddings: dict[str, np.ndarray],
        y: np.ndarray,
        results: pd.DataFrame,
    ) -> None:
        """
        Plot the 2D test-set embedding of every technique side by side.
        
        Same visual style as the original PCA plot (_plot_pca_2d), so the
        techniques can be compared directly. Each panel is annotated with
        its KNN accuracy and silhouette score.
        
        Args:
            embeddings: 2D test embedding for each technique
            y: Test labels (for coloring)
            results: Metrics table from compare_techniques_2d()
        """
        fig, axes = plt.subplots(
            1, len(embeddings),
            figsize=(6 * len(embeddings), 6),
            constrained_layout=True,  # Keeps titles and colorbar from overlapping
        )
        axes = np.atleast_1d(axes)
        metrics = results.set_index("Technique")
        
        for ax, (name, embedding) in zip(axes, embeddings.items()):
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=y,
                cmap='tab10',  # Colormap with 10 distinct colors
                s=30,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.3,
            )
            
            # Label each digit at the centre of its points
            for digit in np.unique(y):
                centre = np.median(embedding[y == digit], axis=0)
                ax.text(
                    centre[0], centre[1], str(digit),
                    fontsize=14, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8),
                )
            
            ax.set_title(
                f"{name}\nKNN acc = {metrics.loc[name, 'KNN Accuracy (2D)']:.3f}, "
                f"silhouette = {metrics.loc[name, 'Silhouette']:.3f}",
                fontsize=12,
                fontweight='bold',
            )
            ax.set_xlabel("Component 1", fontsize=11)
            ax.set_ylabel("Component 2", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        fig.colorbar(scatter, ax=axes, label="Digit Label", shrink=0.85)
        fig.suptitle(
            "2D Embeddings of the Held-Out Test Set (fitted on training data)",
            fontsize=14,
            fontweight='bold',
        )
        self.plots.finish("dr_comparison_2d.png")
    
    def evaluate_downstream_classification(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        component_grid: tuple[int, ...] = (2, 3, 5, 9),
    ) -> pd.DataFrame:
        """
        Measure classification performance after reduction for several
        numbers of components.
        
        Why this metric?
            A common reason to reduce dimensions is to make a later model
            faster or more robust. The question is: how much useful
            information survives the reduction? We answer it with the same
            simple classifier (KNN, k=5) for every technique.
        
        Pipeline for each (technique, n_components):
            StandardScaler → technique → KNeighborsClassifier
            fitted on training data, scored on the test set.
        
        Reference row:
            KNN on all 64 standardized features (no reduction).
        
        Component grid:
            Capped at 9 because LDA can produce at most n_classes - 1 = 9
            components; using the same grid for every technique keeps the
            comparison fair.
        
        PCA-only columns (reported, not forced on other techniques):
            - PCA Variance Retained: cumulative explained variance ratio
            - PCA Reconstruction MSE: mean squared error between the
              standardized test data and its reconstruction from k
              components (LDA and Isomap have no inverse transform)
            - PCA Relative Recon. Error: that MSE divided by the total
              variance of the standardized test data (0 = perfect,
              1 = nothing reconstructed)
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Labels
            component_grid: Numbers of components to evaluate
        
        Returns:
            DataFrame with KNN test accuracy per technique and n_components
        """
        n_classes = len(np.unique(y_train))
        if max(component_grid) > n_classes - 1:
            raise ValueError(
                f"Largest n_components must be <= {n_classes - 1} "
                f"(LDA limit for {n_classes} classes), got {max(component_grid)}"
            )
        
        # Reference: no dimensionality reduction at all
        baseline = make_pipeline(StandardScaler(), KNeighborsClassifier())
        baseline.fit(X_train, y_train)
        baseline_accuracy = baseline.score(X_test, y_test)
        self.logger.info(f"KNN on all {X_train.shape[1]} features: {baseline_accuracy:.4f}")
        
        rows = []
        for n_components in component_grid:
            row: dict[str, Any] = {"Components": n_components}
            
            for name in self.techniques:
                pipeline = make_pipeline(
                    StandardScaler(),
                    self.make_reducer(name, n_components),
                    KNeighborsClassifier(),
                )
                pipeline.fit(X_train, y_train)
                row[f"{name} KNN Accuracy"] = pipeline.score(X_test, y_test)
            
            # PCA-specific measures (only meaningful for PCA)
            scaler = StandardScaler().fit(X_train)
            pca = PCA(n_components=n_components).fit(scaler.transform(X_train))
            X_test_scaled = scaler.transform(X_test)
            reconstructed = pca.inverse_transform(pca.transform(X_test_scaled))
            row["PCA Variance Retained"] = float(pca.explained_variance_ratio_.sum())
            mse = float(np.mean((X_test_scaled - reconstructed) ** 2))
            row["PCA Reconstruction MSE"] = mse
            # Relative error = share of the test set's total (standardized)
            # variance that is lost. Easier to read than raw MSE, because a
            # few rare pixels have extreme standardized values on test data.
            row["PCA Relative Recon. Error"] = mse / float(np.mean(X_test_scaled ** 2))
            
            rows.append(row)
            self.logger.info(
                f"{n_components} components | "
                + " | ".join(f"{t}: {row[f'{t} KNN Accuracy']:.4f}" for t in self.techniques)
            )
        
        results = pd.DataFrame(rows)
        results.attrs["baseline_accuracy"] = baseline_accuracy
        
        self._plot_downstream(results, baseline_accuracy, X_train.shape[1])
        
        return results
    
    def _plot_downstream(
        self,
        results: pd.DataFrame,
        baseline_accuracy: float,
        n_features: int,
    ) -> None:
        """
        Plot KNN test accuracy versus number of components per technique.
        
        Args:
            results: Table from evaluate_downstream_classification()
            baseline_accuracy: KNN accuracy using all original features
            n_features: Number of original features (for the label)
        """
        plt.figure(figsize=(10, 6))
        
        markers = {"PCA": "o", "LDA": "s", "Isomap": "^"}
        for name in self.techniques:
            plt.plot(
                results["Components"],
                results[f"{name} KNN Accuracy"],
                marker=markers.get(name, "o"),
                markersize=8,
                linewidth=2,
                label=name,
            )
        
        plt.axhline(
            baseline_accuracy,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"No reduction ({n_features} features)",
        )
        
        plt.xticks(results["Components"])
        plt.xlabel("Number of Components", fontsize=12)
        plt.ylabel("KNN Test Accuracy", fontsize=12)
        plt.title(
            "Classification Performance After Dimensionality Reduction",
            fontsize=14,
            fontweight='bold',
        )
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.plots.finish("dr_downstream_knn_accuracy.png")


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging
    from data_loaders.dataset_manager import DatasetManager
    
    # Set up
    configure_logging()
    config = AppConfig(save_plots=True, show_plots=False)
    
    # Load digits data
    print("=== Dimensionality Reduction Workflow Demo ===\n")
    dm = DatasetManager()
    X, y = dm.load_digits_data()
    
    print(f"Original data shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}\n")
    
    # Run original PCA workflow
    print("Running PCA dimensionality reduction...")
    workflow = DimensionalityReductionWorkflow(config)
    pca, X_2d, components_95 = workflow.run(X, y)
    print(f"Variance explained by 2 components: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"Components needed for 95% variance: {components_95}\n")
    
    # Compare PCA, LDA and Isomap
    X_train, X_test, y_train, y_test = workflow.split_data(X, y)
    
    print("Comparing techniques in 2D (test set)...")
    results_2d, _ = workflow.compare_techniques_2d(X_train, X_test, y_train, y_test)
    print(results_2d.round(4).to_string(index=False))
    print()
    
    print("Classification after reduction (KNN test accuracy)...")
    downstream = workflow.evaluate_downstream_classification(
        X_train, X_test, y_train, y_test
    )
    print(downstream.round(4).to_string(index=False))
    
    print("\n✓ Dimensionality reduction workflow completed successfully!")
