# Assignment Report: Dimensionality Reduction Workflow Assignment 2 Part 2

All numbers in this report were produced by running `python examples/run_dimensionality_reduction.py` (scikit-learn 1.5.2, `random_state=42`, 80/20 stratified split). Nothing has been estimated or copied from elsewhere.

## 1. What Was Retained from the Original Workflow

- Dataset: Digits (1797 images, 64 pixel features, 10 classes)
- Preprocessing: `StandardScaler` before every technique
- The original `run()` method, unchanged: PCA to 2D, the `pca_digits_2d.png` plot, and the number of components needed for 95% variance
- Configuration, validation, logging, and plotting modules (unchanged)

The original results are reproduced exactly: 2 principal components retain **21.6%** of the variance (PC1 12.0%, PC2 9.6%), and **40 components** are needed for 95%.

### Additions and Changes

| Change | Reason |
|--------|--------|
| Stratified 80/20 train/test split for the comparison | LDA is supervised. Measuring class separation on data it was fitted on would favour it unfairly. |
| `make_reducer`, `compare_techniques_2d`, `evaluate_downstream_classification` | Apply all three techniques consistently and evaluate them with the same metrics |
| Digits loader now validates its data | The original's other loaders validated; this one did not |

## 2. Techniques

**Already present (1):** PCA.

**Added (2):**

1. **Linear Discriminant Analysis (LDA)**: Supervised and linear. It finds the directions that best separate the 10 digits. Compared with PCA, it shows the effect of using labels while keeping the method linear.
2. **Isomap** (`n_neighbors=10`): Unsupervised and non-linear (manifold learning). Compared with PCA, it shows the effect of non-linearity while keeping the method unsupervised. Isomap was chosen over t-SNE because it can transform unseen data, so it can be evaluated on the test set like the other two techniques.

## 3. Results

### 2D Embeddings, Evaluated on the Test Set (360 images)

| Technique | Supervised | Linear | Trustworthiness | Silhouette | K-Means ARI | KNN Accuracy | Fit Time (s) |
|-----------|-----------|--------|-----------------|------------|-------------|--------------|--------------|
| PCA | No | Yes | 0.8036 | 0.0251 | 0.3000 | 0.5222 | 0.0006 |
| LDA | Yes | Yes | 0.7799 | 0.1950 | **0.5180** | 0.7083 | 0.0034 |
| Isomap | No | No | **0.8755** | **0.2252** | 0.5053 | **0.8194** | 0.2506 |

(Fit times are wall-clock times and vary slightly between runs.)

### Classification After Reduction (KNN, k=5, test accuracy)

| Components | PCA | LDA | Isomap | PCA Variance Retained | PCA Relative Reconstruction Error |
|------------|-----|-----|--------|-----------------------|-----------------------------------|
| 2 | 0.5222 | 0.7083 | **0.8194** | 21.7% | 0.833 |
| 3 | 0.7194 | 0.8500 | **0.8944** | 30.0% | 0.762 |
| 5 | 0.8306 | **0.9500** | 0.9306 | 41.6% | 0.679 |
| 9 | 0.9333 | **0.9750** | 0.9583 | 56.2% | 0.556 |

Reference: KNN on all 64 standardized features = **0.9639**.

(PCA variance retained is 21.7% here instead of 21.6% because this PCA is fitted on the training set only. Relative reconstruction error is the reconstruction MSE divided by the total variance of the standardized test data.)

### Robustness Check

The 360-image test set is small, so I repeated the comparison with five different random splits (seeds 0 to 4):

| | PCA | LDA | Isomap |
|---|-----|-----|--------|
| 2D trustworthiness | 0.811 | 0.768 | **0.874** |
| 2D silhouette | 0.053 | 0.165 | **0.221** |
| 2D K-Means ARI | 0.301 | 0.478 | **0.502** |
| KNN accuracy, 2 components | 0.544 | 0.664 | **0.788** |
| KNN accuracy, 5 components | 0.868 | 0.929 | **0.941** |
| KNN accuracy, 9 components | 0.931 | **0.965** | 0.951 |

PCA is last on every measure except trustworthiness in every split. Isomap leads in 2D. LDA and Isomap are within one standard deviation of each other at 5 and 9 components. The single-split result in which LDA had the best K-Means ARI is within noise; Isomap is slightly higher on average.

## 4. How the Techniques Compare and Why

**For visualization and very low dimensions (2 to 3 components), Isomap is the best technique.** It has the highest trustworthiness (its 2D neighbourhoods match the 64D ones), the best silhouette, and the highest KNN accuracy (0.82 in 2D, compared with 0.71 for LDA and 0.52 for PCA). The digit images lie on a curved structure, and Isomap's geodesic distances follow it, while a flat projection folds different digits on top of each other.

**For compact features feeding a classifier (5 to 9 components), LDA is the best technique.** With 9 components it reaches 0.975 test accuracy, slightly above using all 64 features (0.964). This is a 7-fold reduction with no loss in accuracy, because LDA keeps only the class-discriminating directions and discards within-class noise. It needs labels, though, and at most 9 components are possible.

**PCA is the weakest at preserving class structure in few dimensions, but it is the only technique that does what it is designed for: it is unsupervised, extremely fast (under 1 ms), and invertible.** Its 2D projection keeps only 21.7% of the variance and overlaps most digits (silhouette 0.025). It needs 40 components for 95% of the variance. Even with 9 components, 56% of the variance is kept and the relative reconstruction error is 0.56. PCA remains the right choice when the goal is compression with reconstruction, when labels are unavailable, or when speed matters.

**Costs.** Isomap is about 400 times slower to fit than PCA here (0.25 s compared with under 1 ms). Its cost grows quickly with the number of samples because of the neighbour graph and shortest-path computation. LDA is almost as fast as PCA.

### Summary

| Goal | Best Technique | Evidence |
|------|----------------|----------|
| 2D visualization, structure preservation | Isomap | Highest trustworthiness, silhouette, and 2D KNN accuracy in every split |
| Compact features for classification (labels available) | LDA | 0.975 accuracy with 9 components, matching or beating all 64 features |
| Compression, reconstruction, no labels, speed | PCA | Only invertible technique; fastest |

### Why These Metrics

Trustworthiness, silhouette, K-Means ARI, and KNN accuracy apply to all three techniques, so they allow a direct comparison. Explained variance and reconstruction error describe PCA's own objective and are reported for PCA only. Applying them to LDA or Isomap would require an inverse transform they do not have, or would compare quantities with different meanings.
