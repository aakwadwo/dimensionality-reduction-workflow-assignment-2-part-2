"""
Exercise 1: Dimensionality Reduction Challenge
===============================================

Difficulty: Intermediate
Time: 30-45 minutes

SCENARIO:
You work for a postal service that scans handwritten digits. Each scan
is an 8×8 image (64 pixel features). Storage and processing costs grow
with the number of features, so your task is to compress the images to
a few features while keeping as much useful information as possible.

OBJECTIVES:
1. Load and explore the digits dataset
2. Split data properly with stratification
3. Reduce the data with PCA, LDA and Isomap
4. Evaluate each technique with metrics that make sense for it
5. Decide how many components you would keep, and with which technique
6. Answer the questions below

LEARNING GOALS:
- Practice a complete dimensionality reduction workflow
- Understand why scaling and fitting on training data matter
- Compare supervised vs unsupervised, and linear vs non-linear techniques
- Interpret explained variance, reconstruction error and trustworthiness
- Use classification accuracy to judge a reduced representation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import AppConfig
from data_loaders import DatasetManager
from utils import configure_logging


def exercise_1():
    """
    TODO: Complete this function to solve the dimensionality reduction challenge.

    Steps to complete:
    1. Load the digits dataset
    2. Split into train/test (80/20 split, stratified, random_state=42)
    3. Standardize: fit StandardScaler on the TRAINING set only
    4. For each technique (PCA, LDA, Isomap) and n_components in [2, 5, 9]:
       - Fit on the scaled training data (LDA also needs y_train)
       - Transform the test data
       - Train KNeighborsClassifier on the reduced training data
       - Record test accuracy
    5. For PCA only: record explained variance and reconstruction error
    6. Plot the 2D test embedding of each technique side by side

    Hints:
    - Use DatasetManager to load data
    - LDA allows at most n_classes - 1 = 9 components
    - Isomap(n_neighbors=10, n_components=k) can transform new data
    - pca.inverse_transform(pca.transform(X)) reconstructs X
    - trustworthiness(X_high_dim, X_low_dim, n_neighbors=10) measures
      how well local neighbourhoods are preserved
    """

    # Step 1: Setup
    configure_logging()
    dm = DatasetManager()

    # TODO: Load the digits dataset
    # X, y = ...

    print("="*70)
    print("EXERCISE 1: DIMENSIONALITY REDUCTION CHALLENGE")
    print("="*70)
    print()

    # TODO: Print dataset information
    # print(f"Samples: ...")
    # print(f"Features: ...")
    # print(f"Classes: ...")

    # Step 2: Split the data
    # TODO: Split with test_size=0.2, random_state=42, stratify=y
    # X_train, X_test, y_train, y_test = ...

    # Step 3: Standardize
    # TODO: Fit the scaler on X_train, then transform X_train and X_test
    # scaler = StandardScaler()
    # X_train_scaled = ...
    # X_test_scaled = ...

    # Step 4: Compare techniques
    # TODO: Create a function that returns a fresh reducer
    # def make_reducer(name, k):
    #     if name == "PCA": return PCA(n_components=k)
    #     ...

    # TODO: For each technique and k in [2, 5, 9]:
    #   1. Fit on X_train_scaled (LDA: fit(X_train_scaled, y_train))
    #   2. Transform X_train_scaled and X_test_scaled
    #   3. Fit KNN on the reduced training data, score on reduced test data
    #   4. Print results

    # Step 5: PCA-specific evaluation
    # TODO: For each k, print:
    #   - pca.explained_variance_ratio_.sum()
    #   - reconstruction MSE on the test set

    # Step 6: Visualize
    # TODO: Plot the 2D test embedding of each technique side by side,
    #       colored by digit label (cmap="tab10")

    print("\n" + "="*70)
    print("QUESTIONS TO ANSWER:")
    print("="*70)
    print("1. Which technique separates the digits best in 2D? Why?")
    print("2. Why can't you report reconstruction error for LDA or Isomap?")
    print("3. LDA uses the labels. Why must it be fitted on training data only?")
    print("4. How many components would you keep, and with which technique?")
    print("5. When would you still prefer PCA over LDA and Isomap?")
    print("="*70)


# QUESTIONS (write your answers here as comments):

# Q1: Which technique separates the digits best in 2D? Why?
# Answer:

# Q2: Why can't you report reconstruction error for LDA or Isomap?
# Answer:

# Q3: Why must LDA be fitted on training data only?
# Answer:

# Q4: How many components would you keep, and with which technique?
# Answer:

# Q5: When would you still prefer PCA over LDA and Isomap?
# Answer:


if __name__ == "__main__":
    exercise_1()
