"""
Dataset Manager Module
======================

Purpose:
    Loads and summarizes example datasets from scikit-learn.
    Provides a consistent interface for accessing different types of data.

Key Concepts:
    - Scikit-learn datasets: Built-in datasets for learning and testing
    - NumPy arrays: Efficient numeric containers for image data
    - Data validation: Always check data before returning it

Available Datasets:
    1. Digits (Dimensionality Reduction): 8×8 images of handwritten digits

Why Use Built-in Datasets?
    - No download needed
    - Well-documented
    - Standard benchmarks
    - Perfect for learning

    Later, you'll use this same pattern for your own datasets!
"""

import logging

import numpy as np
from sklearn.datasets import load_digits

from utils.data_validator import DataValidator


class DatasetManager:
    """
    Loads and summarizes example datasets for machine learning.

    This class provides a unified interface to load different types of datasets.
    Each method returns validated data ready for machine learning.

    Methods:
        - load_digits_data(): Image data for dimensionality reduction
    """

    def __init__(self) -> None:
        """Initialize the dataset manager with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_digits_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the Digits dataset (handwritten digits 0-9).

        Dataset Details:
            - Samples: 1797 images
            - Features: 64 (8×8 pixel values, flattened)
            - Target: 10 classes (digits 0-9), roughly 180 images each
            - Source: Scikit-learn toy dataset

        Data Format:
            - Each sample is an 8×8 grayscale image
            - Flattened to 64 features (8 * 8 = 64)
            - Pixel values range from 0 (white) to 16 (black)
            - Some border pixels are always 0 (zero variance)

        Returns:
            X (ndarray): Features (1797 rows × 64 columns)
            y (ndarray): Digit labels (1797 values: 0-9)

        Example:
            >>> dm = DatasetManager()
            >>> X, y = dm.load_digits_data()
            >>> print(f"Shape: {X.shape}")
            Shape: (1797, 64)
            >>> print(f"First digit is: {y[0]}")
            First digit is: 0

        Use Cases:
            - Dimensionality reduction (64 features is manageable for PCA)
            - The labels let us MEASURE how well each reduced space keeps
              the digits apart, even for unsupervised techniques
        """
        dataset = load_digits()
        X, y = dataset.data, dataset.target

        # Validate before returning (the original project skipped this step
        # for the digits data; every other loader validated its data)
        DataValidator.validate_features(X)
        DataValidator.validate_target(y)

        self.logger.info(
            "Digits dataset loaded: %s observations, %s features",
            X.shape[0],
            X.shape[1],
        )

        # Return as numpy arrays (more efficient for image data)
        return X, y


# Example usage (for testing this module independently)
if __name__ == "__main__":
    from utils.logger import configure_logging

    # Set up logging
    configure_logging()

    print("=== Dataset Manager Examples ===\n")

    # Create a dataset manager
    dm = DatasetManager()

    # Load digits data
    print("1. Loading Digits Dataset:")
    X_digits, y_digits = dm.load_digits_data()
    print(f"   Features shape: {X_digits.shape}")
    print(f"   Pixel value range: {X_digits.min():.0f} to {X_digits.max():.0f}")
    print(f"   Classes: {np.unique(y_digits)}")
    print(f"   Zero-variance pixels: {int((X_digits.std(axis=0) == 0).sum())}\n")

    print("Dataset loaded successfully!")
