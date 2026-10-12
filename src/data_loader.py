"""
data_loader.py — Data Ingestion, Preprocessing, Scaling, and SMOTE Balancing

Handles dataset loading, feature scaling with StandardScaler, train/test splitting,
and class imbalance mitigation using SMOTE (Synthetic Minority Over-sampling Technique).

Author: Fayez Ahmad Protik
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_feature_matrix(
    csv_path: str = "data/processed/parkinsons_keystroke_features.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the engineered feature dataset and separate features and target.

    Args:
        csv_path: Path to the processed CSV file.

    Returns:
        X: Feature DataFrame with identifier columns removed.
        y: Binary target series (0 = Healthy, 1 = Parkinson's Disease).
    """
    df = pd.read_csv(csv_path)

    # Drop non-feature and identifier columns
    drop_cols = [c for c in ["UserID", "Class", "Gender"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Class"]

    # Fill any remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def prepare_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Perform stratified split, feature scaling, and optional SMOTE oversampling.

    To prevent data leakage, the StandardScaler is fit ONLY on the training split,
    and SMOTE is applied ONLY to the training data. The test set remains strictly
    isolated and unaltered.

    Args:
        X: Feature DataFrame.
        y: Target series.
        test_size: Proportion of dataset to include in test split (default: 0.2).
        random_state: Random seed for reproducibility.
        apply_smote: Whether to apply SMOTE to the training set.

    Returns:
        X_train: Scaled (and SMOTE-balanced) training features.
        X_test: Scaled test features.
        y_train: Training labels (SMOTE-balanced if applied).
        y_test: Test labels (unaltered, preserving natural class distribution).
        scaler: Fitted StandardScaler instance.
    """
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train_raw)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train_raw

    return X_train_final, X_test_scaled, y_train_final, y_test, scaler
