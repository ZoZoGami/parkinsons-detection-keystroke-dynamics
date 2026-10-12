"""
explainability.py — Explainable AI (XAI) using SHAP (SHapley Additive exPlanations)

Computes Shapley values using TreeExplainer on tree-based models to explain
individual predictions and global feature importance. Validates that the model's
diagnostic logic aligns with physiological biomarkers (e.g. Circadian Fluctuation).

Author: Fayez Ahmad Protik
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def compute_shap_explanations(
    tree_model,
    X_test_scaled: np.ndarray,
    feature_names: List[str],
) -> Tuple:
    """Initialize SHAP TreeExplainer and calculate Shapley values for the test split.

    Args:
        tree_model: Trained XGBoost or Random Forest classifier.
        X_test_scaled: Scaled feature matrix of the test set.
        feature_names: List of human-readable feature column names.

    Returns:
        explainer: Fitted SHAP TreeExplainer instance.
        shap_values: Array of Shapley values for each sample and feature.
        X_df: Test set DataFrame with labeled columns.
    """
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test_scaled)
    X_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    return explainer, shap_values, X_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    title: str = "SHAP Summary Plot: Feature Impact on Parkinson's Diagnosis",
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> None:
    """Generate a SHAP beeswarm summary plot showing feature directionality and impact."""
    plt.figure(figsize=(10, 8))
    plt.title(title, weight="bold", fontsize=14, pad=14)
    shap.summary_plot(shap_values, X_df, plot_type="dot", max_display=max_display, show=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved SHAP summary: {save_path}")

    plt.show()


def plot_shap_bar_importance(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    title: str = "Global Feature Importance (Mean |SHAP Value|)",
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> None:
    """Generate a SHAP bar plot displaying absolute mean feature contributions."""
    plt.figure(figsize=(10, 6))
    plt.title(title, weight="bold", fontsize=14, pad=14)
    shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved SHAP bar plot: {save_path}")

    plt.show()
