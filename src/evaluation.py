"""
evaluation.py — Comprehensive Model Evaluation and Benchmark Metrics

Computes Accuracy, Macro-F1, Precision, Recall, Confusion Matrix, and ROC AUC.
Generates publication-quality visualizations for model comparison.

Author: Fayez Ahmad Protik
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained model on the held-out test split.

    Args:
        model: Trained scikit-learn compatible classifier.
        X_test: Scaled test feature matrix.
        y_test: True binary test labels.
        model_name: Name of the model for logging.
        verbose: Whether to print results to stdout.

    Returns:
        Dictionary of computed performance metrics.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # ROC AUC if probabilities available
    roc_auc = 0.5
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

    if verbose:
        print(f"=== {model_name} Evaluation ===")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Macro-F1:  {f1_macro * 100:.2f}%")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  ROC AUC:   {roc_auc:.4f}\n")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Macro_F1": f1_macro,
        "Precision": precision,
        "Recall": recall,
        "ROC_AUC": roc_auc,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ["Healthy", "Parkinson's"],
    title: str = "Ensemble Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(title, weight="bold", fontsize=14, pad=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix: {save_path}")

    plt.show()


def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Soft Voting Ensemble",
    save_path: Optional[str] = None,
) -> None:
    """Plot the Receiver Operating Characteristic (ROC) curve."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba for ROC analysis.")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980B9", lw=2.5, label=f"{model_name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#7F8C8D", lw=1.5, linestyle="--", label="Chance Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve: {model_name}", weight="bold", fontsize=14, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curve: {save_path}")

    plt.show()
