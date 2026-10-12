"""
parkinsons-detection-keystroke-dynamics

A machine learning framework for detecting early-stage Parkinson's Disease
from passive, unconstrained keystroke telemetry using Circadian feature engineering,
SMOTE class rebalancing, a Soft Voting Ensemble, and Explainable AI (SHAP).

Author: Fayez Ahmad Protik
"""

from .feature_extraction import extract_user_keystroke_features, compute_statistical_moments
from .data_loader import load_feature_matrix, prepare_train_test_data
from .models import (
    build_baseline_model,
    build_tuned_random_forest,
    build_xgboost_model,
    build_svm_model,
    build_soft_voting_ensemble,
)
from .evaluation import evaluate_classifier, plot_confusion_matrix, plot_roc_curve
from .explainability import compute_shap_explanations, plot_shap_summary, plot_shap_bar_importance

__version__ = "1.0.0"
__author__ = "Fayez Ahmad Protik"

__all__ = [
    "extract_user_keystroke_features",
    "compute_statistical_moments",
    "load_feature_matrix",
    "prepare_train_test_data",
    "build_baseline_model",
    "build_tuned_random_forest",
    "build_xgboost_model",
    "build_svm_model",
    "build_soft_voting_ensemble",
    "evaluate_classifier",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "compute_shap_explanations",
    "plot_shap_summary",
    "plot_shap_bar_importance",
]
