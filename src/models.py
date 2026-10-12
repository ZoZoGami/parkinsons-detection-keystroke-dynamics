"""
models.py — Machine Learning Models and Soft Voting Ensemble

Defines the individual classifiers and the final Soft Voting Ensemble
designed to detect Parkinson's Disease from noisy unconstrained keystroke data:

    - Baseline: Logistic Regression
    - Random Forest (Hyperparameter-tuned via GridSearchCV)
    - XGBoost (Gradient Boosting optimized for tabular telemetry)
    - Support Vector Machine (RBF Kernel)
    - Soft Voting Ensemble: Probabilistic consensus model

Author: Fayez Ahmad Protik
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


def build_baseline_model(random_state: int = 42) -> LogisticRegression:
    """Build a baseline Logistic Regression classifier."""
    return LogisticRegression(random_state=random_state, max_iter=1000)


def build_tuned_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Optimize Random Forest hyperparameters via GridSearchCV."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def build_xgboost_model(random_state: int = 42) -> xgb.XGBClassifier:
    """Build a tuned XGBoost classifier for tabular keystroke data."""
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )


def build_svm_model(random_state: int = 42) -> SVC:
    """Build an RBF-kernel SVM classifier with probability estimation enabled."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    )


def build_soft_voting_ensemble(
    rf_model: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    svm_model: SVC,
) -> VotingClassifier:
    """Construct a Soft Voting Ensemble combining the three diverse architectures.

    Soft voting calculates the weighted average of predicted class probabilities
    from each model, allowing confident predictions to outweigh uncertain ones
    and outputting continuous clinical risk scores.

    Args:
        rf_model: Trained or initialized Random Forest model.
        xgb_model: Trained or initialized XGBoost model.
        svm_model: Trained or initialized SVM model (with probability=True).

    Returns:
        VotingClassifier instance with voting='soft'.
    """
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("xgb", xgb_model),
            ("svm", svm_model),
        ],
        voting="soft",
    )
    return ensemble
