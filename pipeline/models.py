"""
Phase 4 — Model Training
Trains Logistic Regression (baseline), Random Forest, and XGBoost.
Supports hyperparameter configuration and model serialisation.
"""

from __future__ import annotations

import time
import numpy as np
import joblib
from typing import Dict, Any, Tuple
from pathlib import Path

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier


# ── Model Catalogue ───────────────────────────────────────────────────────────

def get_model_configs() -> Dict[str, Dict]:
    """Return default hyperparameter configurations for all three models."""
    return {
        'Logistic Regression': {
            'description': 'Fast linear baseline. Good for benchmarking.',
            'params': {
                'C':            1.0,
                'max_iter':     1000,
                'solver':       'lbfgs',
                'random_state': 42,
                # No class_weight — SMOTE already balanced training data
            },
        },
        'Random Forest': {
            'description': 'Ensemble of decision trees. Handles non-linearity well.',
            'params': {
                'n_estimators':     200,
                'max_depth':        8,
                'min_samples_leaf': 5,
                'n_jobs':           -1,
                'random_state':     42,
                # No class_weight — SMOTE already balanced training data
            },
        },
        'XGBoost': {
            'description': 'Gradient boosting champion for tabular financial data.',
            'params': {
                'n_estimators':     200,
                'learning_rate':    0.05,
                'max_depth':        5,
                'subsample':        0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1.5,  # mild boost — SMOTE synthetics weaker than real
                'eval_metric':      'logloss',
                'random_state':     42,
                'n_jobs':           -1,
            },
        },
    }


def build_model(name: str, params: Dict[str, Any]):
    """Instantiate a model by name with given params."""
    if name == 'Logistic Regression':
        return LogisticRegression(**params)
    elif name == 'Random Forest':
        return RandomForestClassifier(**params)
    elif name == 'XGBoost':
        # Remove non-XGBoost keys
        return XGBClassifier(**params, verbosity=0)
    else:
        raise ValueError(f'Unknown model: {name}')


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(name: str,
                X_train: np.ndarray,
                y_train: np.ndarray,
                params: Dict[str, Any]) -> Tuple[Any, float]:
    """
    Train a single model. Returns (fitted_model, training_time_seconds).
    """
    model = build_model(name, params)
    t0    = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)
    return model, elapsed


def train_all_models(X_train: np.ndarray,
                     y_train: np.ndarray,
                     configs:  Dict[str, Dict],
                     progress_callback=None) -> Dict[str, Any]:
    """
    Train all three models sequentially.
    progress_callback(name, step, total) is called between models for Streamlit progress.
    Returns dict of {model_name: fitted_model}.
    """
    trained = {}
    total   = len(configs)

    for i, (name, cfg) in enumerate(configs.items()):
        if progress_callback:
            progress_callback(name, i, total)

        model, elapsed = train_model(name, X_train, y_train, cfg['params'])
        trained[name] = {'model': model, 'training_time': elapsed}

    return trained


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


# ── Prediction Helpers ────────────────────────────────────────────────────────

def predict(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (binary_predictions, default_probabilities)."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.50).astype(int)   # standard threshold — SMOTE handles imbalance
    return preds, probs


def predict_single(model, X_single: np.ndarray) -> Dict[str, Any]:
    """
    Predict for a single applicant.
    Returns a dict with prediction, probability, and risk tier.
    """
    prob = model.predict_proba(X_single)[0, 1]
    pred = int(prob >= 0.50)

    if prob < 0.30:
        tier, color = 'Low Risk',    '#10B981'
    elif prob < 0.55:
        tier, color = 'Medium Risk', '#F59E0B'
    elif prob < 0.75:
        tier, color = 'High Risk',   '#F97316'
    else:
        tier, color = 'Very High Risk', '#EF4444'

    return {
        'prediction':        pred,
        'default_prob':      round(float(prob) * 100, 1),
        'paid_prob':         round((1 - float(prob)) * 100, 1),
        'risk_tier':         tier,
        'risk_color':        color,
        'label':             'DEFAULT' if pred == 1 else 'FULLY PAID',
    }
