"""
Phase 3 — Class Imbalance Resolution
Applies SMOTE to the training set to balance the minority class (defaulters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple

from sklearn.model_selection import train_test_split


TARGET_COL = 'loan_status'


# ── Train / Test Split ────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.20,
               random_state: int = 42):
    """
    Stratified 80/20 split.
    IMPORTANT: split is performed BEFORE SMOTE so the test set is real-world.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to training data only.
    Returns (X_resampled, y_resampled).
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(sampling_strategy='auto', k_neighbors=5,
                  random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def get_balance_stats(y_original: np.ndarray,
                      y_resampled: np.ndarray) -> dict:
    """Return before/after balance statistics."""
    orig_counts = np.bincount(y_original.astype(int))
    res_counts  = np.bincount(y_resampled.astype(int))

    return {
        'before': {
            'non_default': int(orig_counts[0]),
            'default':     int(orig_counts[1]),
            'ratio':       round(orig_counts[0] / orig_counts[1], 2),
        },
        'after': {
            'non_default': int(res_counts[0]),
            'default':     int(res_counts[1]),
            'ratio':       round(res_counts[0] / res_counts[1], 2),
        },
        'samples_added': int(res_counts[1] - orig_counts[1]),
    }


# ── Plotly Visualisation ──────────────────────────────────────────────────────

def plot_smote_comparison(stats: dict) -> go.Figure:
    """Side-by-side bar chart showing class balance before and after SMOTE."""
    categories  = ['No Default (0)', 'Default (1)']
    before_vals = [stats['before']['non_default'], stats['before']['default']]
    after_vals  = [stats['after']['non_default'],  stats['after']['default']]

    fig = go.Figure(data=[
        go.Bar(
            name='Before SMOTE',
            x=categories,
            y=before_vals,
            marker_color=['#10B981', '#EF4444'],
            opacity=0.55,
            text=[f'{v:,}' for v in before_vals],
            textposition='outside',
        ),
        go.Bar(
            name='After SMOTE',
            x=categories,
            y=after_vals,
            marker_color=['#10B981', '#EF4444'],
            opacity=1.0,
            text=[f'{v:,}' for v in after_vals],
            textposition='outside',
        ),
    ])

    fig.update_layout(
        title='Class Balance: Before vs After SMOTE',
        barmode='group',
        yaxis_title='Number of Samples',
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig
