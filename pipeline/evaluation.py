"""
Phase 5 — Model Evaluation
Computes Recall, ROC-AUC, Precision, F1, and generates interactive visualisations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any

from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for a single model."""
    return {
        'Recall':    round(recall_score(y_true, y_pred),           4),
        'Precision': round(precision_score(y_true, y_pred,
                                            zero_division=0),       4),
        'F1 Score':  round(f1_score(y_true, y_pred),               4),
        'Accuracy':  round(accuracy_score(y_true, y_pred),         4),
        'ROC-AUC':   round(roc_auc_score(y_true, y_prob),          4),
    }


def compute_cost_benefit(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         avg_loan: float = 15000,
                         fn_cost_pct: float = 0.60,
                         fp_cost_pct: float = 0.05) -> Dict[str, Any]:
    """
    Financial cost-benefit analysis.
    FN (missed default)  = avg_credit × fn_cost_pct  (60% loss on defaulted credit)
    FP (rejected good)   = avg_credit × fp_cost_pct  (5% missed interest revenue)
    """
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fn_cost = fn * avg_loan * fn_cost_pct
    fp_cost = fp * avg_loan * fp_cost_pct
    savings = fn_cost  # money saved vs. no model baseline

    return {
        'TN': int(tn), 'FP': int(fp),
        'FN': int(fn), 'TP': int(tp),
        'fn_cost':  round(fn_cost,  2),
        'fp_cost':  round(fp_cost,  2),
        'net_savings': round(savings - fp_cost, 2),
        'avg_loan': avg_loan,
    }


def evaluate_all(models_dict: Dict[str, Dict],
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> pd.DataFrame:
    """
    Run evaluation on all trained models.
    Returns a DataFrame with one row per model.
    """
    from pipeline.models import predict

    rows = []
    for name, obj in models_dict.items():
        model   = obj['model']
        y_pred, y_prob = predict(model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['Model']         = name
        metrics['Training Time'] = f"{obj['training_time']}s"
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index('Model')
    df = df[['Recall', 'Precision', 'F1 Score', 'Accuracy', 'ROC-AUC', 'Training Time']]
    return df


# ── Plotly Charts ──────────────────────────────────────────────────────────────

PALETTE = ['#6366F1', '#10B981', '#F59E0B']


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all models across all metrics."""
    metric_cols = ['Recall', 'Precision', 'F1 Score', 'Accuracy', 'ROC-AUC']
    fig = go.Figure()

    for i, model_name in enumerate(metrics_df.index):
        vals = [metrics_df.loc[model_name, m] for m in metric_cols]
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_cols,
            y=vals,
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis=dict(title='Score', range=[0, 1.1]),
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig


def plot_roc_curves(models_dict: Dict[str, Dict],
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> go.Figure:
    """Multi-model ROC AUC curves on one plot."""
    from pipeline.models import predict

    fig = go.Figure()

    for i, (name, obj) in enumerate(models_dict.items()):
        _, y_prob  = predict(obj['model'], X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC={auc:.3f})',
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier',
        line=dict(color='#94A3B8', dash='dash', width=1.5),
    ))

    fig.update_layout(
        title='ROC-AUC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = '') -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm     = confusion_matrix(y_true, y_pred)
    labels = ['No Default (0)', 'Default (1)']

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, '#F0FDF4'], [0.5, '#6366F1'], [1, '#1E1B4B']],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont_size=18,
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title=f'Confusion Matrix — {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
    )
    return fig


def plot_cost_benefit(cost_dict: Dict[str, Any], model_name: str = '') -> go.Figure:
    """Bar chart showing financial impact of model decisions."""
    categories = ['FN Cost (Missed Defaults)', 'FP Cost (Good Loans Rejected)', 'Net Savings']
    values = [
        cost_dict['fn_cost'],
        cost_dict['fp_cost'],
        cost_dict['net_savings'],
    ]
    colors = ['#EF4444', '#F59E0B', '#10B981']

    bar_colors = [
        '#EF4444',
        '#F59E0B',
        '#10B981' if values[2] >= 0 else '#EF4444'
    ]
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=bar_colors,
        text=[f'${v:,.0f}' for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f'Financial Impact Analysis — {model_name}',
        yaxis_title='USD ($)',
        height=370,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
    )
    return fig


# ── NEW FEATURE: Precision-Recall Curves ──────────────────────────────────────

def plot_pr_curves(models_dict: Dict[str, Dict],
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> go.Figure:
    """
    Precision-Recall curves for all models.
    PR curves are more informative than ROC for imbalanced datasets because they
    focus on the minority class (defaulters) rather than the full confusion matrix.
    """
    from pipeline.models import predict

    fig = go.Figure()

    for i, (name, obj) in enumerate(models_dict.items()):
        _, y_prob = predict(obj['model'], X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{name} (AP={ap:.3f})',
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))

    # No-skill baseline = proportion of positives
    no_skill = float(np.mean(y_test))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[no_skill, no_skill], mode='lines',
        name=f'No-Skill Baseline ({no_skill:.2f})',
        line=dict(color='#94A3B8', dash='dash', width=1.5),
    ))

    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall  (of all true defaulters, how many did we catch?)',
        yaxis_title='Precision  (of predicted defaults, how many were real?)',
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig


# ── NEW FEATURE: Decision Threshold Tuner ─────────────────────────────────────

def compute_threshold_metrics(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               thresholds: np.ndarray) -> pd.DataFrame:
    """
    Compute Recall, Precision, F1, and Accuracy at every threshold.
    Used to power the interactive threshold tuner slider.
    """
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        # Avoid division by zero for extreme thresholds
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        rows.append({
            'threshold': round(float(t), 3),
            'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
            'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        })
    return pd.DataFrame(rows)


def plot_threshold_tuner(threshold_df: pd.DataFrame,
                          selected_threshold: float) -> go.Figure:
    """
    Line chart showing how Recall, Precision, and F1 trade off as decision
    threshold is varied from 0.1 to 0.9. Marks the user-selected threshold
    with a vertical line.
    """
    colors = {'Recall': '#EF4444', 'Precision': '#10B981', 'F1': '#6366F1', 'Accuracy': '#F59E0B'}

    fig = go.Figure()
    for metric, color in colors.items():
        if metric in threshold_df.columns:
            fig.add_trace(go.Scatter(
                x=threshold_df['threshold'],
                y=threshold_df[metric],
                mode='lines',
                name=metric,
                line=dict(color=color, width=2.5),
            ))

    fig.add_vline(
        x=selected_threshold,
        line_width=2, line_dash='dot', line_color='#FFFFFF',
        annotation_text=f'  Threshold = {selected_threshold:.2f}',
        annotation_position='top right',
        annotation_font_color='#a5b4fc',
    )

    fig.update_layout(
        title='Decision Threshold vs. Performance Metrics',
        xaxis_title='Decision Threshold',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.05]),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig


# ── NEW FEATURE: Calibration Curve ────────────────────────────────────────────

def plot_calibration_curves(models_dict: Dict[str, Dict],
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             n_bins: int = 10) -> go.Figure:
    """
    Reliability / Calibration curve.
    Plots mean predicted probability vs actual fraction of positives per bin.
    A perfectly calibrated model follows the diagonal — if the model says
    '70% default risk', roughly 70% of those applicants actually defaulted.
    """
    from pipeline.models import predict

    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='#94A3B8', dash='dash', width=1.5),
    ))

    for i, (name, obj) in enumerate(models_dict.items()):
        _, y_prob = predict(obj['model'], X_test)

        # Bin predictions and compute actual positive rate per bin
        bin_edges   = np.linspace(0, 1, n_bins + 1)
        mean_pred   = []
        frac_pos    = []

        for j in range(n_bins):
            mask = (y_prob >= bin_edges[j]) & (y_prob < bin_edges[j + 1])
            if mask.sum() == 0:
                continue
            mean_pred.append(float(y_prob[mask].mean()))
            frac_pos.append(float(y_test[mask].mean()))

        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos,
            mode='lines+markers',
            name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title='Calibration Curves — Are Probabilities Trustworthy?',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Actual Fraction of Defaults',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        font=dict(family='Inter'),
    )
    return fig
