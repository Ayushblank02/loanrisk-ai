"""
Phase 6 — Explainable AI (XAI) with SHAP
Generates SHAP summary plots, waterfall plots, and plain-English explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, List


# ── SHAP Computation ──────────────────────────────────────────────────────────

def compute_shap_values(model, X_transformed: np.ndarray,
                        feature_names: List[str]) -> Any:
    """
    Compute SHAP values for the given model.
    Automatically selects TreeExplainer (XGB/RF) or LinearExplainer (LR).
    """
    import shap

    model_type = type(model).__name__
    if 'XGB' in model_type or 'Forest' in model_type:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer   = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    # Normalise to always be a plain 2D numpy array (samples x features)
    sv = np.array(shap_values)
    if sv.ndim == 3:
        # shape: (samples, features, classes) or (classes, samples, features)
        # Try to detect which axis is "classes" (size 2 for binary)
        if sv.shape[2] == 2:          # (samples, features, 2)
            sv = sv[:, :, 1]
        elif sv.shape[0] == 2:        # (2, samples, features)
            sv = sv[1]
        else:
            sv = sv[:, :, 0]
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        sv = np.array(shap_values[1])

    return sv, explainer


def get_global_feature_importance(shap_values: np.ndarray,
                                   feature_names: List[str],
                                   top_n: int = 15) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature → global importance.
    Merges one-hot encoded siblings (e.g. sex_Male + sex_Female → Gender).
    Returns sorted DataFrame.
    """
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[:, 1, :]
    elif sv.ndim == 1 and isinstance(shap_values, list):
        sv = np.array(shap_values[1])

    mean_abs = np.abs(sv).mean(axis=0).flatten()

    n = min(len(feature_names), len(mean_abs))
    df = pd.DataFrame({
        'feature':    list(feature_names)[:n],
        'importance': mean_abs[:n],
    })

    df_clean = df.sort_values('importance', ascending=False).head(top_n)
    return df_clean


# ── Global Summary Plot ───────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray,
                       X_transformed: np.ndarray,
                       feature_names: List[str],
                       top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart showing mean |SHAP| importance per feature.
    Mimics SHAP's summary_plot but in Plotly for Streamlit.
    """
    importance_df = get_global_feature_importance(shap_values, feature_names, top_n)
    importance_df = importance_df.sort_values('importance')  # ascending for horizontal bar

    colors = [
        f'rgba(99, 102, 241, {0.5 + 0.5 * (i / len(importance_df))})'
        for i in range(len(importance_df))
    ]

    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=[_clean_feat_name(f) for f in importance_df['feature']],
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' for v in importance_df['importance']],
        textposition='outside',
    ))
    fig.update_layout(
        title='Global Feature Importance (Mean |SHAP| Value)',
        xaxis_title='Mean |SHAP Value|',
        height=max(380, top_n * 28),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=200, r=80, t=60, b=40),
        font=dict(family='Inter'),
    )
    return fig


# ── Waterfall / Force Plot for Single Prediction ─────────────────────────────

def plot_shap_waterfall(shap_values_single: np.ndarray,
                         feature_names: List[str],
                         feature_values: np.ndarray,
                         base_value: float,
                         top_n: int = 12) -> go.Figure:
    """
    Waterfall chart showing how each feature contributed to a single prediction.
    Positive SHAP → pushes toward Default; Negative → pushes away from Default.
    """
    # Select top_n features by absolute contribution
    abs_shap = np.abs(shap_values_single)
    top_idx  = np.argsort(abs_shap)[::-1][:top_n][::-1]  # reversed for waterfall order

    shap_top  = shap_values_single[top_idx]
    names_top = [_clean_feat_name(feature_names[i]) for i in top_idx]
    vals_top  = [feature_values[i] for i in top_idx]

    colors = ['#EF4444' if v > 0 else '#10B981' for v in shap_top]
    labels = [f'{v:+.4f}' for v in shap_top]

    hover = [
        f'Feature: {n}<br>Value: {fv:.2f}<br>SHAP: {s:+.4f}'
        for n, fv, s in zip(names_top, vals_top, shap_top)
    ]

    fig = go.Figure(go.Bar(
        x=shap_top,
        y=names_top,
        orientation='h',
        marker_color=colors,
        text=labels,
        textposition='outside',
        hovertext=hover,
        hoverinfo='text',
    ))

    fig.add_vline(x=0, line_width=1.5, line_color='#64748B')

    fig.update_layout(
        title='SHAP Waterfall — Why This Prediction?',
        xaxis_title='SHAP Value  (← Lowers Default Risk   |   Raises Default Risk →)',
        height=max(380, top_n * 32),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=220, r=100, t=60, b=40),
        font=dict(family='Inter'),
    )
    return fig


# ── Plain-English Explanation ─────────────────────────────────────────────────

def generate_explanation(shap_values_single: np.ndarray,
                          feature_names: List[str],
                          feature_values: np.ndarray,
                          prediction: Dict[str, Any],
                          top_n: int = 3) -> str:
    """
    Convert top SHAP contributors into a human-readable explanation
    as a bank might communicate to a customer.
    """
    abs_shap    = np.abs(shap_values_single)
    top_idx     = np.argsort(abs_shap)[::-1][:top_n]

    risk_factors    = []
    positive_factors = []

    for idx in top_idx:
        name  = _clean_feat_name(feature_names[idx])
        value = feature_values[idx]
        sv    = shap_values_single[idx]

        if sv > 0.01:
            risk_factors.append(f'**{name}** ({_format_val(name, value)}) — increases default risk')
        elif sv < -0.01:
            positive_factors.append(f'**{name}** ({_format_val(name, value)}) — reduces default risk')

    pred_label = prediction['label']
    prob       = prediction['default_prob']
    tier       = prediction['risk_tier']

    lines = [
        f"### 🏦 Credit Decision Summary",
        f"",
        f"**Outcome:** {pred_label}  |  **Risk Score:** {prob}%  |  **Risk Tier:** {tier}",
        f"",
    ]

    if risk_factors:
        lines.append("**⚠️ Key Risk Factors:**")
        for f in risk_factors:
            lines.append(f"  - {f}")
        lines.append("")

    if positive_factors:
        lines.append("**✅ Positive Indicators:**")
        for f in positive_factors:
            lines.append(f"  - {f}")
        lines.append("")

    if pred_label == 'DEFAULT':
        lines.append(
            "> ℹ️ *This application has been flagged for further review. "
            "The model identified elevated financial risk based on the factors above. "
            "A loan officer may request additional documentation.*"
        )
    else:
        lines.append(
            "> ✅ *This applicant demonstrates acceptable credit risk. "
            "The positive indicators above contributed to this outcome.*"
        )

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_feat_name(name: str) -> str:
    """Make Lending Club feature names human-readable."""
    replacements = {
        'fico_range_low':     'FICO Score',
        'int_rate':           'Interest Rate (%)',
        'dti':                'Debt-to-Income Ratio',
        'dti_ratio':          'DTI Ratio (normalised)',
        'revol_util':         'Revolving Utilisation (%)',
        'credit_util':        'Credit Utilisation',
        'annual_inc':         'Annual Income ($)',
        'loan_amnt':          'Loan Amount ($)',
        'installment':        'Monthly Installment ($)',
        'installment_to_inc': 'Installment-to-Income',
        'loan_to_income':     'Loan-to-Income Ratio',
        'payment_to_income':  'Annual Payments / Income',
        'delinq_2yrs':        'Delinquencies (2 yrs)',
        'pub_rec':            'Public Records',
        'emp_length':         'Employment Length (yrs)',
        'open_acc':           'Open Accounts',
        'total_acc':          'Total Accounts',
        'mort_acc':           'Mortgage Accounts',
        'num_actv_bc_tl':     'Active Bankcards',
    }
    name = name.replace('onehot__', '').replace('scaler__', '').replace('ordinal__', '')
    return replacements.get(name, name.replace('_', ' ').title())


def _format_val(feature: str, val: float) -> str:
    """Format a Lending Club feature value nicely for display."""
    pct_features   = {'int_rate', 'dti', 'revol_util'}
    money_features = {'annual_inc', 'loan_amnt', 'installment'}
    for pf in pct_features:
        if pf in feature:
            return f'{val:.1f}%'
    for mf in money_features:
        if mf in feature:
            return f'${val:,.0f}'
    return f'{val:.2f}'
