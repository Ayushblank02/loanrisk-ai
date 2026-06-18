"""
Phase 1 — Data Loading & EDA
Lending Club-style dataset support.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TARGET_COL = 'loan_status'

NUMERICAL_COLS = [
    'loan_amnt','int_rate','installment','annual_inc','dti',
    'fico_range_low','revol_util','open_acc','total_acc',
    'emp_length','delinq_2yrs','pub_rec','mort_acc','num_actv_bc_tl',
]
CATEGORICAL_COLS = ['grade','home_ownership','purpose']
PALETTE = {'default': '#EF4444', 'paid': '#10B981'}


def load_csv(file_obj) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(file_obj, low_memory=False)
    if TARGET_COL not in df.columns:
        if 'loan_status' in [c.lower() for c in df.columns]:
            df.columns = [c.lower() for c in df.columns]
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map(
            {'Fully Paid': 0, 'Charged Off': 1, 0: 0, 1: 1}
        ).fillna(df[TARGET_COL])
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df, _build_report(df)


def load_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    return df, _build_report(df)


def _build_report(df: pd.DataFrame) -> dict:
    total       = len(df)
    defaults    = int(df[TARGET_COL].sum()) if TARGET_COL in df.columns else 0
    missing     = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    return {
        'total_records':     total,
        'default_count':     defaults,
        'non_default_count': total - defaults,
        'default_rate':      round(defaults / total * 100, 2) if total else 0,
        'num_features':      df.shape[1] - 1,
        'missing_report':    missing_pct[missing_pct > 0].sort_values(ascending=False).to_dict(),
    }


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df  = df.copy()
    log = {}
    num_present = [c for c in NUMERICAL_COLS if c in df.columns]
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]

    for col in num_present:
        n = df[col].isnull().sum()
        if n > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log[col] = f'Imputed {n} nulls with median ({med:.2f})'

    for col in cat_present:
        n = df[col].isnull().sum()
        if n > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            log[col] = f'Imputed {n} nulls with mode ({mode})'

    return df, log


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET_COL].value_counts()
    fig = go.Figure(go.Pie(
        labels=['Fully Paid', 'Default'],
        values=[counts.get(0, 0), counts.get(1, 0)],
        hole=0.55,
        marker_colors=[PALETTE['paid'], PALETTE['default']],
        textinfo='label+percent',
    ))
    fig.update_layout(title='Class Distribution (Loan Status)', title_x=0.5,
                      height=350, paper_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    mp = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    mp = mp[mp > 0]
    if mp.empty:
        fig = go.Figure()
        fig.add_annotation(text='No missing values ✓', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#10B981'))
        fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)')
        return fig
    fig = go.Figure(go.Bar(x=mp.values, y=mp.index, orientation='h',
                            marker_color=PALETTE['default'],
                            text=[f'{v:.1f}%' for v in mp.values],
                            textposition='outside'))
    fig.update_layout(title='Missing Values (%)', height=max(300, len(mp)*30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> go.Figure:
    candidates = ['fico_range_low','int_rate','annual_inc','dti','revol_util','loan_amnt']
    plot_cols  = [c for c in candidates if c in df.columns]

    if not plot_cols:
        plot_cols = [c for c in df.select_dtypes(include='number').columns
                     if c != TARGET_COL][:6]

    if not plot_cols:
        fig = go.Figure()
        fig.add_annotation(text='No numeric features to plot', xref='paper',
                           yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)')
        return fig

    rows = max(1, (len(plot_cols) + 1) // 2)
    fig  = make_subplots(rows=rows, cols=2,
                          subplot_titles=[c.replace('_',' ').title() for c in plot_cols])

    for i, col in enumerate(plot_cols):
        r, c = divmod(i, 2)
        paid    = df[df[TARGET_COL] == 0][col].dropna()
        default = df[df[TARGET_COL] == 1][col].dropna()
        fig.add_trace(go.Histogram(x=paid,    name='Fully Paid', opacity=0.65,
                                    marker_color=PALETTE['paid'],
                                    showlegend=(i == 0)), row=r+1, col=c+1)
        fig.add_trace(go.Histogram(x=default, name='Default',    opacity=0.65,
                                    marker_color=PALETTE['default'],
                                    showlegend=(i == 0)), row=r+1, col=c+1)

    fig.update_layout(title='Feature Distributions by Loan Status',
                      barmode='overlay', height=300*rows,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in NUMERICAL_COLS if c in df.columns] + [TARGET_COL]
    corr = df[cols].corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu_r', zmid=0,
        text=corr.values.round(2), texttemplate='%{text}', textfont_size=8,
    ))
    fig.update_layout(title='Feature Correlation Matrix', height=600,
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(size=9),
                      margin=dict(l=130, r=20, t=60, b=130))
    return fig
