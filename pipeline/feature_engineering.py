"""
Phase 2 — Feature Engineering & Preprocessing
Lending Club-style synthetic dataset.
Creates derived financial ratios and prepares data for ML models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from typing import Tuple, List

TARGET_COL   = 'loan_status'
GRADE_ORDER  = [['G','F','E','D','C','B','A']]
TERM_MAP     = {36: 0, 60: 1}
ORDINAL_COLS = ['grade']
ONEHOT_COLS  = ['home_ownership', 'purpose']
SCALE_COLS   = [
    'loan_amnt','int_rate','installment','annual_inc','dti',
    'fico_range_low','revol_util','open_acc','total_acc',
    'emp_length','delinq_2yrs','pub_rec','mort_acc','num_actv_bc_tl',
    'term',  # BUG FIX: term was mapped to 0/1 but not included in any transformer → was silently dropped
    'dti_ratio','credit_util','installment_to_inc','loan_to_income','payment_to_income',
]


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df    = df.copy()
    added = []

    if 'dti' in df.columns:
        df['dti_ratio'] = df['dti'] / 100
        added.append('dti_ratio')

    if 'revol_util' in df.columns:
        df['credit_util'] = df['revol_util'] / 100
        added.append('credit_util')

    if all(c in df.columns for c in ['installment','annual_inc']):
        monthly_inc = df['annual_inc'] / 12
        df['installment_to_inc'] = (df['installment'] / monthly_inc.replace(0, np.nan)).fillna(0)
        added.append('installment_to_inc')

    if all(c in df.columns for c in ['loan_amnt','annual_inc']):
        df['loan_to_income'] = (df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)).fillna(0)
        added.append('loan_to_income')

    if all(c in df.columns for c in ['installment','annual_inc']):
        df['payment_to_income'] = ((df['installment'] * 12) / df['annual_inc'].replace(0, np.nan)).fillna(0)
        added.append('payment_to_income')

    if 'term' in df.columns:
        df['term'] = df['term'].map(TERM_MAP).fillna(0).astype(int)

    # Safety: remove inf/nan from engineered cols
    for col in added:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    return df, added


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    ord_present = [c for c in ORDINAL_COLS if c in df.columns]
    ohe_present = [c for c in ONEHOT_COLS  if c in df.columns]
    num_present = [c for c in SCALE_COLS   if c in df.columns]

    transformers = []
    if ord_present:
        enc = OrdinalEncoder(categories=GRADE_ORDER,
                             handle_unknown='use_encoded_value', unknown_value=-1)
        transformers.append(('ordinal', enc, ord_present))
    if ohe_present:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore',
                                                      sparse_output=False), ohe_present))
    if num_present:
        transformers.append(('scaler', StandardScaler(), num_present))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, ord_present + ohe_present + num_present


def get_feature_names(preprocessor: ColumnTransformer, df: pd.DataFrame) -> List[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder':
            continue
        try:
            names.extend(trans.get_feature_names_out(cols))
        except Exception:
            names.extend(cols if isinstance(cols, list) else [cols])
    return names


def prepare_X_y(df: pd.DataFrame):
    df_eng, new_feats = engineer_features(df)
    y = df_eng[TARGET_COL]
    X = df_eng.drop(columns=[TARGET_COL])
    return X, y, new_feats


def prepare_single_applicant(applicant: dict,
                              preprocessor: ColumnTransformer,
                              df_sample: pd.DataFrame) -> np.ndarray:
    row = pd.DataFrame([applicant])
    if 'installment' not in row.columns or row['installment'].isnull().all():
        r   = row['int_rate'].iloc[0] / 100 / 12
        n   = row['term'].iloc[0]
        pv  = row['loan_amnt'].iloc[0]
        row['installment'] = pv * r / (1 - (1 + r) ** -n) if r > 0 else pv / n
    row_eng, _ = engineer_features(row)
    return preprocessor.transform(row_eng)
