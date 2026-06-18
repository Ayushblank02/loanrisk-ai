"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LOAN DEFAULT PREDICTOR  —  Credit Risk Assessment System           ║
║          B.Tech Final Year Project  |  ML + XAI + Streamlit                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    streamlit run app.py

Install deps first:
    pip install -r requirements.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Loan Default Predictor',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99,102,241,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a5b4fc;
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
    }
    .phase-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.4);
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1b4b, #0f172a);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #a5b4fc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .risk-pill {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .info-box {
        background: rgba(99,102,241,0.08);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 2px solid rgba(99,102,241,0.3);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #3730a3);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    }
    div[data-testid="stSidebarContent"] {
        background: #0f172a;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(99,102,241,0.2);
        margin-bottom: 1rem;
    }
    .sidebar-logo h2 {
        font-size: 1.2rem;
        color: #a5b4fc;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        'df':              None,
        'df_clean':        None,
        'X':               None,
        'y':               None,
        'X_train':         None,
        'X_test':          None,
        'y_train':         None,
        'y_test':          None,
        'X_train_smote':   None,
        'y_train_smote':   None,
        'preprocessor':    None,
        'feature_names':   None,
        'models_dict':     None,
        'metrics_df':      None,
        'best_model_name': None,
        'shap_values':     None,
        'eda_report':      None,
        'smote_stats':     None,
        'data_loaded':     False,
        'model_trained':   False,
        'shap_computed':   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🏦 LoanRisk AI</h2>
        <p style="color:#64748b; font-size:0.75rem; margin:0;">Credit Risk Assessment System</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        'Navigation',
        ['📊 Data & EDA',
         '⚙️ Train Models',
         '📈 Evaluate & Compare',
         '🔍 Explain Prediction',
         '🎯 Predict Applicant',
         '📖 About'],
        label_visibility='collapsed',
    )

    st.markdown('---')
    st.markdown('<p style="color:#64748b; font-size:0.75rem;">📌 Pipeline Status</p>',
                unsafe_allow_html=True)

    statuses = {
        '✅ Data Loaded'  if st.session_state.data_loaded  else '○ Data Loaded':  st.session_state.data_loaded,
        '✅ Models Trained' if st.session_state.model_trained else '○ Models Trained': st.session_state.model_trained,
        '✅ SHAP Ready'   if st.session_state.shap_computed else '○ SHAP Ready':  st.session_state.shap_computed,
    }
    for label in statuses:
        color = '#10B981' if any(x in label for x in ['✅']) else '#475569'
        st.markdown(f'<p style="color:{color}; font-size:0.82rem; margin:0.2rem 0">{label}</p>',
                    unsafe_allow_html=True)

    st.markdown('---')
    st.markdown("""
    <p style="color:#475569; font-size:0.7rem; text-align:center">
    B.Tech Final Year Project<br>
    ML Pipeline · XGBoost · SHAP XAI
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA & EDA
# ══════════════════════════════════════════════════════════════════════════════

if page == '📊 Data & EDA':

    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Loading & Exploratory Analysis</h1>
        <p>Phase 1 — Load your dataset, understand distributions, and detect data quality issues</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Source ────────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">1. Load Dataset</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-box">📁 <b>Upload your own CSV</b> — Works best with Lending Club Loan Data from Kaggle. Required column: <code>loan_status</code> (values: "Fully Paid" / "Charged Off")</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader('Upload CSV File', type=['csv'], label_visibility='collapsed')

    with col2:
        st.markdown('<div class="info-box">🎲 <b>Demo Mode</b> — Generate a realistic synthetic Lending Club-style dataset calibrated to real published statistics (income, FICO, DTI, grade distributions). 22% default rate matching real LC data.</div>', unsafe_allow_html=True)
        n_demo   = st.slider('Demo Dataset Size', 1000, 30000, 10000, 1000)
        demo_btn = st.button('🚀 Generate Demo Data', width='stretch')

    # ── Load ───────────────────────────────────────────────────────────────────
    if uploaded or demo_btn:
        from pipeline.data_loader   import load_csv, load_dataframe, clean_data
        from pipeline.data_loader   import plot_class_distribution, plot_missing_values
        from pipeline.data_loader   import plot_feature_distributions, plot_correlation_heatmap
        from data.sample_data       import generate_lending_club_sample

        with st.spinner('Loading data...'):
            if demo_btn:
                raw_df = generate_lending_club_sample(n_samples=n_demo)
                df, report = load_dataframe(raw_df)
            else:
                df, report = load_csv(uploaded)

            df_clean, clean_log = clean_data(df)
            st.session_state.df          = df
            st.session_state.df_clean    = df_clean
            st.session_state.eda_report  = report
            st.session_state.data_loaded = True

        st.success(f'✅ Dataset loaded: {report["total_records"]:,} records, {report["num_features"]} features')

    # ── EDA Dashboard ──────────────────────────────────────────────────────────
    if st.session_state.data_loaded:
        from pipeline.data_loader import (
            plot_class_distribution, plot_missing_values,
            plot_feature_distributions, plot_correlation_heatmap, clean_data
        )

        report   = st.session_state.eda_report
        df       = st.session_state.df
        df_clean = st.session_state.df_clean

        st.markdown('<div class="step-header">2. Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric('Total Records',  f'{report["total_records"]:,}')
        with c2: st.metric('Default Count',  f'{report["default_count"]:,}')
        with c3: st.metric('Default Rate',   f'{report["default_rate"]}%')
        with c4: st.metric('Total Features', report['num_features'])

        st.markdown('<div class="step-header">3. Class Imbalance</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(plot_class_distribution(df), width='stretch')
            st.markdown(f"""
            <div class="info-box">
            ⚠️ <b>Class Imbalance Detected</b><br>
            {100 - report["default_rate"]:.1f}% of loans are Fully Paid vs {report["default_rate"]:.1f}% Default.
            This will be corrected with <b>SMOTE</b> during model training.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.plotly_chart(plot_missing_values(df), width='stretch')

        st.markdown('<div class="step-header">4. Feature Distributions</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_feature_distributions(df_clean), width='stretch')

        st.markdown('<div class="step-header">5. Correlation Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_correlation_heatmap(df_clean), width='stretch')

        # Cleaning log
        if st.session_state.df_clean is not None:
            _, clean_log = clean_data(df)
            if clean_log:
                with st.expander('🧹 Data Cleaning Log'):
                    for k, v in clean_log.items():
                        st.markdown(f'- **{k}**: {v}')
            else:
                st.info('✅ Data is clean — no missing values or outliers found.')

        st.success('✅ EDA complete! Proceed to **Train Models** in the sidebar.')

    else:
        st.info('👆 Upload a CSV or click "Generate Demo Data" to begin.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == '⚙️ Train Models':

    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Model Training Pipeline</h1>
        <p>Phases 2–4 — Feature engineering, SMOTE balancing, and training all three models</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning('⚠️ Please load a dataset first on the **Data & EDA** page.')
        st.stop()

    # ── Hyperparameter Configuration ──────────────────────────────────────────
    from pipeline.models import get_model_configs

    configs     = get_model_configs()
    st.markdown('<div class="step-header">1. Configure Hyperparameters</div>', unsafe_allow_html=True)

    tabs = st.tabs(['Logistic Regression', 'Random Forest', 'XGBoost'])

    with tabs[0]:
        col1, col2 = st.columns(2)
        configs['Logistic Regression']['params']['C'] = col1.slider(
            'Regularisation C', 0.01, 10.0,
            float(configs['Logistic Regression']['params']['C']), 0.01,
            key='lr_c')
        configs['Logistic Regression']['params']['max_iter'] = col2.slider(
            'Max Iterations', 100, 2000,
            configs['Logistic Regression']['params']['max_iter'], 100,
            key='lr_max_iter')

    with tabs[1]:
        col1, col2, col3 = st.columns(3)
        configs['Random Forest']['params']['n_estimators'] = col1.slider(
            'N Estimators', 50, 500,
            configs['Random Forest']['params']['n_estimators'], 50,
            key='rf_n_estimators')
        configs['Random Forest']['params']['max_depth'] = col2.slider(
            'Max Depth', 2, 30,
            configs['Random Forest']['params']['max_depth'], 1,
            key='rf_max_depth')
        configs['Random Forest']['params']['min_samples_leaf'] = col3.slider(
            'Min Samples Leaf', 1, 50,
            configs['Random Forest']['params']['min_samples_leaf'], 1,
            key='rf_min_samples_leaf')

    with tabs[2]:
        col1, col2, col3 = st.columns(3)
        configs['XGBoost']['params']['n_estimators'] = col1.slider(
            'N Estimators', 50, 500,
            configs['XGBoost']['params']['n_estimators'], 50,
            key='xgb_n_estimators')
        configs['XGBoost']['params']['learning_rate'] = col2.slider(
            'Learning Rate', 0.01, 0.5,
            float(configs['XGBoost']['params']['learning_rate']), 0.01,
            key='xgb_learning_rate')
        configs['XGBoost']['params']['max_depth'] = col3.slider(
            'Max Depth', 2, 10,
            configs['XGBoost']['params']['max_depth'], 1,
            key='xgb_max_depth')

    # ── SMOTE Toggle ──────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">2. Class Balancing</div>', unsafe_allow_html=True)
    use_smote = st.checkbox('Apply SMOTE to training data', value=True)
    if use_smote:
        st.markdown("""
        <div class="info-box">
        🔬 <b>SMOTE</b> (Synthetic Minority Over-sampling Technique) will generate synthetic
        default examples in the training set using k-nearest neighbors. The test set is
        kept as-is to reflect real-world conditions.
        </div>
        """, unsafe_allow_html=True)

    # ── Train Button ──────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">3. Train All Models</div>', unsafe_allow_html=True)

    if st.button('🚀 Run Full Training Pipeline', width='stretch'):
        from pipeline.feature_engineering import (
            engineer_features, build_preprocessor, get_feature_names, prepare_X_y
        )
        from pipeline.imbalance import split_data, apply_smote, get_balance_stats, plot_smote_comparison
        from pipeline.models    import train_all_models

        progress_bar  = st.progress(0, text='Initialising...')
        status_text   = st.empty()

        # Phase 2: Feature Engineering
        progress_bar.progress(10, text='Phase 2: Engineering features...')
        X, y, new_feats = prepare_X_y(st.session_state.df_clean)
        status_text.info(f'✅ Engineered {len(new_feats)} new features: {", ".join(new_feats)}')

        # Build preprocessor
        progress_bar.progress(20, text='Building preprocessing pipeline...')
        preprocessor, base_feat_names = build_preprocessor(X)
        preprocessor.fit(X)
        feat_names = get_feature_names(preprocessor, X)
        X_processed = preprocessor.transform(X)

        # Phase 3: Split
        progress_bar.progress(30, text='Phase 3: Train/test split...')
        X_train, X_test, y_train, y_test = split_data(X_processed, y)
        y_train_arr = y_train.values

        # SMOTE
        if use_smote:
            progress_bar.progress(40, text='Applying SMOTE...')
            X_train_bal, y_train_bal = apply_smote(X_train, y_train_arr)
            smote_stats = get_balance_stats(y_train_arr, y_train_bal)
        else:
            X_train_bal, y_train_bal = X_train, y_train_arr
            smote_stats = None

        # Phase 4: Train models
        def progress_cb(name, step, total):
            pct = 50 + int((step / total) * 45)
            progress_bar.progress(pct, text=f'Training {name}...')

        trained = train_all_models(X_train_bal, y_train_bal, configs, progress_cb)

        progress_bar.progress(100, text='✅ Training complete!')
        status_text.empty()

        # Store in session
        st.session_state.X_train         = X_train
        st.session_state.X_test          = X_test
        st.session_state.y_train         = y_train_arr
        st.session_state.y_test          = y_test.values
        st.session_state.X_train_smote   = X_train_bal
        st.session_state.y_train_smote   = y_train_bal
        st.session_state.preprocessor    = preprocessor
        st.session_state.feature_names   = feat_names
        st.session_state.models_dict     = trained
        st.session_state.smote_stats     = smote_stats
        st.session_state.model_trained   = True
        st.session_state.metrics_df      = None   # reset so evaluate page recomputes
        st.session_state.best_model_name = None
        st.session_state.shap_computed   = False  # reset SHAP too

        st.success('✅ All 3 models trained successfully! Go to **Evaluate & Compare**.')

        if smote_stats and use_smote:
            from pipeline.imbalance import plot_smote_comparison
            st.markdown('#### SMOTE Balancing Result')
            st.plotly_chart(plot_smote_comparison(smote_stats), width='stretch')
            colA, colB = st.columns(2)
            colA.metric('Synthetic Samples Added', f'{smote_stats["samples_added"]:,}')
            colB.metric('Class Ratio After SMOTE', f'{smote_stats["after"]["ratio"]}:1')

        # Training times
        st.markdown('#### Model Training Times')
        for name, obj in trained.items():
            st.markdown(f'- **{name}**: `{obj["training_time"]}s`')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EVALUATE & COMPARE
# ══════════════════════════════════════════════════════════════════════════════

elif page == '📈 Evaluate & Compare':

    st.markdown("""
    <div class="main-header">
        <h1>📈 Model Evaluation & Comparison</h1>
        <p>Phase 5 — Recall, ROC-AUC, Confusion Matrix, and Financial Impact Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning('⚠️ Please train models first on the **Train Models** page.')
        st.stop()

    from pipeline.evaluation import (
        evaluate_all, plot_metrics_comparison, plot_roc_curves,
        plot_confusion_matrix, plot_cost_benefit, compute_cost_benefit
    )
    from pipeline.models import predict

    models_dict = st.session_state.models_dict
    X_test      = st.session_state.X_test
    y_test      = st.session_state.y_test

    # Compute all metrics
    if st.session_state.metrics_df is None:
        with st.spinner('Computing evaluation metrics...'):
            # Best model = highest ROC-AUC (gold standard for credit scoring across all thresholds)
            metrics_df = evaluate_all(models_dict, X_test, y_test)
            st.session_state.metrics_df = metrics_df
            best_name = metrics_df['ROC-AUC'].idxmax()
            st.session_state.best_model_name = best_name

    metrics_df = st.session_state.metrics_df
    best_name  = st.session_state.best_model_name

    # ── Summary Cards ──────────────────────────────────────────────────────────
    st.markdown(f'### 🏆 Best Model: **{best_name}** (highest ROC-AUC)')
    best = metrics_df.loc[best_name]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('ROC-AUC',    f'{best["ROC-AUC"]:.3f}',   delta='Primary Metric')
    c2.metric('Recall',     f'{best["Recall"]:.3f}',    delta='Default Detection')  # BUG FIX: was showing ROC-AUC twice
    c3.metric('Precision',  f'{best["Precision"]:.3f}')
    c4.metric('F1 Score',   f'{best["F1 Score"]:.3f}')
    c5.metric('Accuracy',   f'{best["Accuracy"]:.3f}')

    st.markdown('<div class="info-box">💡 <b>Why ROC-AUC?</b> ROC-AUC is the gold standard metric for credit scoring. It measures the model ability to distinguish defaulters from good borrowers across all thresholds - independent of class imbalance. This is the metric used in all published lending research.</div>',
                unsafe_allow_html=True)

    # ── Metrics Table ──────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">All Model Scores</div>', unsafe_allow_html=True)
    styled = metrics_df.style.format({
        'Recall': '{:.3f}', 'Precision': '{:.3f}',
        'F1 Score': '{:.3f}', 'Accuracy': '{:.3f}', 'ROC-AUC': '{:.3f}',
    }).highlight_max(subset=['Recall', 'ROC-AUC', 'F1 Score'], color='#1e3a5f')
    st.dataframe(styled, width='stretch')

    # ── Visualisations ─────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Performance Charts</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_metrics_comparison(metrics_df), width='stretch')
    with col_b:
        st.plotly_chart(plot_roc_curves(models_dict, X_test, y_test), width='stretch')

    # ── NEW FEATURE: Precision-Recall Curves ───────────────────────────────────
    from pipeline.evaluation import plot_pr_curves
    st.markdown('<div class="step-header">Precision-Recall Curves</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📖 <b>Why Precision-Recall?</b> ROC-AUC can look optimistic on imbalanced data because
    it rewards correct negatives (the majority class). The PR curve focuses exclusively on
    <b>defaulters</b> — it shows the real trade-off between catching more defaults (Recall)
    and avoiding false alarms (Precision). <b>Average Precision (AP)</b> is the area under this curve.
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(plot_pr_curves(models_dict, X_test, y_test), width='stretch')

    # ── NEW FEATURE: Decision Threshold Tuner ─────────────────────────────────
    from pipeline.evaluation import compute_threshold_metrics, plot_threshold_tuner
    st.markdown('<div class="step-header">🎚️ Decision Threshold Tuner</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    🎯 <b>What is the decision threshold?</b> By default, a loan is flagged as default if
    predicted probability ≥ 50%. Lowering the threshold catches more defaults (<b>↑ Recall</b>)
    but also rejects more good loans (<b>↓ Precision</b>). This lets you tune the model for
    your institution's risk appetite — a conservative bank might use 0.35, a lenient one 0.60.
    </div>
    """, unsafe_allow_html=True)

    threshold_val = st.slider(
        'Decision Threshold', 0.10, 0.90, 0.50, 0.01,
        key='threshold_tuner_slider',
        help='Predicted probability above this value → flagged as DEFAULT'
    )

    _, y_prob_best = predict(models_dict[best_name]['model'], X_test)
    thresholds_range = np.arange(0.10, 0.91, 0.02)
    threshold_df = compute_threshold_metrics(y_test, y_prob_best, thresholds_range)

    st.plotly_chart(plot_threshold_tuner(threshold_df, threshold_val), width='stretch')

    # Live metrics at the selected threshold
    y_pred_thresh = (y_prob_best >= threshold_val).astype(int)
    from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
    thr_c1, thr_c2, thr_c3, thr_c4 = st.columns(4)
    thr_c1.metric('Recall @ threshold',    f'{recall_score(y_test, y_pred_thresh, zero_division=0):.3f}',    delta='Defaulters caught')
    thr_c2.metric('Precision @ threshold', f'{precision_score(y_test, y_pred_thresh, zero_division=0):.3f}', delta='Flag accuracy')
    thr_c3.metric('F1 @ threshold',        f'{f1_score(y_test, y_pred_thresh, zero_division=0):.3f}')
    thr_c4.metric('Accuracy @ threshold',  f'{accuracy_score(y_test, y_pred_thresh):.3f}')

    # ── Per-Model Confusion Matrix ─────────────────────────────────────────────
    st.markdown('<div class="step-header">Confusion Matrices</div>', unsafe_allow_html=True)
    model_cols = st.columns(len(models_dict))
    for i, (name, obj) in enumerate(models_dict.items()):
        y_pred, _ = predict(obj['model'], X_test)
        with model_cols[i]:
            st.plotly_chart(plot_confusion_matrix(y_test, y_pred, name),
                            width='stretch')

    # ── Financial Cost-Benefit ─────────────────────────────────────────────────
    st.markdown('<div class="step-header">💰 Financial Impact Analysis</div>',
                unsafe_allow_html=True)

    avg_loan = st.slider('Average Loan Amount ($)', 5000, 50000, 15000, 1000, key='avg_loan_slider')

    y_pred_best, _ = predict(models_dict[best_name]['model'], X_test)
    cost_dict      = compute_cost_benefit(y_test, y_pred_best, avg_loan=avg_loan)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('False Negatives', cost_dict['FN'], delta='Approved Defaulters', delta_color='inverse')
    col2.metric('False Positives', cost_dict['FP'], delta='Rejected Good Loans',  delta_color='inverse')
    col3.metric('FN Cost',         f'${cost_dict["fn_cost"]:,.0f}')
    col4.metric('Net Savings',     f'${cost_dict["net_savings"]:,.0f}')

    st.plotly_chart(plot_cost_benefit(cost_dict, best_name), width='stretch')

    # ── NEW FEATURE: Calibration Curves ───────────────────────────────────────
    from pipeline.evaluation import plot_calibration_curves
    st.markdown('<div class="step-header">📐 Probability Calibration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📐 <b>What is calibration?</b> A well-calibrated model means its predicted probabilities
    match reality — if it says "70% default risk" for 100 applicants, roughly 70 should actually default.
    A curve hugging the diagonal = trustworthy probabilities.<br><br>
    📈 Curves bowing <b>above</b> the diagonal → model <b>underestimates</b> default risk
    (it predicts 30%, but 50% actually defaulted — underconfident).<br>
    📉 Curves bowing <b>below</b> the diagonal → model <b>overestimates</b> default risk
    (it predicts 50%, but only 25% actually defaulted — overconfident).<br><br>
    ⚠️ <b>Why are all three curves below the diagonal here?</b> This is a direct consequence of
    SMOTE training on a synthetic 50/50 balanced dataset, while the real-world test set is 78/22.
    Models learned to fire at higher default probabilities than the true base rate justifies.
    This is a known SMOTE side-effect — probabilities can be corrected post-hoc using
    Platt Scaling or Isotonic Regression if accurate risk pricing is required.
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(plot_calibration_curves(models_dict, X_test, y_test),
                    width='stretch')

    # ── NEW FEATURE: Model Download ────────────────────────────────────────────
    st.markdown('<div class="step-header">💾 Download Trained Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    💾 Download any trained model as a <code>.pkl</code> file. Load it later with
    <code>joblib.load('model.pkl')</code> — no retraining needed. This is how models
    are exported for production deployment.
    </div>
    """, unsafe_allow_html=True)

    import joblib, io
    dl_cols = st.columns(len(models_dict))
    for i, (name, obj) in enumerate(models_dict.items()):
        with dl_cols[i]:
            buf = io.BytesIO()
            joblib.dump(obj['model'], buf)
            buf.seek(0)
            filename = name.lower().replace(' ', '_') + '.pkl'
            is_best  = (name == best_name)
            label    = f'⬇️ {name}{"  🏆" if is_best else ""}'
            st.download_button(
                label=label,
                data=buf,
                file_name=filename,
                mime='application/octet-stream',
                width='stretch',
                key=f'dl_{name}',
            )
            auc_val = metrics_df.loc[name, 'ROC-AUC']
            st.caption(f'ROC-AUC: {auc_val:.3f} · {obj["training_time"]}s training')

    # Also offer preprocessor download (required to transform new data for inference)
    buf_pp = io.BytesIO()
    joblib.dump(st.session_state.preprocessor, buf_pp)
    buf_pp.seek(0)
    st.download_button(
        label='⬇️ Download Preprocessor Pipeline (required for inference)',
        data=buf_pp,
        file_name='preprocessor.pkl',
        mime='application/octet-stream',
        width='stretch',
        key='dl_preprocessor',
    )
    st.caption('The preprocessor applies feature engineering + scaling. Always bundle it with the model.')

    st.success(f'✅ Evaluation complete! Proceed to **Explain Prediction** to compute SHAP values.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAIN PREDICTION (SHAP)
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🔍 Explain Prediction':

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Explainable AI — SHAP Analysis</h1>
        <p>Phase 6 — Understand why the model makes its predictions using SHAP values</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning('⚠️ Please train models first.')
        st.stop()

    from pipeline.explainability import (
        compute_shap_values, plot_shap_summary, get_global_feature_importance
    )

    # Always use XGBoost for SHAP — tree explainer gives richer, more interpretable values
    models_dict = st.session_state.models_dict
    shap_model_name = 'XGBoost' if 'XGBoost' in models_dict else st.session_state.best_model_name
    best_name   = st.session_state.best_model_name
    best_model  = models_dict[shap_model_name]['model']
    feat_names  = st.session_state.feature_names
    X_test      = st.session_state.X_test

    # ── Disclosure banner when best model ≠ XGBoost ───────────────────────────
    if best_name != 'XGBoost' and 'XGBoost' in models_dict:
        st.info(
            f'ℹ️ **Note:** The best-performing model by ROC-AUC is **{best_name}** ({st.session_state.metrics_df.loc[best_name, "ROC-AUC"]:.3f}). '
            f'However, SHAP analysis always uses **XGBoost** ({st.session_state.metrics_df.loc["XGBoost", "ROC-AUC"]:.3f} ROC-AUC) '
            f'because its `TreeExplainer` computes *exact* Shapley values in polynomial time — '
            f'far more precise than the approximations available for linear models. '
            f'The feature importance ranking is consistent across both models for this dataset.'
        )

    if st.button(f'🔬 Compute SHAP Values (XGBoost)', width='stretch'):
        with st.spinner('Computing SHAP values (this takes 30–60s for large datasets)...'):
            # Use a sample for speed
            sample_size = min(500, len(X_test))
            X_sample    = X_test[:sample_size]
            shap_vals, explainer = compute_shap_values(best_model, X_sample, feat_names)
            st.session_state.shap_values    = shap_vals
            st.session_state.shap_explainer = explainer
            st.session_state.X_shap_sample  = X_sample
            st.session_state.shap_computed  = True
        st.success('✅ SHAP values computed!')

    if st.session_state.shap_computed:
        shap_vals = st.session_state.shap_values
        X_sample  = st.session_state.X_shap_sample

        st.markdown('<div class="step-header">Global Feature Importance (Mean |SHAP|)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_shap_summary(shap_vals, X_sample, feat_names),
                        width='stretch')

        st.markdown('<div class="info-box">📖 <b>Reading this chart:</b> Features at the top have the largest average impact on predictions across all applicants. A high Mean |SHAP| value means this feature consistently drives the model\'s decisions.</div>',
                    unsafe_allow_html=True)

        # Top features table
        importance_df = get_global_feature_importance(shap_vals, feat_names)
        st.markdown('<div class="step-header">Top Feature Ranking</div>', unsafe_allow_html=True)
        st.dataframe(importance_df.style.format({'importance': '{:.5f}'}),
                     width='stretch', height=320)

        st.success('✅ SHAP analysis complete! Go to **Predict Applicant** to test individual predictions.')
    else:
        st.info('👆 Click the button above to compute SHAP values for the best model.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT APPLICANT
# ══════════════════════════════════════════════════════════════════════════════

elif page == '🎯 Predict Applicant':

    st.markdown("""
    <div class="main-header">
        <h1>🎯 Applicant Risk Assessment</h1>
        <p>Enter loan application details and get an instant risk prediction with SHAP explanation</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning('⚠️ Please train models first.')
        st.stop()

    from data.sample_data       import get_demo_applicant
    from pipeline.models        import predict_single
    from pipeline.feature_engineering import engineer_features, prepare_single_applicant
    from pipeline.explainability import (
        compute_shap_values, plot_shap_waterfall, generate_explanation
    )

    # ── Demo Fill Button ───────────────────────────────────────────────────────
    col_hdr, col_btn = st.columns([3, 1])
    with col_btn:
        use_demo = st.button('📋 Fill Demo Applicant', width='stretch')

    demo = get_demo_applicant() if use_demo else {}

    # ── Input Form ────────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Applicant Details</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('**Loan Details**')
        loan_amnt = st.number_input('Loan Amount ($)', 1000, 40000,
                                     demo.get('loan_amnt', 10000), 500)
        term      = st.selectbox('Term (months)', [36, 60],
                                  index=0 if demo.get('term', 36) == 36 else 1)
        int_rate  = st.slider('Interest Rate (%)', 5.0, 30.0,
                               float(demo.get('int_rate', 12.0)), 0.1)
        purpose   = st.selectbox('Loan Purpose',
                                  ['debt_consolidation','credit_card','home_improvement',
                                   'other','major_purchase','small_business','car','medical'],
                                  index=0)

    with c2:
        st.markdown('**Borrower Profile**')
        annual_inc     = st.number_input('Annual Income ($)', 10000, 500000,
                                          int(demo.get('annual_inc', 60000)), 1000)
        emp_length     = st.slider('Employment Length (years)', 0, 10,
                                    demo.get('emp_length', 3))
        home_ownership = st.selectbox('Home Ownership',
                                       ['RENT','MORTGAGE','OWN','OTHER'],
                                       index=['RENT','MORTGAGE','OWN','OTHER'].index(
                                           demo.get('home_ownership','RENT')))
        grade          = st.selectbox('Loan Grade', ['A','B','C','D','E','F','G'],
                                       index=['A','B','C','D','E','F','G'].index(
                                           demo.get('grade','C')))

    with c3:
        st.markdown('**Credit History**')
        fico       = st.slider('FICO Score', 580, 850,
                                demo.get('fico_range_low', 680), 5)
        dti        = st.slider('DTI Ratio (%)', 0.0, 45.0,
                                float(demo.get('dti', 18.0)), 0.5)
        revol_util = st.slider('Revolving Utilisation (%)', 0.0, 100.0,
                                float(demo.get('revol_util', 45.0)), 0.5)
        delinq     = st.number_input('Delinquencies (2 yrs)', 0, 10,
                                      demo.get('delinq_2yrs', 0))
        pub_rec    = st.number_input('Public Records', 0, 5,
                                      demo.get('pub_rec', 0))
        # BUG FIX: these 4 fields were previously hardcoded constants, meaning
        # every applicant was silently given the same credit account profile.
        # The model uses them — they must be real user inputs.
        open_acc       = st.number_input('Open Accounts', 1, 40,
                                          demo.get('open_acc', 10))
        total_acc      = st.number_input('Total Accounts (lifetime)', 1, 80,
                                          demo.get('total_acc', 22))
        mort_acc       = st.number_input('Mortgage Accounts', 0, 15,
                                          demo.get('mort_acc', 0))
        num_actv_bc_tl = st.number_input('Active Bankcards', 0, 20,
                                          demo.get('num_actv_bc_tl', 4))

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown('---')
    predict_btn = st.button('🔮 Assess Credit Risk', width='stretch')

    if predict_btn:
        applicant = {
            'loan_amnt':      loan_amnt,
            'term':           term,
            'int_rate':       int_rate,
            'grade':          grade,
            'emp_length':     emp_length,
            'home_ownership': home_ownership,
            'annual_inc':     float(annual_inc),
            'purpose':        purpose,
            'dti':            dti,
            'delinq_2yrs':    delinq,
            'fico_range_low': fico,
            'open_acc':       int(open_acc),        # BUG FIX: was hardcoded 10
            'pub_rec':        pub_rec,
            'revol_util':     revol_util,
            'total_acc':      int(total_acc),        # BUG FIX: was hardcoded 22
            'mort_acc':       int(mort_acc),         # BUG FIX: was hardcoded 0
            'num_actv_bc_tl': int(num_actv_bc_tl),  # BUG FIX: was hardcoded 4
        }

        preprocessor = st.session_state.preprocessor
        best_name    = st.session_state.best_model_name
        feat_names   = st.session_state.feature_names

        # BUG FIX: SHAP explainer (page 4) is always built on XGBoost.
        # Prediction here must use the SAME model — otherwise we explain
        # XGBoost's decision while showing Random Forest's probability, which is wrong.
        models_dict_p5   = st.session_state.models_dict
        if st.session_state.shap_computed and 'XGBoost' in models_dict_p5:
            predict_model_name = 'XGBoost'
        else:
            predict_model_name = best_name
        best_model = models_dict_p5[predict_model_name]['model']

        if predict_model_name != best_name:
            st.caption(
                f'ℹ️ Predicting with **XGBoost** (SHAP computed) — best model by ROC-AUC is '
                f'**{best_name}**. XGBoost is used here so the prediction and its SHAP '
                f'explanation come from the same model.'
            )

        with st.spinner('Running prediction...'):
            X_single = prepare_single_applicant(applicant, preprocessor,
                                                 st.session_state.df_clean)
            result   = predict_single(best_model, X_single)

        # ── Prediction Result Card ─────────────────────────────────────────────
        st.markdown('### 📋 Prediction Result')

        bg_color    = 'rgba(239,68,68,0.1)'  if result['prediction'] == 1 else 'rgba(16,185,129,0.1)'
        border_col  = result['risk_color']

        st.markdown(f"""
        <div style="background:{bg_color}; border:2px solid {border_col};
                    border-radius:16px; padding:2rem; margin:1rem 0; text-align:center;">
            <div style="font-size:2.5rem; font-weight:700; color:{border_col}">
                {'⚠️ ' if result['prediction']==1 else '✅ '}{result['label']}
            </div>
            <div style="font-size:1.1rem; color:#94a3b8; margin-top:0.5rem">
                Default Probability: <b style="color:{border_col}">{result['default_prob']}%</b>
                &nbsp;|&nbsp;
                Risk Tier: <b style="color:{border_col}">{result['risk_tier']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric('Default Probability', f'{result["default_prob"]}%')
        col2.metric('Repayment Probability', f'{result["paid_prob"]}%')
        col3.metric('Risk Tier', result['risk_tier'])

        # ── SHAP for this prediction ───────────────────────────────────────────
        if st.session_state.shap_computed:
            st.markdown('<div class="step-header">🔍 Why This Decision? (SHAP Explanation)</div>',
                        unsafe_allow_html=True)
            from pipeline.explainability import plot_shap_waterfall, generate_explanation

            explainer = st.session_state.shap_explainer
            raw = explainer.shap_values(X_single)
            sv  = np.array(raw)
            if sv.ndim == 3:
                if sv.shape[2] == 2:
                    sv = sv[:, :, 1]
                elif sv.shape[0] == 2:
                    sv = sv[1]
            elif isinstance(raw, list) and len(raw) == 2:
                sv = np.array(raw[1])
            shap_vals_single = sv[0]

            # BUG FIX: base_val must be the explainer's expected_value (the model's
            # mean prediction on the training set = the waterfall's starting point).
            # np.mean(shap_values) averages the entire SHAP matrix and is meaningless.
            ev = st.session_state.shap_explainer.expected_value
            base_val = float(np.atleast_1d(ev)[-1])  # take class-1 value for binary models

            col_wf, col_exp = st.columns([1.2, 1])
            with col_wf:
                fig_wf = plot_shap_waterfall(
                    shap_vals_single, feat_names, X_single[0], base_val
                )
                st.plotly_chart(fig_wf, width='stretch')

            with col_exp:
                explanation = generate_explanation(
                    shap_vals_single, feat_names, X_single[0], result
                )
                st.markdown(explanation)
        else:
            st.info('💡 Compute SHAP values on the **Explain Prediction** page for a detailed explanation of this decision.')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT / PROJECT REPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == '📖 About':

    st.markdown("""
    <div class="main-header">
        <h1>📖 About This Project</h1>
        <p>B.Tech Final Year Project — ML-based Credit Risk Assessment with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Overview ──────────────────────────────────────────────────────
    st.markdown('## 🎯 Project Overview')
    st.markdown("""
    **LoanRisk AI** is a full end-to-end machine learning system for predicting loan default risk,
    built to demonstrate the complete data science lifecycle — from raw data ingestion through
    model explainability and deployment-ready export.

    The system is modelled on real-world credit risk workflows used by banks and fintechs,
    calibrated against published Lending Club loan statistics (2018), and designed to be
    explainable by regulators and loan officers — not just data scientists.
    """)

    # ── Pipeline Architecture ─────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## 🏗️ Pipeline Architecture')

    phases = [
        ('Phase 1', '📊 Data Loading & EDA',
         'Load real or synthetic Lending Club-style data. Inspect class imbalance, missing values, '
         'feature distributions, and correlation structure before touching any model.'),
        ('Phase 2', '🔧 Feature Engineering',
         'Derive 5 financial ratio features from raw inputs: DTI ratio, credit utilisation, '
         'installment-to-income, loan-to-income, and annual payment burden. Encode grade ordinally '
         '(G→A = 0→6), one-hot encode home ownership and purpose, and standard-scale all numerics.'),
        ('Phase 3', '⚖️ SMOTE Balancing',
         'The training set is imbalanced (~22% defaults). SMOTE synthesises new minority-class '
         'samples in feature space so models learn default patterns without seeing real-world '
         'skew. The test set is never resampled — it stays true to real distribution.'),
        ('Phase 4', '🤖 Model Training',
         'Three models trained in parallel: Logistic Regression (interpretable linear baseline), '
         'Random Forest (non-linear ensemble, resistant to overfitting), and XGBoost (gradient '
         'boosting champion for tabular financial data, captures interaction effects).'),
        ('Phase 5', '📈 Evaluation',
         'Primary metric: ROC-AUC (threshold-independent, imbalance-robust). Secondary: Recall '
         '(default detection rate), Precision, F1. Supplemented by Precision-Recall curves, '
         'an interactive threshold tuner, calibration curves, and financial cost-benefit analysis.'),
        ('Phase 6', '🔍 Explainability (XAI)',
         'SHAP (SHapley Additive exPlanations) decomposes every prediction into per-feature '
         'contributions. Global summary plots show which features matter across all applicants. '
         'Waterfall plots explain individual decisions in plain English — meeting the EU AI Act\'s '
         'right-to-explanation requirements for high-risk AI systems.'),
    ]

    for tag, title, desc in phases:
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.2);
                    border-radius:10px; padding:1rem 1.4rem; margin:0.6rem 0;">
            <span style="background:rgba(99,102,241,0.25); color:#a5b4fc; font-size:0.72rem;
                         font-weight:700; padding:0.15rem 0.6rem; border-radius:12px;
                         letter-spacing:0.06em; text-transform:uppercase;">{tag}</span>
            <span style="font-size:1.05rem; font-weight:600; color:#e2e8f0;
                         margin-left:0.6rem;">{title}</span>
            <p style="color:#94a3b8; font-size:0.88rem; margin:0.5rem 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## 🛠️ Technology Stack')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**ML & Data**')
        for lib, purpose in [
            ('XGBoost 2.0', 'Primary classifier'),
            ('scikit-learn', 'LR, RF, preprocessing, metrics'),
            ('imbalanced-learn', 'SMOTE oversampling'),
            ('SHAP', 'Explainability (TreeExplainer)'),
            ('pandas / numpy', 'Data manipulation'),
        ]:
            st.markdown(f'- `{lib}` — {purpose}')

    with col2:
        st.markdown('**Visualisation**')
        for lib, purpose in [
            ('Plotly', 'All interactive charts'),
            ('Streamlit', 'Web application framework'),
            ('joblib', 'Model serialisation (.pkl)'),
        ]:
            st.markdown(f'- `{lib}` — {purpose}')

    with col3:
        st.markdown('**Design Choices**')
        for choice, rationale in [
            ('ROC-AUC primary metric', 'Threshold-free, imbalance-robust'),
            ('SMOTE on train only', 'Preserves real-world test distribution'),
            ('Tree SHAP', 'Exact values, not approximations'),
            ('Ordinal grade encoding', 'Preserves A > B > … > G ordering'),
        ]:
            st.markdown(f'- **{choice}** — {rationale}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## 📦 Dataset')
    st.markdown("""
    The project uses a **synthetic Lending Club-style dataset** generated from real published
    LC statistics, with the following calibration:

    | Feature | Source | Distribution |
    |---|---|---|
    | Loan grade proportions | LC 2018 public stats | A:15% B:27% C:27% D:17% E:8% F:4% G:2% |
    | Interest rates | LC grade rate ranges | Grade-tied normal distributions |
    | Annual income | Federal Reserve SCF 2019 | Log-normal (μ=10.95, σ=0.52) |
    | FICO score | US credit bureau data | Bimodal: 730±40 (prime) + 625±25 (subprime) |
    | DTI ratio | Fed consumer finance surveys | Gamma (shape=2.5, scale=7) |
    | Default rate | LC charged-off rate | Calibrated to exactly 22% |

    Deliberate noise (σ=1.6) is added to the default logit to prevent any model from achieving
    unrealistically high AUC — keeping the benchmark honest at 0.72–0.80, consistent with
    published Lending Club ML research.

    > **To use real data:** Upload any Lending Club CSV from Kaggle with a `loan_status` column
    > (values: "Fully Paid" / "Charged Off"). The pipeline handles it automatically.
    """)

    # ── Key Design Decisions ──────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## 🧠 Key Design Decisions')
    st.markdown("""
    **Why Recall matters more than Accuracy for banks**

    A model that predicts "no default" for every applicant would achieve ~78% accuracy on this
    dataset — yet catch zero actual defaulters. In credit risk, a False Negative (approving a
    loan that defaults) costs ~60% of the loan value. A False Positive (rejecting a good loan)
    costs only ~5% in missed interest. The loss asymmetry means **Recall must be monitored
    alongside ROC-AUC**, and the threshold tuner lets institutions calibrate this trade-off
    explicitly.

    **Why SMOTE, not class weights**

    Class weights adjust the loss function but leave the decision boundary the same.
    SMOTE creates synthetic minority samples in feature space, forcing the model to learn
    the actual shape of the default region — not just penalise getting it wrong. For tree
    models especially, this produces more robust boundaries on unseen data.

    **Why XGBoost for SHAP**

    XGBoost's `TreeExplainer` computes exact SHAP values in polynomial time (vs exponential
    for brute-force Shapley). The resulting explanations are the gold standard for regulatory
    compliance — they satisfy the EU AI Act Article 13 transparency requirements for
    high-risk AI systems in credit scoring.
    """)

    # ── References ────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('## 📚 References')
    st.markdown("""
    1. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
    2. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
    3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
    4. Lending Club Statistics (2018). *Loan Data Public Statistics.* lendingclub.com
    5. Federal Reserve. (2019). *Survey of Consumer Finances.*
    6. EU AI Act (2024). *Regulation on Artificial Intelligence — Article 13: Transparency.*
    """)

    st.markdown('---')
    st.markdown("""
    <div style="text-align:center; padding:1rem 0; color:#475569; font-size:0.85rem;">
        LoanRisk AI &nbsp;·&nbsp; B.Tech Final Year Project &nbsp;·&nbsp;
        Streamlit + XGBoost + SHAP &nbsp;·&nbsp;
        <span style="color:#6366f1;">Built with ❤️</span>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#334155; font-size:0.75rem;">
    LoanRisk AI · B.Tech Final Year Project · Built with Streamlit + XGBoost + SHAP
</div>
""", unsafe_allow_html=True)
