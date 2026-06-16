import streamlit as st
import pandas as pd
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import RiskEngine

st.set_page_config(
    page_title='Distributor Credit Risk',
    layout='wide'
)

st.markdown("""
<style>
.risk-high {
    background-color: #FCEBEB;
    color: #A32D2D;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}
.risk-low {
    background-color: #EAF3DE;
    color: #3B6D11;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}
.risk-insuff {
    background-color: #FAEEDA;
    color: #854F0B;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}
.metric-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return RiskEngine(base_dir=os.path.dirname(os.path.abspath(__file__)))


engine = load_engine()

DATASET_FIELDS = {
    'DS1 · Give Me Some Credit': [
        'RevolvingUtilizationOfUnsecuredLines', 'age',
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
        'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
    ],
    'DS2 · AR Invoices': [
        'total_open_amount', 'posting_dayofyear', 'posting_year',
        'cust_payment_terms', 'due_in_date', 'invoice_id',
        'converted_usd', 'document_date'
    ],
    'DS3 · Polish Bankruptcy': [
        'Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5',
        'Attr6', 'Attr7', 'Attr8', 'Attr9', 'Attr10'
    ],
    'DS4 · Taiwan Bankruptcy': [
        'ROA(C) before interest and depreciation before interest',
        'Operating Gross Margin', 'Realized Sales Gross Margin',
        'Operating Profit Rate', 'Pre-tax net Interest Rate',
        'After-tax net Interest Rate', 'Non-industry income and expenditure/revenue',
        'Continuous interest rate (after tax)'
    ],
    'DS5 · German Credit': [
        'duration', 'credit_amount', 'installment_rate',
        'present_residence', 'age', 'existing_credits',
        'num_dependents'
    ],
}

RANDOM_RANGES = {
    'RevolvingUtilizationOfUnsecuredLines': (0.0, 1.5, False),
    'age': (18, 85, True),
    'NumberOfTime30-59DaysPastDueNotWorse': (0, 10, True),
    'DebtRatio': (0.0, 2.0, False),
    'MonthlyIncome': (1000, 20000, True),
    'NumberOfOpenCreditLinesAndLoans': (0, 30, True),
    'NumberOfTimes90DaysLate': (0, 10, True),
    'NumberRealEstateLoansOrLines': (0, 5, True),
    'NumberOfTime60-89DaysPastDueNotWorse': (0, 10, True),
    'NumberOfDependents': (0, 8, True),
    'total_open_amount': (100, 500000, False),
    'posting_dayofyear': (1, 365, True),
    'posting_year': (2018, 2024, True),
    'cust_payment_terms': (0, 90, True),
    'due_in_date': (1, 120, True),
    'invoice_id': (100000, 999999, True),
    'converted_usd': (100, 500000, False),
    'document_date': (1, 365, True),
    'Attr1': (0.0, 5.0, False),
    'Attr2': (0.0, 5.0, False),
    'Attr3': (0.0, 5.0, False),
    'Attr4': (0.0, 5.0, False),
    'Attr5': (0.0, 5.0, False),
    'Attr6': (0.0, 5.0, False),
    'Attr7': (0.0, 5.0, False),
    'Attr8': (0.0, 5.0, False),
    'Attr9': (0.0, 5.0, False),
    'Attr10': (0.0, 5.0, False),
    'ROA(C) before interest and depreciation before interest': (-0.5, 0.5, False),
    'Operating Gross Margin': (-0.5, 1.0, False),
    'Realized Sales Gross Margin': (-0.5, 1.0, False),
    'Operating Profit Rate': (-0.5, 1.0, False),
    'Pre-tax net Interest Rate': (-0.5, 0.5, False),
    'After-tax net Interest Rate': (-0.5, 0.5, False),
    'Non-industry income and expenditure/revenue': (-0.5, 0.5, False),
    'Continuous interest rate (after tax)': (-0.5, 0.5, False),
    'duration': (6, 72, True),
    'credit_amount': (500, 20000, True),
    'installment_rate': (1, 4, True),
    'present_residence': (1, 4, True),
    'existing_credits': (1, 4, True),
    'num_dependents': (1, 2, True),
}


def generate_random_sample(fields):
    for field in fields:
        key = f'field_{field}'
        lo, hi, is_int = RANDOM_RANGES.get(field, (0.0, 1.0, False))
        if is_int:
            st.session_state[key] = str(random.randint(int(lo), int(hi)))
        else:
            st.session_state[key] = str(round(random.uniform(lo, hi), 4))


def verdict_badge(verdict, score):
    if verdict == 'INSUFFICIENT_DATA':
        return '<span class="risk-insuff">Insufficient Data</span>'
    elif verdict == 'HIGH_RISK':
        return f'<span class="risk-high">HIGH RISK — {score:.1%}</span>'
    else:
        return f'<span class="risk-low">LOW RISK — {score:.1%}</span>'


def score_bar(score):
    if score is None:
        return ''
    pct = int(score * 100)
    color = '#E24B4A' if score >= 0.3256 else '#639922'
    return f"""
    <div style="background:#eee;border-radius:4px;height:10px;width:100%">
        <div style="width:{pct}%;background:{color};height:10px;border-radius:4px"></div>
    </div>
    <small style="color:#888">{pct}% default probability</small>
    """


st.title('Distributor Credit Risk Assessment')
st.caption('Nestle · 5-dataset ensemble · XGBoost + Autoencoder anomaly detection · Threshold: 0.3256')

tab1, tab2, tab3 = st.tabs(['Manual Entry', 'Batch CSV', 'Model Info'])

# ── TAB 1: Manual Entry ───────────────────────────────────────────────────────
with tab1:
    st.subheader('Single distributor assessment')

    ds_choice = st.selectbox('Select distributor type', list(DATASET_FIELDS.keys()))
    fields = DATASET_FIELDS[ds_choice]

    # Clear fields when dataset changes
    if st.session_state.get('last_ds') != ds_choice:
        for field in [f for fs in DATASET_FIELDS.values() for f in fs]:
            st.session_state.pop(f'field_{field}', None)
        st.session_state['last_ds'] = ds_choice

    if st.button('Load Random Sample'):
        generate_random_sample(fields)

    st.markdown('Enter feature values — leave blank to use training medians')

    cols_per_row = 3
    field_values = {}
    rows = [fields[i:i + cols_per_row] for i in range(0, len(fields), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, field in zip(cols, row):
            val = col.text_input(field, key=f'field_{field}')
            if val.strip() != '':
                try:
                    field_values[field] = float(val)
                except ValueError:
                    field_values[field] = val

    st.markdown('')
    if st.button('Assess Risk', type='primary'):
        if len(field_values) == 0:
            st.warning('Please enter at least one feature value.')
        else:
            with st.spinner('Running assessment...'):
                result = engine.predict(field_values)

            st.markdown('---')
            c1, c2, c3, c4 = st.columns(4)

            score = result['risk_score']
            verdict = result['verdict']

            with c1:
                st.metric('Risk Score', f"{score:.1%}" if score else 'N/A')
            with c2:
                st.metric('Entity Detected', result['entity_type'] or 'None')
            with c3:
                st.metric('Feature Overlap', result['overlap'])
            with c4:
                st.metric('Recon Error', f"{result['recon_error']:.5f}" if result.get('recon_error') else 'N/A')

            st.markdown(verdict_badge(verdict, score if score else 0), unsafe_allow_html=True)
            st.markdown(score_bar(score), unsafe_allow_html=True)

            if verdict == 'HIGH_RISK':
                st.error('This distributor shows elevated default risk. Consider additional credit checks before extending payment terms.')
            elif verdict == 'LOW_RISK':
                st.success('This distributor appears low risk based on available features.')
            else:
                st.warning(result['message'])

# ── TAB 2: Batch CSV ──────────────────────────────────────────────────────────
with tab2:
    st.subheader('Batch assessment')
    st.caption('Upload any CSV — entity type is auto-detected per row, unknown columns are ignored.')

    uploaded = st.file_uploader('Drop your CSV here', type=['csv'])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.info(f'Loaded {len(df)} rows · {len(df.columns)} columns')

        if st.button('Run Batch Assessment', type='primary'):
            progress = st.progress(0, text='Assessing distributors...')
            results = []

            for i, (_, row) in enumerate(df.iterrows()):
                r = engine.predict(row.to_dict())
                results.append(r)
                if i % 100 == 0:
                    progress.progress(min(i / len(df), 1.0), text=f'Processing row {i}/{len(df)}...')

            progress.progress(1.0, text='Done!')

            out = df.copy()
            out['risk_score'] = [r['risk_score'] for r in results]
            out['verdict'] = [r['verdict'] for r in results]
            out['entity_type'] = [r['entity_type'] for r in results]
            out['recon_error'] = [r.get('recon_error') for r in results]
            out['message'] = [r['message'] for r in results]

            high = sum(1 for r in results if r['verdict'] == 'HIGH_RISK')
            low = sum(1 for r in results if r['verdict'] == 'LOW_RISK')
            scores = [r['risk_score'] for r in results if r['risk_score'] is not None]
            avg_scr = np.mean(scores) if scores else 0

            st.markdown('---')
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Total Rows', len(df))
            m2.metric('High Risk', high)
            m3.metric('Low Risk', low)
            m4.metric('Avg Score', f'{avg_scr:.1%}')

            st.markdown('### Results preview')

            def color_verdict(val):
                if val == 'HIGH_RISK':
                    return 'background-color: #FCEBEB; color: #A32D2D'
                elif val == 'LOW_RISK':
                    return 'background-color: #EAF3DE; color: #3B6D11'
                return 'background-color: #FAEEDA; color: #854F0B'

            display_cols = list(df.columns[:4]) + ['risk_score', 'verdict', 'entity_type']
            styled = out[display_cols].head(100).style.applymap(
                color_verdict, subset=['verdict']
            ).format({'risk_score': '{:.1%}'}, na_rep='—')

            st.dataframe(styled, use_container_width=True)

            if len(out) > 100:
                st.caption(f'Showing first 100 of {len(out)} rows.')

            csv_out = out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download full results as CSV',
                data=csv_out,
                file_name='risk_assessment_results.csv',
                mime='text/csv',
            )

# ── TAB 3: Model Info ─────────────────────────────────────────────────────────
with tab3:
    st.subheader('Model performance')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Test AUC', '0.9604')
    c2.metric('Default Recall', '0.84')
    c3.metric('Default F1', '0.66')
    c4.metric('Macro F1', '0.81')

    st.markdown('### AUC by entity')
    auc_data = {
        'Dataset': ['DS1 · Give Me Some Credit', 'DS2 · AR Invoices',
                    'DS3 · Polish Bankruptcy', 'DS4 · Taiwan Bankruptcy',
                    'DS5 · German Credit'],
        'AUC': [0.86, 1.00, 0.97, 0.94, 0.78],
        'Rows': [104805, 50000, 43405, 6819, 1000],
        'Default%': ['6.6%', '20.0%', '4.8%', '3.2%', '30.0%'],
    }
    st.dataframe(pd.DataFrame(auc_data), use_container_width=True, hide_index=True)

    st.markdown('### Known limitations')
    st.warning('''
- DS2 AUC = 1.00 is suspiciously perfect — date features are very strong predictors
- DS5 only 1,000 rows from 1994 — not representative of modern distributors
- All datasets are public benchmarks, not real Nestle distributor data
- No temporal validation performed
- Threshold tuned on training data (0.3256) — held-out validation recommended
    ''')

    st.markdown('### Architecture')
    st.info('''
5 separate PyTorch Autoencoders, each trained only on non-default samples.
Reconstruction error signals how "abnormal" a transaction looks.
Combined 188-feature sparse matrix (186 original + recon error + entity type)
fed to XGBoost, which handles NaN natively.
Both AEs and XGBoost tuned via Optuna (25 and 50 trials respectively).
    ''')
