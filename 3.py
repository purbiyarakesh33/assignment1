"""
Build XGBoost Feature Matrix
=============================
186 sparse original features + 1 recon_error + 1 entity_type = 188 features
+ 1 label = 189 columns total

Saves:
  xgb_matrix.npz          → X (188 features) and y
  xgb_feature_names.pkl   → column names in order

Run:  python build_xgb_matrix.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

BASE     = r'D:\new approch'
SAVE_DIR = os.path.join(BASE, 'ae_models')

# ── Helper: safe label encode (works on all pandas/numpy versions) ────────────
def label_encode_categoricals(X):
    """Encode string/object columns without using select_dtypes(include='str')"""
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == 'str':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X

# ── Loaders ───────────────────────────────────────────────────────────────────

DS1_COLS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]
DS2_COLS = [
    'buisness_year', 'document_create_date', 'document_create_date.1',
    'due_in_date', 'total_open_amount', 'baseline_create_date',
    'cust_payment_terms', 'posting_dayofyear', 'posting_year'
]
DS3_COLS = [f'attr{i}' for i in range(1, 65)]
DS5_COLS = [
    'Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
    'Checking account', 'Credit amount', 'Duration', 'Purpose'
]
DS4_COLS = None  # set dynamically after loading


def load_ds1():
    df = pd.read_csv(os.path.join(BASE, 'train.csv'))
    df = df.drop(columns=['Id'], errors='ignore')
    y  = df['SeriousDlqin2yrs'].values.astype(int)
    X  = df.drop(columns=['SeriousDlqin2yrs'])
    X  = X.apply(pd.to_numeric, errors='coerce')
    X  = X.fillna(X.median())
    return X[DS1_COLS].values.astype(np.float32), y


def load_ds2():
    df  = pd.read_csv(os.path.join(BASE, 'dataset.csv'))
    y   = df['isOpen'].values.astype(int)
    drop_cols = ['isOpen', 'business_code', 'cust_number', 'name_customer',
                 'clear_date', 'doc_id', 'invoice_currency', 'document type',
                 'posting_id', 'invoice_id', 'area_business']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X['posting_date']      = pd.to_datetime(X['posting_date'], errors='coerce')
    X['posting_dayofyear'] = X['posting_date'].dt.dayofyear
    X['posting_year']      = X['posting_date'].dt.year
    X = X.drop(columns=['posting_date'])
    le = LabelEncoder()
    X['cust_payment_terms'] = le.fit_transform(X['cust_payment_terms'].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    return X[DS2_COLS].values.astype(np.float32), y


def load_ds3():
    frames = []
    for yr in range(1, 6):
        data, _ = arff.loadarff(os.path.join(BASE, f'{yr}year.arff'))
        df = pd.DataFrame(data)
        df.columns = [c.lower() for c in df.columns]
        df['class'] = df['class'].apply(
            lambda x: int(x.decode() if isinstance(x, bytes) else x))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    y  = df['class'].values.astype(int)
    X  = df.drop(columns=['class'])
    X  = X.apply(pd.to_numeric, errors='coerce')
    X  = X.fillna(X.median())
    X.columns = [f'attr{i}' for i in range(1, X.shape[1] + 1)]
    return X[DS3_COLS].values.astype(np.float32), y


def load_ds4():
    global DS4_COLS
    df = pd.read_csv(os.path.join(BASE, 'data.csv'))
    df.columns = [c.strip() for c in df.columns]
    y  = df['Bankrupt?'].values.astype(int)
    X  = df.drop(columns=['Bankrupt?'])
    X  = X.apply(pd.to_numeric, errors='coerce')
    stds = X.std()
    X  = X.drop(columns=stds[stds == 0].index.tolist())
    X  = X.fillna(X.median())
    DS4_COLS = X.columns.tolist()
    return X.values.astype(np.float32), y


def load_ds5():
    df = pd.read_csv(os.path.join(BASE, 'german_credit_data.csv'))
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    y  = (df['Risk'] == 'bad').astype(int).values
    X  = df.drop(columns=['Risk'])
    X  = label_encode_categoricals(X)
    X  = X.apply(pd.to_numeric, errors='coerce')
    X  = X.fillna(X.median())
    return X[DS5_COLS].values.astype(np.float32), y


LOADERS = {
    'ds1_credit':  (load_ds1, DS1_COLS),
    'ds2_invoice': (load_ds2, DS2_COLS),
    'ds3_polish':  (load_ds3, DS3_COLS),
    'ds4_taiwan':  (load_ds4, None),
    'ds5_german':  (load_ds5, DS5_COLS),
}

# ── Load everything ───────────────────────────────────────────────────────────

print("Loading datasets and reconstruction errors...")
loaded = {}
for name, (loader_fn, cols) in LOADERS.items():
    X_np, y = loader_fn()
    errors   = np.load(os.path.join(SAVE_DIR, f'{name}_errors.npy'))
    feat_cols = DS4_COLS if name == 'ds4_taiwan' else cols
    assert len(X_np) == len(errors) == len(y), \
        f"{name}: row count mismatch X={len(X_np)} err={len(errors)} y={len(y)}"
    loaded[name] = {'X': X_np, 'y': y, 'errors': errors, 'cols': feat_cols}
    print(f"  {name}: {X_np.shape} | default={y.mean():.1%}")

# ── Build column index map ────────────────────────────────────────────────────

all_feature_cols = []
col_offset = 0
dataset_col_ranges = {}

for name, d in loaded.items():
    prefixed = [f'{name}__{c}' for c in d['cols']]
    all_feature_cols.extend(prefixed)
    dataset_col_ranges[name] = (col_offset, col_offset + len(d['cols']))
    col_offset += len(d['cols'])

all_feature_cols += ['recon_error', 'entity_type']
recon_idx  = col_offset
entity_idx = col_offset + 1
n_features = col_offset + 2

print(f"\nMatrix columns: {n_features} features")
print(f"  {col_offset} original sparse + 1 recon_error + 1 entity_type")

# ── Build sparse matrix ───────────────────────────────────────────────────────

print("\nBuilding sparse combined matrix...")
rows_list, labels_list = [], []

for i, (name, d) in enumerate(loaded.items()):
    X_np   = d['X']
    y      = d['y']
    errors = d['errors'].astype(np.float32)
    start, end = dataset_col_ranges[name]
    n_rows = len(y)

    block = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    block[:, start:end]  = X_np
    block[:, recon_idx]  = errors
    block[:, entity_idx] = i

    rows_list.append(block)
    labels_list.append(y)
    print(f"  {name}: {n_rows} rows | cols {start}-{end-1} filled | rest NaN")

X_combined = np.vstack(rows_list)
y_combined = np.concatenate(labels_list)

print(f"\nFinal shape    : {X_combined.shape}")
print(f"Labels         : 0={(y_combined==0).sum()}  1={(y_combined==1).sum()}")
print(f"recon_error NaN: {np.isnan(X_combined[:, recon_idx]).sum()} (must be 0)")
print(f"entity_type NaN: {np.isnan(X_combined[:, entity_idx]).sum()} (must be 0)")

# ── Save ──────────────────────────────────────────────────────────────────────

matrix_path = os.path.join(BASE, 'xgb_matrix.npz')
names_path  = os.path.join(BASE, 'xgb_feature_names.pkl')

np.savez_compressed(matrix_path, X=X_combined, y=y_combined)
with open(names_path, 'wb') as f:
    pickle.dump(all_feature_cols, f)

print(f"\nSaved → {matrix_path}")
print(f"Saved → {names_path}")
print("Next: run train_xgboost.py")
