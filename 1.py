"""
Distributor Credit Risk — Autoencoder Training
================================================
5 Autoencoders, one per dataset / entity type.
Each AE trained ONLY on non-default (legit) samples.
GPU accelerated (RTX 3050).

Datasets
--------
DS1  → train.csv               (Give Me Some Credit)
DS2  → dataset.csv             (AR Payment / invoice)
DS3  → 1year..5year.arff       (Polish bankruptcy — merged)
DS4  → data.csv                (Taiwan bankruptcy)
DS5  → german_credit_data.csv  (German credit)

Run:  python train_autoencoders.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = r'D:\new approch'       # ← change if needed
SAVE_DIR = os.path.join(BASE, 'ae_models')
os.makedirs(SAVE_DIR, exist_ok=True)

DATA = {
    'ds1_credit':  os.path.join(BASE, 'train.csv'),
    'ds2_invoice': os.path.join(BASE, 'dataset.csv'),
    'ds3_polish':  [os.path.join(BASE, f'{y}year.arff') for y in range(1, 6)],
    'ds4_taiwan':  os.path.join(BASE, 'data.csv'),
    'ds5_german':  os.path.join(BASE, 'german_credit_data.csv'),
}

# ── Config ────────────────────────────────────────────────────────────────────
AE_TRIALS    = 25
TUNE_EPOCHS  = 30
FINAL_EPOCHS = 60
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ds1():
    """Give Me Some Credit — SeriousDlqin2yrs=1 → default"""
    df = pd.read_csv(DATA['ds1_credit'])
    df = df.drop(columns=['Id'], errors='ignore')
    y  = df['SeriousDlqin2yrs'].values.astype(int)
    X  = df.drop(columns=['SeriousDlqin2yrs'])
    return X, y


def load_ds2():
    """
    AR Payment — isOpen=1 → invoice unpaid → default.
    Special handling:
      - posting_date is a date string → extract day-of-year as numeric
      - cust_payment_terms is a code string → label encode
      - area_business is 100% null → drop
      - document type already dropped (near-constant: 99.99% RV)
    """
    df = pd.read_csv(DATA['ds2_invoice'])
    y  = df['isOpen'].values.astype(int)

    drop_cols = ['isOpen', 'business_code', 'cust_number', 'name_customer',
                 'clear_date', 'doc_id', 'invoice_currency', 'document type',
                 'posting_id', 'invoice_id', 'area_business']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # posting_date → numeric day-of-year
    X['posting_date'] = pd.to_datetime(X['posting_date'], errors='coerce')
    X['posting_dayofyear'] = X['posting_date'].dt.dayofyear
    X['posting_year']      = X['posting_date'].dt.year
    X = X.drop(columns=['posting_date'])

    # cust_payment_terms → label encode
    le = LabelEncoder()
    X['cust_payment_terms'] = le.fit_transform(X['cust_payment_terms'].astype(str))

    return X, y


def load_ds3():
    """Polish bankruptcy — merge 5 year files, class=1 → bankrupt"""
    frames = []
    for path in DATA['ds3_polish']:
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        df.columns = [c.lower() for c in df.columns]
        df['class'] = df['class'].apply(
            lambda x: int(x.decode() if isinstance(x, bytes) else x))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    y  = df['class'].values.astype(int)
    X  = df.drop(columns=['class'])
    return X, y


def load_ds4():
    """Taiwan bankruptcy — Bankrupt?=1 → bankrupt"""
    df = pd.read_csv(DATA['ds4_taiwan'])
    df.columns = [c.strip() for c in df.columns]
    y  = df['Bankrupt?'].values.astype(int)
    X  = df.drop(columns=['Bankrupt?'])
    return X, y


def load_ds5():
    """
    German credit — Risk='bad' → 1.
    Has categorical columns that need LabelEncoding:
    Sex, Housing, Saving accounts, Checking account, Purpose
    Also has NaN in Saving accounts and Checking account.
    """
    df = pd.read_csv(DATA['ds5_german'])
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    y  = (df['Risk'] == 'bad').astype(int).values
    X  = df.drop(columns=['Risk'])
    return X, y


LOADERS = {
    'ds1_credit':  load_ds1,
    'ds2_invoice': load_ds2,
    'ds3_polish':  load_ds3,
    'ds4_taiwan':  load_ds4,
    'ds5_german':  load_ds5,
}

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X_df):
    """
    Robust preprocessing pipeline:
    1.  Label-encode any remaining string/object columns
    2.  Coerce everything to numeric
    3.  Drop columns that are 100% null
    4.  Drop columns that are >80% null (too sparse to impute reliably)
    5.  Median imputation — with fallback to 0 if median itself is NaN
    6.  Hard zero-fill for any remaining NaN (safety net)
    7.  Drop zero-variance columns (StandardScaler would produce NaN)
    8.  Clip extreme outliers (±10 std) to prevent exploding loss
    9.  StandardScaler
    10. Final hard assert — nothing bad reaches the autoencoder
    """
    X = X_df.copy()

    # ── Step 1: label encode remaining categoricals ───────────────────────────
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ── Step 2: coerce to numeric ─────────────────────────────────────────────
    X = X.apply(pd.to_numeric, errors='coerce')

    # ── Step 3: drop 100% null columns ───────────────────────────────────────
    all_null = [c for c in X.columns if X[c].isnull().all()]
    if all_null:
        print(f"    Dropping 100%-null cols  : {all_null}")
        X = X.drop(columns=all_null)

    # ── Step 4: drop >80% null columns ───────────────────────────────────────
    null_frac = X.isnull().mean()
    mostly_null = null_frac[null_frac > 0.80].index.tolist()
    if mostly_null:
        print(f"    Dropping >80%-null cols  : {mostly_null}")
        X = X.drop(columns=mostly_null)

    # ── Step 5: median imputation with safe fallback ──────────────────────────
    medians = X.median()
    medians = medians.fillna(0)   # ← KEY FIX: if median is NaN, use 0
    X = X.fillna(medians)

    # ── Step 6: hard zero-fill safety net ────────────────────────────────────
    still_null = X.isnull().sum().sum()
    if still_null > 0:
        print(f"    Hard zero-fill for {still_null} remaining NaN cells")
        X = X.fillna(0)

    # ── Step 7: drop zero-variance columns ───────────────────────────────────
    stds = X.std()
    zero_var = stds[stds == 0].index.tolist()
    if zero_var:
        print(f"    Dropping zero-variance cols: {zero_var}")
        X = X.drop(columns=zero_var)

    feature_names = X.columns.tolist()

    # ── Step 8: clip extreme outliers ────────────────────────────────────────
    # Values beyond ±10 std cause exploding gradients and NaN loss
    col_means = X.mean()
    col_stds  = X.std().replace(0, 1)
    X = X.clip(lower=col_means - 10 * col_stds,
                upper=col_means + 10 * col_stds,
                axis=1)

    # ── Step 9: StandardScaler ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 10: hard asserts — nothing bad reaches the AE ───────────────────
    nan_count = np.isnan(X_scaled).sum()
    inf_count = np.isinf(X_scaled).sum()
    if nan_count > 0 or inf_count > 0:
        # Replace any survivors with 0 and warn
        print(f"    ⚠ WARNING: {nan_count} NaN and {inf_count} Inf after scaling — force-zeroing")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    assert not np.isnan(X_scaled).any(), "NaN survived preprocessing!"
    assert not np.isinf(X_scaled).any(), "Inf survived preprocessing!"

    print(f"    Final shape: {X_scaled.shape} | NaN: 0 ✓ | Inf: 0 ✓")
    return X_scaled, scaler, medians, feature_names


# ── Autoencoder ───────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        enc_layers, prev = [], input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_loader(X_np, batch_size, shuffle=True):
    t = torch.FloatTensor(X_np).to(DEVICE)
    return DataLoader(TensorDataset(t), batch_size=batch_size,
                      shuffle=shuffle, drop_last=True)  # drop_last avoids BatchNorm1d crash on size-1 batch


def train_ae(X_np, input_dim, params, epochs, verbose=False):
    loader = make_loader(X_np, params['batch_size'])
    model  = Autoencoder(input_dim, params['hidden_dims'], params['dropout']).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(),
                               lr=params['lr'],
                               weight_decay=params['weight_decay'])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total = 0
        for (batch,) in loader:
            opt.zero_grad()
            out  = model(batch)
            loss = crit(out, batch)

            # ── NaN loss guard ────────────────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {ep} — "
                    "data has numerical issues or LR is too high.")

            loss.backward()
            # ── Gradient clipping — prevents exploding gradients ───────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()

        sched.step()
        avg = total / len(loader)

        if verbose and ep % 10 == 0:
            print(f"      Epoch {ep:>3}/{epochs}  loss={avg:.6f}")

    return model


def get_errors(model, X_np):
    model.eval()
    t = torch.FloatTensor(X_np).to(DEVICE)
    with torch.no_grad():
        recon  = model(t)
        errors = torch.mean((recon - t) ** 2, dim=1)
    return errors.cpu().numpy()


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_ae(X_normal, input_dim, n_trials):
    """Optuna TPE search for best AE hyperparams."""
    idx   = np.random.permutation(len(X_normal))
    split = int(0.8 * len(X_normal))
    X_tr  = X_normal[idx[:split]]
    X_val = X_normal[idx[split:]]

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)

        # First hidden layer: between input_dim//4 and input_dim//2
        first_h = trial.suggest_int('first_h',
                                     max(8, input_dim // 4),
                                     max(16, input_dim // 2))
        hidden_dims = [first_h]
        prev_h = first_h
        for i in range(1, n_layers):
            h = trial.suggest_int(f'h_{i}',
                                   max(4, prev_h // 4),
                                   max(8, prev_h // 2))
            hidden_dims.append(h)
            prev_h = h

        params = {
            'hidden_dims':  hidden_dims,
            'lr':           trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'dropout':      trial.suggest_float('dropout', 0.0, 0.3),
            'batch_size':   trial.suggest_categorical('batch_size', [64, 128, 256]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }

        # ── KEY FIX: catch RuntimeError so Optuna sees float('inf') ──────────
        # Without this, ALL trials fail with ValueError and study has no winner
        try:
            model   = train_ae(X_tr, input_dim, params, TUNE_EPOCHS, verbose=False)
            val_err = get_errors(model, X_val).mean()
            if np.isnan(val_err) or np.isinf(val_err):
                return float('inf')
            return float(val_err)
        except RuntimeError as e:
            print(f"      Trial skipped: {e}")
            return float('inf')

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Check at least one trial succeeded ───────────────────────────────────
    completed = [t for t in study.trials
                 if t.value is not None and t.value != float('inf')]
    if not completed:
        raise RuntimeError(
            "All Optuna trials failed! "
            "Check your data — possible extreme outliers or all-constant columns.")

    best     = study.best_params
    n_layers = best['n_layers']
    hidden_dims = [best['first_h']]
    for i in range(1, n_layers):
        hidden_dims.append(best[f'h_{i}'])

    best_params = {
        'hidden_dims':  hidden_dims,
        'lr':           best['lr'],
        'dropout':      best['dropout'],
        'batch_size':   best['batch_size'],
        'weight_decay': best['weight_decay'],
    }
    print(f"    Best val loss : {study.best_value:.6f}")
    print(f"    Best params   : {best_params}")
    return best_params


# ── Main loop ─────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("TRAINING 5 AUTOENCODERS")
print("="*65)

results = {}

for name, loader_fn in LOADERS.items():
    print(f"\n>>> {name.upper()}")

    ae_path     = os.path.join(SAVE_DIR, f'{name}_ae.pt')
    scaler_path = os.path.join(SAVE_DIR, f'{name}_scaler.pkl')
    meta_path   = os.path.join(SAVE_DIR, f'{name}_meta.pkl')
    errors_path = os.path.join(SAVE_DIR, f'{name}_errors.npy')
    labels_path = os.path.join(SAVE_DIR, f'{name}_labels.npy')

    # ── Skip if already done ──────────────────────────────────────────────────
    if all(os.path.exists(p) for p in [ae_path, errors_path, labels_path]):
        print("  Saved model found — skipping training.")
        results[name] = {
            'errors': np.load(errors_path),
            'labels': np.load(labels_path),
        }
        n  = len(results[name]['labels'])
        dr = results[name]['labels'].mean()
        print(f"  Loaded {n} rows | default rate: {dr:.1%}")
        continue

    # ── Load ──────────────────────────────────────────────────────────────────
    X_df, y = loader_fn()
    print(f"  Raw shape : {X_df.shape} | default rate: {y.mean():.1%}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_scaled, scaler, medians, feature_names = preprocess(X_df)
    input_dim = X_scaled.shape[1]

    # ── Null check after preprocessing ───────────────────────────────────────
    null_count = np.isnan(X_scaled).sum()
    print(f"  Null check after preprocessing: {null_count} nulls ✓")

    # ── AE trains ONLY on legit rows ──────────────────────────────────────────
    X_normal = X_scaled[y == 0]
    print(f"  Rows: {len(X_scaled)} | Features: {input_dim} | "
          f"Default rate: {y.mean():.1%}")
    print(f"  Normal samples for AE: {len(X_normal)}")

    # ── Tune ──────────────────────────────────────────────────────────────────
    print(f"  Tuning ({AE_TRIALS} trials)...")
    best_params = tune_ae(X_normal, input_dim, AE_TRIALS)

    # ── Train final AE ────────────────────────────────────────────────────────
    print(f"  Training final AE ({FINAL_EPOCHS} epochs)...")
    model = train_ae(X_normal, input_dim, best_params, FINAL_EPOCHS, verbose=True)

    # ── Reconstruction errors on ALL rows ─────────────────────────────────────
    errors = get_errors(model, X_scaled)
    legit_err   = errors[y == 0].mean()
    default_err = errors[y == 1].mean()
    status = '✓' if default_err > legit_err else '⚠ check'
    print(f"  Recon error — legit: {legit_err:.5f} | "
          f"default: {default_err:.5f}  {status}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), ae_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'medians':       medians,
            'feature_names': feature_names,
            'best_params':   best_params,
            'input_dim':     input_dim,
        }, f)
    np.save(errors_path, errors)
    np.save(labels_path, y)
    print(f"  Saved → {SAVE_DIR}")

    results[name] = {'errors': errors, 'labels': y}

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
total = 0
for name, res in results.items():
    n   = len(res['labels'])
    dr  = res['labels'].mean()
    e0  = res['errors'][res['labels'] == 0].mean()
    e1  = res['errors'][res['labels'] == 1].mean()
    total += n
    print(f"  {name:<20} n={n:<7} default={dr:.1%}  "
          f"err_legit={e0:.5f}  err_default={e1:.5f}")
print(f"\n  Total rows: {total}")
print(f"\nAll models saved to: {SAVE_DIR}")
print("Next: run build_xgboost_matrix.py")