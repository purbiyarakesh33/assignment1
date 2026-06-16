"""
Train Autoencoders for DS1 and DS2 only.
DS3, DS4, DS5 already saved — this script skips them.

DS1 → train.csv            (Give Me Some Credit)
DS2 → dataset.csv          (AR Payment / invoice)

Run:  python train_ds1_ds2.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     =r'C:\Users\Admin\OneDrive\Desktop\yes'       # ← change if needed
SAVE_DIR = os.path.join(BASE, 'ae_models')
os.makedirs(SAVE_DIR, exist_ok=True)

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
    df = pd.read_csv(os.path.join(BASE, 'train.csv'))
    df = df.drop(columns=['Id'], errors='ignore')
    y  = df['SeriousDlqin2yrs'].values.astype(int)
    X  = df.drop(columns=['SeriousDlqin2yrs'])
    return X, y


def load_ds2():
    """
    AR Payment — isOpen=1 → unpaid → default.
    Fixes vs old code:
      - area_business dropped (100% null)
      - posting_date parsed to day-of-year + year (numeric)
      - cust_payment_terms label encoded (string codes)
      - document type dropped (99.99% single value — no info)
    """
    df = pd.read_csv(os.path.join(BASE, 'dataset.csv'))
    y  = df['isOpen'].values.astype(int)

    drop_cols = ['isOpen', 'business_code', 'cust_number', 'name_customer',
                 'clear_date', 'doc_id', 'invoice_currency', 'document type',
                 'posting_id', 'invoice_id', 'area_business']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # posting_date → two numeric features
    X = X.copy()
    X['posting_date']     = pd.to_datetime(X['posting_date'], errors='coerce')
    X['posting_dayofyear'] = X['posting_date'].dt.dayofyear
    X['posting_year']      = X['posting_date'].dt.year
    X = X.drop(columns=['posting_date'])

    # cust_payment_terms → label encode
    le = LabelEncoder()
    X['cust_payment_terms'] = le.fit_transform(X['cust_payment_terms'].astype(str))

    return X, y


LOADERS = {
    'ds1_credit':  load_ds1,
    'ds2_invoice': load_ds2,
}

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X_df):
    X = X_df.copy()

    # Label encode any remaining string columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop 100% null columns
    all_null = [c for c in X.columns if X[c].isnull().all()]
    if all_null:
        print(f"    Dropping 100%-null cols: {all_null}")
        X = X.drop(columns=all_null)

    # Median impute
    medians = X.median()
    X = X.fillna(medians)

    # Drop zero-variance columns (StandardScaler → NaN if std=0)
    stds     = X.std()
    zero_var = stds[stds == 0].index.tolist()
    if zero_var:
        print(f"    Dropping zero-variance cols: {zero_var}")
        X = X.drop(columns=zero_var)

    feature_names = X.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.isnan(X_scaled).any(), "NaN survived preprocessing!"
    assert not np.isinf(X_scaled).any(), "Inf survived preprocessing!"

    print(f"    Final shape: {X_scaled.shape} | NaN: 0 ✓ | Inf: 0 ✓")
    return X_scaled, scaler, medians, feature_names

# ── Autoencoder ───────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.LeakyReLU(0.1), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.LeakyReLU(0.1), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_loader(X_np, batch_size, shuffle=True):
    t = torch.FloatTensor(X_np).to(DEVICE)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)


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
            loss = crit(model(batch), batch)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {ep} — data issue.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(loader)
        if verbose and ep % 10 == 0:
            print(f"      Epoch {ep}/{epochs}  loss={avg:.6f}")

    return model


def get_errors(model, X_np):
    model.eval()
    t = torch.FloatTensor(X_np).to(DEVICE)
    with torch.no_grad():
        errors = torch.mean((model(t) - t) ** 2, dim=1)
    return errors.cpu().numpy()


def tune_ae(X_normal, input_dim, n_trials):
    idx   = np.random.permutation(len(X_normal))
    split = int(0.8 * len(X_normal))
    X_tr  = X_normal[idx[:split]]
    X_val = X_normal[idx[split:]]

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        first_h  = trial.suggest_int('first_h',
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

        model   = train_ae(X_tr, input_dim, params, TUNE_EPOCHS, verbose=False)
        val_err = get_errors(model, X_val).mean()
        return float(val_err)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best        = study.best_params
    n_layers    = best['n_layers']
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
print("TRAINING DS1 AND DS2 AUTOENCODERS")
print("="*65)

results = {}

for name, loader_fn in LOADERS.items():
    print(f"\n>>> {name.upper()}")

    ae_path     = os.path.join(SAVE_DIR, f'{name}_ae.pt')
    scaler_path = os.path.join(SAVE_DIR, f'{name}_scaler.pkl')
    meta_path   = os.path.join(SAVE_DIR, f'{name}_meta.pkl')
    errors_path = os.path.join(SAVE_DIR, f'{name}_errors.npy')
    labels_path = os.path.join(SAVE_DIR, f'{name}_labels.npy')

    # Skip if already done
    if all(os.path.exists(p) for p in [ae_path, scaler_path, meta_path, errors_path, labels_path]):
        print("  Already trained — skipping.")
        results[name] = {
            'errors': np.load(errors_path),
            'labels': np.load(labels_path),
        }
        continue

    # Load
    X_df, y = loader_fn()
    print(f"  Raw shape : {X_df.shape} | default rate: {y.mean():.1%}")

    # Preprocess
    X_scaled, scaler, medians, feature_names = preprocess(X_df)
    input_dim = X_scaled.shape[1]

    # Legit rows only for AE
    X_normal = X_scaled[y == 0]
    print(f"  Legit rows for AE: {len(X_normal)}")

    # Tune
    print(f"  Tuning ({AE_TRIALS} trials)...")
    best_params = tune_ae(X_normal, input_dim, AE_TRIALS)

    # Train final AE
    print(f"  Training final AE ({FINAL_EPOCHS} epochs)...")
    model = train_ae(X_normal, input_dim, best_params, FINAL_EPOCHS, verbose=True)

    # Reconstruction errors on ALL rows
    errors      = get_errors(model, X_scaled)
    legit_err   = errors[y == 0].mean()
    default_err = errors[y == 1].mean()
    status      = '✓' if default_err > legit_err else '⚠ check'
    print(f"  Recon error — legit: {legit_err:.5f} | default: {default_err:.5f}  {status}")

    # Save
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

print("\n" + "="*65)
print("DONE")
print("="*65)
for name, res in results.items():
    n   = len(res['labels'])
    dr  = res['labels'].mean()
    e0  = res['errors'][res['labels'] == 0].mean()
    e1  = res['errors'][res['labels'] == 1].mean()
    print(f"  {name:<20} n={n:<7} default={dr:.1%}  "
          f"err_legit={e0:.5f}  err_default={e1:.5f}")

print(f"\nAll 5 AEs now ready. Next: build_xgboost_matrix.py")