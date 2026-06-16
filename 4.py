"""
XGBoost Training on Combined Sparse Matrix
============================================
- Loads xgb_matrix.npz
- Train/test split (stratified)
- Optuna TPE hyperparameter tuning with cross-validation
- Final model trained on full train set
- Saves model + evaluation

Run:  python train_xgboost.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = r'C:\Users\Admin\OneDrive\Desktop\yes'
MATRIX     = os.path.join(BASE, 'xgb_matrix.npz')
NAMES      = os.path.join(BASE, 'xgb_feature_names.pkl')
MODEL_PATH = os.path.join(BASE, 'xgb_final_model.json')

# ── Config ────────────────────────────────────────────────────────────────────
XGB_TRIALS  = 50       # Optuna trials
CV_FOLDS    = 5        # stratified k-fold inside each trial
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading matrix...")
data = np.load(MATRIX)
X    = data['X']    # (206029, 188) — NaN for missing entity features (XGBoost handles natively)
y    = data['y']

with open(NAMES, 'rb') as f:
    feature_names = pickle.load(f)

print(f"Shape : {X.shape}")
print(f"Labels: 0={(y==0).sum()}  1={(y==1).sum()}  imbalance ratio={((y==0).sum()/(y==1).sum()):.1f}x")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── Optuna tuning with CV ─────────────────────────────────────────────────────
print(f"\nTuning XGBoost ({XGB_TRIALS} trials, {CV_FOLDS}-fold CV each)...")

def objective(trial):
    params = {
        # Tree structure
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),

        # Sampling
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.3, 1.0),

        # Learning
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),

        # Regularization
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        # Fixed
        'scale_pos_weight': scale_pos_weight,
        'tree_method':      'hist',
        'device':           'cuda',
        'eval_metric':      'auc',
        'random_state':     RANDOM_SEED,
        'verbosity':        0,
    }

    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs   = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=True)

print(f"\nBest CV AUC : {study.best_value:.4f}")
print(f"Best params : {study.best_params}")

# ── Train final model on full train set ───────────────────────────────────────
print("\nTraining final model on full train set...")

best = study.best_params
final_params = {
    'max_depth':         best['max_depth'],
    'min_child_weight':  best['min_child_weight'],
    'gamma':             best['gamma'],
    'subsample':         best['subsample'],
    'colsample_bytree':  best['colsample_bytree'],
    'colsample_bylevel': best['colsample_bylevel'],
    'learning_rate':     best['learning_rate'],
    'n_estimators':      best['n_estimators'],
    'reg_alpha':         best['reg_alpha'],
    'reg_lambda':        best['reg_lambda'],
    'scale_pos_weight':  scale_pos_weight,
    'tree_method':       'hist',
    'device':            'cuda',
    'eval_metric':       'auc',
    'random_state':      RANDOM_SEED,
    'verbosity':         1,
    'early_stopping_rounds': 30,
}

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=10)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION ON HELD-OUT TEST SET")
print("="*60)

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred       = final_model.predict(X_test)
test_auc     = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest AUC : {test_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── AUC per entity type ───────────────────────────────────────────────────────
print("\nAUC per entity type:")
entity_col_idx = -1  # last feature column is entity_type
entity_names   = ['ds1_credit', 'ds2_invoice', 'ds3_polish', 'ds4_taiwan', 'ds5_german']

for i, name in enumerate(entity_names):
    mask = X_test[:, entity_col_idx] == i
    if mask.sum() > 10:
        auc = roc_auc_score(y_test[mask], y_pred_proba[mask])
        print(f"  {name:<20} AUC={auc:.4f}  n={mask.sum()}")

# ── Feature importance ────────────────────────────────────────────────────────
print("\nTop 20 features by importance:")
importances = final_model.feature_importances_
top20_idx   = np.argsort(importances)[::-1][:20]
for idx in top20_idx:
    print(f"  {feature_names[idx]:<50} {importances[idx]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
final_model.save_model(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
print("Done.")