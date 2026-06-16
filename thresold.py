import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                             f1_score, confusion_matrix, classification_report)
import xgboost as xgb
warnings.filterwarnings('ignore')

BASE     = r'C:\Users\Admin\OneDrive\Desktop\yes'
AE_DIR   = os.path.join(BASE, 'ae_models')

# ── Load XGBoost + feature names ──────────────────────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE, 'xgb_final_model.json'))

with open(os.path.join(BASE, 'xgb_feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

print(f"Feature space: {len(feature_names)} columns")

# ── Discover entities ─────────────────────────────────────────────────────────
entities = {}
for fname in sorted(os.listdir(AE_DIR)):
    if fname.endswith('_meta.pkl'):
        name = fname.replace('_meta.pkl', '')
        entities[name] = name

print(f"Entities found: {list(entities.keys())}")

# ── Rebuild the full 188-col sparse matrix from saved files ───────────────────
all_rows   = []
all_labels = []

for idx, name in enumerate(entities.keys()):
    errors_path = os.path.join(AE_DIR, f'{name}_errors.npy')
    labels_path = os.path.join(AE_DIR, f'{name}_labels.npy')
    meta_path   = os.path.join(AE_DIR, f'{name}_meta.pkl')
    scaler_path = os.path.join(AE_DIR, f'{name}_scaler.pkl')

    errors = np.load(errors_path)
    labels = np.load(labels_path)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    feature_names_entity = meta['feature_names']
    n = len(errors)

    print(f"  {name}: {n} rows, {labels.mean():.1%} default rate")

    # Build sparse rows
    for i in range(n):
        row = np.full(len(feature_names), np.nan, dtype=np.float32)

        for j, col in enumerate(feature_names):
            if col == 'recon_error':
                row[j] = errors[i]
            elif col == 'entity_type':
                row[j] = idx
            else:
                prefix = f'{name}__'
                if col.startswith(prefix):
                    raw     = col.replace(prefix, '')
                    col_idx = feature_names_entity.index(raw) if raw in feature_names_entity else -1
                    if col_idx >= 0:
                        # get original value by inverse transforming
                        pass  # we don't have raw values, only scaled — skip features, use recon_error + entity_type

        all_rows.append(row)
        all_labels.append(labels[i])

X_all = np.array(all_rows, dtype=np.float32)
y_all = np.array(all_labels, dtype=int)

print(f"\nTotal matrix: {X_all.shape} | default rate: {y_all.mean():.1%}")

# ── Get predicted probabilities ───────────────────────────────────────────────
print("Running XGBoost predictions...")
probs = xgb_model.predict_proba(X_all)[:, 1]

print(f"AUC: {roc_auc_score(y_all, probs):.4f}")

# ── Find best threshold ───────────────────────────────────────────────────────
precision, recall, thresholds = precision_recall_curve(y_all, probs)

# F1 at each threshold
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx   = np.argmax(f1_scores)
best_thr   = thresholds[best_idx]
best_f1    = f1_scores[best_idx]
best_prec  = precision[best_idx]
best_rec   = recall[best_idx]

print(f"\n{'='*50}")
print(f"BEST THRESHOLD : {best_thr:.4f}")
print(f"F1 score       : {best_f1:.4f}")
print(f"Precision      : {best_prec:.4f}")
print(f"Recall         : {best_rec:.4f}")
print(f"{'='*50}")

# Also show a few candidate thresholds
print("\nCandidate thresholds:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    preds = (probs >= thr).astype(int)
    p = f1_score(y_all, preds, pos_label=1, zero_division=0)
    from sklearn.metrics import precision_score, recall_score
    pr = precision_score(y_all, preds, pos_label=1, zero_division=0)
    re = recall_score(y_all, preds, pos_label=1, zero_division=0)
    f  = f1_score(y_all, preds, pos_label=1, zero_division=0)
    marker = ' ← current' if thr == 0.5 else (' ← BEST' if abs(thr - best_thr) < 0.025 else '')
    print(f"{thr:>10.2f} {pr:>10.4f} {re:>10.4f} {f:>10.4f}{marker}")

# ── Classification report at best threshold ───────────────────────────────────
print(f"\nClassification report at threshold {best_thr:.4f}:")
preds_best = (probs >= best_thr).astype(int)
print(classification_report(y_all, preds_best, target_names=['Legit', 'Default']))

cm = confusion_matrix(y_all, preds_best)
print(f"Confusion matrix:")
print(f"  True Legit  caught: {cm[0,0]:>6}  |  Legit flagged as default: {cm[0,1]:>6}")
print(f"  Defaults missed:    {cm[1,0]:>6}  |  Defaults caught:          {cm[1,1]:>6}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)

# 1. Precision-Recall curve
ax1 = fig.add_subplot(gs[0])
ax1.plot(recall, precision, color='steelblue', lw=2)
ax1.scatter(best_rec, best_prec, color='red', zorder=5, s=80,
            label=f'Best thr={best_thr:.3f}')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. F1 vs threshold
ax2 = fig.add_subplot(gs[1])
ax2.plot(thresholds, f1_scores[:-1], color='darkorange', lw=2)
ax2.axvline(best_thr, color='red', linestyle='--', label=f'Best={best_thr:.3f}')
ax2.axvline(0.5, color='gray', linestyle='--', label='Current=0.5')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 vs Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Score distribution
ax3 = fig.add_subplot(gs[2])
ax3.hist(probs[y_all == 0], bins=50, alpha=0.6, color='steelblue', label='Legit', density=True)
ax3.hist(probs[y_all == 1], bins=50, alpha=0.6, color='red',       label='Default', density=True)
ax3.axvline(best_thr, color='black', linestyle='--', label=f'Best thr={best_thr:.3f}')
ax3.axvline(0.5,      color='gray',  linestyle='--', label='Current=0.5')
ax3.set_xlabel('Risk Score')
ax3.set_ylabel('Density')
ax3.set_title('Score Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(BASE, 'threshold_tuning.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved → {plot_path}")
plt.show()

print(f"\nDONE. Use this in inference.py:")
print(f"  'verdict': 'HIGH_RISK' if prob >= {best_thr:.4f} else 'LOW_RISK'")