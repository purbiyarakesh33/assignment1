


import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')


# ── Autoencoder (must match training architecture) ─────────────────────────────
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


class RiskEngine:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ae_dir   = os.path.join(base_dir, 'ae_models')
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load XGBoost model
        import xgboost as xgb
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(base_dir, 'xgb_final_model.json'))

        # Load feature names (188 total)
        with open(os.path.join(base_dir, 'xgb_feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)

        # Discover all entity types dynamically from ae_models/
        # Any *_meta.pkl file = one entity type
        self.entities = {}
        for fname in sorted(os.listdir(self.ae_dir)):
            if fname.endswith('_meta.pkl'):
                name = fname.replace('_meta.pkl', '')
                self.entities[name] = self._load_entity(name)

        print(f"Loaded {len(self.entities)} entity types: {list(self.entities.keys())}")
        print(f"Feature space: {len(self.feature_names)} columns")

    def _load_entity(self, name):
        """Load AE + scaler + meta for one entity type."""
        ae_dir = self.ae_dir

        with open(os.path.join(ae_dir, f'{name}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)

        with open(os.path.join(ae_dir, f'{name}_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        # Reconstruct AE from saved params
        params    = meta['best_params']
        input_dim = meta['input_dim']
        model     = Autoencoder(input_dim, params['hidden_dims'], params['dropout'])
        state     = torch.load(
            os.path.join(ae_dir, f'{name}_ae.pt'),
            map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        # The entity's own feature names (prefixed in the 188-col space)
        own_features = [
            col.replace(f'{name}__', '')
            for col in self.feature_names
            if col.startswith(f'{name}__')
        ]

        return {
            'model':         model,
            'scaler':        scaler,
            'medians':       meta['medians'],
            'input_dim':     input_dim,
            'own_features':  own_features,        # unprefixed names
            'feature_names': meta['feature_names'], # exact names after preprocessing
        }

    def _detect_entity(self, row_keys):
        """
        Pick entity type with maximum feature overlap with incoming row.
        row_keys: set of feature names from the incoming transaction.
        Returns (entity_name, entity_index, overlap_count)
        """
        best_name    = None
        best_idx     = -1
        best_overlap = 0

        for idx, (name, info) in enumerate(self.entities.items()):
            known   = set(info['own_features'])
            overlap = len(known.intersection(row_keys))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name    = name
                best_idx     = idx

        return best_name, best_idx, best_overlap

    def _compute_recon_error(self, entity_name, row_dict):
        """
        Build the AE input for this entity, impute missing features
        with saved medians, scale, run AE, return MSE.
        """
        info       = self.entities[entity_name]
        feat_names = info['feature_names']
        medians    = info['medians']
        scaler     = info['scaler']
        model      = info['model']

        row_data = {}
        for col in feat_names:
            if col in row_dict:
                row_data[col] = row_dict[col]
            else:
                row_data[col] = medians.get(col, 0.0)

        X = pd.DataFrame([row_data])[feat_names]
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(medians)

        X_scaled = scaler.transform(X.values)

        t = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            recon = model(t)
            error = torch.mean((recon - t) ** 2).item()

        return error

    def _build_row(self, row_dict, entity_name, entity_idx, recon_error):
        """
        Build the full 188-feature vector.
        Known features placed in correct positions, rest NaN.
        """
        row = np.full(len(self.feature_names), np.nan, dtype=np.float32)

        for i, col in enumerate(self.feature_names):
            if col == 'recon_error':
                row[i] = recon_error
            elif col == 'entity_type':
                row[i] = entity_idx
            else:
                prefix = f'{entity_name}__'
                if col.startswith(prefix):
                    raw = col.replace(prefix, '')
                    if raw in row_dict:
                        try:
                            row[i] = float(row_dict[raw])
                        except (ValueError, TypeError):
                            row[i] = np.nan

        return row

    def _predict_single(self, row_dict):
        """
        Core prediction for one transaction (dict of feature:value).
        Returns dict with risk_score, entity_type, overlap, verdict.
        """
        row_keys = set(row_dict.keys())

        entity_name, entity_idx, overlap = self._detect_entity(row_keys)

        if overlap == 0:
            return {
                'risk_score':  None,
                'verdict':     'INSUFFICIENT_DATA',
                'message':     'Insufficient data to assess this distributor. Please add more data.',
                'entity_type': None,
                'overlap':     0,
            }

        recon_error = self._compute_recon_error(entity_name, row_dict)
        row         = self._build_row(row_dict, entity_name, entity_idx, recon_error)

        prob = self.xgb_model.predict_proba(row.reshape(1, -1))[0][1]

        return {
            'risk_score':  round(float(prob), 4),
            'verdict':     'HIGH_RISK' if prob >= 0.3256 else 'LOW_RISK',
            'entity_type': entity_name,
            'overlap':     overlap,
            'recon_error': round(float(recon_error), 6),
            'message':     None,
        }

    def predict(self, transaction: dict) -> dict:
        """Single transaction prediction."""
        return self._predict_single(transaction)

    def predict_batch(self, transactions: list) -> list:
        """List of dicts → list of results."""
        return [self._predict_single(t) for t in transactions]

    def predict_csv(self, csv_path: str) -> pd.DataFrame:
        """CSV file → DataFrame with predictions appended."""
        df      = pd.read_csv(csv_path)
        results = [self._predict_single(row.to_dict()) for _, row in df.iterrows()]
        out     = df.copy()
        out['risk_score']  = [r['risk_score']  for r in results]
        out['verdict']     = [r['verdict']      for r in results]
        out['entity_type'] = [r['entity_type']  for r in results]
        out['message']     = [r['message']      for r in results]
        return out


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    engine = RiskEngine(base_dir=r'C:\Users\Admin\OneDrive\Desktop\yes')

    # DS1 sample
    r = engine.predict({
        'age': 45,
        'DebtRatio': 0.3,
        'MonthlyIncome': 5000,
        'RevolvingUtilizationOfUnsecuredLines': 0.2,
        'NumberOfTime30-59DaysPastDueNotWorse': 0,
        'NumberOfTimes90DaysLate': 0,
        'NumberOfOpenCreditLinesAndLoans': 5,
    })
    print("Single prediction:", r)

    # Batch
    batch = engine.predict_batch([
        {'age': 30, 'DebtRatio': 0.8, 'MonthlyIncome': 2000},
        {'age': 55, 'DebtRatio': 0.1, 'MonthlyIncome': 9000},
    ])
    print("Batch predictions:", batch)