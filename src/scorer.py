"""Shared scoring used by the service and predict.py — no pickle, data-free."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from features import design

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEFF_PATH = os.path.join(_ROOT, 'model_coefficients.json')

def load_coeffs(path=COEFF_PATH):
    return json.load(open(path, encoding='utf-8'))

def predict_proba(fx: pd.DataFrame, coeffs=None) -> np.ndarray:
    """fx: feature frame with the same raw columns features() produces."""
    if coeffs is None: coeffs = load_coeffs()
    X = design(fx, ref_cols=coeffs['feature_cols']).to_numpy(dtype=float)
    acc = np.zeros(X.shape[0])
    from scipy.special import expit
    for f in coeffs['folds']:
        c = np.asarray(f['coef']); i = f['intercept']
        s = X @ c + i
        acc += expit(-(f['cal_a'] * s + f['cal_b']))
    return acc / len(coeffs['folds'])

def mean_coef(coeffs=None):
    if coeffs is None: coeffs = load_coeffs()
    return np.mean([np.asarray(f['coef']) for f in coeffs['folds']], axis=0)

REASON_PHRASES = {
    'payment_mode_cod': 'cash on delivery orders return more often',
    'payment_mode_emi': 'EMI-paid orders return more often',
    'is_gift_1': 'it is marked as a gift, which returns more often',
    'default_pincode': 'no full address was captured (default pincode)',
    'customer_prior_returns': 'this customer has returned before',
    'prior_return_rate': 'a high share of this customer’s past orders came back',
    'shield_member': 'Shield member (free returns, so returns are more likely)',
    'discount_pct': 'a deep discount on the order',
}

def reasons(fx: pd.DataFrame, coeffs=None, top=3):
    if coeffs is None: coeffs = load_coeffs()
    X = design(fx, ref_cols=coeffs['feature_cols']).to_numpy(dtype=float)
    contribs = X[0] * mean_coef(coeffs)
    s = pd.Series(contribs, index=coeffs['feature_cols'])
    out = []
    for col, c in s[s > 0].sort_values(ascending=False).items():
        if len(out) >= top: break
        if col.startswith('family_'): out.append(f"{col.replace('family_','')} has a higher return rate")
        elif col.startswith('sales_channel_'): out.append(f"sold via {col.replace('sales_channel_','').replace('_',' ')}, which returns more often")
        elif col.startswith('state_'): continue
        elif col in REASON_PHRASES: out.append(REASON_PHRASES[col])
        elif col.startswith('payment_mode_'): out.append(f"paid by {col.replace('payment_mode_','')}")
        else: out.append(col.replace('_', ' ') + ' is above the usual level')
    return out or ['no strong individual risk factor — overall profile looks average']
