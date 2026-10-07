"""Rebuild model_coefficients.json from data/ with a fixed seed."""
import sys, json, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from features import *

train, test, cust, prod = load()
train = dedupe(train)
train = fix_value(train, prod)
ftr = features(train, cust, prod)
ftr = ftr[ftr.returned.notna()]
X = design(ftr); y = ftr.returned.astype(int)
cal = CalibratedClassifierCV(LogisticRegression(max_iter=5000, C=0.05, random_state=0),
                             method='sigmoid', cv=5)
cal.fit(X, y)
folds = []
for cc in cal.calibrated_classifiers_:
    est = cc.estimator
    folds.append({'coef': est.coef_[0].tolist(), 'intercept': float(est.intercept_[0]),
                  'cal_a': float(cc.calibrators[0].a_), 'cal_b': float(cc.calibrators[0].b_)})
out = {'format': 'kestrel-logistic-sigmoid-v1', 'threshold': 0.125,
       'ret_cost': 1150, 'call_cost': 45, 'hold_cancel': 0.12,
       'feature_cols': list(X.columns), 'folds': folds}
json.dump(out, open('model_coefficients.json', 'w'), indent=1)
print('wrote model_coefficients.json, folds:', len(folds), 'cols:', len(X.columns))
