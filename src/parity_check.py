"""Parity: predictions.csv vs predict.py; /score vs predict.py on 20 random rows."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'app')
import pandas as pd, numpy as np
from features import *
import scorer

# 1) predict.py output vs submitted predictions.csv
a = pd.read_csv('outputs/predictions.csv').set_index('order_id').sort_index()
b = pd.read_csv('outputs/predictions_from_predict.csv').set_index('order_id').sort_index()
m = a.join(b, lsuffix='_sub', rsuffix='_pred')
d1 = (m.score_sub - m.score_pred).abs().max()

# 2) service /score vs predict.py on 20 random records
from fastapi.testclient import TestClient
from main import app
test = pd.read_csv('data/test_unlabelled.csv')
samp = test.sample(20, random_state=7)
resp = []
for _, r in samp.iterrows():
    payload = {k: (None if pd.isna(v) else v) for k, v in r.items()}
    payload.pop('returned', None)
    j = TestClient(app).post('/score', json=payload).json()
    resp.append(j['score'])
train, _, cust, prod = load(); t2 = fix_value(test, prod); ftr = features(t2, cust, prod)
proj = pd.Series(scorer.predict_proba(ftr, scorer.load_coeffs()), index=ftr.order_id)
d2 = max(abs(s - round(proj.loc[oid], 4)) for s, oid in zip(resp, samp.order_id))

with open('outputs/parity_check.txt', 'w') as f:
    f.write(f"predict.py vs submitted predictions.csv: {len(m)} rows, max abs diff {d1:.2e}\n")
    f.write(f"service /score vs predict.py on 20 random records: max abs diff {d2:.2e}\n")
print(f"predict.py vs predictions.csv: {len(m)} rows, max abs diff {d1:.2e}")
print(f"service vs predict.py (20 rows): max abs diff {d2:.2e}")
