"""Score test_unlabelled and write predictions.csv in sample_submission shape."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from features import *
import scorer

train, test, cust, prod = load()
test = fix_value(test, prod)
ftr_test = features(test, cust, prod)
p = scorer.predict_proba(ftr_test, scorer.load_coeffs())
out = pd.DataFrame({'order_id': ftr_test['order_id'], 'score': p})
sub = pd.read_csv('data/sample_submission.csv')
assert list(out.columns) == list(sub.columns)
assert len(out) == len(sub) == 2096
assert out.order_id.is_unique and out.order_id.notna().all()
assert out.score.notna().all()
assert set(out.order_id) == set(sub.order_id)
out.to_csv('outputs/predictions.csv', index=False)
print('wrote outputs/predictions.csv')
