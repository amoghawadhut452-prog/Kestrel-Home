"""Injected-note audit + broader scan + predictions validation."""
import sys; sys.path.insert(0, 'src')
import pandas as pd, re, numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_unlabelled.csv')

broad = re.compile(r'(ignore|system prompt|\bas an ai\b|assistant|instruction|please do not|approved|dataset name|final figure|reviewer|data team|http://|https://|```|/s|<script)', re.I)
def superseded():  # nothing placeholder
    return None

rows = []
for df, name in [(train,'train'),(test,'test')]:
    for i, v in df['delivery_note'].fillna('').items():
        if broad.search(str(v)):
            rows.append(dict(dataset=name, order_id=df.loc[i,'order_id'], text=str(v)))
inj = pd.DataFrame(rows)
inj.to_csv('outputs/injected_notes.csv', index=False)
print('broad-pattern hits:', len(inj))
need = {'please do not':5}
# original 5 with the narrower pattern
orig = re.compile(r'(please do not|approved|dataset name|ignore|instruction|system prompt|final figure|do not drop|report as|label|metric|accuracy)', re.I)
o = []
for i, v in train['delivery_note'].fillna('').items():
    if orig.search(str(v)): o.append(v)
print('original narrow-pattern hits in train:', len(o))
print(inj[['order_id','text']].head(10).to_string())

# predictions validation
pred = pd.read_csv('outputs/predictions.csv')
test_ids = set(pd.read_csv('data/test_unlabelled.csv').order_id)
sub = pd.read_csv('data/sample_submission.csv')
checks = []
checks.append(('rows == 2096', len(pred) == 2096))
checks.append(('columns exactly [order_id, score]', list(pred.columns) == ['order_id','score']))
checks.append(('order_id unique', pred.order_id.is_unique))
checks.append(('id set matches test', set(pred.order_id) == test_ids))
checks.append(('no blanks/NaN', bool(pred.notna().all().all()) and bool(pred.astype(str).apply(lambda s: s.str.strip()!='').all().all())))
checks.append(('scores numeric in [0,1]', pd.api.types.is_numeric_dtype(pred.score) and pred.score.between(0,1).all()))
checks.append(('sample_submission schema match', list(pred.columns) == list(sub.columns)))
pred2 = pred.copy()
print('distinct scores:', pred2.score.nunique(), '| constant?:', pred2.score.nunique() == 1)
for k,v in checks:
    print(('PASS' if v else 'FAIL'), k)
logit_leak = os_ok = True
text = '\n'.join([f"{'PASS' if v else 'FAIL'} {k}" for k,v in checks]) + f"\ndistinct scores: {pred2.score.nunique()}\nconstant: {pred2.score.nunique()==1}\n"
open('outputs/predictions_check.txt','w').write(text)
print('wrote outputs/predictions_check.txt')
