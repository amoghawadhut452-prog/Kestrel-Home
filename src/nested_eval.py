"""Nested forward-in-time validation.

Train model on orders before 2026-01-01. Score Jan-Mar 2026 (tune fold) and
Apr-Jun 2026 (untouched report fold). Pick threshold by rupee benefit on the
tune fold; evaluate on the report fold. Bootstrap CIs for net Rs, AUC, PR-AUC,
precision, recall. Save outputs/nested_validation.json.
"""
import sys; sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from features import *

RET_COST = 1150
CALL_COST = 45
RES = json.load(open('outputs/evidence_summary.json'))

train, test, cust, prod = load()
train = dedupe(train); train = fix_value(train, prod)
ftr = features(train, cust, prod); ftr = ftr[ftr.returned.notna()]
t1, t2, t3 = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-04-01'), pd.Timestamp('2026-07-01')
tr = ftr[ftr.order_placed_at < t1]
tune = ftr[(ftr.order_placed_at >= t1) & (ftr.order_placed_at < t2)]
rep = ftr[ftr.order_placed_at >= t2]
print('split sizes:', len(tr), len(tune), len(rep), 'bases', tr.returned.mean().round(4), tune.returned.mean().round(4), rep.returned.mean().round(4))

Xtr = design(tr)
m = CalibratedClassifierCV(LogisticRegression(max_iter=5000, C=0.05), method='sigmoid', cv=5).fit(Xtr, tr.returned.astype(int))
p_tune = m.predict_proba(design(tune, ref_cols=Xtr.columns))[:,1]
p_rep = m.predict_proba(design(rep, ref_cols=Xtr.columns))[:,1]
yt, yr = tune.returned.astype(int).values, rep.returned.astype(int).values

# threshold sweep on the tune fold using 35% effect
best = None
for t in np.arange(0.03, 0.6, 0.005):
    fl = p_tune >= t
    if fl.sum() == 0: continue
    ben = (yt[fl]*0.35*RET_COST).sum() - fl.sum()*CALL_COST
    if best is None or ben > best[0]: best = (ben, t, fl.mean())
ben35, T, tune_rate = best
print(f'threshold from tune fold: {T:.3f} (flag rate {tune_rate:.3f}, tune net@35% Rs {ben35:,.0f})')

# evaluate on untouched report fold
fl = p_rep >= T
prec = precision_score(yr, fl); rec = recall_score(yr, fl)
auc = roc_auc_score(yr, p_rep); pr = average_precision_score(yr, p_rep)

def net(p, y, t, effect, nf=None):
    f = p >= t
    return f.sum()*-CALL_COST + y[f].sum()*effect*RET_COST

res_out = {'threshold_from_tune_fold': T, 'tune_flag_rate': tune_rate,
           'report_n': int(len(rep)), 'report_base': float(yr.mean()),
           'auc': auc, 'pr': pr, 'precision': prec, 'recall': rec,
           'flag_rate': float(fl.mean()), 'flagged_n': int(fl.sum())}
for e in (0.15, 0.25, 0.35):
    res_out[f'net_rs_q_effect_{int(e*100)}'] = net(p_rep, yr, T, e)

# bootstrap CIs (1000 resamples)
rng = np.random.default_rng(0)
B = 1000
idx = np.arange(len(yr))
boot = {k: [] for k in ['auc','pr','prec','rec','net15','net25','net35']}
for _ in range(B):
    b = rng.choice(idx, len(idx), replace=True)
    pb, yb = p_rep[b], yr[b]
    try: boot['auc'].append(roc_auc_score(yb, pb))
    except ValueError: boot['auc'].append(np.nan)
    boot['pr'].append(average_precision_score(yb, pb))
    f = pb >= T
    if f.sum(): boot['prec'].append(yb[f].mean()); boot['rec'].append(yb[f].sum()/max(yb.sum(),1))
    else: boot['prec'].append(np.nan); boot['rec'].append(np.nan)
    for k,e in [('net15',.15),('net25',.25),('net35',.35)]:
        boot[k].append(net(pb, yb, T, e))
ci = {k: (float(np.nanpercentile(v,2.5)), float(np.nanpercentile(v,97.5))) for k,v in boot.items()}
res_out['ci95'] = ci
print('report-fold AUC', round(auc,4), ci['auc'])
print('net Rs per report fold (~1 quarter):')
for e in (15,25,35):
    key = f'net{e}'
    print(f'  effect {e}%:', round(res_out[f"net_rs_q_effect_{e}"],0), 'CI', tuple(round(x) for x in ci[key]))
json.dump(res_out, open('outputs/nested_validation.json','w'), indent=2)
print('saved outputs/nested_validation.json')
