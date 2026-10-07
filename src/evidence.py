"""Evidence pack: forward-fold, calibration, bootstrap CIs, error analysis."""
import sys; sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import *
from features import *

train, test, cust, prod = load()
train = dedupe(train); train = fix_value(train, prod)
ftr = features(train, cust, prod); ftr = ftr[ftr.returned.notna()]
cut = pd.Timestamp('2026-04-01')
tr = ftr[ftr.order_placed_at < cut]; va = ftr[ftr.order_placed_at >= cut]
Xtr = design(tr); Xva = design(va, ref_cols=Xtr.columns)
ytr, yva = tr.returned.astype(int), va.returned.astype(int)
print('forward fold n:', len(tr), len(va), 'base', ytr.mean().round(4), yva.mean().round(4))

cal = CalibratedClassifierCV(LogisticRegression(max_iter=5000, C=0.05), method='sigmoid', cv=5)
cal.fit(Xtr, ytr)
p = cal.predict_proba(Xva)[:,1]
rng = np.random.default_rng(0)
def ci(fn, n=1000):
    vals = []
    idx = np.arange(len(yva))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        vals.append(fn(yva.values[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
auc = roc_auc_score(yva, p); pr = average_precision_score(yva, p)
print('AUC', round(auc,4), 'CI', tuple(round(x,4) for x in ci(roc_auc_score)))
print('PR ', round(pr,4), 'CI', tuple(round(x,4) for x in ci(average_precision_score)))

# best threshold by rupee benefit (call at 35%)
best = None
for t in np.arange(0.03, 0.6, 0.005):
    fl = p >= t
    if fl.sum()==0: continue
    saved = (yva.values[fl] * 0.35 * 1150).sum()
    cost = fl.sum() * 45
    ben = saved - cost
    if best is None or ben > best[0]: best = (ben, t, fl.sum())
ben, t, nflag = best
prec = yva.values[p>=t].mean(); rec = yva.values[p>=t].sum()/yva.sum()
acc = accuracy_score(yva, (p>=t).astype(int))
print(f'threshold {t:.3f}: flags {nflag}/{len(va)} ({nflag/len(va):.1%}), precision {prec:.3f}, recall {rec:.3f}, acc {acc:.3f}')
print(f'   call benefit on fold (35% effect): Rs {ben:,.0f}')

pd.DataFrame({'p':p,'y':yva.values}).assign(predicted=(p>=t).astype(int)).to_csv('outputs/forward_fold_predictions.csv', index=False)
# calibration table
pt = pd.DataFrame({'p':p,'y':yva.values}); pt['bin'] = pd.cut(pt.p, [0,.05,.1,.2,.35,.6,1.0])
ct = pt.groupby('bin', observed=True).agg(n=('y','size'), mean_pred=('p','mean'), actual=('y','mean'))
print(ct); ct.to_csv('outputs/calibration_table.csv')

# decile lift
pt['dec'] = pd.qcut(pt.p, 10, labels=False, duplicates='drop')
lift = pt.groupby('dec').agg(n=('y','size'), mean_pred=('p','mean'), actual=('y','mean'))
lift['lift'] = lift.actual/yva.mean()
print(lift); lift.to_csv('outputs/decile_lift_forward.csv')

# error analysis
seg = va.copy(); seg['p']=p; seg['pred']=(p>=t).astype(int)
err = seg[seg.returned != seg.pred]
print('errors by family:'); print(err.groupby('family').size().sort_values(ascending=False).head())
print('errors by channel:'); print(err.groupby('sales_channel').size())
print('errors by shield:'); print(err.groupby('shield_member').size())
print('errors by prior orders 0 vs >0:', err[err.customer_prior_orders==0].shape[0], err[err.customer_prior_orders>0].shape[0])

json.dump({'auc':auc,'pr':pr,'threshold':t,'flagged':int(nflag),'flag_rate':nflag/len(va),
           'precision':prec,'recall':rec,'accuracy':acc,
           'accuracy_predict_none':accuracy_score(yva, np.zeros_like(yva))},
          open('outputs/evidence_summary.json','w'), indent=2)
seg.drop(columns=[]).to_csv('outputs/forward_fold_segmented.csv', index=False)
print('saved evidence outputs')
