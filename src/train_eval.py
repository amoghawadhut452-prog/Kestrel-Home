"""Forward-in-time validation: logistic baseline vs HistGradientBoosting.
Leakage ablation on the side. Writes metrics CSV/JSON to outputs/.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from features import *

train, test, cust, prod = load()
train = dedupe(train)
train = fix_value(train, prod)
test = fix_value(test, prod)
print('train fixed values flagged:', train.value_fixed.sum(), 'test:', test.value_fixed.sum())
ftr = features(train, cust, prod)
ftr = ftr[ftr.returned.notna()]

# drift / break check
d = ftr.copy(); d['m'] = d.order_placed_at.dt.to_period('M')
print(d.groupby('m').agg(n=('returned','size'), ret=('returned','mean'), shield=('shield_member','mean'), cod=(d['payment_mode']=='cod').groupby(d['m']).mean()) if False else d.groupby('m').apply(lambda g: pd.Series({'n':len(g),'ret':g.returned.mean(),'shield':g.shield_member.mean(),'cod':(g.payment_mode=='cod').mean()}), include_groups=False).T)

# forward split: train < Apr 2026, validate Apr-Jun 2026
cut = pd.Timestamp('2026-04-01')
tr = ftr[ftr.order_placed_at < cut]; va = ftr[ftr.order_placed_at >= cut]
print('forward split:', tr.shape, va.shape, 'base rates', tr.returned.mean(), va.returned.mean())

Xtr, Xva = design(tr), design(tr)  # reindex below
Xva = design(va, ref_cols=Xtr.columns)
ytr, yva = tr.returned.astype(int), va.returned.astype(int)

# random 5-fold for comparison
from sklearn.model_selection import cross_val_predict
Xall = design(ftr)
cross_auc = roc_auc_score(ytr if False else ftr.returned.astype(int),
            cross_val_predict(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300, random_state=0), Xall, ftr.returned.astype(int), cv=5, method='predict_proba')[:,1])
f_fwd = design(ftr)
print('random 5-fold AUC (GBM):', round(cross_auc,4))

def evaluate(model, name, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:,1]
    auc = roc_auc_score(yva, p); pr = average_precision_score(yva, p)
    acc = accuracy_score(yva, (p>=0.5).astype(int))
    acc0 = accuracy_score(yva, np.zeros_like(yva))
    pdp = pd.DataFrame({'p':p,'y':yva.values})
    pdp['dec'] = pd.qcut(pdp.p, 10, labels=False, duplicates='drop')
    lift = pdp.groupby('dec').agg(n=('y','size'), rate=('y','mean'))
    print(f'{name}: AUC {auc:.4f} PR {pr:.4f} acc@.5 {acc:.4f} (predict-none {acc0:.4f})')
    lift.to_csv(f'outputs/decile_lift_{name}.csv')
    return p, auc, pr, acc

lp = LogisticRegression(max_iter=2000, C=0.05)
p_lp, auc_lp, pr_lp, acc_lp = evaluate(lp,'logreg',Xtr,ytr,Xva,yva)
gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=300, random_state=0)
p_gb, auc_gb, pr_gb, acc_gb = evaluate(gb,'gbm',Xtr,ytr,Xva,yva)

# leakage ablation: same GBM on leaked fields
leakX_tr = tr[['discount_pct','qty','order_value_inr','customer_prior_returns','shield_member']].copy()
leak_tr = train.merge(ftr[['order_id','order_placed_at']], on='order_id', how='inner')
leak = (train.set_index('order_id')[['last_service_event_type','pickup_scheduled_at']].notna()
        )
ll = train.copy()
ll['has_pickup'] = ll.pickup_scheduled_at.notna().astype(int)
ll['evt_rp'] = (ll.last_service_event_type=='REVERSE_PICKUP').astype(int)
ll['evt_done'] = ll.last_service_event_type.isin(['DEMO_DONE','INSTALL_DONE']).astype(int)
ll = ll.drop_duplicates('order_id', keep='first').set_index('order_id').loc[ftr.order_id]
y = ftr.returned.astype(int)
from sklearn.model_selection import cross_val_predict as cvp
p_leak = cvp(HistGradientBoostingClassifier(max_iter=300, random_state=0),
             pd.get_dummies(ll[['has_pickup','evt_rp','evt_done']]), y, cv=5, method='predict_proba')[:,1]
print('LEAKAGE ablation AUC (random 5-fold, leaked fields only):', round(roc_auc_score(y,p_leak),4))

json.dump({'logreg': {'auc':auc_lp,'pr':pr_lp,'acc':acc_lp},
           'gbm': {'auc':auc_gb,'pr':pr_gb,'acc':acc_gb},
           'gbm_random_auc': cross_auc,
           'leak_auc_random': float(roc_auc_score(y,p_leak))},
          open('outputs/validation_metrics.json','w'), indent=2)
print('saved outputs/validation_metrics.json')
