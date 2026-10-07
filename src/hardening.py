"""Hardening pass: hold sensitivity, Shield split, calibration/Brier, drift, errors."""
import sys, warnings, json, re
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from features import *

RET_COST, CALL_COST, HOLD_CANCEL = 1150, 45, 0.12
NV = json.load(open('outputs/nested_validation.json'))
T = NV['threshold_from_tune_fold']

train, test, cust, prod = load()
train = dedupe(train); train = fix_value(train, prod)
ftr = features(train, cust, prod); ftr = ftr[ftr.returned.notna()]
t1, t2 = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-04-01')
tr = ftr[ftr.order_placed_at < t1]
tune = ftr[(ftr.order_placed_at >= t1) & (ftr.order_placed_at < t2)]
rep = ftr[ftr.order_placed_at >= t2].copy()

Xtr = design(tr)
m = CalibratedClassifierCV(LogisticRegression(max_iter=5000, C=0.05), method='sigmoid', cv=5).fit(Xtr, tr.returned.astype(int))
p_rep = m.predict_proba(design(rep, ref_cols=Xtr.columns))[:,1]
yr = rep.returned.astype(int).values
fl = p_rep >= T
rep['p'] = p_rep; rep['flagged'] = fl

# --- existing (rule-of-thumb) economics calibrated re-derived below in rupee script.
# --- HOLD SENSITIVITY ---
mean_val = rep.loc[fl, 'order_value_inr'].mean()
hover = pd.DataFrame(index=[0.0,0.10,0.35], columns=[0.15,0.25,0.35])
for h in (0.0,0.10,0.35):
    for gm in (0.15,0.25,0.35):
        prev = fl.sum()*fl.sum()/len(rep)*0  # placeholder replaced below
for i, h in enumerate((0.0,0.10,0.35)):
    for j, gm in enumerate((0.15,0.25,0.35)):
        prevented = fl.sum()*(fl.sum()*0)  # recompute properly:
import itertools
rows = []
expected_returns = (rep.loc[fl,'returned'].values == 1).sum()
expected_returns_rate = rep.loc[fl,'returned'].mean()
nflag = int(fl.sum())
print('flagged expected returns:', int(expected_returns), 'of', nflag)
for h, gm in itertools.product((0.0,0.10,0.35), (0.15,0.25,0.35)):
    prevented_rs = expected_returns * h * RET_COST
    lost_rs = nflag * HOLD_CANCEL * mean_val * gm
    rows.append(dict(hold_prevents_returns=h, gross_margin=gm, net_rs=round(prevented_rs - lost_rs),
                     prevented_rs=round(prevented_rs), lost_rs=round(lost_rs)))
hold_df = pd.DataFrame(rows)
hold_df.to_csv('outputs/hold_sensitivity.csv', index=False)
print(hold_df.pivot(index='hold_prevents_returns', columns='gross_margin', values='net_rs'))

# --- SHIELD ---
rep['shield'] = rep.shield_member
def net_for(df):
    fl2 = df[df.flagged]
    return int(fl2.flagged.sum()*(-CALL_COST) + (fl2.returned*fl2.p.ge(T)).sum()*0*RET_COST)  # placeholder
shield_rows = []
for label, df in [('all flagged', rep), ('flagged non-Shield only', rep[rep.shield==0]), ('flagged Shield only', rep[rep.shield==1])]:
    fl2 = df[df.flagged]
    nret = int(fl2.returned.sum())
    for e in (0.15,0.25,0.35):
        net = nret*e*RET_COST - len(fl2)*CALL_COST
        shield_rows.append(dict(segment=label, effect=e, flagged=int(len(fl2)), expected_returns=nret, net_rs_q=round(net)))
pd.DataFrame(shield_rows).to_csv('outputs/shield_economics.csv', index=False)
print(pd.DataFrame(shield_rows).pivot(index='segment', columns='effect', values='net_rs_q'))
allr = rep
print('Shield: share of orders', round(allr.shield.mean(),3), 'share of returns', round(allr[allr.returned==1].shield.mean(),3),
      'flag rate Shield vs non', round(allr[allr.shield==1].flagged.mean(),3), round(allr[allr.shield==0].flagged.mean(),3))

# --- CALIBRATION BY DECILE + BRIER ---
# uncalibrated logistic for Brier before
raw = LogisticRegression(max_iter=5000, C=0.05).fit(Xtr, tr.returned.astype(int))
p_raw = raw.predict_proba(design(rep, ref_cols=Xtr.columns))[:,1]
print('Brier before calibration:', round(brier_score_loss(yr, p_raw),5), 'after:', round(brier_score_loss(yr, p_rep),5))
pt = pd.DataFrame({'p':p_rep,'y':yr}); pt['dec'] = pd.qcut(pt.p, 10, labels=False, duplicates='drop')
cal = pt.groupby('dec').agg(n=('y','size'), mean_pred=('p','mean'), actual=('y','mean'))
cal.to_csv('outputs/calibration_deciles.csv'); print(cal)
json.dump({'brier_before':brier_score_loss(yr,p_raw),'brier_after':brier_score_loss(yr,p_rep)}, open('outputs/brier.json','w'))

# --- DRIFT: train (all of it) vs test ---
tr_all = ftr
tt = fix_value(test, prod); tt = features(tt, cust, prod)
def mix(df, col, k=None):
    s = df[col].value_counts(normalize=True)
    return s.head(k) if k else s
def drift_report():
    out = {}
    for col in ['sales_channel','family','payment_mode']:
        a = tr_all[col].value_counts(normalize=True); b = tt[col].value_counts(normalize=True)
        idx = sorted(set(a.index)|set(b.index))
        out[col] = {i: (round(a.get(i,0),4), round(b.get(i,0),4)) for i in idx}
    out['price_median'] = [round(tr_all.order_value_inr.median(),1), round(tt.order_value_inr.median(),1)]
    out['discount_median'] = [float(tr_all.discount_pct.median()), float(tt.discount_pct.median())]
    out['shield_share'] = [round(tr_all.shield_member.mean(),4), round(tt.shield_member.mean(),4)]
    out['cod_share'] = [round((tr_all.payment_mode=='cod').mean(),4), round((tt.payment_mode=='cod').mean(),4)]
    out['pincode000000_share'] = [round(tr_all.default_pincode.mean(),4), round(tt.default_pincode.mean(),4)]
    pr = tr_all.customer_prior_returns; pt2 = tt.customer_prior_returns
    out['prior_returns_mean'] = [round(pr.mean(),3), round(pt2.mean(),3)]
    return out
dr = drift_report(); json.dump(dr, open('outputs/drift_report.json','w'), indent=2)
print(json.dumps(dr, indent=1))

# --- ERROR ANALYSIS on report fold at threshold T ---
rep['pred'] = fl.astype(int)
err = rep[rep.returned != rep.pred]
def rates(df, col_vals):
    out = {}
    for name, mask in col_vals:
        sub = rep[mask]
        if len(sub): out[name] = dict(n=int(len(sub)), errors=int((sub.returned!=sub.pred).sum()),
                                      error_rate=round((sub.returned!=sub.pred).mean(),4))
    return out
out = {
 'any': int(len(err)),
 'by_family': rates(err, [(f, rep.family==f) for f in sorted(rep.family.unique())]),
 'by_channel': rates(err, [(c, rep.sales_channel==c) for c in sorted(rep.sales_channel.unique())]),
 'by_shield': rates(err, [('shield', rep.shield==1), ('non-shield', rep.shield==0)]),
 'new_vs_repeat': rates(err, [('new', rep.customer_prior_orders==0), ('repeat', rep.customer_prior_orders>0)]),
 'cod_vs_prepaid': rates(err, [('cod', rep.payment_mode=='cod'), ('prepaid', rep.payment_mode.isin(['prepaid_upi','prepaid_card']))]),
}
json.dump(out, open('outputs/error_analysis.json','w'), indent=2, default=float)
print(json.dumps(out, indent=1, default=float)[:1200])
