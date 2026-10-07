"""Economics: call vs hold vs nothing, per flagged order and per quarter at test volume."""
import sys; sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from features import *
import scorer

RET_COST = 1150
CALL_COST = 45
HOLD_CANCEL = 0.12
MARGIN_ON_LOST_SALE = 0.15  # assumption: contribution forgone on cancelled held order, as share of order value
T = 0.12

assert abs(RET_COST-1150)<1e-9  # per policy v4.1, not Rs 600

train, test, cust, prod = load()
test = fix_value(test, prod)
ftr_test = features(test, cust, prod)
p = scorer.predict_proba(ftr_test, scorer.load_coeffs())
ftr_test = ftr_test.assign(p=p, flagged=p>=T)


exp_ret = 0.114  # validation base rate proxy on flagged set is ~0.266; use flagged-set precision instead
val = pd.read_csv('outputs/forward_fold_predictions.csv')
fl = val[val.predicted==1]
flag_precision = fl.y.mean(); flag_recall = fl.y.sum()/val.y.sum()
flag_rate = val.predicted.mean()
print(f'threshold {T}: forward-fold flag rate {flag_rate:.3f}, precision {flag_precision:.3f}, recall {flag_recall:.3f}')

def quarter(flagged_df, effect):
    n = len(flagged_df)
    n_ret = n * flag_precision           # expected returns among flagged
    calls = n * CALL_COST
    saved = n_ret * effect * RET_COST
    return dict(flagged=int(n), calls_cost=calls, expected_returns=round(n_ret,1),
                saved_at_effect={str(int(e*100))+'%': round(n_ret*e*RET_COST) for e in (0.15,0.25,0.35)})

for shield_mode, df in [('all flagged', ftr_test[ftr_test.flagged]),
                        ('excluding Shield', ftr_test[ftr_test.flagged & (ftr_test.shield_member==0)]),
                        ('Shield only', ftr_test[ftr_test.flagged & (ftr_test.shield_member==1)])]:
    r = quarter(df, 0.35)
    print(shield_mode, '->', {k:v for k,v in r.items() if k!='saved_at_effect'},
          'net at 15/25/35%:', [round(n*flag_precision*e*RET_COST - n*CALL_COST) for e,n in
            [(0.15,len(df)),(0.25,len(df)),(0.35,len(df))]])

# holding economics: benefit only if it also prevents effect x returns; cost = 12% cancellations x forgone margin
flagged_all = ftr_test[ftr_test.flagged]
mean_val = flagged_all.order_value_inr.mean()
def hold_net(effect, margin_share, n=len(flagged_all)):
    prevented = n*flag_precision*effect*RET_COST
    lost = n*HOLD_CANCEL*(mean_val*margin_share)
    return prevented - lost
for eff in (0.15,0.25,0.35):
    row = [round(hold_net(eff, m)) for m in (0.0,0.075,0.15)]
    print(f'hold net per quarter @ {eff:.0%} effect, margin on lost sale 0/7.5/15%: Rs {row}')

call35 = len(flagged_all)*flag_precision*0.35*RET_COST - len(flagged_all)*CALL_COST
print(f'HEADLINE: flag {flag_rate:.0%} of orders, call them: net ~Rs {call35:,.0f}/quarter (15% effect: Rs {len(flagged_all)*flag_precision*0.15*RET_COST - len(flagged_all)*CALL_COST:,.0f}; 25%: Rs {len(flagged_all)*flag_precision*0.25*RET_COST - len(flagged_all)*CALL_COST:,.0f}) vs holding ~Rs {hold_net(0.35,0.15):,.0f}')
# break-even effect size for calls
be = CALL_COST/(flag_precision*RET_COST)
print(f'break-even call effect: {be:.1%}')
json.dump({'ret_cost':RET_COST,'call_cost':CALL_COST,'hold_cancel':HOLD_CANCEL,'T':T,
           'flag_rate':flag_rate,'flag_precision':flag_precision,
           'net_call_15':len(flagged_all)*flag_precision*0.15*RET_COST - len(flagged_all)*CALL_COST,
           'net_call_25':len(flagged_all)*flag_precision*0.25*RET_COST - len(flagged_all)*CALL_COST,
           'net_call_35':call35, 'hold_35_15margin':hold_net(0.35,0.15), 'breakeven_effect':be},
          open('outputs/economics.json','w'), indent=2)
print('saved outputs/economics.json')
