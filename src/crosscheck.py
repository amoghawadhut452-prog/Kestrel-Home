"""Independent crosscheck of the headline rupee range."""
import json
nv = json.load(open('outputs/nested_validation.json'))
exp_ret = nv['flagged_n'] * nv['precision']
lines = []
lines.append(f"NESTED REPORT-FOLD CROSSCHECK (Apr-Jun 2026, n={nv['report_n']})")
lines.append(f"n flagged: {nv['flagged_n']} ({nv['flag_rate']:.3f}); expected returns among flagged: {exp_ret:.1f}")
for e in (0.15, 0.25, 0.35):
    net = exp_ret * e * 1150 - nv['flagged_n'] * 45
    lines.append(f"net Rs/quarter @{int(e*100)}%: {net:,.0f}")
be = 45 / (nv['precision'] * 1150)
lines.append(f"break-even call effect: {be:.1%}")
lines.append(f"AUC {nv['auc']:.4f} / PR {nv['pr']:.4f} / precision {nv['precision']:.3f} / recall {nv['recall']:.3f}")
text = '\n'.join(lines) + '\n'
open('outputs/headline_crosscheck.txt', 'w').write(text)
print(text)
