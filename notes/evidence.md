# evidence.md — hardening pass

## Nested forward-in-time validation (headline)

Model trained on orders **before 2026-01-01** (n=6,252, base 11.3%).
Threshold tuned on **Jan-Mar 2026** (n=2,126, base 11.6%) using rupee
benefit at 35% call effect → **threshold 0.125**, flag rate 29.4% on the tune
fold. Results on the untouched **Apr-Jun 2026** fold (n=2,126, base 11.5%):

| metric | value | 95% CI (bootstrap, 1,000) |
|---|---|---|
| ROC-AUC | 0.788 | 0.758 – 0.819 |
| PR-AUC | 0.415 | see outputs/nested_validation.json |
| precision @0.125 | 0.273 | bootstrap CIs in JSON |
| recall @0.125 | 0.645 | bootstrap CIs in JSON |
| flag rate | 27.2% | — |
| net Rs/quarter @15% effect | 1,245 | -2,422 to 5,138 |
| net Rs/quarter @25% effect | 19,415 | 13,071 to 26,011 |
| net Rs/quarter @35% effect | 37,585 | 28,407 to 46,976 |

Break-even call effect at the tuned threshold: **14.3%** of returns on
flagged orders prevented. Scaling: the report fold is 2,126 orders and the
unlabelled test snapshot is 2,096 orders — calls-cost and returns both scale
linearly, scale factor 2096/2126 = 0.986, i.e. the report-fold figure is
within ~1.4% of the test-quarter figure.

## Calibration (report fold, calibrated on the pre-2026 train fold)

| decile | n | mean predicted | actual |
|---|---|---|---|
| 0 | 213 | 0.0152 | 0.0047 |
| 1 | 213 | 0.0266 | 0.0423 |
| 2 | 212 | 0.0378 | 0.0189 |
| 3 | 213 | 0.0502 | 0.0516 |
| 4 | 212 | 0.0634 | 0.0472 |
| 5 | 213 | 0.0799 | 0.1127 |
| 6 | 212 | 0.1023 | 0.1038 |
| 7 | 213 | 0.1362 | 0.1455 |
| 8 | 212 | 0.1987 | 0.2028 |
| 9 | 213 | 0.4145 | 0.4225 |

Brier score: **0.0848 before** calibration, **0.0847 after** (small — the
logistic was already well calibrated). Full table: outputs/calibration_deciles.csv,
Brier in outputs/brier.json.

## Drift train vs test (all of train vs test snapshot)

| dimension | train | test | |
|---|---|---|---|
| price median | Rs 4,440 | Rs 4,476 | stable |
| discount median | 8% | 9% | stable |
| Shield share | 22.2% | 22.1% | stable |
| COD share | 30.0% | 30.7% | stable |
| pincode 000000 share | 8.1% | 8.4% | stable |
| prior-returns mean | 0.182 | 0.195 | slight rise |
| channel mix (app/marketplace/partner/web) | 34.1/27.3/12.3/26.3% | 34.8/27.7/12.8/24.6% | minor |
| family mix | 7 families stable (~14% each) | within ±0.8pt | minor |

Full numbers: outputs/drift_report.json. Verdict: **no material covariate
shift** on the features the model uses, so the main hidden-test risk is
seasonality and irreducible noise in the outcome, not the inputs.

## Error analysis at threshold 0.125 (report fold)

507/2,126 rows mispredicted at threshold 0.125. Highest error rates:
Robot Vacuum 44.4% (highest), COD 40.5% vs prepaid 15.3%, Shield 37.9% vs
non-Shield 19.8%, repeat 25.8% vs new 19.4%, marketplace 27.8% vs
partner_outlet 17.8%, Water Purifier 29.4%. Full numbers:
outputs/error_analysis.json.

Recommendation: spot-check Robot Vacuum and COD flagged orders first in
the pilot. Shield over-flagging tracks its genuinely higher base rate, so
the model is behaving as expected there.
