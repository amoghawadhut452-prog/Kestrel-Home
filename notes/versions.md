# versions.md / changelog

- v0.1 explore.py — confirmed: train 11,155 rows (1 Apr 2025-30 Jun 2026),
  test 2,096 (1 Jul-30 Sep 2026), 651 train order_id duplicates, dedupe ->
  10,504, base rate 11.4%, joins 1:1, test has only NONE/INSTALL_BOOKED events.
- v0.1 train_eval.py — forward fold (<2026-04 train, Apr-Jun 2026 validate,
  n=2126, base 11.5%): logistic AUC 0.7892 / PR 0.4181 / acc 89.6% vs
  predict-nothing 88.5%; GBM 0.7418 / 0.3489 / 89.2%. Random 5-fold GBM AUC
  0.7593. Leakage-only ablation AUC 0.9868.
- v0.1 model.py — calibrated logistic fitted on all 10,504 rows.
- v0.1 score.py — predictions.csv: 2,096 rows, matching sample_submission
  schema exactly (order_id, score); score range 0.0076-0.9719, mean 0.119.
- v0.1 evidence.py — AUC 0.7891 (95% CI 0.7593-0.8201), PR 0.418 (0.352-0.479);
  threshold 0.12: flags 29.2%, precision 26.6%, recall 67.3%, acc 74.8%;
  decile lift top 3.75x; calibration table binned 0-1; error analysis saved.
- v0.1 economics.py — calls: net Rs ~41k/quarter at 35% effect (Rs ~20.6k at
  25%, ~Rs 0.6k at 15%); break-even effect 14.7% of returns on flagged orders.
  Holding: ~Rs -62k/quarter at 35% effect + 15% lost margin; positive only for
  low lost-margin assumptions.
- v0.1 app — FastAPI /score + static page; 6/6 pytest; uvicorn smoke 200/200.
## v0.2 hardening (same day)
- nested forward validation (train <2026-01-01, tune Jan-Mar, report Apr-Jun).
  Threshold 0.125 from tune fold. Report fold: AUC 0.788 (CI 0.758-0.819),
  PR 0.415, precision 0.273, recall 0.645, flag rate 27.2%. Net Rs/q:
  15% → ~Rs 1.2k (CI -2.4k to 5.1k), 25% → ~Rs 19.4k (13.1k-26.0k),
  35% → ~Rs 37.6k (28.4k-47.0k). Break-even call effect 14.3%. Hold
  sensitivity table added (negative across the grid at 12% cancellation).
  Shield economics split saved; Brier 0.0848→0.0847; drift report; error
  analysis; injected_notes.csv; predictions_check.txt;
  8/8 pytest (added injection + leakage-echo cases).
## v0.3 testability hardening (same day, later)
- exported model to model_coefficients.json; scorer.py mirrors
  CalibratedClassifierCV sigmoid path exactly (max diff 0.0)
- predict.py CLI added; parity vs predictions.csv exact on 2,096 rows;
  /score parity exact on 20 random rows after service 4-dp rounding
- train_model.py reproduces the committed JSON exactly (max coef diff 0.0)
- synthetic_orders.csv + full-reproduce/service-parity pytest added; 10/10 pass
- README gained Data-loading rule and How-to-verify (3 paths); service and
  predict.py both start with data/ absent and fall back to neutral lookups
