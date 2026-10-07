# Time log (2026-10-08)

| time | what |
|---|---|
| 00:18 | copied attachments into data/, read ops-policy.pdf in full, set up venv packages |
| 00:45 | src/explore.py: shapes, 651 dup order_ids, 5 injected delivery_notes, joins 1:1, dedupe -> 10,504 rows, base 11.4% |
| 00:55 | src/train_eval.py: forward-in-time split (<2026-04 train, Apr-Jun validate); logreg AUC 0.789, GBM 0.742, random 5-fold GBM 0.759; leakage ablation AUC 0.987 |
| 01:00 | src/model.py: final calibrated logistic on all 10,504 rows -> outputs/model.pkl |
| 01:05 | src/score.py -> outputs/predictions.csv (2096 rows, shape-asserted) |
| 01:15 | src/evidence.py: AUC 0.7891 CI (0.759, 0.820), PR 0.418, decile lift, calibration table, error analysis |
| 01:25 | src/economics.py: call vs hold vs nothing, per-quarter grid at 15/25/35%, Shield variants |
| 01:35 | app/main.py + index.html + tests/test_api.py — 6/6 pytest pass |
| 01:45 | live uvicorn smoke test: GET / 200, POST /score 200 with sensible band+reasons |
| 02:00 | docs: README, decisions, not_done, versions, ai_usage, expected_score, memo, form_answers, recording_outline |
| 02:30 | hardening item 1: nested eval (train <2026-01-01, tune Jan-Mar w/ 0.125, report Apr-Jun): AUC 0.788, net Rs/q @15/25/35% = 1,245 / 19,415 / 37,585 with bootstrap CIs |
| 02:40 | hardening item 2/3/5: hold sensitivity grid (all net-negative at 12% cancel), Shield split Rs, Brier 0.0848→0.0847, drift report, error analysis → notes/evidence.md |
| 02:50 | hardening item 4/6: injected_notes.csv, broader-scan count = 5 hits (same 5), test for no-leakage echo + 8/8 pytest, predictions_check.txt |
| 03:00 | doc consistency pass: memo, expected_score.md, README, form_answers.md, recording_outline.md, ai_usage.md updated to v0.2 numbers |
| 03:10 | final: full pytest 8/8, deleted + recreated .venv per README, uvicorn smoke (GET / 200, POST /score 200 incl. 100x value), headline crosscheck saved |
| 04:00 | predict-without-pickle: exported model_coefficients.json, scorer.py shared by service + predict.py, train_model.py reproduces committed JSON exactly (cols match, max abs diff 0.0), pickle refs removed (no .pkl/.joblib imports anywhere) |
| 04:05 | parity: predict.py vs outputs/predictions.csv on 2,096 rows max abs diff 0.0; service /score vs predict.py on 20 random records max abs diff 0.0 (after rounding service to 4dp); outputs/parity_check.txt |
| 04:10 | synthetic sample_data/synthetic_orders.csv + 2 new pytest (fallback no-data predict, synthetic parity, full-reproduce hash-skip-if-no-data); 10/10 pass |
| 04:15 | data-loading rule + "How to verify this submission" (3 paths) added to README; service loads with data/ absent (score 0.33, generic behaviour confirmed); fresh-clone check: path (a) service starts GET / 200, POST /score 200; path (b) predict.py reproduces predictions.csv max abs diff 0.0 |

## Final check (fresh venv)

| time | what |
|---|---|
| 02:10 | `py -3.11 -m venv .venv` + `pip install -r requirements.txt` per README — all pinned versions installed |
| 02:15 | fresh venv `pytest tests/` → 6 passed |
| 02:20 | fresh venv uvicorn smoke: GET / 200, POST /score 200, Oct-style 100x value handled |
| 02:25 | second independent derivation: call net @35% = Rs 40,687 (matches outputs/economics.json); holding net negative (sign matches); flag rate 29.2% and precision 26.6% match |
