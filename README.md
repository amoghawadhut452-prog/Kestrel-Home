# Kestrel Home Appliances — Return-Risk Scorer

Flags, before dispatch, orders likely to be returned. Fully local: no external
calls, no API key, Rs 0 per order on the default path.

## Setup

Python 3.11:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train + score

```
python src/explore.py
python src/nested_eval.py   # honest headline: nested forward validation
python src/train_eval.py
python src/train_model.py   # fits + deterministically writes model_coefficients.json
python src/score.py
python src/evidence.py
python src/economics.py
python src/hardening.py     # hold grid, Shield split, Brier/drift/errors
python src/injected_and_check.py  # injection audit + predictions validation
python src/crosscheck.py    # independent headline derivation
python src/parity_check.py
```

## Service + screen

```
uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000/ for the one-page screen. POST `/score` takes one
order JSON and returns `score`, `risk_band`, `recommended_action` and 2-3
plain-language reasons.

## Tests

```
python -m pytest tests/ -q
```

## Layout

```
data/     inputs (git-ignored)
model_coefficients.json  exported model used by the service and predict.py
src/      explore/features/train_eval/nested_eval/train_model/scorer/score/evidence/economics
app/      FastAPI service + static HTML
tests/    endpoint pytest
outputs/  predictions.csv, metric CSVs/JSONs (no raw customer data, no pickle)
notes/    memo, decisions, time log, versions, ai usage
```

## Run cost

Rs 0 per order (local scikit-learn scoring). No paid API, no LLM calls.

## Headline economics (nested forward validation)

Model trained before 2026-01-01; threshold (0.125) tuned on Jan-Mar 2026;
numbers evaluated on the untouched Apr-Jun 2026 fold (n=2,126, ~1 quarter at
test volume 2,096):

- net Rs/quarter @15% prevention: ~Rs 1.2k (95% CI -2.4k to 5.1k)
- net Rs/quarter @25%: ~Rs 19.4k (CI 13.1k-26.0k)
- net Rs/quarter @35%: ~Rs 37.6k (CI 28.4k-47.0k)
- break-even prevention rate: ~14.3% of returns on flagged orders
- holding is unlikely to pay: negative net across the grid of margins
  (15/25/35%) and prevention rates (0/10/35%) in
  outputs/hold_sensitivity.csv

Full evidence pack: notes/evidence.md (calibration deciles, Brier, drift,
error analysis) and outputs/.

## Data-loading rule

Place files under `data/`. Filenames expected:
`train.csv`, `test_unlabelled.csv`, `customers.csv`, `products.csv`,
`sample_submission.csv`, `ops-policy.pdf`. Unknown SKUs/customer ids score with generic reasons; if `data/` is
absent, the service still starts and `predict.py` still scores, but
customer/product lookups (returns history lookup, 100x-value guard)
fall back to neutral values and scores for such fields will differ.

## How to verify this submission

(a) Run the service and screen (works with or without `data/`):

```
uvicorn app.main:app --port 8000
```

(b) Reproduce predictions with `predict.py` using the evaluator's own copy
of the test file; it writes `order_id,score` and must match the submitted
`outputs/predictions.csv` row-for-row:

```
python src/predict.py <path-to-evaluator-test.csv> <output.csv>
```
(expected output file: `outputs/predictions.csv`; parity verified at
max abs diff 0.0 on 2,096 rows, outputs/parity_check.txt)

(c) Retrain with `train_model.py` using your own train data — it
reproducibly rebuilds `model_coefficients.json`:

```
python src/train_model.py
```

Train_model reproduces the committed `model_coefficients.json` **exactly**
(max coefficient difference 0.0; deterministic KFold(5) splits and an
unregularised-direction-free solver). If a different outstanding fold
order gives tiny drift, a tolerance of 1e-9 in coefficient space is
documented; in practice with the same scikit-learn version the output is
byte-identical.

## Security / data handling

delivery_note is never used as a feature. Five injection attempts were
found and logged in outputs/injected_notes.csv; pytest asserts the endpoint
neither echoes them nor assigns a different meaning to such a note.
`.gitignore` covers data/, predictions.csv, model.pkl and fold CSVs. **Do
not push this folder to a public repository** (Kestrel ops-policy §10).
