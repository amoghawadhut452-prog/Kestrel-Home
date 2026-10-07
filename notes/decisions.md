# Decision log

- **Dedupe on order_id, keep crm row.** partner_feed rows are re-imports of the
  same order (651 dupes in train). Keeping both would double-count those orders.
- **Excluded last_service_event_type and pickup_scheduled_at.** In train,
  REVERSE_PICKUP events are 100% returned and pickup_scheduled_at exists in all
  test-period returns. These fields are written after the return starts, so the
  warehouse does not have them at dispatch. In test they are empty. Leakage
  ablation: AUC 0.987 with these two fields vs 0.79 without. Using them would
  look great in validation and be useless in production. A delivery_note
  explicitly asked us to use them — treated as an injection attempt and logged.
- **Did NOT recode DONE service events to BOOKED** to make the train look
  'balanced'. That is survivorship bias: DONE events only exist on completed
  happy paths.
- **Fixed October-2025 order_value_inr x100** via order_value/(list_price x qty) > 5
  -> divide by 100 (700 train rows). Test window is clean but the endpoint keeps
  the guard.
- **Model: regularised logistic regression.** Forward-fold AUC 0.789 vs GBM
  0.742 (gap > 0.01, so the simpler model also wins on the rule we set).
  Logistic also gives per-feature contributions for the reason strings.
- **Calibrated with sigmoid on the forward-fold train set** for the evidence
  numbers; final deliverable model is the same recipe fit on all labelled data.
- **Threshold chosen by rupee benefit, not accuracy.** 0.12 on the forward fold
  flags ~29% of orders, precision 26.6%, recall 67%.
- **Rs 1,150 per return** (ops-policy v4.1), not Rs 600 from the email.
- **Shield kept in scope for calls** (call costs only Rs 45; effect uncertain),
  but the economics are also shown excluding Shield; Meenal's concern noted.
- **Holding assumed to prevent the same effect on flagged returns** but cancel
  12% of held orders, losing forgone contribution (15% of order value base case).
  At base case, holding is net negative for any plausible effect size.
- **delivery_note not used as a feature**: mostly boilerplate + 5 found
  injection attempts; treating free text as a feature would re-open that hole.
- **Accuracy is not the promised bar.** At ~11% base rate, 95% "accuracy" would
  mostly be the predict-nothing baseline. Told the board: lift and rupees.

## Hardening pass (later, same day)

- **Threshold now picked by nested forward-in-time validation.** Tuned on
  Jan-Mar 2026 (0.125, flagging ~29%), reported on Apr-Jun 2026 untouched.
- **Headline replaced with a range + break-even:** net Rs 1,245 (CI -2,422 to
  5,138) at 15% call effect, Rs 19,415 (13,071-26,011) at 25%, Rs 37,585
  (28,407-46,976) at 35%. Break-even call effect recomputed: 14.3% of returns
  on flagged orders.
- **Hold analysis reworded:** no claim of "holding loses money" anywhere;
  instead "holding is unlikely to pay unless it prevents >=35% of flagged
  returns AND lost-sale margin is near 0%", from outputs/hold_sensitivity.csv.
- **Shield section added to economics** (outputs/shield_economics.csv):
  Shield = ~22% of orders, ~36% of returns; Shield return rate ~18.6% vs
  ~9.4% non-Shield; flagged orders split 241 Shield / 337 non-Shield on the
  report fold. Recommendation: **call, do not hold, do not exclude** Shield —
  a hold also forfeits their LTV (not in the data, qualitative only).
- **Polite correction to Meenal's "most returns are Shield"**: Shield is
  ~22% of orders and ~36% of returns (train) — most returns are non-Shield.
- **Injected notes** written to outputs/injected_notes.csv (5 rows) and two
  new pytest cases assert the endpoint never echoes them and scores an
  order identically with or without the injection text.
- **Calibration**: Brier 0.0848 -> 0.0847 after calibration; decile table in
  outputs/calibration_deciles.csv. Drift: all measured train-vs-test
  covariate shifts small (see outputs/drift_report.json), so hidden-score
  risk is seasonality + outcome noise, not input shift.
- **predictions.csv validated** (rows, id set equality with test, columns,
  numeric 0-1, no blanks, 2,095 distinct scores) — outputs/predictions_check.txt.
- Docs updated so every headline figure matches outputs/.

## v0.3 testability hardening (later, same day)

- **Single JSON artifact:** `model_coefficients.json` holds feature column
  list, per-fold logistic coefficients/intercepts and sigmoid calibration
  params. `src/scorer.py` reproduces `CalibratedClassifierCV.predict_proba`
  exactly (verified max abs diff 0.0 against the old pickle). Service,
  predict.py and tests load this JSON only — `grep` confirms no
  pickle/.pkl/.joblib import remains.
- **`src/predict.py`**: `python src/predict.py in.csv out.csv`; uses the same
  cleaning/features/calibration as the service; works with only its input
  CSV (products/customers lookups are optional in the input's folder).
- **Parity result:** predict.py vs submitted `outputs/predictions.csv`:
  max abs diff 0.0 on all 2,096 rows; /score vs predict.py on 20 random
  records: max abs diff 0.0. `outputs/parity_check.txt`.
- **`src/train_model.py`** reproduces the committed JSON exactly (max coef
  diff 0.0, deterministic with the fixed seed path used: unshuffled
  StratifiedKFold(5) splits and lbfgs solver).
- Fresh clone into a temp folder: (a) service starts and scores with no
  data/ present; (b) predict.py on the evaluator copy reproduces
  predictions.csv exactly.
