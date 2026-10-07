# recording_outline.md — 3:00 screencast

0:00-0:15  The ask + the honest framing. "Returns cost Rs 1,150 each.
Ritu wants flag-and-hold. The board was promised 95%." ls data/.

0:15-0:35  Show notes/decisions.md at the leakage bullet; run
`python src/nested_eval.py`: "threshold tuned Jan-Mar (0.125), AUC 0.788
on the untouched Apr-Jun fold."

0:35-0:55  Run `python src/crosscheck.py`; voice the range: "~Rs 1k/q at
15% effect, ~Rs 19k at 25%, ~Rs 38k at 35%; break-even ~14.3%."

0:55-1:20  Show outputs/hold_sensitivity.csv and say: "holding is net
negative unless it prevents >35% AND margin on cancelled holds is ~0."

1:20-1:30  Shield: show outputs/shield_economics.csv one-liners; "call
Shield, don't hold; ~22% of orders, ~36% of returns."

1:30-1:55  Start uvicorn, open http://127.0.0.1:8000/, submit an order
(cod + prior returns), show score + "call" + 3 plain reasons; then run
`python -m pytest tests/ -q`: 8 passed — note the new injection tests.

1:55-2:25  Version diff moment: outputs/validation_metrics.json vs
outputs/nested_validation.json; "what changed: threshold moved from 0.12
to 0.125 and the headline moved from one number to a range."

2:25-2:50  Show outputs/injected_notes.csv row count 5 and
outputs/predictions_check.txt PASS lines; show drift_report.json line:
"covariate drift <1pt, seasonality is the risk."

2:50-3:00  Close: "What to do next week: 2-week pilot, 50/50 call vs no
call on flagged orders."
