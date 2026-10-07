# Memo — Ritu Deshpande, D2C Operations

## The answer up front

Score every order before dispatch; on the ~27-29% of orders above the 0.125
threshold, make a confirmation call — do not hold them. Honest net, sized on
the untouched Apr-Jun 2026 fold (2,126 orders, ~1 quarter at test volume):
**about Rs 1.2k/quarter if the call effect is 15% (CI -2.4k to 5.1k, i.e.
break-even), about Rs 19k/quarter at 25% (CI 13k-26k), about Rs 38k/quarter
at 35% (CI 28k-47k)**. Our spring pilot suggested ~35%; treat 25% as the
conservative planning number. The break-even call effect is ~14.3%.

## What the model can and cannot do

A calibrated logistic model on dispatch-time fields only. Nested, leakage-free
validation (model trained before 2026-01-01, threshold tuned on Jan-Mar 2026,
reported on Apr-Jun 2026): **ROC-AUC 0.788 (95% CI 0.758-0.819), PR-AUC 0.415,
precision 27%, recall 65%, top-decile lift ~3.7x.** It does **not** meet 95%
accuracy, and it cannot: at an ~11% return rate, predicting "nothing returns"
is already ~88.5% accurate, so that number is noise. Tell the board: at the
0.125 cut-off the model puts ~43% of returns into the top decile, and calls on
flagged orders pay for themselves above a 14.3% prevention rate.

Two traps were removed. Post-outcome columns (service event, pickup
timestamp) made train look 99% accurate — excluded (using them gives AUC
0.987 on a random split, useless at dispatch). Five customer notes were
instructions to the analysis tooling — logged, never used.

## The rupees

A return costs Rs 1,150 (policy v4.1); a call costs Rs 45. On the report
fold the model flags 578 of 2,126 orders, precision 27% → ~158 expected
returns; preventing 35% saves ~Rs 64k against Rs 26k of call cost. Holding:
assume >24h holds get cancelled 12% of the time. Across a grid of lost-sale
margins (15/25/35% of order value) and prevented-return rates (0/10/35%),
holding is **net negative in every cell** — i.e. it only pays if it prevents
>35% of returns AND the margin lost on cancellations is near zero. We do not
recommend it. On Shield orders the case is weaker still: Shield returns are
free for the customer, and Meenal (rightly) points to their lifetime value.

## Shield

Shield is ~22% of orders but ~36% of train returns (18.6% return rate vs
9.4% non-Shield). The model flags Shield at ~2.5x the non-Shield rate, so
undeauted of 578 flagged report-fold orders, 241 are Shield. Recommendation:
**call Shield orders, do not hold them, do not exclude them.** The call is
only Rs 45, Shield-only calls net ~Rs 2.8k/q at 15% effect and ~Rs 21k/q at
35%. We did not model Shield lifetime value (not in the data) but losing a
high-LTV Shield order to a dispatch hold is an extra cost that makes holding
Shield strictly worse. Note, politely: "most returns are Shield" is not what
the data shows — about 64% of returns are from non-Shield customers.

## What to do next week

Two-week controlled pilot on ~2,000 flagged orders: callers work a random
50/50 split of flagged orders with no-call holdout. If the prevention rate
clears ~15%, roll out. Re-score weekly, retrain quarterly.

## What we did not do and why

No delivery-note text features (injection attempts found and logged); no
LLM/embedding calls; no hyperparameter search; no SHAP (coefficients drive
the reason strings); no timestamp normalisation (no time features used);
no festive-season term (the test window has no festive month). Data never
left the machine and nothing was pushed to a public repo (policy §10).
