# Not done (COULD tier, deliberately skipped)

- Fairness/proxy analysis on delivery_pincode — skipped, but default_pincode is
  a model feature; Meenal/Ritu should review it before a wider rollout.
- Timestamp normalisation (IST vs legacy UTC events) — skipped; no time
  features are used.
- Hyperparameter search — skipped; one regularised logistic setting chosen via
  the forward fold.
- SHAP — skipped; logistic coefficients + per-feature contributions cover the
  reason strings.
- LLM use anywhere — none. delivery_note was scanned for injected instructions
  (5 found, logged) and otherwise ignored as a feature.
- Festive-season modelling — test window has no festive month; endpoint guards
  the Oct-style 100x values but no seasonality term.
- Shield-specific retention economics (LTV of Shield) — Meenal flagged it; we
  show Shield-excluded numbers but did not size LTV impact.
