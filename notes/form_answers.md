# form_answers.md

**What AI tools were used?**
OpenCode coding agent (Fledge Alpha Free) ran the exploration, model,
service and tests under a fixed ruleset. No LLM calls are in the product
path — the scorer is scikit-learn on a machine.

**What did they cost?**
Session LLM cost: HUMAN TO CONFIRM (free-tier environment; ask the
operator). Deployed scorer runtime cost: Rs 0 per order, no API keys.
Operator hours: HUMAN TO CONFIRM.

**What was discarded?**
HistGradientBoosting (worse on forward fold); leaked post-outcome fields
(AUC 0.987, rejected); delivery_note as a feature (injection attempts
found); Rs 600-per-return figure (replaced by policy Rs 1,150); single
headline rupee number (replaced by a range from nested validation).

**Key decisions?**
Dedupe keeping crm rows; fix Oct-2025 payment values; nested forward
validation only; threshold (0.125) by rupee value; recommend calls over
holding; break-even prevention ~14.3%.

**What was left out?**
SHAP, hyperparameter search, LLM/embeddings, seasonality terms,
timestamp normalisation, pincode fairness analysis — see
notes/not_done.md.

**Business goal as one number?**
Net rupees per quarter from prevented returns at the same ops cost:
~Rs 19k-38k /quarter at a 25-35% prevention effect (roughly break-even
at ~14%), vs ~zero-to-negative for holding.
