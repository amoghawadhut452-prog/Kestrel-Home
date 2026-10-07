# expected_score.md

Written from the **nested forward-in-time** estimate (see
outputs/nested_validation.json and notes/evidence.md):

- Model trained on orders before 2026-01-01, threshold tuned on Jan-Mar
  2026 (0.125), numbers below evaluated on the untouched Apr-Jun 2026 fold
  (n=2,126, base rate 11.5%). That fold is ~1 quarter at the test volume
  (2,096 orders), so the quarterly figures stand as-is (scale factor
  2096/2126 = 0.986).
- Expected metric profile on a hidden dispatch snapshot:
  - ROC-AUC ~0.76-0.82 (point 0.788, bootstrap 95% CI 0.758-0.819)
  - PR-AUC ~0.35-0.48 (point 0.415)
  - precision ~25-30%, recall ~60-68% at threshold 0.125
  - top-decile lift ~3.7x base rate
  - accuracy at the business cut-off is high only because the metric is
    dominated by the ~89% of orders that stay — state this to the board.
- Expected net rupees per quarter from calls on flagged orders:
  - 15% prevention: ~Rs 1k (CI -2.4k to 5.1k) — i.e. roughly break-even
  - 25% prevention: ~Rs 19k (CI 13k-26k)
  - 35% prevention: ~Rs 38k (CI 28k-47k)
  - break-even prevention rate: ~14.3%

Why the hidden set may still score lower:

1. **Seasonality**: the test window (Jul-Sep 2026) contains no festive
   orders; training and validation windows include some.
2. **Outcome noise**: returns partly trigger on service interactions we
   deliberate excluded (leakage), so part of the outcome is irreducible from
   dispatch-visible data.
3. **Irreducible covariate drift is small**: measured train-vs-test shifts
   (channel mix, family mix, price, discount, Shield share, COD share,
   pincode 000000 share, prior-return distribution) are all within a point
   or two (see outputs/drift_report.json), so drift alone should not move
   the score much — seasonality and pilot-effect noise are the bigger risks.
4. **Injection attempts were excluded** by design: we do not read or act on
   delivery_note, so we cannot score the way the injected text requested —
   that is intentional, not a weakness.
