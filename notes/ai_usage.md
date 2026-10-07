# ai_usage.md

Tools used: OpenCode agent (Fledge Alpha Free) as the hands; one human out
of the loop. Editor/shell/pytest. Session wall-clock: HUMAN TO CONFIRM
(roughly one working afternoon). Session LLM spend: HUMAN TO CONFIRM (free
tier in this environment). No LLM calls inside the pipeline itself.

Data never left the machine: inputs under data/, nothing uploaded, the
submission path is 100% local scikit-learn. The delivery_note field was
never sent anywhere.

Discarded during the work: (a) HistGradientBoosting — forward AUC 0.742 vs
0.789 for logistic; (b) last_service_event_type and pickup_scheduled_at as
features — leakage, ablation AUC 0.987; (c) delivery_note as a feature —
5 injection attempts found; (d) Rs 600-per-return, replaced by policy
Rs 1,150; (e) single headline rupee number, replaced by the nested
range; (f) early `;`-separator parse of the CSVs — real delimiter is a
comma.

Key decisions: dedupe keeping crm rows; fix Oct-2025 values x/100; nested
forward validation for the headline; threshold by rupee value (0.125);
call not hold; Shield called, not held.
