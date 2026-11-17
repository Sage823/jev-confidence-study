# Experiment protocol

Source: `../proposal/proposal_v2.tex`. Engineering reference: [Rafe & Das repository](https://github.com/pozapas/system-one-models-survey). We use its frozen-input/manifest and append-only JSONL patterns; our implementation is independent.

## Research unit

The unit is one **fixed base answer** to one labeled multiple-choice item. Its correctness is `answer_key == gold_key`. Jev and Qwen3-8B/32B see the same frozen question, keyed options, and display order. Extra confidence calls never replace the base answer. Majority-vote accuracy is a separate outcome.

Datasets: SciQ (no support passage), HellaSwag, single-answer MedMCQA, StrategyQA, and MMLU. The chosen proposal total is **3,000 canonical test / 900 calibration items**. Allocate those totals across the five datasets before collecting real data; the proposal table's 4,500/1,400 totals are superseded. Freeze source revisions, split names, item IDs, hashes, option order, and seed. Reverse option order for **200 fixed test items per domain** while keeping key-to-text and gold identity fixed. Source JSONL and normalized items stay under ignored `data/`.

## Measurement

For each fixed answer, record:

1. Selected-option probability from the native option distribution.
2. `1 - entropy/log(number of options)` from that distribution.
3. A separate self-reported probability of the fixed answer being correct, where supported.
4. Fraction of sampled reasoning answers matching the fixed answer, at 5/10/20 samples, where supported.

Jev's Choice response provides native probabilities. The current Jev adapter has no self-report or reasoning-sampling path, so those signals are **unavailable**, not zero. Qwen uses a pinned Transformers checkpoint: non-thinking next-token option logits for the fixed answer, a separate non-thinking self-report, and thinking-mode sampled traces. Raw outputs, parsed values, parse failures, token counts, and latency are retained. The Qwen next-token measurement must be checked in a small real pilot because prompt/tokenizer behavior may differ from the fixture.

## Analysis

On held-out test items report answer accuracy; multiclass Brier/NLL; signal-level binary Brier, AUROC (correct answer positive), AURC, ECE (10/15 bins, equal-width/frequency), risk at 80% coverage, reliability/risk curves, cost/latency, and signal availability. Show per-domain results and equal-weight domain averages. Fit monotone scalar calibration and distribution temperature scaling on **calibration rows only**, in three modes: same domain, other domains, and all domains. The reversed-order test uses canonical calibration rows. Report 5/10/20 sample majority-vote accuracy separately, order agreement/probability movement, signal Spearman correlations, and paired bootstrap intervals for matched-model accuracy and raw signal AUROC/AURC/ECE differences. The report includes case IDs for confident errors, uncertain correct answers, and model disagreements.

Raw model outputs and errors go in new ignored `runs/` directories, with input hash and run metadata. Failed items are logged; they must be reviewed before interpreting results. Bootstrap intervals for calibrated metrics and accuracy-matched comparisons remain protocol work.

## Boundaries

No benchmark assets or model weights are distributed. SciQ is CC BY-NC 3.0; verify each exact source/revision and intended use before preparing data. The project has no `LICENSE` yet, so its original code has no public reuse grant. Optional Qwen dependencies and model weights are fetched by the user, not bundled. Real Jev runs require an API key; no live API or model call occurs in the test suite.
