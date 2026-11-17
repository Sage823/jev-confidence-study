# Design review — 2026-11-17

The local pipeline now covers five dataset shapes, frozen items and manifests, Jev/Qwen adapters, shared JSONL runs, confidence signals, calibration modes, and report outputs. Offline tests use fixtures and a fake Qwen engine. They do **not** validate real model behavior or establish empirical findings.

## Decisions

- The team chose the proposal prose total: **3,000 test / 900 calibration**. The five per-domain counts still need to be frozen. The proposal specifies **200 order items per domain**; `combine --protocol-totals` checks all three counts.
- A common item set does not imply equal model accuracy. Reports show answer accuracy and paired differences. Any accuracy-matched comparison requires an explicit additional protocol.
- Calibration fits use canonical calibration rows only. Cross-domain fitting excludes the target domain. Temperature scaling of option distributions and scalar correctness calibration are separate measures.
- Jev supplies native probabilities only. Self-report and sample consistency are marked unavailable for Jev. Qwen's local path computes all four signals and records invalid self-report or sample parses.
- The reference repository's code/data are not copied. It declares MIT for code and separate terms for curated/third-party data; our source data need independent license checks.
- Optional dependencies are not bundled: [PyTorch](https://pypi.org/project/torch/) lists BSD-style terms, [Transformers](https://pypi.org/project/transformers/) and [Accelerate](https://pypi.org/project/accelerate/) list Apache terms, and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) lists Apache-2.0. These do not clear the five benchmark datasets for redistribution.

## Before a full real run

1. Record each dataset's exact source/revision, license and allowed use, then choose source splits and per-domain counts. The proposed StrategyQA Hugging Face mirror currently has a broken preview; use a verified labeled source export.
2. Pilot Jev and Qwen on a few shared items; verify returned model IDs, Qwen option-token scoring, parsers, and failure records.
3. Confirm compute: Qwen3-32B in BF16 does not fit on a single 24 GB GPU; arrange larger memory, offload, or a documented quantized condition.
4. Set API/GPU budget and user-supplied price rates if dollar estimates are required. There is no automatic retry or resume after an interrupted run.
5. Decide whether to add uncertainty intervals for calibrated metrics and a prespecified accuracy-matched comparison before making strong comparative claims.

The code repository has no project `LICENSE` pending agreement among its authors. Do not treat a public GitHub upload as permission for others to reuse it.
