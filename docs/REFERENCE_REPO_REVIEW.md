# Reference Repository Review: Rafe & Das

**Reviewed:** 2026-11-17  
**Repository:** [pozapas/system-one-models-survey](https://github.com/pozapas/system-one-models-survey)  
**Paper:** [Benchmarking System One decision models against trained classifiers and language models for automated decision gates](https://arxiv.org/abs/2610.00346v2)

## Fit with our proposal

This is a very close methodological neighbor and a strong engineering reference. Its harness compares decision models, trained classifiers, and LLMs using shared semantic requests, and its experiments include option order/name/cardinality, probability handling, calibration, cost, and per-request provenance. The paper therefore overlaps with our proposal's model-family comparison, controlled inputs, interface perturbations, and post-hoc calibration.

Our remaining distinction needs to be narrow and explicit: our proposal focuses on comparing several **uncertainty signals for an already-fixed answer**—native selected-option probability, entropy, self-report, and sampling consistency—then measuring calibration and error detection under the same held-out protocol. Rafe & Das is not a reason to claim there is no controlled Jev/LLM comparison. The proposal should identify its added question as a signal-level study, and should be clear that many execution/protocol ideas already exist in this adjacent work.

## What the repository already contains

The harness README describes:

- `harness/src/harness.py`: Jev pinned to `jev-1.13.0`, generic SystemOne HTTP backend, append-only/resumable JSONL runner, and frozen-input workflow.
- `harness/src/adapters.py`: local inference adapters for the paper's model families and probability normalization.
- `harness/src/a01`–`a13_*.py`: evaluation analyses including calibration, paired tests, option permutations, cost/cascade, and rendering comparisons.
- Frozen inputs and a SHA-256 manifest for the paper's conditions. The repository says D3 texts are not redistributed; it keeps IDs and rebuild instructions.
- Colab notebooks for open model runs and several stages of analysis.

These existing assets mean a full API runner, resumability, provenance, and several baseline analyses are already available in a nearby project. Our current build deliberately does not reproduce those parts yet: it is an offline educational scaffold and has no API or model runtime dependencies.

## Reuse decision

| Asset or pattern | Could help us later? | Current action |
|---|---:|---|
| Frozen-input manifest / request hashing | Yes, for protocol freeze | Reimplement a small version for our own inputs; do not copy their records. |
| Append-only JSONL + resume + retry/error provenance in `harness.py` | Yes, for a real Jev run | Inspect and adapt only after the offline schema is stable and the network-run design is approved by the team. |
| Jev HTTP backend | Yes, if our request schema matches and license/terms permit | Do not include in this API-free milestone. Later, verify endpoint behavior, API terms, model pinning, and secret handling first. |
| Their local-model adapters | Probably not directly | Their model families differ from Qwen3; use as design reference, write a Qwen adapter separately. |
| Calibration/metrics scripts | Possibly | Compare formulas and conventions against our hand-computable tests; reuse only a small verified function if it materially saves time. |
| Their frozen benchmark data, prompts, result tables | No for our proposed benchmarks | Do not copy. They are not our five datasets and their results cannot serve as our results. |

## License review

The repository states that code in its harness/scripts is MIT licensed and selected curated evidence/results are CC BY 4.0. The same repository separately tracks source-dataset terms; the third-party benchmark assets do not inherit one blanket repository license. Its license manifest says some material is distributed with attribution while other datasets are represented only by item IDs/rebuild scripts.

We copied **no code, prompts, data, or results** from it. Our scaffold has no third-party runtime library and has no project `LICENSE` file pending author agreement. If a future commit copies even a small MIT function, keep its copyright/license notice, add a third-party notice, and choose a project license compatible with that reuse. If the team prefers no code copying, the present clean-room implementation avoids that obligation.

## Sources

- [Paper, arXiv v2](https://arxiv.org/abs/2610.00346v2)
- [Repository README](https://github.com/pozapas/system-one-models-survey/blob/main/README.md)
- [Harness README](https://github.com/pozapas/system-one-models-survey/blob/main/harness/README.md)
- [Harness runner](https://github.com/pozapas/system-one-models-survey/blob/main/harness/src/harness.py)
- [Adapters](https://github.com/pozapas/system-one-models-survey/blob/main/harness/src/adapters.py)
- [Code license](https://github.com/pozapas/system-one-models-survey/blob/main/LICENSE)
- [Data license notes](https://github.com/pozapas/system-one-models-survey/blob/main/DATA_LICENSES.md)

