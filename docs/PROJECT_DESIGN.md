# Jev Confidence Study — Project Design

**Status:** reviewed design for an offline-first course-project scaffold  
**Basis:** `proposal/proposal_v2.tex`, the proposal handoff, and the Rafe & Das companion repository  
**Scope of this version:** a reproducible local analysis pipeline and synthetic simulator. It does not call Jev or another model and does not download or redistribute benchmark data.

For the repository-specific suitability and reuse review, see `REFERENCE_REPO_REVIEW.md`.

## 1. Purpose and research question

The project compares how well different confidence signals identify when a model's already-fixed answer is wrong. The planned signals are (a) native probability assigned to the selected option, (b) normalized entropy of the option distribution, (c) an LLM's self-reported probability of correctness, and (d) agreement among independently sampled answers. Jev and Qwen3-8B/32B are the proposed model conditions. The same item, gold answer, option text, and canonical option order must be linked across conditions. A separate permutation run measures sensitivity to order.

The implementation must keep answer quality and confidence quality distinct: the baseline answer is fixed first; confidence signals are then evaluated against whether that answer was correct. Majority-vote answers are a separate analysis condition and cannot silently replace the fixed answer.

The Rafe & Das paper and repository are the nearest implementation reference. Their repository offers useful patterns for request provenance, resumable JSONL records, model adapters, probability normalization, calibration metrics, and option-name/cardinality analyses. We will study those patterns but write this project independently. No source files, data, prompts, or result tables will be copied into this scaffold.

## 2. Scope and milestones

### Scaffold milestone (this build)

- Python standard library only; Python 3.11 or later.
- Deterministic synthetic data generator with explicit `simulation` provenance.
- Strict JSONL schema and validation, including probability-vector/option alignment and split checks.
- Pure-Python metrics: accuracy, multiclass Brier score, clipped NLL, ECE, AUROC, AURC, risk at requested coverage, and order agreement.
- Signal extraction for selected-option probability, normalized entropy, scalar self-report, and sample agreement.
- A calibration interface that accepts calibration rows only and applies the fitted mapping to held-out rows; report raw and calibrated scores separately.
- Command-line entry points for `simulate`, `validate`, and `analyze`, deterministic seeds, and outputs under ignored `runs/`.
- Offline tests using synthetic fixtures. There must be no HTTP client, credentials, API adapters, or hidden network activity in this milestone.

### Later research milestones (not implemented here)

1. Freeze a protocol and resolve the sample-count contradiction in the proposal (see design review).
2. Add dataset adapters after recording source, version/revision, license, required attribution, and allowed use for every source. Keep raw source datasets outside the code repository unless their licenses and course rules clearly permit redistribution.
3. Add Jev and local Qwen execution adapters only after the offline pipeline is stable; store credentials outside source control, cache raw responses, and record model revision, latency, tokens, failures, and option order.
4. Add the minimum viable three-domain pilot, then expand only when time and compute allow.
5. Generate the proposal's plots and tables from frozen run artifacts.

## 3. Data flow

`synthetic fixture or approved frozen input → normalized item records → model-condition records → validation → signal extraction → raw metrics → calibration fit on calibration split → held-out metrics → aggregate report`

Every derived record retains `run_id`, `source_kind`, `item_id`, `domain`, `split`, `model_id`, and protocol/config hash. `source_kind` is one of `simulation`, `benchmark`, or `model_output`; simulation artifacts must never be presented as benchmark findings. The validator rejects mixed split use in a calibration fit, duplicate `(run_id, item_id, model_id, condition)` keys, missing options, probability vectors with the wrong length, non-finite/negative probabilities, and gold labels not in the option keys.

### Normalized item record

- `item_id`: stable source ID or a namespaced synthetic ID.
- `domain`, `split`: domain name and one of `calibration` or `test`.
- `question`, `options`: option keys and text in canonical order.
- `gold_key`: correct option key.
- `source`: dataset/revision reference, or explicit simulation generator/version.

### Prediction record

- Item identity fields above (question text may be stored only when permitted).
- `model_id`, `condition_id`, `answer_key`, `option_order`.
- `option_probabilities`: key-to-probability mapping aligned to the presented option order; include raw values and normalized values when an adapter must repair a non-unit sum. Never silently discard original values.
- Optional signal inputs: `self_report_probability`, `sampled_answer_keys`.
- Provenance: `source_kind`, `run_id`, `protocol_hash`, `model_revision`, `created_at_utc`, `latency_ms`, `input_tokens`, `output_tokens`, `error`.

The scaffold's simulator emits a small, seeded collection of the same record shapes, with fake item content and fake model IDs. These are not real model outputs.

## 4. Analysis rules

- Correctness is `answer_key == gold_key`; each confidence signal estimates or ranks this event.
- Selected-option probability is read by key, not by positional index.
- Entropy is computed from the full normalized option distribution and divided by `log(number_of_options)` when there is more than one option, so it is comparable across different option counts. The corresponding confidence is `1 - normalized_entropy`.
- Self-report is converted to a correctness probability. Its prompt and output parser must be versioned in a later protocol.
- Sample consistency is the fraction of valid sampled answers equal to the fixed baseline answer. Invalid samples are counted and reported, not silently dropped.
- AUROC uses a correctness score with correct answers as the positive class; AURC sorts from most confident to least confident and reports the chosen convention. The report must state whether it uses confidence or error risk at each point.
- ECE binning choices (equal-width/equal-frequency and bin count) are explicit configuration, not defaults hidden in code.
- Calibration transforms are fitted on calibration rows only. The first scaffold uses one documented scalar monotone logistic mapping and distribution temperature scaling only if both can be implemented and reviewed in the offline milestone; no fitted parameter may be learned from test labels.
- Every summary reports per-domain values and an equal-weight domain average. A pooled metric may be included as a clearly labeled secondary value.
- Cost fields remain missing when unavailable; simulated costs are labeled synthetic. No claim of equal answer accuracy follows merely from using the same items. Any accuracy-matched analysis must be explicitly specified and reported separately.

## 5. Proposed repository layout

```text
jev_confidence_study/
  README.md
  pyproject.toml             # minimal metadata; no runtime dependencies
  .gitignore
  docs/
    PROJECT_DESIGN.md
    DESIGN_REVIEW.md
    REFERENCE_REPO_REVIEW.md
  src/jevstudy/
    __init__.py
    __main__.py
    cli.py
    schema.py
    simulate.py
    metrics.py
    signals.py
    calibration.py
    report.py
  tests/
    test_schema.py
    test_metrics.py
    test_simulation.py
    test_calibration.py
  runs/                       # generated, ignored, never committed by default
```

The workspace-level novice guide and code-review walkthrough live in `proposal/`, outside this project folder. The code repository should contain only project-facing documentation and implementation material.

## 6. Commands and artifacts

- `python -m jevstudy simulate --seed 4901 --out runs/demo` writes synthetic input/prediction JSONL and run metadata.
- `python -m jevstudy validate runs/demo` checks schema, identity alignment, provenance, and split discipline.
- `python -m jevstudy analyze runs/demo --out runs/demo/report.json` writes machine-readable summary metrics.
- `python -m unittest discover -s tests` runs offline tests; tests require no APIs, external packages, or benchmark downloads.

Each run contains a metadata file with command arguments, seed, package version, timestamp, protocol hash, source kind, and output file hashes. Results are append-only or immutable per run; reruns use a new `run_id` rather than overwriting evidence.

## 7. Reuse and licensing policy

The Rafe & Das companion repository is useful as a design reference. Its repository declares MIT for code and CC BY 4.0 for curated data/results, while raw benchmark assets can carry separate terms. We will not copy its code or frozen assets. This project will begin without a `LICENSE` file until all student authors decide the project-code license. That means it must not be described as open-source or redistributed with an implied license before that decision.

No benchmark records are bundled. This is especially important because the proposal itself identifies SciQ as CC BY-NC 3.0, which restricts commercial reuse, while other dataset mirrors may have missing or conflicting license metadata. For each chosen source/revision, maintain a provenance record with canonical URL, exact revision, license text/link, attribution, intended use, and whether redistribution is permitted. When uncertain, distribute only adapter code and item IDs/hashes or a fetch script, subject to the dataset's own terms and instructor requirements.

No third-party runtime libraries are planned in this scaffold. If that changes, record package name/version/license and review compatibility before adding it. The Qwen model weights have their own license and terms and will not be included in this repository.

## 8. Security and reproducibility boundaries

- No secrets, `.env` files, raw API keys, or user home paths in tracked files.
- `runs/`, caches, downloaded datasets, model weights, and raw prompts/responses are ignored by default until a deliberate sharing policy is agreed.
- Deterministic seed and stable sorting are mandatory for simulation and summaries.
- Preserve raw response fields needed to audit normalization; derived values do not overwrite raw evidence.
- Report missing, invalid, and failed items. Never improve a score by silently excluding difficult failures.

