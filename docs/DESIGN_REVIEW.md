# Design Review

**Review status:** original offline scaffold approved; API addendum reviewed for a bounded Jev integration, with research-protocol issues recorded below.
**Reviewed artifact:** `PROJECT_DESIGN.md`  
**Review date:** 2026-11-17

## 1. Summary

The design is suitable for a first implementation because it isolates the reusable scientific pipeline from API access, real benchmark data, and model downloads. It gives a beginner a runnable vertical slice, keeps synthetic output visibly labeled, and sets a boundary against copying the Rafe & Das repository. The metrics, schema, and calibration implementation must be checked against hand-worked examples before being used on real runs.

This approval covers only the offline scaffold. It does not approve the full empirical protocol or establish that benchmark redistribution is permitted.

## 2. Blocking protocol inconsistencies found in the proposal

### 2.1 Dataset counts do not add up

The table in `proposal_v2.tex` specifies test counts of 1,000 + 1,000 + 1,000 + 500 + 1,000 = **4,500**, and calibration counts of 300 + 300 + 300 + 200 + 300 = **1,400**. The prose instead says **3,000 test / 900 calibration**. These are incompatible plans. The implementation therefore must not hard-code either total. Before a real-data run, the team must pick a consistent table and prose total and record exact item IDs and split sizes in a frozen config.

### 2.2 “Equal answer accuracy” is not ensured by identical items

The RQ says “at equal answer accuracy and equal item sets,” but identical items and fixed answers alone do not make Jev and Qwen equally accurate. The project can compare confidence conditional on each model's own correctness, report accuracy alongside it, and optionally run a pre-specified accuracy-matched/stratified comparison. It must not claim accuracy equality unless the protocol actually enforces or statistically adjusts for it.

### 2.3 Calibration and validation split need an explicit protocol

The proposal has a calibration split, but the code still needs a frozen rule for selecting calibration examples, fitting each domain/joint/cross-domain mapping, selecting hyperparameters, handling small classes, and producing uncertainty intervals. In-sample fit metrics must be labeled as fit diagnostics, not held-out evidence. A calibration-size curve is recommended before interpreting many subgroup fits.

### 2.4 Benchmark licenses require source-by-source review

The proposal names SciQ as CC BY-NC 3.0, HellaSwag as MIT, and MedMCQA as Apache-2.0. This scan found direct source/mirror metadata is not consistently populated: the SciQ dataset metadata JSON and MMLU metadata report an empty license field, while some dataset cards or secondary listings declare a license. StrategyQA's official GitHub repo has a license file, but the exact dataset download/terms should be checked at its canonical page. A mirror's metadata alone is not legal clearance. The scaffold bundles no benchmark content and no dataset mirror code.

### 2.5 Full-proposal calibration wording needs care

The design separates distribution calibration from scalar correctness calibration. The report should keep these as different estimands: temperature scaling of an option distribution can be scored with multiclass Brier/NLL and classwise calibration; a scalar monotone mapping predicts whether the chosen answer is correct. Do not compare their ECE values as if they estimate the same thing without stating the target event and binning method.

## 3. Implementation acceptance checks

The scaffold is ready to code if each check below is covered by a deterministic synthetic test:

1. Identical seed yields byte-stable logical records; changing the seed changes generated records.
2. Validation catches malformed keys, missing options, duplicate condition/item keys, invalid probability values, and calibration/test leakage.
3. Metric implementations match hand-computable examples, including ties in AUROC and edge cases with one class.
4. Every signal is oriented so larger values mean more likely correct before AUROC/risk-coverage computation.
5. Calibration parameters are fitted using calibration rows only; changing test labels cannot change fitted parameters.
6. A permutation changes presented order but preserves key-to-text and key-to-gold identity; agreement is computed per item.
7. All generated output says `source_kind=simulation` and documentation says synthetic output is not empirical evidence.
8. Running the full simulated path requires no network, API key, third-party package, or downloaded dataset.

## 4. License and reuse disposition

- Rafe & Das code, data, and results are not copied. Repository-level MIT and curated-data CC BY 4.0 statements are useful reference information, not a license for their underlying third-party benchmark assets.
- We author new scaffold code and synthetic fixtures. No third-party code or dataset material is included.
- A project `LICENSE` is intentionally deferred until all project authors choose and approve one. Until then the code is not licensed for public reuse by default.
- The scaffold can be built locally without distributing a license file. Before a public GitHub repository is created, choose a license for original code and complete the benchmark/model asset inventory.

## 5. Review outcome

**Approved for implementation:** offline schema, deterministic simulator, metrics, signals, calibration boundary, reports, and synthetic tests.  
**Not approved for implementation in this milestone:** real benchmark downloads, model weights, Jev/Qwen API integrations, or claims based on simulated results.  
**Required before empirical study:** reconcile the counts in §2.1, freeze a complete split/calibration protocol, check each dataset's canonical license and reuse terms, and correct the RQ language about equal accuracy.

## 6. API integration addendum (2026-11-17)

The API adapter in `src/jevstudy/adapters/jev_api.py` was added after checking TypeSafe's official [Quick start](https://docs.typesafe.ai/introduction/quickstart). It submits `model`, `state`, and typed `questions` to the documented System One endpoint, requests a `Choice`, reads the returned choice/probability map and usage fields, and records the resolved model and raw response. The adapter uses only Python's standard-library HTTP client. `run-jev` is the only command that sends requests; payload preview and all tests are offline.

The API contract is covered by fixture-based tests, not a live call. The integration is intentionally not yet a production runner: there is no automatic retry, checkpoint/resume, rate scheduler, dollar-cost calculation, or Qwen adapter. Since a timeout may occur after the provider received a request, automatic retries are disabled to avoid unintentionally duplicating billable calls. Before a large experiment, pin a model version, do a small authorized pilot, check the response's actual model field, agree on quota/cost limits, and complete the protocol/license items above.

