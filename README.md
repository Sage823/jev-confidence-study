# Jev Confidence Study

This repository implements the proposal's fixed-answer confidence comparison for Jev and local Qwen3-8B/32B. It follows the frozen-input and append-only run pattern of [Rafe & Das's companion repository](https://github.com/pozapas/system-one-models-survey), with independently written code. No benchmark data, model weights, or API keys are tracked.

## Offline check

Python 3.11+; from this folder:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m jevstudy simulate --out runs/demo
python -m jevstudy validate runs/demo
python -m jevstudy analyze runs/demo --out runs/demo/report.json
```

The demo is synthetic and provides no empirical result. Use a fresh `runs/` directory each time.

## Real experiment path

1. Export licensed, labeled source rows as local JSONL. Run `prepare` once per dataset, passing source split names, pinned source revision, and explicit calibration/test counts. Output belongs under ignored `data/`.
2. Combine the five `items.jsonl` files with `combine --protocol-totals`. This checks **900 calibration and 3,000 canonical test items**, per the proposal prose. The proposal table uses different counts; per-domain allocations still need a team decision.
3. Run the same frozen `data/all/items.jsonl` through Jev and Qwen, writing each model to a fresh `runs/` folder.
4. Run `analyze` with `--extra-runs` to compare matched item IDs.

```powershell
python -m jevstudy prepare --dataset sciq --calibration-source data/raw/sciq_train.jsonl --test-source data/raw/sciq_test.jsonl --calibration-source-split train --test-source-split test --revision SOURCE_COMMIT --source-license CC-BY-NC-3.0 --source-license-url https://creativecommons.org/licenses/by-nc/3.0/ --n-calibration 180 --n-test 600 --order-items 200 --out data/sciq
python -m jevstudy combine --inputs data/sciq/items.jsonl data/hellaswag/items.jsonl data/medmcqa/items.jsonl data/strategyqa/items.jsonl data/mmlu/items.jsonl --out data/all --protocol-totals
python -m jevstudy run-jev --input data/all/items.jsonl --out runs/jev-pilot --model jev-1.13.0 --limit 2
python -m pip install -e '.[qwen]'
python -m jevstudy run-qwen --input data/all/items.jsonl --out runs/qwen8-pilot --model-id Qwen/Qwen3-8B --revision MODEL_COMMIT --samples 5 --limit 2
python -m jevstudy analyze runs/jev-full --extra-runs runs/qwen8-full runs/qwen32-full --out runs/combined-report.json
```

The SciQ numbers above are **illustrative only**; do not run `combine --protocol-totals` until the five per-domain allocations sum to 900/3,000. `run-jev` requires `TYPESAFE_API_KEY` and sends billable requests. Qwen requires model weights and suitable hardware. See [project protocol](docs/PROJECT_DESIGN.md), [design review](docs/DESIGN_REVIEW.md), and [Jev setup](docs/JEV_API_INTEGRATION.md).

The report contains per-domain and equal-weight macro results, four confidence signals where available, in-domain/cross-domain/joint calibration, 5/10/20 sample budgets, order sensitivity, paired model accuracy intervals, and token/latency counts. A missing signal stays visible as a coverage count. Dollar estimates appear only when explicit rates are supplied.
