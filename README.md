# Jev Confidence Study

An offline-first project scaffold for studying confidence signals. The current version uses **synthetic data only**. It makes no Jev/LLM calls, downloads no benchmarks, and its output is not empirical evidence.

## Run it

From this folder, with Python 3.11 or later:

```powershell
python -m jevstudy simulate --seed 4901 --out runs/demo
python -m jevstudy validate runs/demo
python -m jevstudy analyze runs/demo --out runs/demo/report.json
python -m unittest discover -s tests -v
```

The `src` layout means the package is not importable directly from a checkout without installation. For these commands, either install it in editable mode with `python -m pip install -e .` (no runtime dependencies), or set `PYTHONPATH=src` in PowerShell for the current terminal:

```powershell
$env:PYTHONPATH = "src"
```

Outputs go under `runs/`, which is ignored by Git. Use a new output directory for each run.

## Current limits

- Fake domains, questions, labels, predictions, confidence values, and costs.
- No real data or model adapters.
- No project license has been selected yet. Ask all project authors before adding a license or publishing the repository.
- Read `docs/PROJECT_DESIGN.md` and `docs/DESIGN_REVIEW.md` before extending the pipeline.
- `docs/REFERENCE_REPO_REVIEW.md` explains how the related Rafe & Das repository fits and what may be useful later.

