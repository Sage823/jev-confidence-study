# Jev Confidence Study

An offline-first project scaffold for studying confidence signals. It supports synthetic runs and has an optional explicit Jev API command. No API request happens unless you run `run-jev`; synthetic outputs are not empirical evidence.

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
- No benchmark datasets or Qwen model adapter.
- `run-jev` sends live requests and may incur charges; follow `docs/JEV_API_INTEGRATION.md` and start with `--limit 2`.
- No project license has been selected yet. Ask all project authors before adding a license or publishing the repository.
- Read `docs/PROJECT_DESIGN.md` and `docs/DESIGN_REVIEW.md` before extending the pipeline.
- `docs/REFERENCE_REPO_REVIEW.md` explains how the related Rafe & Das repository fits and what may be useful later.
- `docs/JEV_API_INTEGRATION.md` explains the optional live API path, input format, dry-run preview, and cost-bearing command.

