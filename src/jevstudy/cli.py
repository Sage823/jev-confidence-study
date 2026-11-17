"""Command-line interface for the offline scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .adapters.jev_api import build_choice_payload, run_one
from .adapters.qwen_local import TransformersQwenEngine, run_one_qwen
from .prepare import DATASET_URLS, combine_frozen, freeze_dataset, validate_items, write_frozen
from .report import analyze_records, write_report
from .runner import run_items
from .schema import read_jsonl, validate_records, write_jsonl
from .simulate import generate_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevstudy", description="Offline confidence-study scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sim = subparsers.add_parser("simulate", help="write deterministic synthetic prediction records")
    sim.add_argument("--seed", type=int, default=4901)
    sim.add_argument("--items-per-domain", type=int, default=24)
    sim.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate prediction records in a run folder")
    validate.add_argument("run_dir", type=Path)
    analyze = subparsers.add_parser("analyze", help="summarize test rows in a run folder")
    analyze.add_argument("run_dir", type=Path)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--extra-runs", type=Path, nargs="*", default=[],
                         help="other run folders for matched-model comparison")
    analyze.add_argument("--allow-incomplete", action="store_true",
                         help="permit mismatched model item sets for pilot diagnostics")
    analyze.add_argument("--bootstrap-draws", type=int, default=200)
    payload_cmd = subparsers.add_parser("jev-payload", help="preview a Jev request without calling the API")
    payload_cmd.add_argument("item_json", type=Path)
    payload_cmd.add_argument("--model", required=True, help="use a pinned model version for reproducibility")
    run_jev = subparsers.add_parser("run-jev", help="call Jev for each normalized item in a JSONL file")
    run_jev.add_argument("--input", type=Path, required=True)
    run_jev.add_argument("--out", type=Path, required=True, help="new output directory under runs/")
    run_jev.add_argument("--model", required=True, help="use a pinned model version for reproducibility")
    run_jev.add_argument("--limit", type=int, help="optional request cap for a small live smoke run")
    run_jev.add_argument("--input-usd-per-million", type=float)
    run_jev.add_argument("--output-usd-per-million", type=float)
    run_qwen = subparsers.add_parser("run-qwen", help="run a pinned local Qwen3 model")
    run_qwen.add_argument("--input", type=Path, required=True)
    run_qwen.add_argument("--out", type=Path, required=True)
    run_qwen.add_argument("--model-id", choices=("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"), required=True)
    run_qwen.add_argument("--revision", required=True, help="pin a Hugging Face commit or release")
    run_qwen.add_argument("--samples", type=int, choices=(0, 5, 10, 20), default=20)
    run_qwen.add_argument("--no-self-report", action="store_true")
    run_qwen.add_argument("--seed", type=int, default=4901)
    run_qwen.add_argument("--max-sample-tokens", type=int, default=512)
    run_qwen.add_argument("--limit", type=int)
    run_qwen.add_argument("--gpu-usd-per-hour", type=float)
    prepare = subparsers.add_parser("prepare", help="freeze locally supplied benchmark rows")
    prepare.add_argument("--dataset", choices=sorted(DATASET_URLS), required=True)
    prepare.add_argument("--calibration-source", type=Path, required=True)
    prepare.add_argument("--test-source", type=Path, required=True)
    prepare.add_argument("--calibration-source-split", required=True)
    prepare.add_argument("--test-source-split", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--source-url", help="canonical URL for the actual source export")
    prepare.add_argument("--source-license", required=True, help="license identifier for the exact source revision")
    prepare.add_argument("--source-license-url", required=True, help="URL to the source license or terms")
    prepare.add_argument("--n-calibration", type=int, required=True)
    prepare.add_argument("--n-test", type=int, required=True)
    prepare.add_argument("--order-items", type=int, default=200)
    prepare.add_argument("--seed", type=int, default=4901)
    prepare.add_argument("--out", type=Path, required=True)
    combine = subparsers.add_parser("combine", help="combine five frozen dataset item files")
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--out", type=Path, required=True)
    combine.add_argument("--protocol-totals", action="store_true",
                         help="require five domains and 900 calibration / 3000 canonical test items")
    args = parser.parse_args(argv)

    if args.command in {"prepare", "combine"}:
        data_root = Path("data").resolve()
        output_path = args.out.resolve()
        if output_path == data_root or not output_path.is_relative_to(data_root):
            parser.error("frozen benchmark outputs must be stored in a new directory inside data/")
        if args.command == "prepare":
            if args.dataset == "strategyqa" and not args.source_url:
                parser.error("StrategyQA needs --source-url for a verified labeled export")
            items = freeze_dataset(
                args.dataset, read_jsonl(args.calibration_source), read_jsonl(args.test_source),
                calibration_source_split=args.calibration_source_split,
                test_source_split=args.test_source_split, revision=args.revision,
                n_calibration=args.n_calibration, n_test=args.n_test,
                seed=args.seed, order_items=args.order_items, source_url=args.source_url)
            manifest = {"dataset": args.dataset, "source_url": args.source_url or DATASET_URLS[args.dataset],
                        "revision": args.revision, "source_license": args.source_license,
                        "source_license_url": args.source_license_url, "seed": args.seed,
                        "n_calibration": args.n_calibration, "n_test": args.n_test,
                        "order_items": args.order_items,
                        "calibration_source_split": args.calibration_source_split,
                        "test_source_split": args.test_source_split,
                        "calibration_source_sha256": hashlib.sha256(args.calibration_source.read_bytes()).hexdigest(),
                        "test_source_sha256": hashlib.sha256(args.test_source.read_bytes()).hexdigest()}
        else:
            items = combine_frozen(args.inputs)
            if args.protocol_totals:
                calibration_count = sum(item["split"] == "calibration" for item in items)
                test_count = sum(item["split"] == "test" and item["condition_id"] == "canonical_order" for item in items)
                order_counts = {domain: sum(item["domain"] == domain and item["split"] == "test"
                                            and item["condition_id"] == "reversed_order" for item in items)
                                for domain in DATASET_URLS}
                if (set(item["domain"] for item in items) != set(DATASET_URLS)
                        or calibration_count != 900 or test_count != 3000
                        or any(count != 200 for count in order_counts.values())):
                    parser.error("protocol requires five domains, 900 calibration, 3000 canonical test, and 200 order items per domain")
            manifest = {"source_item_files": [{"name": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.inputs],
                        "datasets": sorted({item["domain"] for item in items}),
                        "protocol_totals_checked": args.protocol_totals}
        write_frozen(args.out, items, manifest)
        print(f"Froze {len(items)} items to {args.out / 'items.jsonl'}")
        return 0

    if args.command == "simulate":
        if args.items_per_domain < 4:
            parser.error("--items-per-domain must be at least 4")
        args.out.mkdir(parents=True, exist_ok=False)
        rows = generate_records(args.seed, args.items_per_domain)
        records_path = args.out / "predictions.jsonl"
        write_jsonl(records_path, rows)
        digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
        metadata = {
            "source_kind": "simulation",
            "warning": "Synthetic only; not empirical model output.",
            "command": ["simulate", "--seed", args.seed, "--items-per-domain", args.items_per_domain,
                        "--out", str(args.out)],
            "seed": args.seed,
            "items_per_domain": args.items_per_domain,
            "package_version": __version__,
            "protocol_hash": rows[0]["protocol_hash"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(rows),
            "predictions_sha256": digest,
        }
        (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(rows)} synthetic records to {records_path}")
        return 0

    if args.command == "jev-payload":
        item = json.loads(args.item_json.read_text(encoding="utf-8"))
        print(json.dumps(build_choice_payload(item, args.model), ensure_ascii=False, indent=2))
        return 0

    if args.command in {"run-jev", "run-qwen"}:
        output_path = args.out.resolve()
        run_root = Path("runs").resolve()
        if output_path == run_root or not output_path.is_relative_to(run_root) or output_path.exists():
            parser.error("--out must be a new directory inside runs/")
        if args.command == "run-jev" and not os.environ.get("TYPESAFE_API_KEY"):
            parser.error("set TYPESAFE_API_KEY in this PowerShell session before calling Jev")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        items = read_jsonl(args.input)
        if args.limit:
            items = items[:args.limit]
        if not items:
            parser.error("input JSONL contains no items")
        validate_items(items)
        if args.command == "run-jev":
            rates = (args.input_usd_per_million, args.output_usd_per_million)
            if any(rate is not None and rate < 0 for rate in rates):
                parser.error("token prices must be nonnegative")
            if (rates[0] is None) != (rates[1] is None):
                parser.error("provide both input and output token prices")
            model_name = f"jev:{args.model}"
            configuration = {"model": args.model, "api_url": "https://api.typesafe.ai/v1/systemone",
                             "retries": 0, "input_usd_per_million": rates[0],
                             "output_usd_per_million": rates[1]}
            infer = lambda item, run_id: run_one(item, args.model, run_id)
        else:
            if args.max_sample_tokens < 1:
                parser.error("--max-sample-tokens must be positive")
            if args.gpu_usd_per_hour is not None and args.gpu_usd_per_hour < 0:
                parser.error("--gpu-usd-per-hour must be nonnegative")
            engine = TransformersQwenEngine(args.model_id, args.revision)
            model_name = args.model_id
            configuration = {"model": args.model_id, "revision": args.revision,
                             "torch_version": engine.torch_version,
                             "transformers_version": engine.transformers_version,
                             "samples": args.samples, "self_report": not args.no_self_report,
                             "seed": args.seed, "max_sample_tokens": args.max_sample_tokens,
                             "gpu_usd_per_hour": args.gpu_usd_per_hour}
            def infer(item, run_id):
                item_seed = int(hashlib.sha256(f"{args.seed}:{item['item_id']}".encode()).hexdigest()[:8], 16)
                return run_one_qwen(item, engine, run_id, sample_count=args.samples,
                                    self_report=not args.no_self_report, seed=item_seed,
                                    max_sample_tokens=args.max_sample_tokens)
        metadata = run_items(items, args.out, model_name=model_name,
                             configuration=configuration, infer=infer,
                             input_hash=hashlib.sha256(args.input.read_bytes()).hexdigest())
        print(f"Finished: {metadata['success_count']} successes, {metadata['failure_count']} failures; see {args.out}")
        return 0 if metadata["failure_count"] == 0 else 1

    run_dirs = [args.run_dir] + (args.extra_runs if args.command == "analyze" else [])
    rows = [row for run_dir in run_dirs for row in read_jsonl(run_dir / "predictions.jsonl")]
    validate_records(rows)
    if args.command == "validate":
        print(f"Valid: {len(rows)} records; source kinds: {', '.join(sorted({r['source_kind'] for r in rows}))}")
        return 0
    report = analyze_records(rows, allow_incomplete=args.allow_incomplete,
                             bootstrap_draws=args.bootstrap_draws)
    write_report(args.out, report)
    print(f"Wrote summary to {args.out}")
    return 0

