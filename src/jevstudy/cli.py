"""Command-line interface for the offline scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import __version__
from .adapters.jev_api import build_choice_payload, run_one
from .report import analyze_records, write_report
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
    payload_cmd = subparsers.add_parser("jev-payload", help="preview a Jev request without calling the API")
    payload_cmd.add_argument("item_json", type=Path)
    payload_cmd.add_argument("--model", required=True, help="use a pinned model version for reproducibility")
    run_jev = subparsers.add_parser("run-jev", help="call Jev for each normalized item in a JSONL file")
    run_jev.add_argument("--input", type=Path, required=True)
    run_jev.add_argument("--out", type=Path, required=True, help="new output directory under runs/")
    run_jev.add_argument("--model", required=True, help="use a pinned model version for reproducibility")
    run_jev.add_argument("--limit", type=int, help="optional request cap for a small live smoke run")
    args = parser.parse_args(argv)

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

    if args.command == "run-jev":
        if not os.environ.get("TYPESAFE_API_KEY"):
            parser.error("set TYPESAFE_API_KEY in this PowerShell session before calling Jev")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        runs_root = Path("runs").resolve()
        output_path = args.out.resolve()
        if output_path == runs_root or not output_path.is_relative_to(runs_root):
            parser.error("API outputs must be stored in a new directory inside runs/")
        items = read_jsonl(args.input)
        if args.limit:
            items = items[:args.limit]
        if not items:
            parser.error("input JSONL contains no items")
        args.out.mkdir(parents=True, exist_ok=False)
        run_id = f"jev-{uuid4().hex[:12]}"
        predictions_path = args.out / "predictions.jsonl"
        errors_path = args.out / "errors.jsonl"
        successes = 0
        failures = 0
        for item in items:
            try:
                record = run_one(item, args.model, run_id)
                with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                successes += 1
                print(f"OK {item['item_id']}")
            except Exception as exc:
                error = {"run_id": run_id, "item_id": item.get("item_id"),
                         "model": args.model, "error_type": type(exc).__name__, "error": str(exc)}
                with errors_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")
                failures += 1
                print(f"ERROR {item.get('item_id')}: {type(exc).__name__}: {exc}")
        metadata = {"source_kind": "model_output", "model_requested": args.model,
                    "api_url": "https://api.typesafe.ai/v1/systemone", "run_id": run_id,
                    "input_file_name": args.input.name,
                    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    "requested_count": len(items),
                    "success_count": successes, "failure_count": failures,
                    "package_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "note": "No automatic retries. Review errors.jsonl; a timeout can leave request billing uncertain."}
        (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if successes:
            validate_records(read_jsonl(predictions_path))
        print(f"Finished: {successes} successes, {failures} failures; see {args.out}")
        return 0 if failures == 0 else 1

    records_path = args.run_dir / "predictions.jsonl"
    rows = read_jsonl(records_path)
    validate_records(rows)
    if args.command == "validate":
        print(f"Valid: {len(rows)} records; source kinds: {', '.join(sorted({r['source_kind'] for r in rows}))}")
        return 0
    report = analyze_records(rows)
    write_report(args.out, report)
    print(f"Wrote summary to {args.out}")
    return 0

