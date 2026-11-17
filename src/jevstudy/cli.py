"""Command-line interface for the offline scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
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

