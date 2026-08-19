#!/usr/bin/env python3
"""Validate and analyze activation-dependent sparse-streaming frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_study.sparse_streaming_analysis import analyze, schema_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply validation-only promotion gates, freeze decisions, and report "
            "held-out sparse-streaming results."
        )
    )
    parser.add_argument("--input", type=Path, action="append", default=[], help="Run output directory; repeat to merge bounded runs.")
    parser.add_argument("--config", type=Path, help="Locked sparse-streaming JSON configuration.")
    parser.add_argument("--output", type=Path, help="Analysis output directory.")
    parser.add_argument("--validate-only", action="store_true", help="Validate all inputs and compute decisions in memory without writing artifacts.")
    parser.add_argument("--write-schema", type=Path, help="Write the canonical input schema and exit; no experimental data are fabricated.")
    args = parser.parse_args()
    if args.write_schema is not None:
        args.write_schema.parent.mkdir(parents=True, exist_ok=True)
        args.write_schema.write_text(json.dumps(schema_document(), indent=2, sort_keys=True) + "\n")
        return
    if not args.input or args.config is None:
        parser.error("--input and --config are required unless --write-schema is used")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is used")
    result = analyze(
        args.input,
        args.config,
        args.output if args.output is not None else Path("."),
        validate_only=args.validate_only,
    )
    if args.validate_only:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
