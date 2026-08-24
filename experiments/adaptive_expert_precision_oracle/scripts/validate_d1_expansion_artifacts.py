#!/usr/bin/env python3
"""Validate raw promoted D1 cells against the immutable expansion config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_tail_quality import (  # noqa: E402
    index_oracle_cells,
    sha256,
    validate_expansion_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    cells = index_oracle_cells(args.layer_dir)
    result = validate_expansion_artifacts(config, cells)
    result["config_file"] = str(args.config)
    result["config_sha256"] = sha256(args.config)
    result["validator_sha256"] = sha256(Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"validated {len(cells)} cells against {args.config.name}; "
        f"config_sha256={result['config_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
