#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path.insert(0, str(experiment / "src"))
    from oracle_study.capture import extract_records, save_extraction

    arrays, facts = extract_records(
        Path(config["remote"]["capture_root"]),
        config["layers"],
        int(config["position_stride"]),
        int(config["seed"]),
    )
    save_extraction(args.output, arrays, facts)
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
