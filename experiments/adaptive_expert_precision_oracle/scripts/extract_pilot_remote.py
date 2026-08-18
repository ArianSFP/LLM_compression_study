#!/usr/bin/env python3
"""Extract evenly spaced audited capture segments for a cost-bounded pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    experiment = args.config.resolve().parents[1]
    sys.path.insert(0, str(experiment / "src"))
    import oracle_study.capture as capture

    original = capture._segment_dirs
    requested = int(config["capture_segments"])

    def varied_segments(root: Path) -> list[Path]:
        available = original(root)
        if requested >= len(available):
            return available
        indices = np.linspace(0, len(available) - 1, requested, dtype=np.int64)
        selected = [available[int(index)] for index in indices]
        print(json.dumps({
            "available_segments": len(available),
            "selected_segments": [path.name for path in selected],
        }), flush=True)
        return selected

    capture._segment_dirs = varied_segments
    arrays, facts = capture.extract_records(
        Path(config["remote"]["capture_root"]),
        config["layers"],
        int(config["position_stride"]),
        int(config["seed"]),
        progress_every=1,
    )
    facts["sampling_design"] = "six evenly spaced audited capture segments; deterministic token stride; complete-request split"
    facts["pilot_only"] = True
    capture.save_extraction(args.output, arrays, facts)
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
