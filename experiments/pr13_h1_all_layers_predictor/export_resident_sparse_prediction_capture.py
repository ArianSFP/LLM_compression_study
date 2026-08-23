#!/usr/bin/env python3
"""Run the resident+sparse exporter with an aligned resident-rollout parent state."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def align(root: Path, layer: int, dataset: dict[str, Any]) -> torch.Tensor:
    captured: dict[tuple[str, int, str], np.ndarray] = {}
    for split in ("train", "validation"):
        with np.load(root / f"{split}_layer_{layer:02d}.npz", allow_pickle=False) as data:
            for index in range(len(data["position"])):
                key = (
                    str(data["request_id"][index]),
                    int(data["position"][index]),
                    str(data["prefix_hash"][index]),
                )
                if key in captured:
                    raise RuntimeError(f"duplicate identity {key}")
                captured[key] = np.asarray(data["x"][index], np.float32)
    return torch.from_numpy(np.stack([
        captured[(str(request), int(position), str(prefix))]
        for request, position, prefix in zip(
            dataset["request_id"], dataset["target_position"], dataset["prefix_hash"]
        )
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parent-capture", type=Path, required=True)
    parser.add_argument("--base-script", type=Path, default=Path("/workspace/export_resident_sparse_prediction.py"))
    known, remaining = parser.parse_known_args()
    layer_parser = argparse.ArgumentParser(add_help=False)
    layer_parser.add_argument("--layer", type=int, default=14)
    layer_parser.add_argument("--output", type=Path, required=True)
    base_args, _ = layer_parser.parse_known_args(remaining)
    spec = importlib.util.spec_from_file_location("resident_sparse_capture_base", known.base_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(known.base_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_load = module.torch.load

    def patched_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        value = original_load(path, *args, **kwargs)
        if isinstance(value, dict) and "parent_target_x" in value and "prefix_hash" in value:
            value = dict(value)
            value["parent_target_x"] = align(known.parent_capture, base_args.layer, value).half()
        return value

    module.torch.load = patched_load
    sys.argv = [sys.argv[0], *remaining]
    module.main()
    manifest_path = base_args.output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_capture"] = str(known.parent_capture)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "resident_capture_parent_recorded", "parent_capture": str(known.parent_capture)}), flush=True)


if __name__ == "__main__":
    main()
