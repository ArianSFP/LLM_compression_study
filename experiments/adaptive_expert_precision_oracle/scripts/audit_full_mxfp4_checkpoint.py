#!/usr/bin/env python3
"""Losslessly audit every routed expert tensor in four study layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 4, 20, 39])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.experiment / "src"))
    from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert, pack_leaf_codes

    audit = audit_checkpoint(args.checkpoint, args.layers)
    if not audit["passed"]:
        raise RuntimeError(audit)
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    aggregate = hashlib.sha256()
    for layer in args.layers:
        for expert in range(256):
            for projection in ("gate", "up", "down"):
                tensor = load_compressed_mxfp4_expert(args.checkpoint, weight_map, layer, expert, projection)
                packed, scales = tensor.exact_streams()
                if not np.array_equal(pack_leaf_codes(tensor.codes), packed):
                    raise AssertionError("lossless nibble roundtrip failed")
                packed_hash = digest(packed); scale_hash = digest(scales)
                aggregate.update(tensor.tensor_name.encode()); aggregate.update(bytes.fromhex(packed_hash)); aggregate.update(bytes.fromhex(scale_hash))
                facts = tensor.storage_bytes()
                rows.append({
                    "layer": layer, "expert_id": expert, "projection": projection,
                    "tensor_name": tensor.tensor_name, "packed_dtype": str(packed.dtype), "scale_dtype": str(scales.dtype),
                    "rows": tensor.shape[0], "columns": tensor.shape[1], "packed_bytes": facts["packed_leaf_bytes"],
                    "scale_bytes": facts["scale_bytes"], "total_bytes": facts["total_bytes"],
                    "effective_bpw": facts["effective_bpw"], "packed_sha256": packed_hash, "scale_sha256": scale_hash,
                    "max_leaf_code": int(tensor.codes.max()), "min_leaf_code": int(tensor.codes.min()),
                })
        print(json.dumps({"audited_layer": layer, "pairs": len(rows)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output / "exact_tensor_audit.parquet", index=False)
    audit.update({
        "opened_and_verified_pairs": len(rows), "expected_pairs": len(args.layers) * 256 * 3,
        "all_pairs_uint8": all(row["packed_dtype"] == "uint8" and row["scale_dtype"] == "uint8" for row in rows),
        "all_pairs_4_25_bpw": all(row["effective_bpw"] == 4.25 for row in rows),
        "aggregate_exact_stream_sha256": aggregate.hexdigest(),
        "proof": "every pair was opened from safetensors; nibbles were losslessly unpacked/repacked; original packed bytes and E8M0 scale bytes were hashed",
    })
    (args.output / "full_checkpoint_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
