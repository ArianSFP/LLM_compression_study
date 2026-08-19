#!/usr/bin/env python3
"""Serialize bounded exact-MXFP4 winner bundles for named candidates/experts.

This stage intentionally requires an explicit expert list.  It cannot silently
materialize every expert or overwrite a previous result namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_expert(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("expert must be LAYER:EXPERT")
    try:
        layer, expert = map(int, fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expert must be LAYER:EXPERT") from error
    if layer < 0 or expert < 0:
        raise argparse.ArgumentTypeError("layer and expert IDs must be non-negative")
    return layer, expert


def _strata_from_fit_facts(path: Path, experts: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    facts = json.loads(path.read_text())
    result: dict[tuple[int, int], str] = {}
    for layer, expert in experts:
        layer_facts: dict[str, Any] | None = facts.get("layers", {}).get(str(layer))
        if layer_facts is None:
            raise KeyError(f"layer {layer} is absent from {path}")
        value = layer_facts.get("strata", {}).get(str(expert))
        if value is None:
            # In-memory integer keys become strings in JSON; accept hand-edited
            # manifests that used a list of pairs as well.
            for item in layer_facts.get("strata", []):
                if isinstance(item, (list, tuple)) and len(item) == 2 and int(item[0]) == expert:
                    value = item[1]
                    break
        if value is None:
            raise KeyError(f"layer {layer} expert {expert} has no fitted stratum in {path}")
        result[(layer, expert)] = str(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--study-output", type=Path, required=True,
        help="Existing codebook study output containing serialized_manifests/codecs",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--experts", nargs="+", type=parse_expert, required=True, metavar="LAYER:EXPERT")
    args = parser.parse_args()

    experiment = args.config.resolve().parents[1]
    sys.path[:0] = [str(experiment / "src"), str(experiment / "scripts")]
    from oracle_study.codebook_serialization import serialize_winner_bundle
    from oracle_study.mxfp4_embed import audit_checkpoint, load_compressed_mxfp4_expert
    from run_codebook_granularity_study import load_candidate

    config = json.loads(args.config.read_text())
    config_path = args.checkpoint / "config.json"
    index_path = args.checkpoint / "model.safetensors.index.json"
    expected = config["reference"]
    hashes = {"config_sha256": sha256(config_path), "index_sha256": sha256(index_path)}
    if hashes["config_sha256"] != expected["config_sha256"]:
        raise RuntimeError("authoritative checkpoint config hash changed")
    if hashes["index_sha256"] != expected["index_sha256"]:
        raise RuntimeError("authoritative checkpoint index hash changed")

    experts = sorted(set(args.experts))
    layers = sorted({layer for layer, _ in experts})
    audit = audit_checkpoint(args.checkpoint, layers)
    if not audit["passed"]:
        raise RuntimeError(audit)
    fit_facts_path = args.study_output / "serialized_manifests" / "fit_run_facts.json"
    strata = _strata_from_fit_facts(fit_facts_path, experts)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    tensors = {
        (layer, expert): {
            projection: load_compressed_mxfp4_expert(
                args.checkpoint, weight_map, layer, expert, projection,
            )
            for projection in ("gate", "up", "down")
        }
        for layer, expert in experts
    }

    if args.destination.exists() and any(args.destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty destination: {args.destination}")
    args.destination.mkdir(parents=True, exist_ok=True)
    codec_root = args.study_output / "serialized_manifests" / "codecs"
    index: dict[str, Any] = {
        "format": "bounded_exact_mxfp4_codebook_winners",
        "checkpoint": {"path": str(args.checkpoint), **hashes, "audit": audit},
        "fit_run_facts": {"path": str(fit_facts_path), "sha256": sha256(fit_facts_path)},
        "experts": [{"layer": layer, "expert": expert, "stratum": strata[(layer, expert)]} for layer, expert in experts],
        "candidates": [],
    }
    for name in args.candidates:
        source_directory = codec_root / name
        candidate = load_candidate(source_directory)
        package = args.destination / name
        manifest = serialize_winner_bundle(
            package,
            candidate,
            tensors,
            strata,
            source={
                "candidate_directory": str(source_directory),
                "candidate_manifest_sha256": sha256(source_directory / "manifest.json"),
                "candidate_research_state_sha256": sha256(source_directory / "research_state.npz"),
                "checkpoint_config_sha256": hashes["config_sha256"],
                "checkpoint_index_sha256": hashes["index_sha256"],
            },
        )
        index["candidates"].append({
            "name": name,
            "manifest": str((package / "manifest.json").relative_to(args.destination)),
            "manifest_sha256": sha256(package / "manifest.json"),
            "rate_accounting": manifest["rate_accounting"],
            "exact_q4_code_equality": manifest["exact_q4_code_equality"],
        })

    index_path_out = args.destination / "index.json"
    index_path_out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "destination": str(args.destination),
        "index": str(index_path_out),
        "candidates": [item["name"] for item in index["candidates"]],
        "experts": len(experts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

