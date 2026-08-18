#!/usr/bin/env python3
"""Consolidate ephemeral RRQ package manifests without copying large codes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = []
    for path in sorted(args.package_root.glob("*/manifest.json")):
        record = json.loads(path.read_text())
        record["manifest_path"] = str(path)
        record["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        packages.append(record)
    payload = {
        "package_root": str(args.package_root),
        "package_root_ephemeral_on_stopped_pod": True,
        "packages": packages,
        "package_count": len(packages),
        "serialized_bytes": sum(int(item["serialized_bytes"]) for item in packages),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(json.dumps({"package_count": len(packages), "serialized_bytes": payload["serialized_bytes"], "manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
