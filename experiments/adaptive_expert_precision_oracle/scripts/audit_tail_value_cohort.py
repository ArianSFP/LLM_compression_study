#!/usr/bin/env python3
"""Input-only audit and prefix-disjoint development/validation/reserve manifests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def effective(row, position=None):
    p = row["decode_position"] if position is None else position
    return tuple(row["prompt_token_ids"][:p + 1])


def audit(rows):
    result = {}
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        result[split] = {"requests": len(selected), "unique_effective_inputs": len({effective(r) for r in selected}),
                         "domains": {d: dict(requests=len(s := [r for r in selected if r["domain"] == d]),
                                             unique_effective_inputs=len({effective(r) for r in s}))
                                     for d in sorted({r["domain"] for r in selected})}}
    if {"calibration", "evaluation"} <= {r["split"] for r in rows}:
        cal = {effective(r) for r in rows if r["split"] == "calibration"}
        result["evaluation_inputs_seen_in_calibration"] = sum(effective(r) in cal for r in rows if r["split"] == "evaluation")
    return result


def corrected(source, old):
    old_hashes = {r["prompt_sha256"] for r in old}
    available = [dict(r, request_id=str(r["request_id"]), decode_position=min(63, len(r["prompt_token_ids"]) - 2))
                 for r in source if r["prompt_sha256"] not in old_hashes]
    if any(r["decode_position"] < 8 for r in available):
        raise ValueError("insufficient prefix")
    counts = Counter(effective(r) for r in available)
    if any(n != 1 for n in counts.values()):
        raise ValueError("new effective prefixes are not unique")
    old_inputs = {effective(r) for r in old}
    if any(effective(r) in old_inputs for r in available):
        raise ValueError("old effective input overlap")
    result = []
    for domain in sorted({r["domain"] for r in available}):
        candidates = sorted([r for r in available if r["domain"] == domain],
                            key=lambda r: digest([20260909, domain, r["prompt_sha256"]]))
        if len(candidates) != 16:
            raise ValueError(f"expected 16 unused requests for {domain}, got {len(candidates)}")
        for i, r in enumerate(candidates):
            r.update(split="development" if i < 4 else "validation" if i < 8 else "reserve",
                     effective_input_sha256=digest(effective(r)), schema="tail_value_prefix_v1")
            result.append(r)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--old-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = [json.loads(l) for l in a.source.read_text().splitlines()]
    old = [json.loads(l) for l in a.old_manifest.read_text().splitlines()]
    rows = corrected(source, old)
    a.output.mkdir(parents=True, exist_ok=True)
    facts = {"schema": "tail_value_cohort_audit_v1", "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
             "old_manifest_sha256": hashlib.sha256(a.old_manifest.read_bytes()).hexdigest(),
             "old": audit(old), "new": audit(rows), "position_rule": "min(63,token_count-2)",
             "old_full_request_overlap": 0, "effective_input_overlap": 0,
             "reserve_outcomes_may_be_executed": False,
             "limitation": "Prefix uniqueness does not prove independence of source templates or documents; source supplies no grouping IDs. Reserve is unused in this experiment, not certified unseen in all prior projects."}
    for split in ("development", "validation", "reserve"):
        path = a.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows if r["split"] == split))
        facts[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output / "cohort_audit.json").write_text(json.dumps(facts, indent=2) + "\n")
    (a.output / "old_effective_inputs.json").write_text(json.dumps([
        dict(request_id=str(r["request_id"]), domain=r["domain"], split=r["split"],
             effective_input_sha256=digest(effective(r))) for r in old], indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
