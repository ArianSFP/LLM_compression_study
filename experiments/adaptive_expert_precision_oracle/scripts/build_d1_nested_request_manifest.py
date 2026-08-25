#!/usr/bin/env python3
"""Build a deterministic, label-free nested request manifest for D1 studies.

The source is pinned byte-for-byte before it is parsed.  Selection is balanced
within exactly eight configured domains: four calibration and twelve evaluation
requests per domain.  No model execution or terminal-quality label is involved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Collection, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "pr13_d1_nested_request_manifest_v1"
FACTS_SCHEMA = "pr13_d1_nested_request_manifest_facts_v1"
REQUIRED_DOMAIN_COUNT = 8
CALIBRATION_PER_DOMAIN = 4
EVALUATION_PER_DOMAIN = 12
SELECTED_PER_DOMAIN = CALIBRATION_PER_DOMAIN + EVALUATION_PER_DOMAIN
REQUIRED_FIELDS = (
    "request_id",
    "domain",
    "prompt_sha256",
    "prompt_token_ids",
)


class ManifestValidationError(ValueError):
    """Raised when an input cannot satisfy the immutable sampling contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError(f"{field} must be a SHA-256 string")
    normalised = value.lower()
    if len(normalised) != 64 or any(
        character not in "0123456789abcdef" for character in normalised
    ):
        raise ManifestValidationError(f"{field} must contain 64 hexadecimal digits")
    return normalised


def _normalise_domains(domains: Sequence[str]) -> tuple[str, ...]:
    if len(domains) != REQUIRED_DOMAIN_COUNT:
        raise ManifestValidationError(
            f"exactly {REQUIRED_DOMAIN_COUNT} domains are required; got {len(domains)}"
        )
    clean: list[str] = []
    for domain in domains:
        if not isinstance(domain, str) or not domain or domain != domain.strip():
            raise ManifestValidationError(
                "configured domains must be non-empty strings without edge whitespace"
            )
        clean.append(domain)
    if len(set(clean)) != len(clean):
        raise ManifestValidationError("configured domains must be unique")
    return tuple(sorted(clean))


def _selection_hash(
    *, seed: int, domain: str, prompt_sha256: str, request_id: str
) -> str:
    # Canonical JSON avoids delimiter ambiguity while fixing the ranking contract.
    identity = json.dumps(
        [SCHEMA, seed, domain, prompt_sha256, request_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(identity)


def _validate_source_row(row: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ManifestValidationError(
            f"source line {line_number} must contain a JSON object"
        )
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ManifestValidationError(
            f"source line {line_number} is missing fields: {', '.join(missing)}"
        )

    raw_request_id = row["request_id"]
    domain = row["domain"]
    if isinstance(raw_request_id, bool):
        raise ManifestValidationError(
            f"source line {line_number} has an invalid request_id"
        )
    if isinstance(raw_request_id, int):
        if raw_request_id < 0:
            raise ManifestValidationError(
                f"source line {line_number} has an invalid request_id"
            )
        request_id = str(raw_request_id)
    elif (
        isinstance(raw_request_id, str)
        and raw_request_id
        and raw_request_id == raw_request_id.strip()
    ):
        request_id = raw_request_id
    else:
        raise ManifestValidationError(
            f"source line {line_number} has an invalid request_id"
        )
    if not isinstance(domain, str) or not domain or domain != domain.strip():
        raise ManifestValidationError(
            f"source line {line_number} has an invalid domain"
        )

    prompt_sha256 = _normalise_sha256(
        row["prompt_sha256"], field=f"source line {line_number} prompt_sha256"
    )
    tokens = row["prompt_token_ids"]
    if not isinstance(tokens, list):
        raise ManifestValidationError(
            f"source line {line_number} prompt_token_ids must be a list"
        )
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in tokens
    ):
        raise ManifestValidationError(
            f"source line {line_number} prompt_token_ids must be non-negative integers"
        )
    return {
        "request_id": request_id,
        "domain": domain,
        "prompt_sha256": prompt_sha256,
        "prompt_token_ids": list(tokens),
    }


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ManifestValidationError(
                    f"source line {line_number} is blank; JSONL rows must be contiguous"
                )
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"source line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            rows.append(_validate_source_row(decoded, line_number=line_number))
    if not rows:
        raise ManifestValidationError("source JSONL contains no requests")

    request_counts = Counter(row["request_id"] for row in rows)
    duplicate_requests = sorted(
        request_id for request_id, count in request_counts.items() if count != 1
    )
    if duplicate_requests:
        raise ManifestValidationError(
            "duplicate request_id values in source: "
            + ", ".join(duplicate_requests[:8])
        )
    prompt_counts = Counter(row["prompt_sha256"] for row in rows)
    duplicate_prompts = sorted(
        prompt_sha256 for prompt_sha256, count in prompt_counts.items() if count != 1
    )
    if duplicate_prompts:
        raise ManifestValidationError(
            "duplicate prompt_sha256 values in source: "
            + ", ".join(duplicate_prompts[:8])
        )
    return rows


def load_excluded_prompt_hashes(
    direct_hashes: Sequence[str], files: Sequence[Path]
) -> tuple[set[str], list[dict[str, Any]]]:
    hashes = {
        _normalise_sha256(value, field="excluded prompt SHA-256")
        for value in direct_hashes
    }
    provenance: list[dict[str, Any]] = []
    for path in files:
        values: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                values.append(
                    _normalise_sha256(
                        value,
                        field=f"{path} line {line_number} excluded prompt SHA-256",
                    )
                )
        hashes.update(values)
        provenance.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "hashes": len(values),
            }
        )
    return hashes, provenance


def build_manifest_rows(
    source_rows: Sequence[dict[str, Any]],
    *,
    domains: Sequence[str],
    excluded_prompt_hashes: Collection[str],
    seed: int,
    max_position: int,
    min_prefix_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configured_domains = _normalise_domains(domains)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ManifestValidationError("seed must be a non-negative integer")
    if (
        isinstance(max_position, bool)
        or not isinstance(max_position, int)
        or max_position < 0
    ):
        raise ManifestValidationError("max_position must be a non-negative integer")
    if (
        isinstance(min_prefix_tokens, bool)
        or not isinstance(min_prefix_tokens, int)
        or min_prefix_tokens < 0
    ):
        raise ManifestValidationError(
            "min_prefix_tokens must be a non-negative integer"
        )
    if max_position < min_prefix_tokens:
        raise ManifestValidationError(
            "max_position must be at least min_prefix_tokens"
        )
    excluded = {
        _normalise_sha256(value, field="excluded prompt SHA-256")
        for value in excluded_prompt_hashes
    }

    by_domain: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in configured_domains
    }
    domain_stats: dict[str, dict[str, int]] = {
        domain: {
            "source_rows": 0,
            "excluded_prompt_hash_rows": 0,
            "short_prefix_rows": 0,
            "eligible_rows": 0,
            "calibration": 0,
            "evaluation": 0,
            "selected": 0,
        }
        for domain in configured_domains
    }
    configured = set(configured_domains)
    unconfigured_rows = 0
    for raw_row in source_rows:
        # Validate programmatic callers just as strictly as JSONL callers.
        row = _validate_source_row(raw_row, line_number=0)
        domain = row["domain"]
        if domain not in configured:
            unconfigured_rows += 1
            continue
        stats = domain_stats[domain]
        stats["source_rows"] += 1
        if row["prompt_sha256"] in excluded:
            stats["excluded_prompt_hash_rows"] += 1
            continue
        tokens = row["prompt_token_ids"]
        decode_position = min(max_position, len(tokens) - 2)
        if decode_position < min_prefix_tokens:
            stats["short_prefix_rows"] += 1
            continue
        selection_hash = _selection_hash(
            seed=seed,
            domain=domain,
            prompt_sha256=row["prompt_sha256"],
            request_id=row["request_id"],
        )
        by_domain[domain].append(
            {
                "schema": SCHEMA,
                "request_id": row["request_id"],
                "domain": domain,
                "prompt_sha256": row["prompt_sha256"],
                "prompt_token_ids": tokens,
                "decode_position": decode_position,
                "prefix_tokens": decode_position,
                "next_token_id": tokens[decode_position + 1],
                "selection_hash": selection_hash,
            }
        )
        stats["eligible_rows"] += 1

    selected: list[dict[str, Any]] = []
    for domain in configured_domains:
        ranked = sorted(
            by_domain[domain],
            key=lambda row: (
                row["selection_hash"],
                row["prompt_sha256"],
                row["request_id"],
            ),
        )
        if len(ranked) < SELECTED_PER_DOMAIN:
            raise ManifestValidationError(
                f"domain {domain!r} has {len(ranked)} eligible requests; "
                f"{SELECTED_PER_DOMAIN} are required"
            )
        for domain_rank, row in enumerate(ranked[:SELECTED_PER_DOMAIN]):
            split = (
                "calibration"
                if domain_rank < CALIBRATION_PER_DOMAIN
                else "evaluation"
            )
            split_rank = (
                domain_rank
                if split == "calibration"
                else domain_rank - CALIBRATION_PER_DOMAIN
            )
            selected.append(
                {
                    **row,
                    "split": split,
                    "domain_rank": domain_rank,
                    "split_rank": split_rank,
                }
            )
            domain_stats[domain][split] += 1
            domain_stats[domain]["selected"] += 1

    calibration = [row for row in selected if row["split"] == "calibration"]
    evaluation = [row for row in selected if row["split"] == "evaluation"]
    calibration_requests = {row["request_id"] for row in calibration}
    evaluation_requests = {row["request_id"] for row in evaluation}
    calibration_prompts = {row["prompt_sha256"] for row in calibration}
    evaluation_prompts = {row["prompt_sha256"] for row in evaluation}
    if calibration_requests & evaluation_requests:
        raise ManifestValidationError("calibration/evaluation request overlap")
    if calibration_prompts & evaluation_prompts:
        raise ManifestValidationError("calibration/evaluation prompt overlap")
    if len({row["request_id"] for row in selected}) != len(selected):
        raise ManifestValidationError("duplicate selected request_id")
    if len({row["prompt_sha256"] for row in selected}) != len(selected):
        raise ManifestValidationError("duplicate selected prompt_sha256")
    if len(calibration) != REQUIRED_DOMAIN_COUNT * CALIBRATION_PER_DOMAIN:
        raise ManifestValidationError("incorrect calibration selection size")
    if len(evaluation) != REQUIRED_DOMAIN_COUNT * EVALUATION_PER_DOMAIN:
        raise ManifestValidationError("incorrect evaluation selection size")

    stats: dict[str, Any] = {
        "source_rows": len(source_rows),
        "configured_domain_rows": sum(
            values["source_rows"] for values in domain_stats.values()
        ),
        "unconfigured_domain_rows": unconfigured_rows,
        "excluded_prompt_hash_rows": sum(
            values["excluded_prompt_hash_rows"] for values in domain_stats.values()
        ),
        "short_prefix_rows": sum(
            values["short_prefix_rows"] for values in domain_stats.values()
        ),
        "eligible_rows": sum(
            values["eligible_rows"] for values in domain_stats.values()
        ),
        "selected_rows": len(selected),
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "domain_split": domain_stats,
    }
    return selected, stats


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _sha256_string_set(values: Collection[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return sha256_bytes(payload)


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def build_nested_request_manifest(
    *,
    source_jsonl: Path,
    expected_source_sha256: str,
    output_jsonl: Path,
    facts_json: Path,
    domains: Sequence[str],
    seed: int,
    max_position: int,
    min_prefix_tokens: int,
    excluded_prompt_hashes: Collection[str] = (),
    exclusion_file_provenance: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    source_jsonl = Path(source_jsonl)
    output_jsonl = Path(output_jsonl)
    facts_json = Path(facts_json)
    expected_sha = _normalise_sha256(
        expected_source_sha256, field="expected source SHA-256"
    )
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"source JSONL not found: {source_jsonl}")
    resolved = {
        source_jsonl.resolve(),
        output_jsonl.resolve(),
        facts_json.resolve(),
    }
    if len(resolved) != 3:
        raise ManifestValidationError(
            "source, selected JSONL, and facts JSON paths must be distinct"
        )
    actual_sha = sha256_file(source_jsonl)
    if actual_sha != expected_sha:
        raise ManifestValidationError(
            f"source SHA-256 mismatch: expected {expected_sha}, observed {actual_sha}"
        )

    source_rows = load_source_rows(source_jsonl)
    selected, stats = build_manifest_rows(
        source_rows,
        domains=domains,
        excluded_prompt_hashes=excluded_prompt_hashes,
        seed=seed,
        max_position=max_position,
        min_prefix_tokens=min_prefix_tokens,
    )
    selected_payload = _jsonl_bytes(selected)
    selected_sha = sha256_bytes(selected_payload)
    normalised_domains = _normalise_domains(domains)
    normalised_exclusions = sorted(
        _normalise_sha256(value, field="excluded prompt SHA-256")
        for value in set(excluded_prompt_hashes)
    )
    selection_spec = {
        "schema": SCHEMA,
        "seed": seed,
        "domains": list(normalised_domains),
        "required_domain_count": REQUIRED_DOMAIN_COUNT,
        "calibration_per_domain": CALIBRATION_PER_DOMAIN,
        "evaluation_per_domain": EVALUATION_PER_DOMAIN,
        "max_position": max_position,
        "min_prefix_tokens": min_prefix_tokens,
        "excluded_prompt_sha256": normalised_exclusions,
        "rank_algorithm": (
            "sha256(canonical_json([schema,seed,domain,prompt_sha256,request_id]))"
        ),
    }
    selection_spec_payload = (
        json.dumps(selection_spec, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    calibration = [row for row in selected if row["split"] == "calibration"]
    evaluation = [row for row in selected if row["split"] == "evaluation"]
    facts: dict[str, Any] = {
        "schema": FACTS_SCHEMA,
        "completed": True,
        "source": {
            "path": str(source_jsonl.resolve()),
            "bytes": source_jsonl.stat().st_size,
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "rows": len(source_rows),
            "fields_read": list(REQUIRED_FIELDS),
        },
        "selection": selection_spec,
        "selection_spec_sha256": sha256_bytes(selection_spec_payload),
        "counts": {key: value for key, value in stats.items() if key != "domain_split"},
        "domain_split": stats["domain_split"],
        "split_identity_hashes": {
            "calibration_request_ids_sha256": _sha256_string_set(
                {row["request_id"] for row in calibration}
            ),
            "calibration_prompt_sha256_set_sha256": _sha256_string_set(
                {row["prompt_sha256"] for row in calibration}
            ),
            "evaluation_request_ids_sha256": _sha256_string_set(
                {row["request_id"] for row in evaluation}
            ),
            "evaluation_prompt_sha256_set_sha256": _sha256_string_set(
                {row["prompt_sha256"] for row in evaluation}
            ),
        },
        "exclusion_files": list(exclusion_file_provenance),
        "output": {
            "path": str(output_jsonl.resolve()),
            "bytes": len(selected_payload),
            "rows": len(selected),
            "sha256": selected_sha,
        },
        "invariants": {
            "exactly_one_decode_position_per_request": True,
            "calibration_evaluation_request_overlap": 0,
            "calibration_evaluation_prompt_overlap": 0,
            "duplicate_selected_request_ids": 0,
            "duplicate_selected_prompt_hashes": 0,
            "model_or_gpu_used": False,
            "terminal_quality_labels_read_or_used": False,
            "next_token_id_retained": True,
        },
    }
    facts_payload = (
        json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    selected_temporary: Path | None = None
    facts_temporary: Path | None = None
    try:
        selected_temporary = _stage_bytes(output_jsonl, selected_payload)
        facts_temporary = _stage_bytes(facts_json, facts_payload)
        os.replace(selected_temporary, output_jsonl)
        selected_temporary = None
        os.replace(facts_temporary, facts_json)
        facts_temporary = None
    finally:
        for temporary in (selected_temporary, facts_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return facts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument(
        "--source-sha256",
        required=True,
        help="Pinned SHA-256 of the source JSONL bytes",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--facts-json", required=True, type=Path)
    parser.add_argument(
        "--domains",
        required=True,
        nargs=REQUIRED_DOMAIN_COUNT,
        metavar="DOMAIN",
        help="Exactly eight balanced source domains",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-position", required=True, type=int)
    parser.add_argument("--min-prefix-tokens", required=True, type=int)
    parser.add_argument(
        "--exclude-prompt-sha256",
        action="append",
        default=[],
        metavar="SHA256",
        help="Prompt hash to exclude; may be repeated",
    )
    parser.add_argument(
        "--exclude-prompt-sha256-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="Newline-delimited prompt hashes to exclude; may be repeated",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        excluded, exclusion_provenance = load_excluded_prompt_hashes(
            args.exclude_prompt_sha256,
            args.exclude_prompt_sha256_file,
        )
        facts = build_nested_request_manifest(
            source_jsonl=args.source_jsonl,
            expected_source_sha256=args.source_sha256,
            output_jsonl=args.output_jsonl,
            facts_json=args.facts_json,
            domains=args.domains,
            seed=args.seed,
            max_position=args.max_position,
            min_prefix_tokens=args.min_prefix_tokens,
            excluded_prompt_hashes=excluded,
            exclusion_file_provenance=exclusion_provenance,
        )
    except (ManifestValidationError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "completed": True,
                "selected_rows": facts["counts"]["selected_rows"],
                "selected_sha256": facts["output"]["sha256"],
                "facts": str(args.facts_json),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
