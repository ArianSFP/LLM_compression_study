from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))

from build_d1_nested_request_manifest import (  # noqa: E402
    ManifestValidationError,
    build_manifest_rows,
    build_nested_request_manifest,
    load_source_rows,
    sha256_file,
)


DOMAINS = tuple(f"domain-{index}" for index in range(8))


def _prompt_hash(domain: str, index: int) -> str:
    return hashlib.sha256(f"{domain}:{index}".encode()).hexdigest()


def _rows(*, per_domain: int = 20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_index, domain in enumerate(DOMAINS):
        for index in range(per_domain):
            rows.append(
                {
                    "request_id": f"request-{domain_index:02d}-{index:03d}",
                    "domain": domain,
                    "prompt_sha256": _prompt_hash(domain, index),
                    "prompt_token_ids": [
                        1000 + domain_index,
                        2000 + index,
                        3000,
                        3001,
                        3002,
                        3003,
                        3004,
                        3005,
                    ],
                    # The builder must project away unrelated or terminal fields.
                    "terminal_logit_kl": 123.0,
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_balanced_nested_manifest_is_deterministic_and_label_free(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source_sha = _write_jsonl(source, _rows())
    selected_path = tmp_path / "selected.jsonl"
    facts_path = tmp_path / "facts.json"

    facts = build_nested_request_manifest(
        source_jsonl=source,
        expected_source_sha256=source_sha,
        output_jsonl=selected_path,
        facts_json=facts_path,
        domains=tuple(reversed(DOMAINS)),
        seed=20260825,
        max_position=5,
        min_prefix_tokens=2,
    )
    selected = _read_jsonl(selected_path)

    assert len(selected) == 128
    assert sum(row["split"] == "calibration" for row in selected) == 32
    assert sum(row["split"] == "evaluation" for row in selected) == 96
    assert len({row["request_id"] for row in selected}) == 128
    assert len({row["prompt_sha256"] for row in selected}) == 128
    assert all(row["decode_position"] == 5 for row in selected)
    assert all(row["prefix_tokens"] == 5 for row in selected)
    assert all(row["next_token_id"] == row["prompt_token_ids"][6] for row in selected)
    assert all("terminal_logit_kl" not in row for row in selected)
    for domain in DOMAINS:
        domain_rows = [row for row in selected if row["domain"] == domain]
        assert sum(row["split"] == "calibration" for row in domain_rows) == 4
        assert sum(row["split"] == "evaluation" for row in domain_rows) == 12

    assert facts["counts"]["selected_rows"] == 128
    assert facts["invariants"]["terminal_quality_labels_read_or_used"] is False
    assert facts["output"]["sha256"] == sha256_file(selected_path)
    assert json.loads(facts_path.read_text()) == facts

    reversed_source = tmp_path / "source-reversed.jsonl"
    reversed_sha = _write_jsonl(reversed_source, list(reversed(_rows())))
    reversed_output = tmp_path / "selected-reversed.jsonl"
    build_nested_request_manifest(
        source_jsonl=reversed_source,
        expected_source_sha256=reversed_sha,
        output_jsonl=reversed_output,
        facts_json=tmp_path / "facts-reversed.json",
        domains=DOMAINS,
        seed=20260825,
        max_position=5,
        min_prefix_tokens=2,
    )
    assert reversed_output.read_bytes() == selected_path.read_bytes()


def test_uint64_request_ids_are_normalized_losslessly_before_ranking(
    tmp_path: Path,
) -> None:
    rows = _rows(per_domain=16)
    uint64_max = (1 << 64) - 1
    for index, row in enumerate(rows):
        row["request_id"] = uint64_max - index
    source = tmp_path / "uint64-source.jsonl"
    _write_jsonl(source, rows)

    loaded = load_source_rows(source)
    assert loaded[0]["request_id"] == str(uint64_max)
    assert all(isinstance(row["request_id"], str) for row in loaded)

    selected, _ = build_manifest_rows(
        loaded,
        domains=DOMAINS,
        excluded_prompt_hashes=(),
        seed=20260825,
        max_position=5,
        min_prefix_tokens=2,
    )
    assert len(selected) == 128
    assert str(uint64_max) in {row["request_id"] for row in selected}
    assert all(isinstance(row["request_id"], str) for row in selected)


def test_exclusions_and_short_prefix_gate_precede_hash_ranking() -> None:
    rows = _rows(per_domain=19)
    excluded = {
        _prompt_hash(domain, 0)
        for domain in DOMAINS
    }
    for row in rows:
        if row["prompt_sha256"] == _prompt_hash(str(row["domain"]), 1):
            row["prompt_token_ids"] = [1, 2, 3]

    selected, stats = build_manifest_rows(
        rows,
        domains=DOMAINS,
        excluded_prompt_hashes=excluded,
        seed=7,
        max_position=6,
        min_prefix_tokens=2,
    )

    assert len(selected) == 128
    assert not ({row["prompt_sha256"] for row in selected} & excluded)
    assert stats["excluded_prompt_hash_rows"] == 8
    assert stats["short_prefix_rows"] == 8
    assert all(row["decode_position"] >= 2 for row in selected)
    assert all(row["next_token_id"] == row["prompt_token_ids"][row["decode_position"] + 1] for row in selected)


@pytest.mark.parametrize("duplicate_field", ["request_id", "prompt_sha256"])
def test_source_rejects_duplicate_request_or_prompt_identity(
    tmp_path: Path, duplicate_field: str
) -> None:
    rows = _rows()
    rows[1][duplicate_field] = rows[0][duplicate_field]
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, rows)

    with pytest.raises(ManifestValidationError, match="duplicate"):
        load_source_rows(source)


def test_requires_exactly_eight_domains_and_sufficient_domain_coverage() -> None:
    with pytest.raises(ManifestValidationError, match="exactly 8 domains"):
        build_manifest_rows(
            _rows(),
            domains=DOMAINS[:-1],
            excluded_prompt_hashes=(),
            seed=1,
            max_position=5,
            min_prefix_tokens=1,
        )

    sparse = [
        row
        for row in _rows()
        if row["domain"] != DOMAINS[0] or int(str(row["request_id"]).split("-")[-1]) < 15
    ]
    with pytest.raises(ManifestValidationError, match="16 are required"):
        build_manifest_rows(
            sparse,
            domains=DOMAINS,
            excluded_prompt_hashes=(),
            seed=1,
            max_position=5,
            min_prefix_tokens=1,
        )


def test_pinned_source_mismatch_does_not_replace_existing_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, _rows())
    output = tmp_path / "selected.jsonl"
    facts = tmp_path / "facts.json"
    output.write_text("selected sentinel\n")
    facts.write_text("facts sentinel\n")

    with pytest.raises(ManifestValidationError, match="source SHA-256 mismatch"):
        build_nested_request_manifest(
            source_jsonl=source,
            expected_source_sha256="0" * 64,
            output_jsonl=output,
            facts_json=facts,
            domains=DOMAINS,
            seed=1,
            max_position=5,
            min_prefix_tokens=1,
        )

    assert output.read_text() == "selected sentinel\n"
    assert facts.read_text() == "facts sentinel\n"
