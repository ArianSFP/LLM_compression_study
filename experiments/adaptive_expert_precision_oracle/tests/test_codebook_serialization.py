from pathlib import Path

import numpy as np
import pytest

from oracle_study.codebook_granularity import book_from_order, fit_learned_scalar_book
from oracle_study.codebook_serialization import (
    HEADER_BYTES,
    serialize_progressive_projection,
    serialize_winner_bundle,
    read_progressive_projection,
)
from oracle_study.mxfp4_embed import MXFP4LeafTensor, pack_leaf_codes


def synthetic_tensor(label: str = "synthetic") -> MXFP4LeafTensor:
    # Eight rows exercise one exact coordinate-major output byte; 64 columns
    # exercise two native MXFP4 blocks and two 32-weight family selectors.
    codes = np.resize(np.arange(16, dtype=np.uint8), (8, 64))
    scales = (np.arange(16, dtype=np.uint8).reshape(8, 2) + np.uint8(119)).copy()
    return MXFP4LeafTensor(codes, scales, label)


def exact_books():
    masses = np.arange(1, 17, dtype=np.float64)
    return [
        book_from_order(range(16), masses, "ascending"),
        book_from_order(reversed(range(16)), masses, "descending"),
    ]


def test_projection_payload_roundtrips_exact_codes_scales_and_layout(tmp_path: Path) -> None:
    source = synthetic_tensor()
    books = exact_books()
    selectors = np.arange(16, dtype=np.uint16) % 2
    path = tmp_path / "gate.cgw"

    facts = serialize_progressive_projection(path, source, books, selectors, 32, "gate")
    restored = read_progressive_projection(path, books, selectors, expected_projection="gate")

    assert facts["header_bytes"] == HEADER_BYTES
    assert facts["bytes"] == path.stat().st_size
    assert facts["bytes"] % 64 == 0
    assert all(item["offset"] % 64 == 0 for item in facts["streams"])
    assert all(item["offset"] % 512 == 0 for item in facts["streams"])
    assert [item["layout"] for item in facts["streams"]] == [
        "row_major", "row_major", "coordinate_major", "coordinate_major",
    ]
    assert np.array_equal(restored.tensor.codes, source.codes)
    assert np.array_equal(restored.tensor.scale_codes, source.scale_codes)
    assert pack_leaf_codes(restored.tensor.codes).tobytes() == pack_leaf_codes(source.codes).tobytes()
    assert facts["source_packed_leaf_sha256"] == facts["reconstructed_packed_leaf_sha256"]
    assert facts["resident_bytes"] + facts["q3_streamed_bytes"] + facts["q4_streamed_bytes"] == facts["bytes"]


def test_winner_bundle_counts_actual_metadata_headers_and_stat_bytes(tmp_path: Path) -> None:
    books = exact_books()
    selectors = np.arange(16, dtype=np.uint16) % 2
    candidate = {
        "name": "B2_g32_synthetic",
        "design": "B",
        "design_id": 2,
        "sharing": "global_projection",
        "families": 2,
        "group_size": 32,
        "selector_bits": 1,
        "experts": [[0, 7]],
        "books": {f"global/{projection}": books for projection in ("gate", "up", "down")},
        "selectors": {
            f"l00_e007_{projection}": selectors.copy() for projection in ("gate", "up", "down")
        },
        "modifiers": {},
    }
    tensors = {
        (0, 7): {
            projection: synthetic_tensor(projection) for projection in ("gate", "up", "down")
        }
    }
    root = tmp_path / "bundle"
    manifest = serialize_winner_bundle(root, candidate, tensors, {(0, 7): "hot"})

    files = manifest["table_files"] + manifest["expert_metadata_files"] + manifest["projection_files"]
    assert all(item["bytes"] == (root / item["path"]).stat().st_size for item in files)
    assert len(manifest["table_files"]) == 3
    assert all(item["header_bytes"] == 64 for item in manifest["table_files"])
    assert len(manifest["expert_metadata_files"]) == 1
    assert manifest["expert_metadata_files"][0]["header_bytes"] == 64
    accounting = manifest["rate_accounting"]
    assert accounting["codec_stat_bytes"] == sum(item["bytes"] for item in files)
    assert accounting["codec_stat_bytes"] == (
        accounting["resident_bytes"]
        + accounting["q3_streamed_bytes"]
        + accounting["q4_streamed_bytes"]
    )
    assert manifest["exact_q4_code_equality"] is True
    assert accounting["manifest_bytes_excluded_control_plane"] == (root / "manifest.json").stat().st_size


def test_exact_bundle_rejects_learned_q4_diagnostic(tmp_path: Path) -> None:
    source = synthetic_tensor()
    learned = fit_learned_scalar_book(np.arange(1, 17, dtype=np.float64))
    with pytest.raises(ValueError, match="exact-MXFP4"):
        serialize_progressive_projection(tmp_path / "bad.cgw", source, [learned], None, 32, "gate")
