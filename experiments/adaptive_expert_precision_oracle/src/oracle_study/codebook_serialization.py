"""Deployment-byte serialization for exact progressive codebook candidates.

The fitting code stores enough research state to resume an experiment, but that
state is not the deployment representation.  This module writes a bounded,
auditable deployment bundle containing:

* row-major two-bit Q2 parent codes;
* the checkpoint's byte-exact E8M0 scale codes;
* coordinate-major Q3 and Q4 one-bit planes (the selective page layout);
* the winning codebook tables and per-expert selector/modifier metadata.

Every projection payload has a fixed 64-byte header and 512-byte stream
alignment so selective suffix pages are physically aligned. Shared codebook
tables have a 64-byte header/payload alignment. The generated
manifest charges headers and padding from actual file sizes and records SHA256
digests for both files and logical streams.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from .codebook_granularity import (
    ProgressiveBook,
    book_code_arrays,
    serialize_metadata_file,
)
from .mxfp4_embed import (
    BLOCK_SIZE,
    MXFP4LeafTensor,
    pack_1bit,
    pack_2bit,
    pack_leaf_codes,
    unpack_1bit,
    unpack_2bit,
)


PROJECTIONS = ("gate", "up", "down")
PROJECTION_IDS = {"gate": 0, "up": 1, "down": 2}
PROJECTION_NAMES = {value: key for key, value in PROJECTION_IDS.items()}

WEIGHT_MAGIC = b"CGBWGHT\0"
TABLE_MAGIC = b"CGBTABL\0"
FORMAT_VERSION = 1
HEADER_BYTES = 64
PROJECTION_STREAM_ALIGNMENT_BYTES = 512
TABLE_PAYLOAD_ALIGNMENT_BYTES = 64
STREAM_NAMES = ("q2_parent", "e8m0_scale", "q3_refinement", "q4_refinement")

# Exactly 32 bytes, followed by four fixed-order (offset, logical length)
# entries.  The final entry ends at byte 64.
_WEIGHT_HEADER = struct.Struct("<8sHHBBHIIHHI")
_STREAM_ENTRY = struct.Struct("<II")
assert _WEIGHT_HEADER.size == 32
assert _WEIGHT_HEADER.size + len(STREAM_NAMES) * _STREAM_ENTRY.size == HEADER_BYTES

# The remaining bytes in the 64-byte table header are reserved and zero.
_TABLE_HEADER = struct.Struct("<8sHHHHII")

FLAG_EXACT_MXFP4 = 1 << 0
FLAG_SUFFIX_COORDINATE_MAJOR = 1 << 1
FLAG_E8M0_UINT8 = 1 << 2


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _safe_key(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace("-", "_")


def _record_key(layer: int, expert: int, projection: str) -> str:
    return f"l{layer:02d}_e{expert:03d}_{projection}"


def _candidate_scope(
    candidate: Mapping[str, Any], layer: int, expert: int, projection: str, stratum: str,
) -> str:
    sharing = candidate["sharing"]
    if sharing in ("baseline_global", "global_projection"):
        return f"global/{projection}"
    if sharing == "layer_projection":
        return f"layer_{layer:02d}/{projection}"
    if sharing == "cluster_projection":
        return f"cluster_{stratum}/{projection}"
    if sharing in ("expert_projection", "independent_vector"):
        return f"layer_{layer:02d}/expert_{expert:03d}/{projection}"
    raise ValueError(f"unsupported sharing scope: {sharing}")


def _book_grid(
    shape: tuple[int, int], books: Sequence[ProgressiveBook], selectors: np.ndarray | None,
    group_size: int, projection: str,
) -> np.ndarray:
    rows, columns = shape
    if selectors is None:
        vector_count = rows if projection in ("gate", "up") else columns
        if len(books) != vector_count:
            raise ValueError(
                f"{projection} independent table count {len(books)} != functional vectors {vector_count}"
            )
        ids = np.arange(vector_count, dtype=np.int64)
        return (
            np.broadcast_to(ids[:, None], shape)
            if projection in ("gate", "up")
            else np.broadcast_to(ids[None, :], shape)
        )
    if columns % group_size:
        raise ValueError(f"group size {group_size} does not divide {shape}")
    ids = np.asarray(selectors, np.int64)
    expected = rows * (columns // group_size)
    if ids.size != expected:
        raise ValueError(f"selector count {ids.size} != expected {expected} for {shape}")
    if ids.size and (ids.min() < 0 or ids.max() >= len(books)):
        raise ValueError("selector references a missing book")
    return np.repeat(ids.reshape(rows, columns // group_size), group_size, axis=1)


def _inverse_leaf_table(books: Sequence[ProgressiveBook]) -> np.ndarray:
    inverse = np.empty((len(books), 4, 2, 2), np.uint8)
    for book_id, book in enumerate(books):
        if not isinstance(book, ProgressiveBook) or hasattr(book, "c4"):
            raise ValueError("deployment serializer accepts exact-MXFP4 ProgressiveBook tables only")
        for leaf in range(16):
            inverse[
                book_id,
                int(book.parent[leaf]),
                int(book.r1[leaf]),
                int(book.r2[leaf]),
            ] = leaf
    return inverse


def reconstruct_leaf_codes(
    parent: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    books: Sequence[ProgressiveBook],
    selectors: np.ndarray | None,
    group_size: int,
    projection: str,
) -> np.ndarray:
    """Invert the stored hierarchy digits and recover exact MXFP4 leaf codes."""
    p = np.asarray(parent, np.uint8)
    first = np.asarray(r1, np.uint8)
    second = np.asarray(r2, np.uint8)
    if p.shape != first.shape or p.shape != second.shape:
        raise ValueError("hierarchy planes have different shapes")
    grid = _book_grid(tuple(map(int, p.shape)), books, selectors, group_size, projection)
    return _inverse_leaf_table(books)[grid, p, first, second]


@dataclass(frozen=True)
class ProjectionPayload:
    """Decoded contents of one ``.cgw`` projection file."""

    tensor: MXFP4LeafTensor
    parent: np.ndarray
    q3_refinement: np.ndarray
    q4_refinement: np.ndarray
    projection: str
    group_size: int


def serialize_progressive_projection(
    path: Path,
    tensor: MXFP4LeafTensor,
    books: Sequence[ProgressiveBook],
    selectors: np.ndarray | None,
    group_size: int,
    projection: str,
) -> dict[str, Any]:
    """Write all hierarchy streams for one projection and prove Q4 equality."""
    if projection not in PROJECTION_IDS:
        raise ValueError(projection)
    rows, columns = tensor.shape
    if columns % 4:
        raise ValueError("Q2 parent rows must be divisible by four")
    _inverse_leaf_table(books)  # Reject learned-Q4/non-exact books before writing.
    parent, first, second = book_code_arrays(
        tensor.codes, list(books), selectors, int(group_size), projection,
    )
    reconstructed = reconstruct_leaf_codes(
        parent, first, second, books, selectors, int(group_size), projection,
    )
    source_packed = pack_leaf_codes(tensor.codes).tobytes(order="C")
    restored_packed = pack_leaf_codes(reconstructed).tobytes(order="C")
    if source_packed != restored_packed or not np.array_equal(reconstructed, tensor.codes):
        raise AssertionError("Q4 leaf/code reconstruction is not exact")

    # Q2 is row-major for the resident decoder.  Suffix bits are transposed
    # before packing so all output-channel bits for one input coordinate remain
    # contiguous, matching selective page actions.
    stream_payloads = (
        pack_2bit(parent).tobytes(order="C"),
        np.asarray(tensor.scale_codes, np.uint8).tobytes(order="C"),
        pack_1bit(first.T).tobytes(order="C"),
        pack_1bit(second.T).tobytes(order="C"),
    )
    body = bytearray()
    entries: list[tuple[int, int]] = []
    paddings: list[int] = []
    cursor = HEADER_BYTES
    for payload in stream_payloads:
        aligned = _align(cursor, PROJECTION_STREAM_ALIGNMENT_BYTES)
        padding = aligned - cursor
        body.extend(b"\0" * padding)
        offset = HEADER_BYTES + len(body)
        body.extend(payload)
        entries.append((offset, len(payload)))
        paddings.append(padding)
        cursor = offset + len(payload)
    trailing_padding = _align(cursor, PROJECTION_STREAM_ALIGNMENT_BYTES) - cursor
    body.extend(b"\0" * trailing_padding)

    flags = FLAG_EXACT_MXFP4 | FLAG_SUFFIX_COORDINATE_MAJOR | FLAG_E8M0_UINT8
    header = bytearray(HEADER_BYTES)
    _WEIGHT_HEADER.pack_into(
        header,
        0,
        WEIGHT_MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        PROJECTION_IDS[projection],
        flags,
        len(STREAM_NAMES),
        rows,
        columns,
        BLOCK_SIZE,
        int(group_size),
        len(body),
    )
    for index, entry in enumerate(entries):
        _STREAM_ENTRY.pack_into(header, _WEIGHT_HEADER.size + index * _STREAM_ENTRY.size, *entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(body))
    stat_bytes = path.stat().st_size
    expected_bytes = HEADER_BYTES + len(body)
    if stat_bytes != expected_bytes:
        raise AssertionError((stat_bytes, expected_bytes))

    stream_facts = []
    for index, (name, payload, (offset, length), padding) in enumerate(
        zip(STREAM_NAMES, stream_payloads, entries, paddings, strict=True)
    ):
        if index == len(STREAM_NAMES) - 1:
            padding += trailing_padding
        stream_facts.append({
            "name": name,
            "offset": offset,
            "logical_bytes": length,
            "padding_bytes": padding,
            "stored_bytes": length + padding,
            "sha256": _sha256_bytes(payload),
            "layout": "coordinate_major" if name.startswith(("q3_", "q4_")) else "row_major",
        })
    resident_bytes = HEADER_BYTES + sum(
        item["stored_bytes"] for item in stream_facts if item["name"] in ("q2_parent", "e8m0_scale")
    )
    q3_bytes = next(item["stored_bytes"] for item in stream_facts if item["name"] == "q3_refinement")
    q4_bytes = next(item["stored_bytes"] for item in stream_facts if item["name"] == "q4_refinement")
    if resident_bytes + q3_bytes + q4_bytes != stat_bytes:
        raise AssertionError("projection byte classes do not sum to stat bytes")
    return {
        "path": str(path),
        "bytes": stat_bytes,
        "sha256": _sha256_file(path),
        "header_bytes": HEADER_BYTES,
        "alignment_bytes": PROJECTION_STREAM_ALIGNMENT_BYTES,
        "rows": rows,
        "columns": columns,
        "weights": rows * columns,
        "projection": projection,
        "native_block_size": BLOCK_SIZE,
        "hierarchy_group_size": int(group_size),
        "suffix_layout": "coordinate_major_input_then_packed_output",
        "streams": stream_facts,
        "resident_bytes": resident_bytes,
        "q3_streamed_bytes": q3_bytes,
        "q4_streamed_bytes": q4_bytes,
        "source_packed_leaf_sha256": _sha256_bytes(source_packed),
        "reconstructed_packed_leaf_sha256": _sha256_bytes(restored_packed),
        "source_e8m0_sha256": _sha256_bytes(np.asarray(tensor.scale_codes, np.uint8).tobytes(order="C")),
        "exact_q4_code_equality": True,
    }


def read_progressive_projection(
    path: Path,
    books: Sequence[ProgressiveBook],
    selectors: np.ndarray | None,
    *,
    expected_projection: str | None = None,
) -> ProjectionPayload:
    """Read and validate a ``.cgw`` file, including exact hierarchy inversion."""
    payload = path.read_bytes()
    if len(payload) < HEADER_BYTES:
        raise ValueError("truncated progressive projection header")
    (
        magic, version, header_bytes, projection_id, flags, stream_count, rows, columns,
        native_block_size, group_size, body_bytes,
    ) = _WEIGHT_HEADER.unpack_from(payload, 0)
    if magic != WEIGHT_MAGIC or version != FORMAT_VERSION or header_bytes != HEADER_BYTES:
        raise ValueError("unsupported progressive projection format")
    if flags != FLAG_EXACT_MXFP4 | FLAG_SUFFIX_COORDINATE_MAJOR | FLAG_E8M0_UINT8:
        raise ValueError("projection flags do not describe the exact selective layout")
    if stream_count != len(STREAM_NAMES) or native_block_size != BLOCK_SIZE:
        raise ValueError("unexpected stream/block geometry")
    if body_bytes != len(payload) - HEADER_BYTES:
        raise ValueError("recorded body length does not match stat bytes")
    projection = PROJECTION_NAMES.get(projection_id)
    if projection is None or (expected_projection is not None and projection != expected_projection):
        raise ValueError(f"projection mismatch: {projection} != {expected_projection}")

    streams: list[bytes] = []
    previous_end = HEADER_BYTES
    for index in range(len(STREAM_NAMES)):
        offset, length = _STREAM_ENTRY.unpack_from(
            payload, _WEIGHT_HEADER.size + index * _STREAM_ENTRY.size,
        )
        if offset % PROJECTION_STREAM_ALIGNMENT_BYTES or offset < previous_end or offset + length > len(payload):
            raise ValueError("invalid stream offset/alignment")
        streams.append(payload[offset: offset + length])
        previous_end = offset + length

    parent_expected = rows * (columns // 4)
    scale_expected = rows * (columns // BLOCK_SIZE)
    suffix_expected = columns * ((rows + 7) // 8)
    expected_lengths = (parent_expected, scale_expected, suffix_expected, suffix_expected)
    if tuple(map(len, streams)) != expected_lengths:
        raise ValueError(f"stream size mismatch: {tuple(map(len, streams))} != {expected_lengths}")

    parent_packed = np.frombuffer(streams[0], np.uint8).reshape(rows, columns // 4)
    parent = unpack_2bit(parent_packed, columns).copy()
    scale_codes = np.frombuffer(streams[1], np.uint8).reshape(rows, columns // BLOCK_SIZE).copy()
    suffix_shape = (columns, (rows + 7) // 8)
    first = unpack_1bit(np.frombuffer(streams[2], np.uint8).reshape(suffix_shape), rows).T.copy()
    second = unpack_1bit(np.frombuffer(streams[3], np.uint8).reshape(suffix_shape), rows).T.copy()
    codes = reconstruct_leaf_codes(parent, first, second, books, selectors, group_size, projection)
    tensor = MXFP4LeafTensor(codes.astype(np.uint8, copy=False), scale_codes, str(path))
    return ProjectionPayload(tensor, parent, first, second, projection, int(group_size))


def serialize_book_table_file(path: Path, books: Sequence[ProgressiveBook]) -> dict[str, Any]:
    """Serialize one shared/local exact book table with counted framing bytes."""
    values = list(books)
    if not values:
        raise ValueError("cannot serialize an empty codebook table")
    for book in values:
        if not isinstance(book, ProgressiveBook) or hasattr(book, "c4"):
            raise ValueError("learned-Q4 books are diagnostic and cannot enter an exact bundle")
    sizes = {len(book.to_bytes()) for book in values}
    if sizes != {32}:
        raise ValueError(f"exact book size changed: {sizes}")
    logical = b"".join(book.to_bytes() for book in values)
    padding = _align(len(logical), TABLE_PAYLOAD_ALIGNMENT_BYTES) - len(logical)
    body = logical + b"\0" * padding
    header = bytearray(HEADER_BYTES)
    _TABLE_HEADER.pack_into(
        header, 0, TABLE_MAGIC, FORMAT_VERSION, HEADER_BYTES, 32, len(values), len(logical), len(body),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + body)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "header_bytes": HEADER_BYTES,
        "alignment_bytes": TABLE_PAYLOAD_ALIGNMENT_BYTES,
        "payload_bytes": len(logical),
        "padding_bytes": padding,
        "books": len(values),
        "bytes_per_book": 32,
    }


def _relative_fact(fact: Mapping[str, Any], root: Path) -> dict[str, Any]:
    result = dict(fact)
    result["path"] = str(Path(str(fact["path"])).relative_to(root))
    return result


def serialize_winner_bundle(
    output: Path,
    candidate: Mapping[str, Any],
    tensors: Mapping[tuple[int, int], Mapping[str, MXFP4LeafTensor]],
    strata: Mapping[tuple[int, int], str],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize selected candidate/expert pairs into one non-overwriting bundle."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty bundle directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted((int(layer), int(expert)) for layer, expert in tensors)
    if not selected:
        raise ValueError("at least one expert must be selected")
    available = {tuple(map(int, pair)) for pair in candidate.get("experts", selected)}
    missing = set(selected) - available
    if missing:
        raise ValueError(f"candidate was not fitted for selected experts: {sorted(missing)}")

    needed_scopes: dict[str, Sequence[ProgressiveBook]] = {}
    projection_scopes: dict[tuple[int, int, str], str] = {}
    for layer, expert in selected:
        if (layer, expert) not in strata:
            raise KeyError(f"missing stratum for layer {layer} expert {expert}")
        for projection in PROJECTIONS:
            scope = _candidate_scope(candidate, layer, expert, projection, strata[(layer, expert)])
            if scope not in candidate["books"]:
                raise KeyError(f"missing fitted book scope {scope}")
            needed_scopes[scope] = candidate["books"][scope]
            projection_scopes[(layer, expert, projection)] = scope

    table_files = []
    for scope in sorted(needed_scopes):
        fact = serialize_book_table_file(
            output / "metadata" / "tables" / f"{_safe_key(scope)}.cgt",
            needed_scopes[scope],
        )
        table_files.append({"role": "resident_codebook_table", "scope": scope, **_relative_fact(fact, output)})

    expert_metadata_files = []
    projection_files = []
    selector_bits = int(candidate["selector_bits"])
    modifiers_all = candidate.get("modifiers", {})
    for layer, expert in selected:
        selectors_by_projection: dict[str, np.ndarray] = {}
        modifiers_by_projection: dict[str, np.ndarray] = {}
        for projection in PROJECTIONS:
            key = _record_key(layer, expert, projection)
            selectors_by_projection[projection] = np.asarray(
                candidate.get("selectors", {}).get(key, np.empty(0, np.uint16)), np.uint16,
            )
            if key in modifiers_all:
                modifiers_by_projection[projection] = np.asarray(modifiers_all[key], np.float32)
        if modifiers_by_projection and set(modifiers_by_projection) != set(PROJECTIONS):
            raise ValueError("vector modifiers must be present for every projection")
        if selector_bits or modifiers_by_projection:
            fact = serialize_metadata_file(
                output / "metadata" / "experts" / f"layer_{layer:02d}_expert_{expert:03d}.cgb",
                int(candidate["design_id"]),
                selector_bits,
                selectors_by_projection,
                modifiers_by_projection or None,
            )
            expert_metadata_files.append({
                "role": "resident_expert_metadata", "layer": layer, "expert": expert,
                "alignment_bytes": 16, **_relative_fact(fact, output),
            })

        for projection in PROJECTIONS:
            key = _record_key(layer, expert, projection)
            selector = candidate.get("selectors", {}).get(key)
            scope = projection_scopes[(layer, expert, projection)]
            fact = serialize_progressive_projection(
                output / "weights" / f"layer_{layer:02d}" / f"expert_{expert:03d}" / f"{projection}.cgw",
                tensors[(layer, expert)][projection],
                needed_scopes[scope],
                selector,
                int(candidate["group_size"]),
                projection,
            )
            # Read from disk and repeat the exact code/scale equality test.  This
            # catches header, layout, padding, and unpacking bugs, not just an
            # in-memory hierarchy error.
            restored = read_progressive_projection(
                Path(fact["path"]), needed_scopes[scope], selector, expected_projection=projection,
            )
            original = tensors[(layer, expert)][projection]
            if not np.array_equal(restored.tensor.codes, original.codes):
                raise AssertionError("serialized Q4 codes differ from the checkpoint")
            if not np.array_equal(restored.tensor.scale_codes, original.scale_codes):
                raise AssertionError("serialized E8M0 codes differ from the checkpoint")
            projection_files.append({
                "role": "progressive_weight_streams", "layer": layer, "expert": expert,
                "scope": scope, **_relative_fact(fact, output),
            })

    metadata_bytes = sum(int(item["bytes"]) for item in table_files + expert_metadata_files)
    weight_resident_bytes = sum(int(item["resident_bytes"]) for item in projection_files)
    q3_streamed_bytes = sum(int(item["q3_streamed_bytes"]) for item in projection_files)
    q4_streamed_bytes = sum(int(item["q4_streamed_bytes"]) for item in projection_files)
    total_weights = sum(int(item["weights"]) for item in projection_files)
    codec_bytes = metadata_bytes + sum(int(item["bytes"]) for item in projection_files)
    resident_bytes = metadata_bytes + weight_resident_bytes
    if codec_bytes != resident_bytes + q3_streamed_bytes + q4_streamed_bytes:
        raise AssertionError("bundle byte classes do not sum to serialized stat bytes")

    excluded = {"books", "selectors", "modifiers", "table_files", "metadata_files", "book_scopes"}
    candidate_metadata = {key: value for key, value in candidate.items() if key not in excluded}
    manifest = {
        "format": "exact_mxfp4_progressive_codebook_bundle",
        "version": FORMAT_VERSION,
        "candidate": candidate_metadata,
        "source": dict(source or {}),
        "selected_experts": [
            {"layer": layer, "expert": expert, "stratum": strata[(layer, expert)]}
            for layer, expert in selected
        ],
        "layout": {
            "projection_header_bytes": HEADER_BYTES,
            "projection_stream_alignment_bytes": PROJECTION_STREAM_ALIGNMENT_BYTES,
            "q2_parent": "row_major_little_endian_2bit",
            "e8m0_scale": "checkpoint_exact_uint8_row_major_native_block_32",
            "q3_refinement": "coordinate_major_little_endian_1bit",
            "q4_refinement": "coordinate_major_little_endian_1bit",
            "book_table_header_bytes": HEADER_BYTES,
            "expert_metadata_header_bytes": 64,
            "expert_metadata_alignment_bytes": 16,
        },
        "rate_accounting": {
            "weights": total_weights,
            "weight_resident_bytes": weight_resident_bytes,
            "resident_metadata_bytes": metadata_bytes,
            "resident_bytes": resident_bytes,
            "q3_streamed_bytes": q3_streamed_bytes,
            "q4_streamed_bytes": q4_streamed_bytes,
            "all_suffix_streamed_bytes": q3_streamed_bytes + q4_streamed_bytes,
            "codec_stat_bytes": codec_bytes,
            "resident_bpw": 8.0 * resident_bytes / total_weights,
            "q3_streamed_bpw": 8.0 * q3_streamed_bytes / total_weights,
            "q4_streamed_bpw": 8.0 * q4_streamed_bytes / total_weights,
            "all_suffix_streamed_bpw": 8.0 * (q3_streamed_bytes + q4_streamed_bytes) / total_weights,
            "total_effective_bpw": 8.0 * codec_bytes / total_weights,
            "manifest_bytes_excluded_control_plane": None,
        },
        "table_files": table_files,
        "expert_metadata_files": expert_metadata_files,
        "projection_files": projection_files,
        "exact_q4_code_equality": all(item["exact_q4_code_equality"] for item in projection_files),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["rate_accounting"]["manifest_bytes_excluded_control_plane"] = manifest_path.stat().st_size
    # Record the final manifest size.  A second write can change the number of
    # digits, so converge before returning.
    for _ in range(3):
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        size = manifest_path.stat().st_size
        if manifest["rate_accounting"]["manifest_bytes_excluded_control_plane"] == size:
            break
        manifest["rate_accounting"]["manifest_bytes_excluded_control_plane"] = size
    return manifest
