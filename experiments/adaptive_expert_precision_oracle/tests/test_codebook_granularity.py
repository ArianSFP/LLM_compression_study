from pathlib import Path

import numpy as np

from oracle_study.codebook_granularity import (
    ProgressiveBook,
    book_code_arrays,
    book_from_order,
    decode_with_books,
    fit_family_books,
    fit_learned_scalar_book,
    pack_selector,
    serialize_metadata_file,
    unpack_selector,
)
from oracle_study.mxfp4_embed import MXFP4LeafTensor


def tensor() -> MXFP4LeafTensor:
    codes = np.tile(np.arange(16, dtype=np.uint8), (2, 4))
    scales = np.full((2, 2), 127, np.uint8)
    return MXFP4LeafTensor(codes, scales, "synthetic")


def test_book_is_exact_32_byte_roundtrip() -> None:
    book = book_from_order(range(16), np.arange(1, 17, dtype=np.float64))
    payload = book.to_bytes()
    restored = ProgressiveBook.from_bytes(payload)
    assert len(payload) == 32
    assert np.array_equal(restored.parent, book.parent)
    assert np.array_equal(restored.r1, book.r1)
    assert np.array_equal(restored.r2, book.r2)
    assert np.array_equal(restored.c2, book.c2)
    assert np.array_equal(restored.c3, book.c3)


def test_non_byte_aligned_selector_roundtrip() -> None:
    values = np.asarray([0, 7, 3, 5, 1, 6, 2], np.uint16)
    payload = pack_selector(values, 3)
    assert unpack_selector(payload, len(values), 3).tolist() == values.tolist()


def test_family_decode_preserves_q4_codes() -> None:
    source = tensor()
    masses = np.stack((np.arange(1, 17), np.arange(16, 0, -1), np.ones(16), np.eye(16)[3] + 0.1))
    books, assignment, _ = fit_family_books(masses, 2, 0.005, 19, order_count=16, max_iterations=3)
    selectors = np.resize(assignment, source.shape[0] * (source.shape[1] // 32))
    parent, r1, r2 = book_code_arrays(source.codes, books, selectors, 32, "gate")
    inverse = np.empty((len(books), 4, 2, 2), np.uint8)
    for family, book in enumerate(books):
        for leaf in range(16):
            inverse[family, book.parent[leaf], book.r1[leaf], book.r2[leaf]] = leaf
    grid = np.repeat(selectors.reshape(2, 2), 32, axis=1)
    assert np.array_equal(inverse[grid, parent, r1, r2], source.codes)
    assert np.array_equal(decode_with_books(source, books, selectors, 32, "gate", 4), source.dequantize())


def test_design_c_file_counts_actual_header_and_padding(tmp_path: Path) -> None:
    selectors = {
        "gate": np.arange(4, dtype=np.uint16) % 8,
        "up": np.arange(4, dtype=np.uint16) % 8,
        "down": np.arange(4, dtype=np.uint16) % 8,
    }
    modifiers = {name: np.ones((2, 4), np.float32) for name in selectors}
    facts = serialize_metadata_file(tmp_path / "one.cgb", 3, 3, selectors, modifiers)
    assert facts["bytes"] == (tmp_path / "one.cgb").stat().st_size
    assert facts["header_bytes"] == 64
    assert facts["bytes"] % 16 == 0


def test_learned_leaf_diagnostic_is_explicitly_non_bijective_capable() -> None:
    book = fit_learned_scalar_book(np.arange(1, 17, dtype=np.float64))
    assert len(book.to_bytes()) == 64
    assert book.normalized_table(4).shape == (16,)
    assert np.isfinite(book.distortion_vector(4)).all()
