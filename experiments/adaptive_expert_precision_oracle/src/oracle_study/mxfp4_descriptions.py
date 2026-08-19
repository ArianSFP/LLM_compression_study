"""Multiple-description refinement for an exact embedded MXFP4 hierarchy.

The resident parent code is unchanged.  The two exact suffix bits are named
``a`` and ``b``; an optional third stored description is ``q = a xor b``.
Any one description chooses one of the three balanced 2+2 partitions within
the parent.  Any two distinct descriptions recover the exact E2M1 leaf.

This module contains no activation-dependent fitting.  Reconstruction tables
are fit from training-only leaf masses and serialized as FP16 where the locked
tree uses free centroids.  The authoritative four-bit endpoint always comes
from the original packed leaf code, never from a requantized value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .mxfp4_embed import EmbeddedTree, LEAF_VALUES, MXFP4LeafTensor, UNIQUE_E2M1_VALUES


DESCRIPTIONS = ("a", "b", "q")
DESCRIPTION_MASK = {"a": 1, "b": 2, "q": 4}


def _centroid(leaves: np.ndarray, masses: np.ndarray, constrained: bool) -> float:
    weights = masses[leaves]
    if float(weights.sum()) <= 0:
        value = float(np.mean(LEAF_VALUES[leaves]))
    else:
        value = float(np.sum(weights * LEAF_VALUES[leaves]) / weights.sum())
    if constrained:
        value = float(UNIQUE_E2M1_VALUES[np.argmin(np.abs(UNIQUE_E2M1_VALUES - value))])
    return value


def description_bits(tree: EmbeddedTree, leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return parent, a, b and parity descriptions for exact leaf codes."""
    parent, a, b = tree.bitplanes(leaves)
    return parent, a, b, np.bitwise_xor(a, b)


def fit_description_table(
    tree: EmbeddedTree,
    masses: np.ndarray,
    description: str,
) -> np.ndarray:
    """Fit a training-weighted Q3 reconstruction table for one partition."""
    if description not in DESCRIPTIONS:
        raise ValueError(description)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (16,):
        raise ValueError("leaf masses must have shape (16,)")
    parent, a, b, q = description_bits(tree, np.arange(16, dtype=np.uint8))
    bits = {"a": a, "b": b, "q": q}[description]
    constrained = tree.centroid_mode == "e2m1"
    table = np.empty((4, 2), dtype=np.float32)
    for p in range(4):
        for bit in range(2):
            leaves = np.flatnonzero((parent == p) & (bits == bit))
            if len(leaves) != 2:
                raise AssertionError("description does not induce a balanced 2+2 split")
            table[p, bit] = _centroid(leaves, masses, constrained)
    if tree.centroid_mode == "free_fp16":
        table = table.astype(np.float16).astype(np.float32)
    return table


@dataclass(frozen=True)
class MultipleDescriptionTree:
    """Locked parent tree plus independently decoded one-bit descriptions."""

    base: EmbeddedTree
    c3a: np.ndarray
    c3b: np.ndarray
    c3q: np.ndarray

    def __post_init__(self) -> None:
        for name in ("c3a", "c3b", "c3q"):
            if np.asarray(getattr(self, name)).shape != (4, 2):
                raise ValueError(f"{name} must have shape (4, 2)")

    @classmethod
    def fit(cls, base: EmbeddedTree, masses: np.ndarray) -> "MultipleDescriptionTree":
        # Preserve the exact locked A table instead of silently refitting it.
        c3a = np.asarray(base.c3, dtype=np.float32).copy()
        return cls(
            base=base,
            c3a=c3a,
            c3b=fit_description_table(base, masses, "b"),
            c3q=fit_description_table(base, masses, "q"),
        )

    def decode_normalized(self, leaves: np.ndarray, descriptions: Iterable[str]) -> np.ndarray:
        """Decode Q2, one-description Q3, or an exact two-description leaf."""
        selected = tuple(dict.fromkeys(descriptions))
        if any(value not in DESCRIPTIONS for value in selected):
            raise ValueError(selected)
        if len(selected) > 2:
            raise ValueError("at most two descriptions are needed")
        parent, a, b, q = description_bits(self.base, leaves)
        if not selected:
            return np.asarray(self.base.c2, dtype=np.float32)[parent]
        if len(selected) == 1:
            name = selected[0]
            bits = {"a": a, "b": b, "q": q}[name]
            table = {"a": self.c3a, "b": self.c3b, "q": self.c3q}[name]
            return np.asarray(table, dtype=np.float32)[parent, bits]
        # Two distinct binary descriptions determine (a, b) and hence the leaf.
        return LEAF_VALUES[np.asarray(leaves, dtype=np.uint8)]

    def decode(self, tensor: MXFP4LeafTensor, descriptions: Iterable[str]) -> np.ndarray:
        return tensor.expanded_scales * self.decode_normalized(tensor.codes, descriptions)

    def matrices(self, tensor: MXFP4LeafTensor, include_parity: bool) -> dict[str, np.ndarray]:
        names = ("a", "b", "q") if include_parity else ("a", "b")
        result = {"base": self.decode(tensor, ())}
        result.update({name: self.decode(tensor, (name,)) for name in names})
        result["exact"] = tensor.dequantize()
        return result

    def reconstruct_leaf_codes(
        self,
        parent: np.ndarray,
        first_name: str,
        first_bits: np.ndarray,
        second_name: str,
        second_bits: np.ndarray,
    ) -> np.ndarray:
        """Recover exact leaf codes from any two distinct descriptions."""
        if first_name == second_name or first_name not in DESCRIPTIONS or second_name not in DESCRIPTIONS:
            raise ValueError("two distinct valid descriptions are required")
        known = {
            first_name: np.asarray(first_bits, dtype=np.uint8),
            second_name: np.asarray(second_bits, dtype=np.uint8),
        }
        if "a" in known and "b" in known:
            a, b = known["a"], known["b"]
        elif "a" in known and "q" in known:
            a, b = known["a"], np.bitwise_xor(known["a"], known["q"])
        else:
            b, a = known["b"], np.bitwise_xor(known["b"], known["q"])
        return self.base.leaf_by_bits[np.asarray(parent, dtype=np.uint8), a, b]

    def metadata_bytes(self, include_parity: bool) -> int:
        # A is already counted in the locked tree. B/Q add one 4x2 FP16 table each.
        return self.base.metadata_bytes() + (2 if include_parity else 1) * 4 * 2 * 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.as_dict(),
            "c3a": np.asarray(self.c3a, dtype=np.float32).tolist(),
            "c3b": np.asarray(self.c3b, dtype=np.float32).tolist(),
            "c3q": np.asarray(self.c3q, dtype=np.float32).tolist(),
            "metadata_bytes_bidirectional": self.metadata_bytes(False),
            "metadata_bytes_parity": self.metadata_bytes(True),
        }


def description_distortion(
    hierarchy: MultipleDescriptionTree,
    masses: np.ndarray,
    description: str,
) -> float:
    approx = hierarchy.decode_normalized(np.arange(16, dtype=np.uint8), (description,))
    return float(np.sum(np.asarray(masses, dtype=np.float64) * (LEAF_VALUES - approx) ** 2))
