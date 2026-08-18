#!/usr/bin/env python3
"""Run progressive projection sensitivity with mean-absolute-scale W1."""

from __future__ import annotations

import oracle_study.quant as quant
from oracle_study.quant import binary_quantize as reference_binary_quantize
import run_progressive_remote as implementation


def simple_binary(weights, group_size=64, activation_second_moment=None):
    return reference_binary_quantize(weights, group_size, None)


if __name__ == "__main__":
    quant.binary_quantize = simple_binary
    implementation.main()
