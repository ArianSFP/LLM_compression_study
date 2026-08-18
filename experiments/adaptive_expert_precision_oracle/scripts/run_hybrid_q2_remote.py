#!/usr/bin/env python3
"""Run the complete-expert hybrid oracle with the deterministic Q2 base."""

from __future__ import annotations

import oracle_study.quant as quant
from run_progressive_q2_remote import q2_quantize
import run_hybrid_remote as implementation


if __name__ == "__main__":
    quant.binary_quantize = q2_quantize
    implementation.main()
