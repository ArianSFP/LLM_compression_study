#!/usr/bin/env python3
"""Sequential pilot wrapper with invertible generalized basis and prefix paging."""

from __future__ import annotations

import oracle_study.bases as bases
import run_phase_a_v2_remote as phase_a_fix
import run_phase_b_remote as implementation
import run_phase_b_v2_remote as paging_fix


if __name__ == "__main__":
    bases.generalized_transform = phase_a_fix.generalized_full_rank
    implementation.candidate_counts = paging_fix.prefix_candidate_counts
    implementation.main()
