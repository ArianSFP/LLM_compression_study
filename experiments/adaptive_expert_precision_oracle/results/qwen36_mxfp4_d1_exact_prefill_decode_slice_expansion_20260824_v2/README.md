# Exact-prefill decode-only D1 slice expansion

This package contains the complete compact evidence for the cached,
single-token D1 objective expansion run on 2026-08-24. It covers layers 0, 1,
4, 6, 12, and 23; 384 and 749 refinement pages per expert; three validation
requests; and 29 isolated decode-surrogate positions per layer/rate cell.

The principal held-out result is that a calibration-selected strict D1 policy
reduces exact next-router top-8 crossings from 9/87 to 2/87 at 384 pages and
from 5/87 to 1/87 at 749 pages. Exact-combined local qenergy alone produces
10/87 and 3/87. This supports the proposed router-boundary mechanism, but it is
not yet evidence of lower terminal KL or production latency.

See [the full methodology and findings](analysis/D1_EXACT_PREFILL_DECODE_SLICE_REPORT.md).
The machine-readable aggregate gate record is
[`d1_decode_analysis_facts.json`](analysis/d1_decode_analysis_facts.json).
The complete compact-package hashes are in
[`PACKAGE_MANIFEST.json`](PACKAGE_MANIFEST.json).

Directory contents:

- `analysis/`: aggregate Parquet tables, identity audits, calibration and
  held-out results, and the report;
- `results/`: all 12 finalized cell directories, including complete
  expert-level allocations, full 512-entry selected-state vectors, exact
  route metrics, cache parity, VJP timing, selected deltas, and per-cell facts;
- `evidence/`: the 12 successful execution logs and the preserved initial
  in-place-cache autograd failure.

The 20 GiB rate-independent option-geometry cache and exact code snapshots are
retained at
`/workspace/pr13_d1_decode_slice_pilot_20260824_v2` on network storage. They
are intentionally not duplicated in Git. Every scientific output is hashed by
its cell fact file, and every aggregate output is hashed by the analysis fact
file.
