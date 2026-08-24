# Matched-rate cached-decode D1 allocation package

This package contains the complete six-layer cached single-token allocation
cells at 360 and 725 refinement pages per expert. These are the
runtime-metadata-matched companions to the 384- and 749-page mechanism cells.

## Scope

- Layers: 0, 1, 4, 6, 12, and 23
- Rates: 360 and 725 pages/expert
- Requests: mxfp4-confirm-006, 007, and 008
- Isolated decode groups per cell: 29
- D1 eta grid: 0, 0.00005, 0.0001, 0.00025, 0.0005, and 0.001
- Temperature: 0.0625
- Frontier: rate-specific column-generated complete options
- Hardware: NVIDIA GeForce RTX 3090
- Prompt-position coupling: forbidden
- Historical PR #13 claim at these nonprincipal caps: none

The router-square comparator is regenerated with the unchanged PR #13
algorithm and is labelled regenerated_same_algorithm_nonhistorical_cap.

## Calibration result

Fixed policy selection uses only layers 0, 12, and 23.

| Rate | Selected fixed eta | Calibration crossings | All-six-layer crossings |
| ---: | ---: | ---: | ---: |
| 360 | 0.001 | 2/87 | 6/174 |
| 725 | 0.0001 | 0/87 | 1/174 |

The downstream-tail frontier companion at 360 pages is eta 0.0005, which has
7/174 crossings and lower local qenergy than eta 0.001. At 725 pages it is eta
0.0005, which has 0/174 crossings across all six layers and higher local
qenergy than the fixed eta 0.0001 point.

These are immediate-D1 mechanism observations, not terminal-logit or
deployable-controller results.

## Contents

The results directory contains all 12 finalized cell directories. Every cell
includes:

- complete group and expert allocation tables;
- all 512 precision states for each of eight routed experts;
- exact cached D1 route metrics;
- exact token-oracle selections;
- token-dependent stored selected deltas;
- VJP metrics;
- cache parity metrics; and
- a hash-bearing finalized fact sidecar.

The evidence directory contains the complete stdout/stderr log for each cell.
The artifact_hashes.sha256 ledger verifies all 132 files.

Manifest SHA-256:

650fcffbd1e07f859764ab76003fc22d655f338e595f97da04097bcf1c3dc3a2

## Storage

The authoritative network-storage copy is:

/workspace/pr13_d1_decode_matched_rates_20260824_v2

This repository copy was transferred from that root and verified byte for
byte with sha256sum -c.

## Scientific boundary

The stored delta vectors are activation-dependent slice results. The
product-facing full-tail runner does not inject them as static effects. It
uses the preserved expert IDs and 512-state vectors to reconstruct each
selected allocation on the live full-model cached activation; stored deltas
are transfer diagnostics only.

See ../../D1_CACHED_DECODE_TAIL_KL_METHODS_20260824.md for the exact-prefix
full-tail design and its PRO 6000 hardware gate.
