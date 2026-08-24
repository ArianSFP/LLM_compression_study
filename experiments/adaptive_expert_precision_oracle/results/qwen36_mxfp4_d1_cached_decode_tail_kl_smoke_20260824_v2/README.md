# Cached-decode downstream-tail KL smoke package

This directory packages the completed validation stage for the exact-prefix,
single-current-token downstream-tail experiment.

## Status

- The complete 24-cell policy input grid validates.
- All 132 matched-rate inputs pass their immutable SHA-256 manifest.
- Terminal quality replay has not been run.

No KL, NLL, final-hidden, or cache-drift claim is made by this package.

## Preflight artifacts

- `preflight/d1_cached_decode_tail_fixed_calibration.parquet`
- `preflight/d1_cached_decode_tail_policy_bank_manifest.json`

SHA-256:

```text
3d01cb47477c8bb71cb3c5169cbf1114e240751de1ae2e2ee02113999c459405  d1_cached_decode_tail_fixed_calibration.parquet
24865a22808e8eee6c5132de97445725b48ea545ea0608881a855282a6139e3d  d1_cached_decode_tail_policy_bank_manifest.json
```

## Boundary

The run processes one current decode token behind a complete exact prefix
cache, recomputes each policy's selected precision states on that token's live
activation, and never couples deltas across positions. The configured RTX PRO
6000 hardware gate deliberately prevents the 3090 from producing the terminal
evidence. See [the methods](../../D1_CACHED_DECODE_TAIL_KL_METHODS_20260824.md).
