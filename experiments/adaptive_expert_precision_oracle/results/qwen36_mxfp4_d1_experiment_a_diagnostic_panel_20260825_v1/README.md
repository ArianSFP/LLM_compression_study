# D1 Experiment A allocator-diagnosis package

This is the compact, immutable evidence package for the 2026-08-25 diagnosis
of the PR #13 D1 allocator. The study stays within Experiment A and assumes
exact prefill. Each observation injects one current decode token at one layer
behind a private exact-prefix cache.

The principal conclusion is narrow: downstream expert membership execution is
the largest identified amplifier, but binary next-router crossing count is not
a sufficient global ranking objective for terminal KL. Hard-saturating repair
of the PR #13 incumbent was the lower-variance tested alternative. Nested local
completion inside a calibrated D1-safe set is the proposed next oracle, not an
executed result, deployable predictor, or Experiment B controller.

Reviewer entry points:

- [Canonical scientific report](analysis/D1_EXPERIMENT_A_ALLOCATOR_DIAGNOSIS_REPORT.md)
- [Frozen methodology](../../D1_EXPERIMENT_A_ALLOCATOR_DIAGNOSIS_METHODS_20260825.md)
- [Immutable run configuration](../../configs/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1.json)
- [Package manifest](PACKAGE_MANIFEST.json)
- [Package SHA-256 ledger](artifact_hashes.sha256)

The exact panel contains 24 token/layer identities over layers 0, 1, 4, 6, 12
and 23; four page caps; five realizable policies; two explanatory control
families; and three downstream route modes. It records 2,304 terminal-quality
rows, 74,784 route rows and 72 zero-dose rows. The analysis directory contains
17 deterministic Parquet tables plus its machine-readable facts and cost
accounting.

The package also links to two supporting result directories checked into this
branch:

- `../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/`
  holds the complete 696-pair crossing decomposition, adjacent-rate geometry,
  heavy-tail analysis and D1-only remedy ablations.
- `../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/transfer_diagnosis/`
  holds the corrected source-to-live D1 calibration audit.

No allocation is selected from terminal KL or from routes after D1. Full-tail
routes are retained only as descriptive causal outcomes. An exploratory
multi-layer route selector was rejected and is not part of this package.

The complete raw per-layer shards, logs and final code snapshot are retained
on network storage at:

`/workspace/pr13_d1_experiment_a_diagnostic_panel_20260825_v1`

The Git package intentionally omits large reconstruction caches and model
weights. Every included artifact is covered by `artifact_hashes.sha256`; raw
and aggregate scientific outputs are independently bound by their facts files.
