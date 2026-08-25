# D1 source-slice to live full-model transfer diagnosis

This is a GPU-free audit of finalized Experiment A artifacts. It performs no
allocator or terminal-KL selection.

## Result

### Candidate signed-margin prediction and live calibration

The source candidate margin retains ranking information, but its zero point is
not a reliable full-model route certificate. Across all policies and rates,
margin-risk ROC AUC is 0.813449, while exact
source crossing labels agree with live full-model D1 outcomes only
90.03%. For the promoted fixed policy the corresponding values
are 0.802699 and 90.37%. That
accuracy is base-rate dominated: the source crossing label misses
93.33% of actual fixed-policy crossings.

This distinction matters: AUC measures ordering and is invariant to a shifted
decision threshold; crossing agreement at a chosen margin threshold measures
certification. The threshold table therefore reports 0, 1/64, 2/64 and 4/64
explicitly rather than interpreting a useful ranking as a calibrated
certificate. For the fixed policy at threshold zero, pooled recall is
20.00% and the false-positive rate is 8.81%.

### Baseline D1 anchor transfer

The raw-source exact D1 Q4 rank-8/rank-9 margin has Pearson
0.961537, Spearman 0.900124, and MAE
0.016433 against the separately captured exact-PRO D1 baseline.
Its source-minus-PRO bias is 0.001167 and its
maximum absolute error is 0.062500.

### Delta reconstruction transfer

Fixed-policy stored/live delta cosine averages
0.990959; mean transfer MSE is
1.11902e-08. Delta transfer is close,
but not bit-exact.

### Remaining error (inference)

The baseline D1 anchor and reconstructed delta transfer substantially better
than the candidate crossing threshold. The remaining gap therefore lies in
candidate signed-effect transfer: source-slice candidate execution, BF16
boundary quantization, and—upstream of the exact source replay—the VJP
linearization used to choose allocations. Because the audited candidate margin
is exact inside the source slice, its live miscalibration cannot be attributed
to the VJP alone. This audit does not identify the shares of those effects.

## Patch stratum

The twelve Q4 route-mismatch identities were regenerated on the PRO host. For
the fixed policy, the PR13-paired mean terminal-KL delta is
-0.000515953 on patched rows and
-6.72295e-05 on unpatched
rows. These KL values are a post-hoc patch-stratum outcome audit only; they were
not used to select any policy, threshold, or allocation.

## Files and interpretation boundary

- `d1_source_live_transfer_rows.parquet` is the exact joined row-level audit.
- `d1_source_live_transfer_crossing_summary.parquet` separates confusion from
  ranking AUC.
- `d1_source_live_transfer_threshold_calibration.parquet` audits four fixed
  safety thresholds.
- `d1_source_live_transfer_baseline_margin_summary.parquet` audits D1 Q4 margin
  transfer independently of candidate allocations.
- `d1_source_live_transfer_delta_fidelity.parquet` audits stored/live delta
  reconstruction fidelity.
- `d1_source_live_transfer_fixed_pr13_patch_*.parquet` is the narrowly scoped
  patched-versus-unpatched KL comparison.

The data support using source margins as screening scores, not as zero-threshold
certificates. They do not establish a terminal-KL minimization target and do not
promote Experiment B.
