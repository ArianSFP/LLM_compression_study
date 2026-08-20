# Low-rank signed interaction-field study

This is an exact-H4 algebraic geometry ceiling, not a deployable candidate
predictor, prefetch, downstream-routing, logit, or token-quality result.

## Outcome

- Best geometry: `exact_proxy_plus_eigh_tail_tail32_exact4_fp32`.
- One-bpw recovery: p10 96.2668%, median 98.1596%.
- One-bpw set-gain retention versus the PR #10 exact hybrid teacher: p10
  99.9693%, median 100.0433%.
- Best physically quantized factor: `joint_eigh_tail16_exact0_int8_per_row`.
- Quantized one-bpw recovery: p10 96.2149%, median
  98.0639%; retention p10 99.9516%.
- Best two-seed factor: `exact_proxy_plus_eigh_tail_tail32_exact4_fp32`.
- Two-seed one-bpw recovery: p10 96.2668%, median
  98.1596%; retention p10 99.9693%;
  compute 3493888 MACs.
- Best quantized two-seed factor:
  `joint_eigh_tail16_exact0_int8_per_row`.
- Quantized two-seed recovery: p10 96.2149%,
  median 98.0639%; retention p10
  99.9657%; compute
  1594368 MACs.
- Best all-gates quantized factor: `joint_eigh_tail8_exact0_int8_per_row`; recovery p10 96.1354%, median 98.0678%; retention p10 99.9313%; metadata 0.03389 bpw; compute 834560 MACs.
- Quantized two-seed geometry gate passed:
  `True`.
- Geometry gate passed: `True`.
- Quantized geometry gate passed: `True`.
- Quantized two-seed geometry+metadata+compute gate passed:
  `True`.
- Status: `diagnostic_only_exact_h4_no_candidate_or_horizon_claim`.

The primary compute candidate uses only all-00 and independent-A/B/C seeds.
The broader geometry diagnostic additionally charges residual-forward and the
continuous relaxation. Both receive identical coordinate and local repair;
neither may borrow the other's recovery or accounting result.

## Artifact rows

- Factor/budget summary rows: 120
- Solver-control rows: 45
- Layer-summary rows: 240
