# Split gate/up/down interaction-field study

## Outcome

- Frozen decision: **continue_to_predicted_mixed_hidden_interface**.
- Passing compressed configurations: **joint_eigh_tail8_exact0_int4_per_row_hadamard, joint_eigh_tail8_exact0_int8_per_row**.
- The action-space attribution now compares compressed four-state and eight-state solvers with the same exact-self DP/all-00 seeds, coordinate descent, local repair, and factor.
- Exact-Gram same-solver eight-state median gain over four states: +0.7353 percentage points.
- This is an exact mixed-H4 geometry ceiling, not a deployment or downstream-quality claim.

## Fixed one-correction-bpw compressed results

| factor | p10 recovery | median recovery | action gain | gain vs PR11 | DP/solver headroom | metadata bpw | max MACs |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint_eigh_tail8_exact0_fp32 | 99.108% | 99.497% | +0.812 pp | +1.371 pp | +0.552 pp | 0.091176 | 1,329,152 |
| joint_eigh_tail8_exact0_int8_per_row | 99.094% | 99.488% | +0.776 pp | +1.386 pp | +0.499 pp | 0.033895 | 1,329,152 |
| exact_proxy_plus_eigh_tail_tail0_exact4_fp32 | 99.084% | 99.477% | +0.817 pp | +1.460 pp | +0.549 pp | 0.049510 | 736,256 |
| joint_eigh_tail8_exact0_int4_per_row_hadamard | 99.065% | 99.476% | +0.844 pp | +1.430 pp | +0.575 pp | 0.023478 | 1,329,152 |

## Strict all-in one-bpw comparison

The correction-page budget is reduced so encoded A/B/C plus the factor plus correction pages fit within 393,216 bytes.

| factor | page cap | total bpw | p10 recovery | median recovery | metadata bytes | max MACs |
|---|---:|---:|---:|---:|---:|---:|
| joint_eigh_tail8_exact0_fp32 | 697 | 0.998728 | 98.572% | 99.237% | 35,852 | 1,329,152 |
| exact_proxy_plus_eigh_tail_tail0_exact4_fp32 | 729 | 0.998728 | 98.822% | 99.337% | 19,468 | 736,256 |
| joint_eigh_tail8_exact0_int8_per_row | 741 | 0.998739 | 98.887% | 99.388% | 13,328 | 1,329,152 |
| joint_eigh_tail8_exact0_int4_per_row_hadamard | 749 | 0.998739 | 98.916% | 99.408% | 9,232 | 1,329,152 |

Frozen all-in recommendation: **joint_eigh_tail8_exact0_int4_per_row_hadamard** at a 749-page cap.
INT8 minus INT4 paired all-in recovery p10/median/p90: -0.0590/-0.0129/+0.0138 percentage points.
Rank-4 exact-proxy control: 729 pages, 98.822% p10 / 99.337% median, 736,256 max charged MACs.

## Actual solver work and wall time

| factor | regime | pages | runtime p10/median/p90 ms | coordinate sweeps median/p90 | local passes median/p90 |
|---|---|---:|---:|---:|---:|
| exact_proxy_plus_eigh_tail_tail0_exact4_fp32 | fixed_correction_budget | 768 | 3341.992/4250.881/4352.506 | 12.0/15.0 | 12.0/12.0 |
| joint_eigh_tail8_exact0_fp32 | fixed_correction_budget | 768 | 3403.219/4229.707/4324.186 | 13.0/14.2 | 12.0/12.0 |
| joint_eigh_tail8_exact0_int4_per_row_hadamard | fixed_correction_budget | 768 | 3641.997/4233.936/4338.964 | 12.0/15.0 | 12.0/12.0 |
| joint_eigh_tail8_exact0_int8_per_row | fixed_correction_budget | 768 | 3483.004/4117.422/4273.921 | 13.0/14.0 | 12.0/12.0 |
| exact_proxy_plus_eigh_tail_tail0_exact4_fp32 | strict_all_in_one_bpw | 729 | 3641.281/4148.172/4238.906 | 13.0/15.0 | 12.0/12.0 |
| joint_eigh_tail8_exact0_fp32 | strict_all_in_one_bpw | 697 | 3678.157/4089.983/4159.938 | 12.0/16.0 | 12.0/12.0 |
| joint_eigh_tail8_exact0_int4_per_row_hadamard | strict_all_in_one_bpw | 749 | 3580.801/4213.962/4303.704 | 12.0/16.0 | 12.0/12.0 |
| joint_eigh_tail8_exact0_int8_per_row | strict_all_in_one_bpw | 741 | 3662.163/4164.216/4273.917 | 12.0/15.0 | 12.0/12.0 |

The recommended all-in path evaluates median/p90 12.0/12.0 local passes and accepts a median 12.0 bundles; it exhausts the frozen 12-pass cap, so this is a bounded-search result, not a convergence claim.
Measured wall times are the actual Python reference selector under the 24-worker quota-saturating run; they are not optimized kernel latency.

## Interpretation boundary

- The legacy PR #11 four-state result is retained separately from the new compressed four-state same-solver control.
- Coordinate damage is updated incrementally; exact recomputation is retained as a terminal parity assertion.
- Gate, up, and down are distinct one-page refinements; state cost is their popcount.
- Exact A/B/C self terms and the unchanged L4/L2 factor score all eight states; no new interaction metadata is introduced.
- Exact-Gram controls use training-only teacher geometry and are nonpromotable.
- No test scientific value, candidate predictor, H4 prefetch, causal replay, routing, logit, or token metric is used.

## Integrity

- 69 validation invocations across 12 frozen layer/expert cells.
- Four frozen PR #11 factors, three fixed correction budgets, four factor-specific strict all-in budgets, and exact row/state/page/accounting reconstruction.
- Every exact eight-state result is checked not to lose to its embedded four-state result.
