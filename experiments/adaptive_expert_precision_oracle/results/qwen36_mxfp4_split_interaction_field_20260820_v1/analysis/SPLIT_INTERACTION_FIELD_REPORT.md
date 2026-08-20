# Split gate/up/down interaction-field study

## Outcome

- Frozen decision: **continue_to_predicted_mixed_hidden_interface**.
- Passing compressed configurations: **joint_eigh_tail8_exact0_int4_per_row_hadamard, joint_eigh_tail8_exact0_int8_per_row**.
- No single winner is selected because the frozen protocol did not predeclare a tie-break.
- Exact same-solver eight-state median gain versus its embedded four-state solution: +0.7353 percentage points.
- This is an exact mixed-H4 geometry ceiling, not a deployment or downstream-quality claim.

## One-bpw compressed results

| factor | p10 recovery | median recovery | p10 retention | median gain | metadata bpw | max MACs | median split-only units |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint_eigh_tail8_exact0_fp32 | 99.108% | 99.497% | 100.010% | +1.371 pp | 0.091176 | 1,329,152 | 240.0 |
| joint_eigh_tail8_exact0_int8_per_row | 99.094% | 99.488% | 100.010% | +1.386 pp | 0.033895 | 1,329,152 | 238.0 |
| exact_proxy_plus_eigh_tail_tail0_exact4_fp32 | 99.084% | 99.477% | 100.010% | +1.460 pp | 0.049510 | 736,256 | 241.0 |
| joint_eigh_tail8_exact0_int4_per_row_hadamard | 99.065% | 99.476% | 100.010% | +1.430 pp | 0.023478 | 1,329,152 | 239.0 |

## Exact action-space control

At one physical bpw the exact-Gram eight-state solver reaches 99.168% p10 / 99.512% median recovery. Its paired p10/median gain over exact restricted four-state search is +0.005/+0.735 percentage points.

## Interpretation boundary

- Gate, up, and down are distinct one-page refinements; state cost is their popcount.
- Exact A/B/C self terms and the unchanged L4/L2 factor score all eight states; no new interaction metadata is introduced.
- The primary solver uses all-00 plus exact-self DP seeds, coordinate descent, and bounded 1/2/3-unit repair.
- Inherited-four-state warm starts are charged diagnostics and are not promotion-eligible.
- Exact-Gram controls use training-only teacher geometry and are nonpromotable.
- No test scientific value, candidate predictor, H4 prefetch, causal replay, routing, logit, or token metric is used.

## Integrity

- 69 validation invocations across 12 frozen layer/expert cells.
- Four frozen PR #11 factors, three page budgets, and exact row/state/page/accounting reconstruction.
- Every exact eight-state result is checked not to lose to its embedded four-state result.
