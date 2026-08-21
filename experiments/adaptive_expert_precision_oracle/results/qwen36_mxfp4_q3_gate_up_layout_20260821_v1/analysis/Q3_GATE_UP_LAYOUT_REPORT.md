# Gate/up Q3 physical-layout study

## Outcome

Gate/up-Q3 status: **continue_to_predictive_q3_study**.
Selected physical layout: `q3_physical_fixed_gate_up_pairing`.
Down-Q3 status: **eligible_for_separate_followup**.

This is an exact-H4 Rank-4 Hadamard geometry/layout study. It is not a deployable predictor, latency, downstream, routing, logit, or token-quality result.
The ideal-plane row is a heuristic solver result, not a certified global ceiling. The best physical state vector is feasible in the ideal space at equal-or-lower cost.
At strict rate the heuristic ideal solve trails the `q3_physical_fixed_gate_up_pairing` feasible witness by 0.1836 median pp and 0.4438 p10 pp; this is optimizer headroom.

## Strict all-in one-bpw comparison

| Layout | p10 recovery | Median recovery | Median remaining-damage ratio vs eight-state | Total bpw |
|---|---:|---:|---:|---:|
| `q2q4_monolithic_same_solver_reference` | 95.9911% | 98.1115% | 1.0000 | 0.999390 |
| `q3_ideal_logical_256_byte_plane_ceiling` | 95.9537% | 98.3971% | — | 0.999390 |
| `q3_physical_fixed_gate_up_pairing` | 96.3975% | 98.5807% | 0.7516 | 0.999390 |
| `q3_physical_training_coselection_replicated2` | 96.5234% | 98.4701% | 0.8101 | 0.999471 |
| `q3_physical_training_coselection_single` | 96.5708% | 98.4557% | 0.8177 | 0.999430 |

## Average-rate accuracy curve: `q3_physical_fixed_gate_up_pairing`

| Allowed total bpw | Mean actual total bpw | p10 | Median | p90 | Median residual ratio vs eight-state |
|---:|---:|---:|---:|---:|---:|
| 0.268270 | 0.267863 | 88.2971% | 94.9717% | 98.5207% | 1.0557 |
| 0.289103 | 0.288741 | 89.0264% | 95.3700% | 98.6503% | 1.0548 |
| 0.309937 | 0.309497 | 89.5641% | 95.6919% | 98.7159% | 1.0404 |
| 0.330770 | 0.330321 | 90.1719% | 96.0070% | 98.7980% | 1.0214 |
| 0.351603 | 0.351164 | 90.7283% | 96.2844% | 98.8701% | 0.9731 |
| 0.372437 | 0.372074 | 91.2067% | 96.5080% | 98.9068% | 0.9591 |
| 0.393270 | 0.392838 | 91.8991% | 96.6960% | 98.9537% | 0.9477 |
| 0.414103 | 0.413738 | 92.3040% | 96.8931% | 99.0141% | 0.9237 |
| 0.434937 | 0.434554 | 92.6115% | 97.0238% | 99.0446% | 0.9146 |
| 0.455770 | 0.455411 | 92.8250% | 97.1422% | 99.0981% | 0.9153 |
| 0.476603 | 0.476228 | 93.4109% | 97.2940% | 99.1251% | 0.8938 |
| 0.497437 | 0.497050 | 93.5825% | 97.3723% | 99.1806% | 0.8983 |
| 0.518270 | 0.517911 | 93.9501% | 97.4860% | 99.2216% | 0.8758 |
| 0.539103 | 0.538705 | 94.1954% | 97.5594% | 99.2550% | 0.8857 |
| 0.559937 | 0.559582 | 94.3849% | 97.6703% | 99.2918% | 0.8616 |
| 0.580770 | 0.580325 | 94.8101% | 97.7612% | 99.3053% | 0.8390 |
| 0.601603 | 0.601222 | 95.0209% | 97.7730% | 99.3336% | 0.8557 |
| 0.622437 | 0.622032 | 95.1244% | 97.8535% | 99.3409% | 0.8473 |
| 0.643270 | 0.642743 | 95.2629% | 97.8873% | 99.3705% | 0.8610 |
| 0.664103 | 0.663731 | 95.3681% | 97.9220% | 99.3891% | 0.8647 |
| 0.684937 | 0.684465 | 95.5348% | 97.9557% | 99.4136% | 0.8818 |
| 0.705770 | 0.705348 | 95.8092% | 98.0116% | 99.4285% | 0.8812 |
| 0.726603 | 0.726214 | 95.9774% | 98.0590% | 99.4284% | 0.8716 |
| 0.747437 | 0.747000 | 96.1043% | 98.1070% | 99.4453% | 0.8696 |
| 0.768270 | 0.767853 | 96.1372% | 98.1914% | 99.4587% | 0.8460 |
| 0.789103 | 0.788586 | 96.1499% | 98.2582% | 99.4760% | 0.8300 |
| 0.809937 | 0.809434 | 96.1940% | 98.3011% | 99.4899% | 0.8202 |
| 0.830770 | 0.830193 | 96.2126% | 98.3748% | 99.4949% | 0.7964 |
| 0.851603 | 0.851040 | 96.2252% | 98.4125% | 99.4963% | 0.7861 |
| 0.872437 | 0.871985 | 96.2647% | 98.4326% | 99.4974% | 0.7862 |
| 0.893270 | 0.892469 | 96.3191% | 98.4638% | 99.4966% | 0.7801 |
| 0.914103 | 0.913237 | 96.3222% | 98.4920% | 99.4978% | 0.7815 |
| 0.934937 | 0.933764 | 96.3394% | 98.5163% | 99.4983% | 0.7753 |
| 0.955770 | 0.954231 | 96.3654% | 98.5377% | 99.4984% | 0.7677 |
| 0.976603 | 0.974508 | 96.3917% | 98.5606% | 99.4980% | 0.7595 |
| 0.997437 | 0.994329 | 96.3952% | 98.5747% | 99.4979% | 0.7535 |
| 0.998739 | 0.995571 | 96.3975% | 98.5793% | 99.4979% | 0.7525 |
| 0.999390 | 0.996206 | 96.3975% | 98.5807% | 99.4980% | 0.7516 |
| 1.018270 | 1.013381 | 96.3996% | 98.6237% | 99.4979% | 0.7318 |
| 1.039103 | 1.031712 | 96.4000% | 98.6605% | 99.4982% | 0.7137 |
| 1.059937 | 1.049328 | 96.4064% | 98.6739% | 99.5062% | 0.7103 |
| 1.080770 | 1.066503 | 96.4016% | 98.6761% | 99.5079% | 0.7163 |
| 1.101603 | 1.083138 | 96.4059% | 98.6850% | 99.5141% | 0.7146 |

## Minimum total bpw at target quality

| Layout | Statistic | Target | Minimum total bpw |
|---|---|---:|---:|
| `q2q4_monolithic_same_solver_reference` | median | 99.00% | not reached |
| `q2q4_monolithic_same_solver_reference` | median | 99.50% | not reached |
| `q2q4_monolithic_same_solver_reference` | median | 99.80% | not reached |
| `q2q4_monolithic_same_solver_reference` | median | 99.90% | not reached |
| `q2q4_monolithic_same_solver_reference` | p10 | 99.00% | not reached |
| `q2q4_monolithic_same_solver_reference` | p10 | 99.50% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.00% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.50% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.80% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.90% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | p10 | 99.00% | not reached |
| `q3_ideal_logical_256_byte_plane_ceiling` | p10 | 99.50% | not reached |
| `q3_physical_fixed_gate_up_pairing` | median | 99.00% | not reached |
| `q3_physical_fixed_gate_up_pairing` | median | 99.50% | not reached |
| `q3_physical_fixed_gate_up_pairing` | median | 99.80% | not reached |
| `q3_physical_fixed_gate_up_pairing` | median | 99.90% | not reached |
| `q3_physical_fixed_gate_up_pairing` | p10 | 99.00% | not reached |
| `q3_physical_fixed_gate_up_pairing` | p10 | 99.50% | not reached |
| `q3_physical_training_coselection_replicated2` | median | 99.00% | not reached |
| `q3_physical_training_coselection_replicated2` | median | 99.50% | not reached |
| `q3_physical_training_coselection_replicated2` | median | 99.80% | not reached |
| `q3_physical_training_coselection_replicated2` | median | 99.90% | not reached |
| `q3_physical_training_coselection_replicated2` | p10 | 99.00% | not reached |
| `q3_physical_training_coselection_replicated2` | p10 | 99.50% | not reached |
| `q3_physical_training_coselection_single` | median | 99.00% | not reached |
| `q3_physical_training_coselection_single` | median | 99.50% | not reached |
| `q3_physical_training_coselection_single` | median | 99.80% | not reached |
| `q3_physical_training_coselection_single` | median | 99.90% | not reached |
| `q3_physical_training_coselection_single` | p10 | 99.00% | not reached |
| `q3_physical_training_coselection_single` | p10 | 99.50% | not reached |

## Runtime

| Layout | p50 full multi-budget frontier time | p90 | p99 | Median sweeps | Median local passes |
|---|---:|---:|---:|---:|---:|
| `q2q4_monolithic_same_solver_reference` | 161638.64 ms | 165682.76 ms | 168900.32 ms | 2518.5 | 671.5 |
| `q3_ideal_logical_256_byte_plane_ceiling` | 47198.24 ms | 48203.98 ms | 49913.75 ms | 2772.0 | 641.0 |
| `q3_physical_fixed_gate_up_pairing` | 181918.70 ms | 187221.20 ms | 195407.48 ms | 2978.5 | 563.0 |
| `q3_physical_training_coselection_replicated2` | 273262.33 ms | 281943.96 ms | 286724.17 ms | 2929.0 | 548.5 |
| `q3_physical_training_coselection_single` | 179394.53 ms | 185486.32 ms | 188899.23 ms | 2927.5 | 554.5 |

## Accounting and interpretation

- Ideal Q3 charges independent 256-byte planes, but the reported coordinate/local solve is a nonphysical heuristic control rather than a certified global ceiling.
- Every physical Q3 result charges the exact union of 512-byte page IDs; the replicated control chooses one whole layout and never mixes replicas.
- The eight-state comparator uses the same coordinate/local solver and Rank-4 factor, restricted to inherited states with monolithic Q2→Q4 projection pages.
- Learned layouts use only routed train occurrences. Validation values never fit pairings or factors; test scientific values are never admitted.
- The table reports allowed total bpw, including A/B/C, factor, amortized learned-layout descriptor, and correction bytes.
