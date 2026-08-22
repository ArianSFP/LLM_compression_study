# Gate/up Q3 physical-layout study

## Outcome

Gate/up-Q3 status: **stop_embedded_gate_up_q3**.
Selected physical layout: `None`.
Down-Q3 status: **deferred_gate_up_q3_did_not_pass**.

This is an exact-H4 Rank-4 Hadamard geometry/layout study. It is not a deployable predictor, latency, downstream, routing, logit, or token-quality result.
The frozen PR #13 target is 99.5903% p10 / 99.8869% median at 0.998739 total bpw.
The restored restricted-eight-state DP control reaches 99.5855% p10 / 99.8824% median; reproduction gate: **True**.
Every Q3 candidate frontier includes the reproduced Q2/Q4 states. The ideal candidate frontier additionally includes every physical candidate after re-costing it in 256-byte quanta.
The ideal row remains a heuristic solver result rather than a globally certified exact-recovery ceiling; constructive inclusion certifies compressed-objective dominance, not exact-qenergy ordering.
At strict rate, witness-minus-ideal recovery gaps versus `q3_physical_training_coselection_replicated2` are -0.0397 median pp and -0.1494 p10 pp (positive means the ideal solver trails); this diagnoses optimizer headroom without reversing the sign.

## Frozen-target all-in comparison

| Layout | p10 recovery | Median recovery | Median remaining-damage ratio vs frozen PR #13 | Total bpw |
|---|---:|---:|---:|---:|
| `q2q4_monolithic_same_solver_reference` | 99.5855% | 99.8824% | 1.0401 | 0.998739 |
| `q3_ideal_logical_256_byte_plane_ceiling` | 99.7775% | 99.9352% | 0.5729 | 0.998739 |
| `q3_physical_fixed_gate_up_pairing` | 99.6256% | 99.8917% | 0.9577 | 0.998739 |
| `q3_physical_training_coselection_replicated2` | 99.6281% | 99.8955% | 0.9243 | 0.998820 |
| `q3_physical_training_coselection_single` | 99.6281% | 99.8927% | 0.9490 | 0.998779 |

## Average-rate accuracy curve: `q3_physical_training_coselection_replicated2`

| Allowed total bpw | Mean actual total bpw | p10 | Median | p90 | Median residual ratio vs eight-state |
|---:|---:|---:|---:|---:|---:|
| 0.268351 | 0.268155 | 90.4477% | 96.9175% | 99.1912% | 0.9822 |
| 0.289185 | 0.288994 | 91.2770% | 97.2534% | 99.3131% | 0.9710 |
| 0.310018 | 0.309867 | 92.0719% | 97.5265% | 99.4090% | 0.9701 |
| 0.330851 | 0.330666 | 92.6728% | 97.7685% | 99.4999% | 0.9772 |
| 0.351685 | 0.351504 | 93.3176% | 97.9874% | 99.5876% | 0.9954 |
| 0.372518 | 0.372344 | 93.8794% | 98.1595% | 99.6522% | 0.9827 |
| 0.393351 | 0.393174 | 94.4591% | 98.3255% | 99.7115% | 0.9807 |
| 0.414185 | 0.413949 | 94.9499% | 98.4911% | 99.7630% | 0.9488 |
| 0.435018 | 0.434799 | 95.3474% | 98.5943% | 99.7968% | 0.9624 |
| 0.455851 | 0.455631 | 95.7703% | 98.7111% | 99.8266% | 0.9674 |
| 0.476685 | 0.476446 | 96.2091% | 98.8354% | 99.8510% | 0.9559 |
| 0.497518 | 0.497271 | 96.5042% | 98.9376% | 99.8756% | 0.9408 |
| 0.518351 | 0.518077 | 96.8461% | 99.0177% | 99.8934% | 0.9639 |
| 0.539185 | 0.538984 | 97.0636% | 99.0929% | 99.9090% | 0.9771 |
| 0.560018 | 0.559804 | 97.2732% | 99.1871% | 99.9235% | 0.9568 |
| 0.580851 | 0.580676 | 97.5131% | 99.2463% | 99.9338% | 0.9756 |
| 0.601685 | 0.601487 | 97.7001% | 99.3208% | 99.9433% | 0.9700 |
| 0.622518 | 0.622341 | 97.9108% | 99.3881% | 99.9527% | 0.9475 |
| 0.643351 | 0.643141 | 98.1141% | 99.4327% | 99.9615% | 0.9901 |
| 0.664185 | 0.663935 | 98.2716% | 99.4781% | 99.9691% | 1.0048 |
| 0.685018 | 0.684794 | 98.4130% | 99.5237% | 99.9751% | 0.9673 |
| 0.705851 | 0.705499 | 98.5546% | 99.5737% | 99.9791% | 0.9533 |
| 0.726685 | 0.726367 | 98.6952% | 99.6138% | 99.9834% | 0.9490 |
| 0.747518 | 0.747232 | 98.8039% | 99.6490% | 99.9869% | 0.9410 |
| 0.768351 | 0.768087 | 98.9067% | 99.6861% | 99.9891% | 0.9206 |
| 0.789185 | 0.788892 | 99.0179% | 99.7173% | 99.9911% | 0.9055 |
| 0.810018 | 0.809727 | 99.1389% | 99.7400% | 99.9926% | 0.9346 |
| 0.830851 | 0.830510 | 99.2247% | 99.7691% | 99.9944% | 0.9143 |
| 0.851685 | 0.851381 | 99.2909% | 99.7949% | 99.9957% | 0.9058 |
| 0.854289 | 0.854043 | 99.3020% | 99.7946% | 99.9957% | 0.9155 |
| 0.856893 | 0.856644 | 99.3095% | 99.7980% | 99.9960% | 0.8968 |
| 0.859497 | 0.859229 | 99.3175% | 99.8031% | 99.9961% | 0.8831 |
| 0.862101 | 0.861820 | 99.3202% | 99.8005% | 99.9962% | 0.9059 |
| 0.864705 | 0.864400 | 99.3274% | 99.8039% | 99.9964% | 0.9084 |
| 0.867310 | 0.867007 | 99.3314% | 99.8064% | 99.9965% | 0.9075 |
| 0.869914 | 0.869592 | 99.3456% | 99.8092% | 99.9966% | 0.8901 |
| 0.872518 | 0.872299 | 99.3461% | 99.8085% | 99.9967% | 0.9112 |
| 0.875122 | 0.874786 | 99.3621% | 99.8129% | 99.9969% | 0.9035 |
| 0.877726 | 0.877401 | 99.3692% | 99.8152% | 99.9969% | 0.8959 |
| 0.880330 | 0.880033 | 99.3715% | 99.8171% | 99.9970% | 0.9034 |
| 0.882935 | 0.882659 | 99.3763% | 99.8208% | 99.9972% | 0.8913 |
| 0.885539 | 0.885236 | 99.3980% | 99.8201% | 99.9972% | 0.9011 |
| 0.888143 | 0.887845 | 99.3931% | 99.8234% | 99.9974% | 0.9016 |
| 0.890747 | 0.890477 | 99.4017% | 99.8243% | 99.9975% | 0.9121 |
| 0.893351 | 0.893064 | 99.4128% | 99.8277% | 99.9975% | 0.8953 |
| 0.895955 | 0.895668 | 99.4178% | 99.8296% | 99.9977% | 0.9005 |
| 0.898560 | 0.898212 | 99.4160% | 99.8312% | 99.9977% | 0.9108 |
| 0.901164 | 0.900918 | 99.4336% | 99.8328% | 99.9978% | 0.9082 |
| 0.903768 | 0.903436 | 99.4411% | 99.8378% | 99.9979% | 0.8933 |
| 0.906372 | 0.906031 | 99.4433% | 99.8375% | 99.9980% | 0.9028 |
| 0.908976 | 0.908601 | 99.4518% | 99.8398% | 99.9980% | 0.9263 |
| 0.911580 | 0.911310 | 99.4545% | 99.8435% | 99.9981% | 0.9007 |
| 0.914185 | 0.913831 | 99.4594% | 99.8448% | 99.9982% | 0.9124 |
| 0.916789 | 0.916442 | 99.4646% | 99.8474% | 99.9983% | 0.9010 |
| 0.919393 | 0.919097 | 99.4698% | 99.8479% | 99.9983% | 0.8979 |
| 0.921997 | 0.921612 | 99.4784% | 99.8498% | 99.9984% | 0.9076 |
| 0.924601 | 0.924234 | 99.4863% | 99.8499% | 99.9985% | 0.8994 |
| 0.927205 | 0.926875 | 99.4917% | 99.8540% | 99.9986% | 0.8935 |
| 0.929810 | 0.929484 | 99.5005% | 99.8551% | 99.9986% | 0.8989 |
| 0.932414 | 0.932067 | 99.5011% | 99.8556% | 99.9987% | 0.9066 |
| 0.935018 | 0.934708 | 99.5088% | 99.8565% | 99.9987% | 0.9113 |
| 0.937622 | 0.937290 | 99.5138% | 99.8606% | 99.9988% | 0.9033 |
| 0.940226 | 0.939911 | 99.5211% | 99.8615% | 99.9989% | 0.8989 |
| 0.942830 | 0.942479 | 99.5279% | 99.8619% | 99.9989% | 0.9093 |
| 0.945435 | 0.945085 | 99.5377% | 99.8665% | 99.9990% | 0.8990 |
| 0.948039 | 0.947628 | 99.5356% | 99.8677% | 99.9990% | 0.8946 |
| 0.950643 | 0.950267 | 99.5408% | 99.8665% | 99.9991% | 0.9151 |
| 0.953247 | 0.952920 | 99.5494% | 99.8716% | 99.9992% | 0.8995 |
| 0.955851 | 0.955526 | 99.5541% | 99.8728% | 99.9992% | 0.8912 |
| 0.958455 | 0.958066 | 99.5613% | 99.8729% | 99.9992% | 0.8919 |
| 0.961060 | 0.960690 | 99.5658% | 99.8734% | 99.9993% | 0.9086 |
| 0.963664 | 0.963275 | 99.5645% | 99.8726% | 99.9993% | 0.9270 |
| 0.966268 | 0.965883 | 99.5730% | 99.8771% | 99.9993% | 0.9045 |
| 0.968872 | 0.968510 | 99.5844% | 99.8781% | 99.9994% | 0.9043 |
| 0.971476 | 0.971057 | 99.5822% | 99.8789% | 99.9994% | 0.9095 |
| 0.974080 | 0.973718 | 99.5837% | 99.8820% | 99.9994% | 0.9041 |
| 0.976685 | 0.976323 | 99.5932% | 99.8813% | 99.9994% | 0.9118 |
| 0.979289 | 0.979005 | 99.6016% | 99.8857% | 99.9994% | 0.8746 |
| 0.981893 | 0.981511 | 99.6066% | 99.8851% | 99.9995% | 0.8984 |
| 0.984497 | 0.984165 | 99.6027% | 99.8863% | 99.9995% | 0.9082 |
| 0.987101 | 0.986748 | 99.6143% | 99.8850% | 99.9995% | 0.9193 |
| 0.989705 | 0.989318 | 99.6201% | 99.8905% | 99.9996% | 0.8955 |
| 0.992310 | 0.991962 | 99.6107% | 99.8905% | 99.9996% | 0.9034 |
| 0.994914 | 0.994554 | 99.6221% | 99.8932% | 99.9996% | 0.8840 |
| 0.997518 | 0.997122 | 99.6215% | 99.8921% | 99.9996% | 0.9017 |
| 0.998820 | 0.998446 | 99.6281% | 99.8955% | 99.9996% | 0.8887 |
| 0.999471 | 0.999096 | 99.6277% | 99.8957% | 99.9996% | 0.8821 |
| 1.000122 | 0.999817 | 99.6265% | 99.8937% | 99.9996% | 0.9073 |
| 1.002726 | 1.002375 | 99.6251% | 99.8954% | 99.9997% | 0.9089 |
| 1.005330 | 1.004901 | 99.6321% | 99.8973% | 99.9997% | 0.8957 |
| 1.007935 | 1.007490 | 99.6384% | 99.8985% | 99.9997% | 0.8928 |
| 1.010539 | 1.010098 | 99.6422% | 99.8993% | 99.9997% | 0.9004 |
| 1.013143 | 1.012741 | 99.6523% | 99.9012% | 99.9997% | 0.8934 |
| 1.015747 | 1.015307 | 99.6488% | 99.9007% | 99.9997% | 0.9121 |
| 1.018351 | 1.017961 | 99.6567% | 99.9031% | 99.9997% | 0.8921 |
| 1.039185 | 1.038798 | 99.6882% | 99.9103% | 99.9998% | 0.9130 |
| 1.060018 | 1.059606 | 99.7189% | 99.9202% | 99.9999% | 0.8966 |
| 1.080851 | 1.080298 | 99.7389% | 99.9258% | 99.9999% | 0.8953 |
| 1.101685 | 1.101098 | 99.7646% | 99.9332% | 99.9999% | 0.8812 |

## Minimum rate matching frozen PR #13 p10 and median

These are grid-certified minima. The dense target region is sampled every 4 quanta (0.002604 total bpw); the preceding sampled point is the lower edge of the certified interval.

| Physical layout | Previous sampled bpw (fails one or both targets) | Minimum sampled total bpw | Delta vs PR #13 |
|---|---:|---:|---:|
| q3_physical_fixed_gate_up_pairing | 0.989624 | 0.992228 | -0.006510 |
| q3_physical_training_coselection_single | 0.987061 | 0.989665 | -0.009074 |
| q3_physical_training_coselection_replicated2 | 0.987101 | 0.989705 | -0.009033 |

## Minimum total bpw at target quality

| Layout | Statistic | Target | Minimum total bpw |
|---|---|---:|---:|
| `q2q4_monolithic_same_solver_reference` | median | 99.00% | 0.539103 |
| `q2q4_monolithic_same_solver_reference` | median | 99.50% | 0.684937 |
| `q2q4_monolithic_same_solver_reference` | median | 99.80% | 0.885457 |
| `q2q4_monolithic_same_solver_reference` | median | 99.90% | 1.039103 |
| `q2q4_monolithic_same_solver_reference` | p10 | 99.00% | 0.809937 |
| `q2q4_monolithic_same_solver_reference` | p10 | 99.50% | 0.955770 |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.00% | 0.476603 |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.50% | 0.601603 |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.80% | 0.789103 |
| `q3_ideal_logical_256_byte_plane_ceiling` | median | 99.90% | 0.921916 |
| `q3_ideal_logical_256_byte_plane_ceiling` | p10 | 99.00% | 0.705770 |
| `q3_ideal_logical_256_byte_plane_ceiling` | p10 | 99.50% | 0.851603 |
| `q3_physical_fixed_gate_up_pairing` | median | 99.00% | 0.518270 |
| `q3_physical_fixed_gate_up_pairing` | median | 99.50% | 0.684937 |
| `q3_physical_fixed_gate_up_pairing` | median | 99.80% | 0.864624 |
| `q3_physical_fixed_gate_up_pairing` | median | 99.90% | 1.018270 |
| `q3_physical_fixed_gate_up_pairing` | p10 | 99.00% | 0.789103 |
| `q3_physical_fixed_gate_up_pairing` | p10 | 99.50% | 0.934937 |
| `q3_physical_training_coselection_replicated2` | median | 99.00% | 0.518351 |
| `q3_physical_training_coselection_replicated2` | median | 99.50% | 0.685018 |
| `q3_physical_training_coselection_replicated2` | median | 99.80% | 0.859497 |
| `q3_physical_training_coselection_replicated2` | median | 99.90% | 1.013143 |
| `q3_physical_training_coselection_replicated2` | p10 | 99.00% | 0.789185 |
| `q3_physical_training_coselection_replicated2` | p10 | 99.50% | 0.929810 |
| `q3_physical_training_coselection_single` | median | 99.00% | 0.539144 |
| `q3_physical_training_coselection_single` | median | 99.50% | 0.684977 |
| `q3_physical_training_coselection_single` | median | 99.80% | 0.867269 |
| `q3_physical_training_coselection_single` | median | 99.90% | 1.013102 |
| `q3_physical_training_coselection_single` | p10 | 99.00% | 0.789144 |
| `q3_physical_training_coselection_single` | p10 | 99.50% | 0.929769 |

## Runtime

| Layout | p50 total selector | p50 frontier | p50 allocation | total p90 | total p99 | Median sweeps | Median local passes | Median DP tables | Median DP state updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q2q4_monolithic_same_solver_reference` | 168380.45 ms | 133776.20 ms | 34422.18 ms | 176319.68 ms | 185594.54 ms | 2401.0 | 511.5 | 8.0 | 100597760.0 |
| `q3_ideal_logical_256_byte_plane_ceiling` | 285786.74 ms | 40382.79 ms | 244255.62 ms | 350902.64 ms | 428366.38 ms | 2562.5 | 422.0 | 8.0 | 226344960.0 |
| `q3_physical_fixed_gate_up_pairing` | 258203.63 ms | 177815.59 ms | 77548.25 ms | 277931.29 ms | 306563.77 ms | 2925.5 | 395.5 | 8.0 | 226279424.0 |
| `q3_physical_training_coselection_replicated2` | 344232.00 ms | 267705.19 ms | 74621.24 ms | 360528.82 ms | 374363.27 ms | 2929.5 | 550.5 | 0.0 | 0.0 |
| `q3_physical_training_coselection_single` | 265005.12 ms | 182876.97 ms | 77874.64 ms | 283562.10 ms | 292750.35 ms | 2925.5 | 552.0 | 0.0 | 0.0 |

## Accounting and interpretation

- Exact-self dynamic programming seeds both the full 18-state and restricted inherited-eight-state additive layouts before the identical coordinate/local solver.
- Ideal Q3 charges independent 256-byte planes; its candidate set is a constructive superset of every physical frontier, but its selected exact-recovery row is still a nonphysical heuristic control rather than a certified global optimum.
- Every physical Q3 result charges the exact union of 512-byte page IDs and includes a fully charged legacy Q2/Q4 packing replica, making every restored PR #13 state physically feasible at its original page cost.
- Each expert chooses one complete packing replica and never mixes pages across replicas. External-storage accounting charges every extra gate/up refinement copy.
- The restored eight-state comparator uses the same DP seed, coordinate/local solver, and Rank-4 factor, restricted to inherited states with monolithic Q2→Q4 projection pages.
- Learned layouts use only routed train occurrences. Validation values never fit pairings or factors; test scientific values are never admitted.
- The table reports allowed total bpw, including A/B/C, factor, amortized learned-layout descriptor, and correction bytes.
