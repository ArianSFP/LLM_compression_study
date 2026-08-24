# D1 layer-slice pilot report

## Scientific boundary

This is a paired RTX 3090, validation-only D1 oracle pilot over layers
0, 12, 23. D1 is exactly the same token's next-layer router. The experiment loads
complete PR #13 frontier options and performs true pre-residual routed-output
replacement, but executes only through the next router. It makes no terminal
KL, D2-D4, all-layer, or production-controller claim.

The immutable PRO 6000 captures remain the activation/allocation foundation.
Small cross-device BF16 differences are measured separately. Every policy is
scored against a bit-repeatable paired 3090 Q4 slice baseline, so hardware
drift is not counted as allocator-caused switching.

## Exact D1 result

| rate_pages_per_expert | policy | groups | exact_d1_crossings | exact_d1_crossing_rate | mean_local_qenergy_damage | local_damage_ratio_to_pr13 |
| --- | --- | --- | --- | --- | --- | --- |
| 384 | d1_exact_request_group_repaired | 96 | 0 | 0 | 0.00208258 | 0.981479 |
| 384 | d1_exact_request_reranked | 96 | 1 | 0.0104167 | 0.00208709 | 0.983606 |
| 384 | exact_combined_local_column_generated_frontier | 96 | 5 | 0.0520833 | 0.00208222 | 0.98131 |
| 384 | pr13_router_square_column_generated | 96 | 6 | 0.0625 | 0.00212188 | 1 |
| 749 | d1_exact_request_group_repaired | 64 | 0 | 0 | 0.000405318 | 0.981571 |
| 749 | d1_exact_request_reranked | 64 | 0 | 0 | 0.00040925 | 0.991095 |
| 749 | exact_combined_local_column_generated_frontier | 64 | 3 | 0.046875 | 0.000404539 | 0.979685 |
| 749 | pr13_router_square_column_generated | 64 | 3 | 0.046875 | 0.000412927 | 1 |

The request-level reranker is oracle-only. It selects among one-sided D1
finalists by exact D1 crossings first, then local qenergy, routing-mass loss,
pages, and deterministic policy name. Exact labels are never proposed as
runtime inputs.

The promoted request-coupled repair is also oracle-only. It mixes complete
one-sided finalists when an exact replay exposes a causal-prefix interaction.
Its pair search is restricted to source positions at or before the earliest
remaining crossing. This repair is a scientific upper-bound/reranking step,
not part of the measured deployable scorer.

### By layer

| layer | rate_pages_per_expert | policy | exact_d1_crossings | exact_d1_crossing_rate | mean_local_qenergy_damage | local_damage_ratio_to_pr13 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 384 | d1_exact_request_group_repaired | 0 | 0 | 4.45591e-05 | 0.988516 |
| 0 | 384 | d1_exact_request_reranked | 1 | 0.03125 | 4.49403e-05 | 0.996971 |
| 0 | 384 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 4.44761e-05 | 0.986675 |
| 0 | 384 | pr13_router_square_column_generated | 2 | 0.0625 | 4.50768e-05 | 1 |
| 12 | 384 | d1_exact_request_group_repaired | 0 | 0 | 0.00249976 | 0.986081 |
| 12 | 384 | d1_exact_request_reranked | 0 | 0 | 0.00250125 | 0.986668 |
| 12 | 384 | exact_combined_local_column_generated_frontier | 1 | 0.03125 | 0.00249921 | 0.985862 |
| 12 | 384 | pr13_router_square_column_generated | 2 | 0.0625 | 0.00253505 | 1 |
| 12 | 749 | d1_exact_request_group_repaired | 0 | 0 | 0.000322157 | 0.982534 |
| 12 | 749 | d1_exact_request_reranked | 0 | 0 | 0.000328447 | 1.00172 |
| 12 | 749 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 0.000320805 | 0.978409 |
| 12 | 749 | pr13_router_square_column_generated | 2 | 0.0625 | 0.000327884 | 1 |
| 23 | 384 | d1_exact_request_group_repaired | 0 | 0 | 0.00370342 | 0.978314 |
| 23 | 384 | d1_exact_request_reranked | 0 | 0 | 0.00371509 | 0.981397 |
| 23 | 384 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 0.00370298 | 0.978198 |
| 23 | 384 | pr13_router_square_column_generated | 2 | 0.0625 | 0.00378551 | 1 |
| 23 | 749 | d1_exact_request_group_repaired | 0 | 0 | 0.000488478 | 0.980937 |
| 23 | 749 | d1_exact_request_reranked | 0 | 0 | 0.000490053 | 0.984101 |
| 23 | 749 | exact_combined_local_column_generated_frontier | 1 | 0.03125 | 0.000488272 | 0.980525 |
| 23 | 749 | pr13_router_square_column_generated | 1 | 0.03125 | 0.00049797 | 1 |

### Finalist choices

| layer | rate_pages_per_expert | policy | requests |
| --- | --- | --- | --- |
| 0 | 384 | d1_strict_tau_0.0625_eta_0 | 2 |
| 0 | 384 | d1_strict_tau_0.0625_eta_5e-05 | 1 |
| 12 | 384 | d1_strict_tau_0.0625_eta_0 | 2 |
| 12 | 384 | d1_strict_tau_0.0625_eta_0.0001 | 1 |
| 12 | 749 | d1_strict_tau_0.0625_eta_0 | 2 |
| 12 | 749 | d1_strict_tau_0.0625_eta_0.00025 | 1 |
| 23 | 384 | d1_strict_tau_0.0625_eta_0.0001 | 1 |
| 23 | 384 | d1_strict_tau_0.0625_eta_0.00025 | 1 |
| 23 | 384 | d1_strict_tau_0.0625_eta_5e-05 | 1 |
| 23 | 749 | d1_strict_tau_0.0625_eta_0 | 2 |
| 23 | 749 | d1_strict_tau_0.0625_eta_5e-05 | 1 |

## Runtime cost boundary

The one-adjoint controller estimate is
933,888 MAC/group, or
1.819% of the frozen
PR #13 selector estimate (51,343,360
MAC/group). Joint D1 metadata changes the matched page caps to
{'0.523478190104': 363, '0.773478190104': 555, '0.998738606771': 728, '1.0234781901': 747}; with uncertainty metadata they are
{'0.523478190104': 360, '0.773478190104': 552, '0.998738606771': 725, '1.0234781901': 744}.

On NVIDIA GeForce RTX 3090, the device scorer measured 0.852 ms median and 0.868 ms p90 per refresh.

Exact frontier generation, VJPs, coordinate/pair search, and exact replay in
this pilot are oracle-label costs and are excluded from the deployable runtime
budget.

## Interpretation

Across the supplied slices, changing which complete refinement options are
selected can remove the observed D1 membership changes at matched page caps
while retaining a tight additive local-qenergy guard. This establishes the D1
allocation mechanism on the sampled layers; it does not yet establish final
KL improvement. The next scientific gate is exact downstream replay of the
reranked finalists, followed only then by predictor/controller work.
