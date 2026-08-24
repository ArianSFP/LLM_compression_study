# D1 stratified layer-slice report

## Scientific boundary

This is a paired RTX 3090, validation-only D1 oracle pilot over layers
0, 1, 4, 6, 12, 23. D1 is exactly the same token's next-layer router. The experiment loads
complete PR #13 frontier options and performs true pre-residual routed-output
replacement, but executes only through the next router. It makes no terminal
KL, D2-D4, all-layer, or production-controller claim.

The layer-6 slice crosses into the model's full-attention layer 7; the other sampled D1 boundaries use linear attention.

The immutable PRO 6000 captures remain the activation/allocation foundation.
Small cross-device BF16 differences are measured separately. Every policy is
scored against a bit-repeatable paired 3090 Q4 slice baseline, so hardware
drift is not counted as allocator-caused switching.

## Exact D1 result

| rate_pages_per_expert | policy | groups | exact_d1_crossings | exact_d1_crossing_rate | mean_local_qenergy_damage | local_damage_ratio_to_pr13 |
| --- | --- | --- | --- | --- | --- | --- |
| 384 | d1_exact_request_group_repaired | 192 | 1 | 0.00520833 | 0.00159983 | 0.982793 |
| 384 | d1_exact_request_reranked | 192 | 4 | 0.0208333 | 0.00162147 | 0.996085 |
| 384 | exact_combined_local_column_generated_frontier | 192 | 12 | 0.0625 | 0.00159809 | 0.981727 |
| 384 | pr13_router_square_column_generated | 192 | 20 | 0.104167 | 0.00162784 | 1 |
| 749 | d1_exact_request_group_repaired | 192 | 0 | 0 | 0.000205715 | 0.983981 |
| 749 | d1_exact_request_reranked | 192 | 1 | 0.00520833 | 0.000212717 | 1.01747 |
| 749 | exact_combined_local_column_generated_frontier | 192 | 7 | 0.0364583 | 0.000204892 | 0.980045 |
| 749 | pr13_router_square_column_generated | 192 | 11 | 0.0572917 | 0.000209064 | 1 |

The request-level reranker is oracle-only. It selects among one-sided D1
finalists by exact D1 crossings first, then local qenergy, routing-mass loss,
pages, and deterministic policy name. Exact labels are never proposed as
runtime inputs.

The promoted request-coupled repair is also oracle-only. It mixes complete
one-sided finalists when an exact replay exposes a causal-prefix interaction.
Its pair search is restricted to source positions at or before the earliest
remaining crossing. This repair is a scientific upper-bound/reranking step,
not part of the measured deployable scorer.

### Fixed one-sided policies

| rate_pages_per_expert | policy | groups | exact_d1_crossings | exact_d1_crossing_rate | mean_local_qenergy_damage | local_damage_ratio_to_pr13 |
| --- | --- | --- | --- | --- | --- | --- |
| 384 | d1_strict_tau_0.0625_eta_0 | 192 | 12 | 0.0625 | 0.00159809 | 0.981727 |
| 384 | d1_strict_tau_0.0625_eta_0.0001 | 192 | 10 | 0.0520833 | 0.00160593 | 0.98654 |
| 384 | d1_strict_tau_0.0625_eta_0.00025 | 192 | 11 | 0.0572917 | 0.001623 | 0.997025 |
| 384 | d1_strict_tau_0.0625_eta_0.0005 | 192 | 9 | 0.046875 | 0.00165388 | 1.016 |
| 384 | d1_strict_tau_0.0625_eta_0.001 | 192 | 7 | 0.0364583 | 0.00171852 | 1.0557 |
| 384 | d1_strict_tau_0.0625_eta_5e-05 | 192 | 14 | 0.0729167 | 0.00160046 | 0.983181 |
| 749 | d1_strict_tau_0.0625_eta_0 | 192 | 7 | 0.0364583 | 0.000204892 | 0.980045 |
| 749 | d1_strict_tau_0.0625_eta_0.0001 | 192 | 6 | 0.03125 | 0.000216778 | 1.0369 |
| 749 | d1_strict_tau_0.0625_eta_0.00025 | 192 | 2 | 0.0104167 | 0.000235963 | 1.12866 |
| 749 | d1_strict_tau_0.0625_eta_0.0005 | 192 | 2 | 0.0104167 | 0.000268038 | 1.28208 |
| 749 | d1_strict_tau_0.0625_eta_0.001 | 192 | 2 | 0.0104167 | 0.000328701 | 1.57225 |
| 749 | d1_strict_tau_0.0625_eta_5e-05 | 192 | 8 | 0.0416667 | 0.000210224 | 1.00555 |

This table applies one eta to every request and layer at each displayed rate.
It is therefore a stricter mechanism check than request-level exact reranking;
the reranker and repair rows above are oracle upper bounds, not deployable
controller measurements.

### By layer

| layer | rate_pages_per_expert | policy | exact_d1_crossings | exact_d1_crossing_rate | mean_local_qenergy_damage | local_damage_ratio_to_pr13 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 384 | d1_exact_request_group_repaired | 0 | 0 | 4.45591e-05 | 0.988516 |
| 0 | 384 | d1_exact_request_reranked | 1 | 0.03125 | 4.49403e-05 | 0.996971 |
| 0 | 384 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 4.44761e-05 | 0.986675 |
| 0 | 384 | pr13_router_square_column_generated | 2 | 0.0625 | 4.50768e-05 | 1 |
| 0 | 749 | d1_exact_request_group_repaired | 0 | 0 | 5.4691e-07 | 0.968088 |
| 0 | 749 | d1_exact_request_reranked | 0 | 0 | 5.4691e-07 | 0.968088 |
| 0 | 749 | exact_combined_local_column_generated_frontier | 0 | 0 | 5.4691e-07 | 0.968088 |
| 0 | 749 | pr13_router_square_column_generated | 0 | 0 | 5.64938e-07 | 1 |
| 1 | 384 | d1_exact_request_group_repaired | 0 | 0 | 0.000452329 | 0.980504 |
| 1 | 384 | d1_exact_request_reranked | 1 | 0.03125 | 0.000451233 | 0.978129 |
| 1 | 384 | exact_combined_local_column_generated_frontier | 1 | 0.03125 | 0.000451233 | 0.978129 |
| 1 | 384 | pr13_router_square_column_generated | 4 | 0.125 | 0.000461323 | 1 |
| 1 | 749 | d1_exact_request_group_repaired | 0 | 0 | 5.6864e-05 | 1.00582 |
| 1 | 749 | d1_exact_request_reranked | 1 | 0.03125 | 5.7629e-05 | 1.01935 |
| 1 | 749 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 5.57219e-05 | 0.985614 |
| 1 | 749 | pr13_router_square_column_generated | 2 | 0.0625 | 5.65352e-05 | 1 |
| 4 | 384 | d1_exact_request_group_repaired | 0 | 0 | 0.00128105 | 0.983993 |
| 4 | 384 | d1_exact_request_reranked | 1 | 0.03125 | 0.00131337 | 1.00882 |
| 4 | 384 | exact_combined_local_column_generated_frontier | 2 | 0.0625 | 0.0012795 | 0.9828 |
| 4 | 384 | pr13_router_square_column_generated | 3 | 0.09375 | 0.00130189 | 1 |
| 4 | 749 | d1_exact_request_group_repaired | 0 | 0 | 0.000160444 | 0.981493 |
| 4 | 749 | d1_exact_request_reranked | 0 | 0 | 0.000164353 | 1.00541 |
| 4 | 749 | exact_combined_local_column_generated_frontier | 1 | 0.03125 | 0.000160466 | 0.981629 |
| 4 | 749 | pr13_router_square_column_generated | 4 | 0.125 | 0.000163469 | 1 |
| 6 | 384 | d1_exact_request_group_repaired | 1 | 0.03125 | 0.00161785 | 0.987589 |
| 6 | 384 | d1_exact_request_reranked | 1 | 0.03125 | 0.00170291 | 1.03951 |
| 6 | 384 | exact_combined_local_column_generated_frontier | 4 | 0.125 | 0.00161116 | 0.983506 |
| 6 | 384 | pr13_router_square_column_generated | 7 | 0.21875 | 0.00163818 | 1 |
| 6 | 749 | d1_exact_request_group_repaired | 0 | 0 | 0.000205801 | 0.989616 |
| 6 | 749 | d1_exact_request_reranked | 0 | 0 | 0.000235274 | 1.13134 |
| 6 | 749 | exact_combined_local_column_generated_frontier | 1 | 0.03125 | 0.000203541 | 0.978747 |
| 6 | 749 | pr13_router_square_column_generated | 2 | 0.0625 | 0.000207961 | 1 |
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
| 0 | 749 | d1_strict_tau_0.0625_eta_0 | 3 |
| 1 | 384 | d1_strict_tau_0.0625_eta_0 | 3 |
| 1 | 749 | d1_strict_tau_0.0625_eta_0 | 2 |
| 1 | 749 | d1_strict_tau_0.0625_eta_5e-05 | 1 |
| 4 | 384 | d1_strict_tau_0.0625_eta_0 | 2 |
| 4 | 384 | d1_strict_tau_0.0625_eta_0.001 | 1 |
| 4 | 749 | d1_strict_tau_0.0625_eta_0 | 2 |
| 4 | 749 | d1_strict_tau_0.0625_eta_0.0001 | 1 |
| 6 | 384 | d1_strict_tau_0.0625_eta_0.0005 | 2 |
| 6 | 384 | d1_strict_tau_0.0625_eta_0.001 | 1 |
| 6 | 749 | d1_strict_tau_0.0625_eta_0 | 2 |
| 6 | 749 | d1_strict_tau_0.0625_eta_0.0005 | 1 |
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

At 384 pages/expert, repair reduces 20 PR #13 crossings to 1. At 749 pages/expert, repair reduces 11 PR #13 crossings to 0. Thus, changing which complete refinement options are selected
removes nearly all sampled D1 membership changes at matched page caps while
retaining a tight additive local-qenergy guard. The single residual low-rate
crossing is on the layer-6 to full-attention-layer-7 boundary. This establishes
the D1 allocation mechanism on the sampled layers; it does not yet establish
final KL improvement or runtime predictor accuracy. The next scientific gate
is exact downstream replay of the repaired finalists, followed only then by
the separate sequential predictor/controller experiment.
