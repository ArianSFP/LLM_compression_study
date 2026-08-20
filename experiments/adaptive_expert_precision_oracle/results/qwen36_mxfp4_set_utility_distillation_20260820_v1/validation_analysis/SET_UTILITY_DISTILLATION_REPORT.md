# Set-utility distillation study

## Outcome

Hybrid: **continue**. Templates: **regime_exists**. No predicted or independent-ABC H0-late policy passes every frozen gate.

This is a strict **exact-checkpoint validation-only** pilot. The runner necessarily physically
loads immutable monolithic capture archives that contain test rows. It inspects only the `split`
labels and `request_id` arrays for train/validation/test, solely to verify within-capture request
separation and cross-capture split compatibility. No test scientific tensor or value is admitted,
evaluated for a metric, used for tuning, summarized, or used in a decision. It consumes actual
x/Q2 features only at H0, does not train or evaluate
an H4 predictor or demonstrate early prefetch latency, and makes no router, logit, token-quality,
or measured-latency claim. A passing validation configuration requires a new sealed holdout
before any confirmatory claim.

## Frozen scientific boundary

- Base experiment: `56fe7764ec28c92947c83cb3d7dbd16e5630311b`.
- Checkpoint revision: `7eceff3a9f7e6f916c824d197266d86676bce695`.
- Selected-tree SHA-256: `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.
- Exact capture SHA-256: `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`.
- Cross-reference capture SHA-256 (fit only): `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`.
- Evaluation: 69 exact-checkpoint validation invocations.
- Candidate interface: 256 coherent packets fetched (768 pages; 393,216 bytes; 1.0 physical bpw),
  followed by 192 applications (0.75 logical bpw).

The locked Q2→Q3→Q4 codec, trees, checkpoint, request splits, and future-proxy qenergy metric
are unchanged. Actual routed training occurrences and synthetic activation/expert augmentation
are reported separately.

## Evidence coverage

| artifact | layer | rows | unique_invocations | unique_requests | experts | split | capture_source |
|---|---|---|---|---|---|---|---|
| hybrid | 0 | 165 | 11 | 3 | 3 | validation | exact_checkpoint |
| hybrid | 4 | 285 | 19 | 3 | 3 | validation | exact_checkpoint |
| hybrid | 20 | 300 | 20 | 3 | 3 | validation | exact_checkpoint |
| hybrid | 39 | 285 | 19 | 3 | 3 | validation | exact_checkpoint |
| template | 0 | 1408 | 11 | 3 | 3 | validation | exact_checkpoint |
| template | 4 | 2432 | 19 | 3 | 3 | validation | exact_checkpoint |
| template | 20 | 2560 | 20 | 3 | 3 | validation | exact_checkpoint |
| template | 39 | 2432 | 19 | 3 | 3 | validation | exact_checkpoint |
| selector | 0 | 187 | 11 | 3 | 3 | validation | exact_checkpoint |
| selector | 4 | 323 | 19 | 3 | 3 | validation | exact_checkpoint |
| selector | 20 | 340 | 20 | 3 | 3 | validation | exact_checkpoint |
| selector | 39 | 323 | 19 | 3 | 3 | validation | exact_checkpoint |
| candidate | 0 | 6380 | 11 | 3 | 3 | validation | exact_checkpoint |
| candidate | 4 | 11020 | 19 | 3 | 3 | validation | exact_checkpoint |
| candidate | 20 | 11600 | 20 | 3 | 3 | validation | exact_checkpoint |
| candidate | 39 | 11020 | 19 | 3 | 3 | validation | exact_checkpoint |

Before any summaries or gates, the analyzer independently requires exactly 69 unique validation
invocations spanning every one of the 12 config-locked `(layer, expert)` cells. It then matches
each validation-scoped fit-manifest selector entry to a complete per-cell invocation grid in both
selector artifacts and all four candidate semantics. A wholly absent selector configuration,
expert cell, or candidate interface is therefore a fatal evidence error rather than an omitted row.
The same pre-summary audit requires the runner's five named hybrid families at each configured
0.5/0.75/1.0-bpw budget for all 69 identities. Template evidence is matched to every
validation-scoped fit-manifest cohort and its config-frozen K grid, both top-1 and top-2-union
interfaces, and repair counts 0/16/32/64; each combination and its four rerank interfaces must
cover the exact expert-cell identities. Candidate-cap flags and physical charging are recomputed;
intentional top-2 unions above 256 candidates remain visible as negative, nonpromotable evidence
and cannot pass the existing candidate-max gate. Template results remain explicitly
nonpromotable existence oracles.

## Training provenance: actual routed versus synthetic augmentation

| layer | training_cohort | pairing_semantics | teacher_rows | fit_split | validation_or_test_rows_used |
|---|---|---|---|---|---|
| 0 | combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 402 | train | False |
| 0 | exact_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 210 | train | False |
| 0 | routed_exact_train_primary | actual_routed_occurrences | 14 | train | False |
| 4 | combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 402 | train | False |
| 4 | exact_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 210 | train | False |
| 4 | routed_exact_train_primary | actual_routed_occurrences | 37 | train | False |
| 20 | combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 402 | train | False |
| 20 | exact_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 210 | train | False |
| 20 | routed_exact_train_primary | actual_routed_occurrences | 38 | train | False |
| 39 | combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 402 | train | False |
| 39 | exact_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | 210 | train | False |
| 39 | routed_exact_train_primary | actual_routed_occurrences | 32 | train | False |

Synthetic all-activation/expert pairs add teacher examples, but they are not routed occurrences
and are never counted as validation evidence.

## 1. Coherent/G/D hybrid H0 oracle

| selector_family | physical_budget_bpw | recovery_p10 | recovery_median | recovery_p90 |
|---|---|---|---|---|
| coherent_exact_set_fixed_greedy_teacher | 0.500000 | 0.798267 | 0.872993 | 0.956471 |
| coherent_exact_set_fixed_greedy_teacher | 0.750000 | 0.840910 | 0.906806 | 0.960234 |
| coherent_exact_set_fixed_greedy_teacher | 1.000000 | 0.859512 | 0.922308 | 0.964705 |
| coherent_independent_abc_pr9_control | 0.500000 | 0.710755 | 0.845760 | 0.987287 |
| coherent_independent_abc_pr9_control | 0.750000 | 0.837335 | 0.924381 | 0.997779 |
| coherent_independent_abc_pr9_control | 1.000000 | 0.913045 | 0.956004 | 0.999573 |
| direct_GD_hybrid_forward | 0.500000 | 0.831381 | 0.893643 | 0.956856 |
| direct_GD_hybrid_forward | 0.750000 | 0.862802 | 0.917745 | 0.963431 |
| direct_GD_hybrid_forward | 1.000000 | 0.875410 | 0.924420 | 0.966249 |
| factorized_fixed_greedy_pr9_control | 0.500000 | 0.820295 | 0.892445 | 0.957308 |
| factorized_fixed_greedy_pr9_control | 0.750000 | 0.856061 | 0.916927 | 0.962154 |
| factorized_fixed_greedy_pr9_control | 1.000000 | 0.866240 | 0.922204 | 0.964006 |
| hybrid_forward_plus_1_2_3_unit_local_search | 0.500000 | 0.866639 | 0.929897 | 0.994576 |
| hybrid_forward_plus_1_2_3_unit_local_search | 0.750000 | 0.925015 | 0.961717 | 0.999134 |
| hybrid_forward_plus_1_2_3_unit_local_search | 1.000000 | 0.960458 | 0.978865 | 0.999882 |

The continuation gate is an absolute ≥0.005 improvement in one-bpw median recovery over the
`coherent_exact_set_fixed_greedy_teacher`. Paired deltas versus the
`coherent_independent_abc_pr9_control` are reported separately. It is an exact H0 oracle comparison, not a deployable predictor result.

| selector_family | pr9_independent_median | exact_set_coherent_median | hybrid_median | median_recovery_improvement_vs_exact_set_coherent | paired_delta_vs_exact_set_p10 | paired_delta_vs_exact_set_median | paired_delta_vs_pr9_p10 | paired_delta_vs_pr9_median | passes |
|---|---|---|---|---|---|---|---|---|---|
| hybrid_forward_plus_1_2_3_unit_local_search | 0.956004 | 0.922308 | 0.978865 | 0.056556 | 0.022615 | 0.054389 | 0.000310 | 0.021227 | True |
| direct_GD_hybrid_forward | 0.956004 | 0.922308 | 0.924420 | 0.002112 | -0.010119 | 0.006335 | -0.097153 | -0.030270 | False |
| factorized_fixed_greedy_pr9_control | 0.956004 | 0.922308 | 0.922204 | -0.000104 | -0.016867 | 0.003792 | -0.110575 | -0.029782 | False |

## 2. Per-expert support-template existence oracle

| template_cohort | pairing_semantics | template_selector | template_count_requested | repair_count | validation_oracle_path_utility_retained_p10 | validation_oracle_path_utility_retained_median | candidate_max | complete_validation_coverage | passes |
|---|---|---|---|---|---|---|---|---|---|
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 4 | 0 | 0.374856 | 0.521918 | 192.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 4 | 16 | 0.823922 | 0.889773 | 208.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 4 | 32 | 0.888748 | 0.937910 | 224.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 4 | 64 | 0.958487 | 0.979844 | 256.000000 | True | True |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 8 | 0 | 0.465825 | 0.596274 | 192.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 8 | 16 | 0.826390 | 0.900694 | 208.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 8 | 32 | 0.896115 | 0.938359 | 224.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 8 | 64 | 0.959848 | 0.979844 | 256.000000 | True | True |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 16 | 0 | 0.517336 | 0.626651 | 192.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 16 | 16 | 0.829785 | 0.905562 | 208.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 16 | 32 | 0.898403 | 0.942372 | 224.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 16 | 64 | 0.964622 | 0.981701 | 256.000000 | True | True |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 32 | 0 | 0.546199 | 0.640538 | 192.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 32 | 16 | 0.837945 | 0.908582 | 208.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 32 | 32 | 0.900749 | 0.947121 | 224.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 32 | 64 | 0.966455 | 0.983943 | 256.000000 | True | True |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 64 | 0 | 0.572220 | 0.694613 | 192.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 64 | 16 | 0.842863 | 0.915714 | 208.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 64 | 32 | 0.905982 | 0.949654 | 224.000000 | True | False |
| combined_exact_cross_train_all_x_augmentation | synthetic_all_x_by_expert_augmentation | top1_template_validation_oracle | 64 | 64 | 0.968068 | 0.983943 | 256.000000 | True | True |

_Shows 20 of 232 rows._

Template membership is chosen with validation teacher utility in this experiment. Therefore
this table answers whether a compact support regime exists; it does **not** evaluate a template
classifier. Top-1 requires median gain retention ≥0.95. Top-2 requires median ≥0.98 and p10
≥0.95. Both require at most 256 charged candidates and complete validation coverage.

## 3. PQ and high-rank/low-bit synopsis direct score

| selector_config_id | selector_family | recovery_p10 | recovery_median | selector_metadata_bpw_max | selector_compute_macs_max |
|---|---|---|---|---|---|
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 0.686832 | 0.832359 | 0.092478 | 68608.000000 |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 0.694118 | 0.858315 | 0.177114 | 134144.000000 |
| high_rank_r128_int2 | high_rank_low_bit_linear_response | 0.633105 | 0.854853 | 0.101593 | 396288.000000 |
| high_rank_r128_ternary | high_rank_low_bit_linear_response | 0.669487 | 0.843864 | 0.101593 | 396288.000000 |
| high_rank_r256_ternary | high_rank_low_bit_linear_response | 0.695428 | 0.850251 | 0.190135 | 789504.000000 |
| high_rank_r64_fp8_e4m3fn_per_row | high_rank_low_bit_linear_response | 0.700128 | 0.833522 | 0.182322 | 199680.000000 |
| high_rank_r64_int8_per_row | high_rank_low_bit_linear_response | 0.704694 | 0.835095 | 0.182322 | 199680.000000 |

Block-PQ uses activation-second-moment-weighted, 32-weight blocks with physically packed code
indices. The high-rank/low-bit control is reported as least-squares response fitting followed by
quantization, not as jointly optimized low-bit training. It is a transductive sampled-expert
diagnostic and cannot be promoted. The runner and analyzer both charge the 1,024 output-scale
multiplications required by the two row-scaled 512-output synthesis matrices per invocation.
The fit manifest marks these controls nonpromotable, and their transductive sampled-expert scope
independently enforces that exclusion.

## 4. Direct set predictor

| selector_config_id | selector_family | recovery_p10 | recovery_median | selector_metadata_bpw_max | selector_compute_macs_max |
|---|---|---|---|---|---|
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | 0.430192 | 0.720174 | 0.007853 | 188512.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | 0.451165 | 0.692320 | 0.092491 | 278656.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | 0.293299 | 0.576540 | 0.092490 | 270464.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_single | direct_set_predictor | 0.368037 | 0.633634 | 0.092490 | 270464.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_static_unit_bias_control_listwise_boundary_hard_set_single | direct_set_predictor | 0.456490 | 0.714604 | 0.010458 | 196720.000000 |
| direct_exact_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | 0.427586 | 0.729483 | 0.007853 | 188512.000000 |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | 0.347209 | 0.688696 | 0.092491 | 278656.000000 |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | 0.343814 | 0.655597 | 0.092490 | 270464.000000 |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_single | direct_set_predictor | 0.377440 | 0.698400 | 0.092490 | 270464.000000 |
| direct_exact_train_all_x_augmentation_q2_abc_static_unit_bias_control_listwise_boundary_hard_set_single | direct_set_predictor | 0.462543 | 0.721962 | 0.010458 | 196720.000000 |

Training may use exact marginal paths and correction Grams, while inference is restricted to
declared Q2-side features and charged synopsis/static metadata. Soft or straight-through masks
are training surrogates only; every reported validation decision is a hard discrete 256/192 set.
The joint-head variant supervises coverage of the teacher top-192 under the actual nested
top-192 apply-head plus top-64 non-duplicate candidate-head fetch construction. Although the
training helper supports optional exclusion-regret targets, this run supplies and optimizes no
exclusion-regret supervision.

## 5. Candidate interface: four distinct meanings

| selector_config_id | selector_family | rerank_semantics | selection_regime | recovery_p10 | recovery_median | oracle_set_gain_retention_p10 | selector_metadata_bpw_max | selector_compute_macs_max | rerank_incremental_compute_macs_max | total_selector_compute_macs_max |
|---|---|---|---|---|---|---|---|---|---|---|
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | contained_fetched_target_exact_h0 | information_deployable_but_expensive_interaction_oracle | 0.696950 | 0.826182 | 0.819388 | 0.092478 | 68608.000000 | 138674176.000000 | 138742784.000000 |
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | 0.743691 | 0.878299 | 0.836796 | 0.092478 | 68608.000000 | 1050112.000000 | 1118720.000000 |
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | full_target_restricted_teacher_oracle | teacher_oracle_not_deployable | 0.808976 | 0.885465 | 0.951106 | 0.092478 | 68608.000000 | 546308096.000000 | 546376704.000000 |
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | predicted_direct_no_rerank | deployable_predicted_application | 0.686832 | 0.832359 | 0.763899 | 0.092478 | 68608.000000 | 0.000000 | 68608.000000 |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | contained_fetched_target_exact_h0 | information_deployable_but_expensive_interaction_oracle | 0.694371 | 0.841074 | 0.838975 | 0.177114 | 134144.000000 | 138674176.000000 | 138808320.000000 |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | 0.787213 | 0.886332 | 0.904480 | 0.177114 | 134144.000000 | 1050112.000000 | 1184256.000000 |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | full_target_restricted_teacher_oracle | teacher_oracle_not_deployable | 0.811905 | 0.890373 | 0.952118 | 0.177114 | 134144.000000 | 546308096.000000 | 546442240.000000 |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | predicted_direct_no_rerank | deployable_predicted_application | 0.694118 | 0.858315 | 0.789216 | 0.177114 | 134144.000000 | 0.000000 | 134144.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | contained_fetched_target_exact_h0 | information_deployable_but_expensive_interaction_oracle | 0.454271 | 0.746263 | 0.548602 | 0.007853 | 188512.000000 | 138674176.000000 | 138862688.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | 0.503124 | 0.781051 | 0.605072 | 0.007853 | 188512.000000 | 1050112.000000 | 1238624.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | full_target_restricted_teacher_oracle | teacher_oracle_not_deployable | 0.691094 | 0.828439 | 0.825170 | 0.007853 | 188512.000000 | 546308096.000000 | 546496608.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | 0.430192 | 0.720174 | 0.510789 | 0.007853 | 188512.000000 | 0.000000 | 188512.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | contained_fetched_target_exact_h0 | information_deployable_but_expensive_interaction_oracle | 0.474616 | 0.670633 | 0.570589 | 0.092491 | 278656.000000 | 138674176.000000 | 138952832.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | 0.538996 | 0.756040 | 0.608263 | 0.092491 | 278656.000000 | 1050112.000000 | 1328768.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | full_target_restricted_teacher_oracle | teacher_oracle_not_deployable | 0.679550 | 0.823946 | 0.814497 | 0.092491 | 278656.000000 | 546308096.000000 | 546586752.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | 0.451165 | 0.692320 | 0.552430 | 0.092491 | 278656.000000 | 0.000000 | 278656.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | contained_fetched_target_exact_h0 | information_deployable_but_expensive_interaction_oracle | 0.388532 | 0.674789 | 0.448742 | 0.092490 | 270464.000000 | 138674176.000000 | 138944640.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | 0.425686 | 0.725799 | 0.472385 | 0.092490 | 270464.000000 | 1050112.000000 | 1320576.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | full_target_restricted_teacher_oracle | teacher_oracle_not_deployable | 0.668717 | 0.810440 | 0.792183 | 0.092490 | 270464.000000 | 546308096.000000 | 546578560.000000 |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | 0.293299 | 0.576540 | 0.340842 | 0.092490 | 270464.000000 | 0.000000 | 270464.000000 |

_Shows 20 of 580 rows._

- `predicted_direct_no_rerank` applies the predicted 192 with no exact reranking.
- `exact_independent_abc_within_fetched_candidates` computes exact independent A/B/C scores for fetched units. It is the primary
  realistic H0-late reranker and charges Q4 gate/up response work for all 256 candidates,
  including their resident Q2 gate/up parent-code payload, conditional E8M0 scale payload, and
  reported Q4-response block-scale applications. The fetched 1,536-byte packet per unit remains
  suffix-only and excludes resident Q2 data.
- `contained_fetched_target_exact_h0` forms fetched correction vectors and builds the rank-4-proxy-augmented 256-action
  interaction Gram (about 138.7M incremental analytical MACs). It charges Q4 responses,
  the same 256-unit gate/up parent codes, conditional E8M0 scales, Q4 block-scale applications,
  correction formation/workspace, the Euclidean and proxy Gram components, and proxy metadata.
  It remains a precise lower bound only because resident Q2-down parent-code/scale payload,
  Q2-down tree/E8M0 decode and block-scale application, fixed-greedy initial correlation, and
  per-selection vector updates remain excluded. It is an expensive H0 systems oracle and is
  never used for promotion.
- `full_target_restricted_teacher_oracle` also uses omitted-unit effects. It is a nondeployable teacher-information
  control and is never used for promotion. Its declared arithmetic charges all 512 Q4 teacher
  responses and their gate/up parent codes, conditional E8M0 scales and block-scale applications,
  correction formation, the all-512 correction workspace, and Euclidean plus rank-4 proxy Gram
  arithmetic. It has the same explicit Q2-down/fixed-greedy lower-bound exclusions as the
  contained path.
  Every exact path is
  `exact_marginal_fixed_greedy` with fixed coefficients and no least-squares refit—not a global
  top-k optimum. Restricted and unrestricted fixed-greedy paths can differ through path
  dependence, so the reported gain-retention ratio may legitimately exceed 1 and is not clipped.

## Frozen gates

| selector_config_id | selector_family | rerank_semantics | selection_regime | rerank_role | unique_invocations | unique_requests | recovery_n | recovery_p10 | recovery_median | recovery_p90 | recovery_min | recovery_max | oracle_set_gain_retention_n | oracle_set_gain_retention_p10 | oracle_set_gain_retention_median | oracle_set_gain_retention_p90 | oracle_set_gain_retention_min | oracle_set_gain_retention_max | selector_metadata_bpw_n | selector_metadata_bpw_p10 | selector_metadata_bpw_median | selector_metadata_bpw_p90 | selector_metadata_bpw_min | selector_metadata_bpw_max | selector_compute_macs_n | selector_compute_macs_p10 | selector_compute_macs_median | selector_compute_macs_p90 | selector_compute_macs_min | selector_compute_macs_max | rerank_incremental_compute_macs_n | rerank_incremental_compute_macs_p10 | rerank_incremental_compute_macs_median | rerank_incremental_compute_macs_p90 | rerank_incremental_compute_macs_min | rerank_incremental_compute_macs_max | total_selector_compute_macs_n | total_selector_compute_macs_p10 | total_selector_compute_macs_median | total_selector_compute_macs_p90 | total_selector_compute_macs_min | total_selector_compute_macs_max | storage_multiplier_n | storage_multiplier_p10 | storage_multiplier_median | storage_multiplier_p90 | storage_multiplier_min | storage_multiplier_max | promotion_eligible_selector | promotion_exclusion_reason | passes_recovery | passes_set_gain | passes_metadata | passes_compute | passes_storage | passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.743691 | 0.878299 | 0.997062 | 0.610778 | 0.999915 | 69 | 0.836796 | 0.976070 | 1.046086 | 0.731025 | 1.227839 | 69 | 0.092478 | 0.092478 | 0.092478 | 0.092478 | 0.092478 | 69 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1118720.000000 | 1118720.000000 | 1118720.000000 | 1118720.000000 | 1118720.000000 | 69 | 1.492348 | 1.492348 | 1.492348 | 1.492348 | 1.492348 | True | none | False | False | True | True | True | False |
| block_pq_s1_separate_gate_up | block_pq_residual_synopsis | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.686832 | 0.832359 | 0.991405 | 0.486702 | 0.998876 | 69 | 0.763899 | 0.950316 | 1.034614 | 0.536721 | 1.174932 | 69 | 0.092478 | 0.092478 | 0.092478 | 0.092478 | 0.092478 | 69 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 68608.000000 | 69 | 1.492348 | 1.492348 | 1.492348 | 1.492348 | 1.492348 | True | none | False | False | True | True | True | False |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.787213 | 0.886332 | 0.997543 | 0.522626 | 0.999915 | 69 | 0.904480 | 0.982630 | 1.045318 | 0.604639 | 1.207349 | 69 | 0.177114 | 0.177114 | 0.177114 | 0.177114 | 0.177114 | 69 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1184256.000000 | 1184256.000000 | 1184256.000000 | 1184256.000000 | 1184256.000000 | 69 | 1.512262 | 1.512262 | 1.512262 | 1.512262 | 1.512262 | True | none | False | False | True | True | True | False |
| block_pq_s2_separate_gate_up | block_pq_residual_synopsis | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.694118 | 0.858315 | 0.992128 | 0.495034 | 0.998984 | 69 | 0.789216 | 0.956143 | 1.037105 | 0.572716 | 1.169225 | 69 | 0.177114 | 0.177114 | 0.177114 | 0.177114 | 0.177114 | 69 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 134144.000000 | 69 | 1.512262 | 1.512262 | 1.512262 | 1.512262 | 1.512262 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.503124 | 0.781051 | 0.991333 | -0.174785 | 0.999915 | 69 | 0.605072 | 0.871023 | 1.026048 | -0.198998 | 1.153461 | 69 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 69 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.430192 | 0.720174 | 0.964642 | -0.286691 | 0.989027 | 69 | 0.510789 | 0.786279 | 1.004284 | -0.326407 | 1.116260 | 69 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.538996 | 0.756040 | 0.984537 | 0.279913 | 0.999915 | 69 | 0.608263 | 0.844677 | 1.009600 | 0.332423 | 1.153461 | 69 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.451165 | 0.692320 | 0.913994 | 0.116547 | 0.984662 | 69 | 0.552430 | 0.761090 | 0.954440 | 0.137721 | 1.071798 | 69 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.425686 | 0.725799 | 0.973428 | 0.160432 | 0.999915 | 69 | 0.472385 | 0.791063 | 0.996225 | 0.207553 | 1.153461 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.293299 | 0.576540 | 0.908608 | 0.093435 | 0.981018 | 69 | 0.340842 | 0.634650 | 0.951400 | 0.097733 | 1.110676 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.442425 | 0.710659 | 0.995848 | 0.241891 | 0.999915 | 69 | 0.516784 | 0.791906 | 1.028946 | 0.257384 | 1.153461 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.368037 | 0.633634 | 0.944910 | 0.177814 | 0.991897 | 69 | 0.431667 | 0.704284 | 0.989852 | 0.211170 | 1.104780 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_static_unit_bias_control_listwise_boundary_hard_set_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.538662 | 0.781546 | 0.995666 | 0.252221 | 0.999915 | 69 | 0.644353 | 0.857542 | 1.038323 | 0.287161 | 1.153461 | 69 | 0.010458 | 0.010458 | 0.010458 | 0.010458 | 0.010458 | 69 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1246832.000000 | 1246832.000000 | 1246832.000000 | 1246832.000000 | 1246832.000000 | 69 | 1.473049 | 1.473049 | 1.473049 | 1.473049 | 1.473049 | False | transductive_sampled_expert_upper_bound | False | False | True | True | True | False |
| direct_combined_exact_cross_train_all_x_augmentation_q2_abc_static_unit_bias_control_listwise_boundary_hard_set_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.456490 | 0.714604 | 0.931762 | 0.216422 | 0.992579 | 69 | 0.526536 | 0.770391 | 0.978621 | 0.246403 | 1.028965 | 69 | 0.010458 | 0.010458 | 0.010458 | 0.010458 | 0.010458 | 69 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 196720.000000 | 69 | 1.473049 | 1.473049 | 1.473049 | 1.473049 | 1.473049 | False | transductive_sampled_expert_upper_bound | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.524946 | 0.769916 | 0.995376 | 0.173205 | 0.999915 | 69 | 0.618696 | 0.859678 | 1.044846 | 0.197121 | 1.154763 | 69 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 1238624.000000 | 69 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | True | none | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_control_listwise_boundary_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.427586 | 0.729483 | 0.953861 | -0.017716 | 0.991937 | 69 | 0.498287 | 0.811455 | 0.998252 | -0.021181 | 1.063817 | 69 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 0.007853 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 188512.000000 | 69 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | 1.472436 | True | none | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.406050 | 0.750920 | 0.995156 | 0.119569 | 0.999915 | 69 | 0.468361 | 0.850326 | 1.038344 | 0.136133 | 1.153461 | 69 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 1328768.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_joint_nested_listwise_boundary_hard_set_joint_nested | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.347209 | 0.688696 | 0.969233 | -0.562095 | 0.992875 | 69 | 0.405929 | 0.745013 | 0.998403 | -0.639962 | 1.120594 | 69 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 0.092491 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 278656.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | exact_independent_abc_within_fetched_candidates | deployable_after_candidate_fetch | primary_realistic_h0_rerank | 69 | 3 | 69 | 0.475592 | 0.757232 | 0.992947 | 0.325085 | 0.999915 | 69 | 0.582461 | 0.820013 | 1.032858 | 0.354978 | 1.153461 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 1050112.000000 | 69 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 1320576.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |
| direct_exact_train_all_x_augmentation_q2_abc_pq1_listwise_boundary_hard_set_single | direct_set_predictor | predicted_direct_no_rerank | deployable_predicted_application | primary_predictor_output | 69 | 3 | 69 | 0.343814 | 0.655597 | 0.967566 | 0.096303 | 0.990045 | 69 | 0.405475 | 0.722174 | 0.989759 | 0.119757 | 1.120518 | 69 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 0.092490 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 69 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 270464.000000 | 69 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | 1.492351 | True | none | False | False | True | True | True | False |

_Shows 20 of 34 rows._

Every promoted predicted or independent-ABC H0-late configuration must have median recovery ≥0.94, p10 recovery
≥0.88, p10 exact-set-gain retention ≥0.97, metadata ≤0.35 bpw, total selector plus rerank compute strictly below
1,572,864 MACs, and total storage below 5×. Physical traffic charges all 256 fetched packets;
logical traffic reflects only 192 applied packets.

## Recomputed compute and storage accounting

| layer | expert_id | selector_config_id | selector_family | expert_specific_metadata_bytes | layer_shared_metadata_bytes | layer_shared_amortized_bytes_per_expert | abc_metadata_bytes | selector_metadata_bytes_per_expert | selector_metadata_bpw | selector_metadata_bpw_recomputed | selector_compute_macs | selector_abc_score_units | selector_abc_score_compute_macs | selector_abc_score_macs_per_unit | selector_abc_score_mac_convention | selector_abc_score_units_recomputed | selector_abc_score_compute_macs_recomputed | selector_abc_score_macs_per_unit_recomputed | selector_compute_macs_recomputed | selector_compute_additions | selector_compute_additions_recomputed | selector_scale_multiplications | selector_scale_multiplications_recomputed | selector_bytes_read | selector_bytes_read_recomputed | selector_bytes_read_semantics | selector_activation_payload_already_read | selector_activation_payload_bytes_read | selector_q2_unit_feature_bytes_read | selector_abc_metadata_bytes_read | storage_multiplier | storage_multiplier_recomputed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 62 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 0 | 72 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 0 | 160 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 4 | 17 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 4 | 110 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 4 | 154 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 20 | 191 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 20 | 242 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 20 | 251 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 39 | 68 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 39 | 108 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 39 | 231 | block_pq_s1_separate_gate_up | block_pq_residual_synopsis | 32768 | 131072 | 512.000000 | 3084 | 36364.000000 | 0.092478 | 0.092478 | 68608 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 68608.000000 | 64512 | 64512.000000 | 65536 | 65536.000000 | 501772 | 501772.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.492348 | 1.492348 |
| 0 | 62 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 0 | 72 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 0 | 160 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 4 | 17 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 4 | 110 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 4 | 154 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 20 | 191 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |
| 20 | 242 | block_pq_s2_separate_gate_up | block_pq_residual_synopsis | 65536 | 262144 | 1024.000000 | 3084 | 69644.000000 | 0.177114 | 0.177114 | 134144 | 512 | 3072 | 6 | six_scalar_multiplications_per_unit_for_A_h4_squared_minus_2B_h4_h2_plus_C_h2_squared;_minus_2_is_prefolded_into_B_without_changing_metadata_size;_two_scalar_additions_are_not_MACs | 512.000000 | 3072.000000 | 6.000000 | 134144.000000 | 130048 | 130048.000000 | 65536 | 65536.000000 | 927756 | 927756.000000 | logical_unique_payload_plus_LUT_reads_not_measured_DRAM_traffic | True | 4096 | 3072 | 3084 | 1.512262 | 1.512262 |

_Shows 20 of 204 rows._

The analyzer recomputes page bytes/bpw, logical actions/bpw, amplification, expert and amortized
layer metadata, storage multiplier, and family-specific MACs. Ordinary three-scalar FP16 A/B/C
metadata is 3,084 bytes per expert (three 512-value FP16 arrays plus three FP32 scales). The
nonpromotable static-unit-bias control adds one FP16 scalar per unit and therefore charges 4,108
bytes. Standalone PQ and high-rank scores also charge the exact 3,072-byte Q2 g/u/h payload and
serialized A/B/C payload plus 3,072 analytical MACs (six per unit) for the 512 A/B/C scores;
direct predictors charge serialized A/B/C bytes rather than a vector
dimension approximation. MACs, additions, scale multiplications, non-MAC/decode/control
operations, and logical bytes are reported as distinct quantities. Every MAC column is an
analytical linear/algebraic contract; nonlinear SiLU/log/sqrt, elementwise and control operations,
top-k/sort, and runtime overhead are excluded from it and disclosed separately. Every read-byte
column is logical unique payload plus explicitly declared LUT reads, not measured DRAM, cache, or
PCIe traffic. The precise candidate components and bound strings are preserved in
`candidate_rerank_accounting.csv`. Support-template existence-oracle rows have no deployable base
selector, so their raw base-selector addition and scale-multiplication cells remain empty in that
CSV and are represented as explicit `null` values in the accounting JSON; their rerank and total
charges remain finite and explicit. The explicitly disclosed diagnostic accounting caveats above
are never used for promotion. The analyzer refuses evidence if a reported value differs.

## H0, deployable approximation, and H4 boundary

| Result class | Activation timing | Status |
|---|---|---|
| Hybrid exact oracle | H0 | Oracle only |
| Best-template choice | H0 teacher utility | Existence oracle only |
| PQ/high-rank direct score | H0 actual activation | H0-late scoring; external fetch follows |
| Direct set predictor | H0 Q2-side features | H0-late scoring; not H4 prefetch |
| Independent A/B/C rerank | H0 after candidate fetch | Primary realistic H0-late rerank |
| Contained-target interaction rerank | H0 after candidate fetch | Expensive systems oracle; not promoted |
| Full-target restricted rerank | H0 teacher with omitted effects | Nondeployable information control |
| H4 candidate predictor | H4 | Not trained or evaluated in this branch |

## Primary-source context, not proof

[ShadowLLM](https://arxiv.org/abs/2406.16635) studies predictor-based contextual sparsity and
reports that criterion predictability/objective choice matters. [DejaVu](https://arxiv.org/abs/2310.17157)
supports predicting input-dependent model structures. [SOFT top-k](https://arxiv.org/abs/2002.06504)
and [SIMPLE](https://arxiv.org/abs/2210.01941) motivate differentiable surrogates for discrete
selection; they do not remove the need for hard-cardinality evaluation. [AQLM](https://arxiv.org/abs/2401.06118)
and [GPTVQ](https://arxiv.org/abs/2402.15319) support activation/Hessian-aware vector-quantization
objectives. None of these papers proves that this activation-dependent MXFP4 streaming selector
works, and no latency number is imported from them.

## Reproducibility and interpretation

The canonical decision is `set_utility_promotions.json`. `analysis_manifest.json` hashes every
input, generated artifact, and the analyzer wrapper/core sources using repository-relative paths.
The 14 generated CSVs (seven frontier tables, three gate tables, provenance, coverage, selector
accounting, and candidate-rerank accounting) and all plots are descriptive views of the same
immutable validation rows. The old published test split has already informed the research direction and
is deliberately absent; any promoted method requires a newly captured, sealed holdout.
