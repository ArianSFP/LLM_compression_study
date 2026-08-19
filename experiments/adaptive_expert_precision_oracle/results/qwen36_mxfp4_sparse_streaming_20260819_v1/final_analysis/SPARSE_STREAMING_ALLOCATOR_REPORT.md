# Activation-Dependent Sparse Weight Streaming Study

Run: `qwen36_mxfp4_sparse_streaming_20260819_v1`

## Scope and claim boundary

This study measures **expert-output qenergy recovery**, not task accuracy. It does not measure logits, routing quality, token quality, perplexity, or downstream benchmarks. The exact checkpoint, embedded Q2→Q3→Q4 codec, selected trees, and train/validation/test request split remain locked.

Every selector retains a precise `selection_regime`; a derived category groups it as `h0_oracle`, `h0_proxy`, or `deployable_h0_proxy` without erasing whether it reads the full suffix, runs late after resident Q2, or needs deployable metadata. H0 choices may use the current activation or exact suffix corrections and are not H4 prefetch claims. No H4 predictor is trained here. The broader cross-reference capture is a secondary sensitivity check; primary claims use the fresh exact-checkpoint capture.

Router rank and router coefficient are occurrence metadata only. This study does not evaluate router decisions or logit agreement, and qenergy recovery must not be restated as model accuracy.

## Outcome map

The frozen family decisions are neuron-major **`promote`**, activation shortlist **`stop`**, and tile streaming **`promote`**. The promoted neuron and tile configurations are H0 oracles. A deployable-H0 selector was **not selected**. Consequently, any positive held-out result below establishes allocator headroom, not a deployable streaming policy and not H4 prefetch feasibility.

Fresh exact-checkpoint validation selected configurations; fresh exact-checkpoint test rows provide the primary held-out estimate. The broader capture is reported separately as `secondary_cross_reference_sensitivity` and never pooled with the primary cohort. It cannot change a frozen selection.

## Held-out success-gate disposition

This disposition uses only the frozen selected neuron and tile identities on the primary fresh `exact_checkpoint` test rows at exactly **1.0 physical correction bpw** and only configured audited layers `[0, 4, 20, 39]`. Full-run rows are preferred over duplicate pilot sanity rows. No cross-reference row or other budget enters these statistics.

The configured PR #6 H0-hybrid comparator is p10 **0.8724** and median **0.9237** on its all-layer cohort. Overall fresh exact-checkpoint results and descriptive deltas to that configured comparator are:

| path | selector | n | p10 | median | delta_p10_vs_configured_pr6_h0_hybrid | delta_median_vs_configured_pr6_h0_hybrid | median_selector_runtime_ms | median_unique_footprint_bytes_lower_bound | median_page_amplification | median_storage_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neuron_major | independent_unit_correction_norm | 85 | 0.9268 | 0.9661 | 0.0544 | 0.0424 | 6.3850 | 8388608 | 1.0000 | 1.4706 |
| tile_64x16 | exact_dynamic_tile_marginal_refresh_1 | 85 | 0.9375 | 0.9683 | 0.0651 | 0.0446 | 1386.5348 | 25174016 | 1.0000 | 1.4706 |

By-layer primary audit:

| path | layer | n | p10 | median | p90 |
| --- | --- | --- | --- | --- | --- |
| neuron_major | 0 | 13 | 0.9964 | 1.0000 | 1.0000 |
| neuron_major | 4 | 30 | 0.9168 | 0.9549 | 0.9819 |
| neuron_major | 20 | 19 | 0.9254 | 0.9455 | 0.9665 |
| neuron_major | 39 | 23 | 0.9436 | 0.9805 | 0.9964 |
| tile_64x16 | 0 | 13 | 0.9715 | 0.9797 | 0.9859 |
| tile_64x16 | 4 | 30 | 0.9396 | 0.9626 | 0.9803 |
| tile_64x16 | 20 | 19 | 0.9395 | 0.9608 | 0.9737 |
| tile_64x16 | 39 | 23 | 0.9294 | 0.9718 | 0.9952 |

Difficult-layer pooled audit (configured layers other than layer 0):

| path | layers | n | p10 | median | delta_p10_vs_configured_all_layer_pr6_h0_hybrid | delta_median_vs_configured_all_layer_pr6_h0_hybrid | improves_both_configured_pr6_statistics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neuron_major | 4,20,39 | 72 | 0.9225 | 0.9598 | 0.0501 | 0.0361 | True |
| tile_64x16 | 4,20,39 | 72 | 0.9364 | 0.9632 | 0.0640 | 0.0395 | True |

The difficult-layer deltas use the configured **all-layer** PR #6 H0-hybrid control as a descriptive comparator because no layer-matched PR #6 values are present in this configuration. They are not presented as a layer-matched control.

These recovery results are **H0 oracle headroom only**, not a deployable selector result:

- **Deployable proxy: FAIL.** No deployable proxy was selected; the frozen neuron and tile winners inspect the full suffix.
- **Tile refresh limit:** **FAIL.** The selected exact tile oracle refreshes after every selected page (`refresh_interval=1`): **768 global refreshes** at 1.0 bpw, not **<=16**.
- **Physical amplification and storage:** **PASS for these accounting limits.** The maximum selected-path median physical page amplification **1.0** (limit <=1.2) and median storage multiplier **1.470588** (limit <5.0).
- **Rank <= 64 / low-rank gain-retention gate: NOT APPLICABLE and NOT DEMONSTRATED.** These structured H0 oracles do not instantiate a rank-64-or-lower approximation and therefore cannot establish the requested >=90% retained exact-over-diagonal gain with no more than 16 refreshes.

## Promotion protocol

`sparse_streaming_promotions.json` is frozen from validation rows only. Test rows were joined only after configuration IDs had been selected. A changed decision cannot overwrite the frozen JSON. The exact action labels remain evidence, not training data for any held-out decision in this analysis.

Every candidate must cover layers `[0, 4, 20, 39]` with at least **4 invocations and 2 distinct requests per layer**. `promotion_validation_coverage.csv` records the actual count for every candidate/layer, and each promoted validation record reports its total `n`.

Promotion gates use the frozen pooled p10/median rules, but a broad-effect claim requires the separate layer audit. `neuron_major_by_layer_summary.csv`, `activation_shortlist_by_layer_summary.csv`, and `tile_streaming_by_layer_summary.csv` report n/p10/median/p90 and physical/logical accounting for every layer, split, capture source, and selector identity. Coverage alone is never presented as evidence of consistent effect size.

## Completed inputs and pilot/full deduplication

Only run directories with `completed=true` and matching expected/observed unique-invocation counts are accepted. Pilot and full rows are keyed by invocation plus scientific configuration. Duplicate recovery, support, and action evidence must agree within the frozen tolerance; the full row is then retained and the pilot copy dropped. Pre/post counts are: `{"concentration": {"duplicate_scientific_key_groups": 32, "kept_rows_by_run_mode": {"full": 1816, "pilot": 440}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 2256, "pre_rows": 2288, "pre_rows_by_run_mode": {"full": 1816, "pilot": 472}, "preference": "full_over_pilot", "rows_removed": 32}, "labels": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 483328}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 483328, "pre_rows": 483328, "pre_rows_by_run_mode": {"pilot": 483328}, "preference": "full_over_pilot", "rows_removed": 0}, "neuron": {"duplicate_scientific_key_groups": 104, "kept_rows_by_run_mode": {"full": 5902, "pilot": 3200}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 9102, "pre_rows": 9206, "pre_rows_by_run_mode": {"full": 5902, "pilot": 3304}, "preference": "full_over_pilot", "rows_removed": 104}, "shortlist": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 9912}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 9912, "pre_rows": 9912, "pre_rows_by_run_mode": {"pilot": 9912}, "preference": "full_over_pilot", "rows_removed": 0}, "stability": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 2794}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 2794, "pre_rows": 2794, "pre_rows_by_run_mode": {"pilot": 2794}, "preference": "full_over_pilot", "rows_removed": 0}, "tile": {"duplicate_scientific_key_groups": 48, "kept_rows_by_run_mode": {"full": 4086, "pilot": 3678}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 7764, "pre_rows": 7812, "pre_rows_by_run_mode": {"full": 4086, "pilot": 3726}, "preference": "full_over_pilot", "rows_removed": 48}}`. Duplicate invocations are never silently reweighted.

## Method context, not evidence for this result

- [Prox](https://arxiv.org/abs/2607.27591) motivates using the SwiGLU intermediate state as a channel-salience signal. Prox studies channel omission in a different serving design; it does not establish the packet-recovery result measured here.
- [WINA](https://arxiv.org/abs/2505.19427) and its [reference implementation](https://github.com/microsoft/wina) motivate activation-times-weight proxy scores. WINA's exact orthogonality argument does not hold for these unrotated correction columns, so the score is treated only as a heuristic.
- [R-Sparse](https://arxiv.org/abs/2504.19449) motivates combining concentrated input support with another structured representation. It is not evidence that the correction-action Gram is sparse.
- [TEAL](https://arxiv.org/abs/2408.14690) and its [reference implementation](https://github.com/FasterDecoding/TEAL) motivate a separately labeled resident-Q2 sparsification control. That intervention changes resident computation and is not conflated with lossless selective suffix streaming.

## Measured activation concentration

`activation_concentration.parquet` and `activation_concentration_summary.csv` regenerate the heavy-tail diagnostics from the locked captures: exact-zero fraction, top-K energy fraction, and coordinates required for 90% energy. No previously quoted 0.069%/81%/762 figure is treated as evidence unless it is reproduced in these artifacts. Plot 06 shows the measured held-out curves and layer medians.

## Neuron-major progressive refinement

Neuron packets make 512 SwiGLU hidden units the decision axis. Full Q2→Q4 refinement costs three 512-byte pages per unit; the progressive form costs two pages for Q3 and one additional page for Q4. At one correction bpw, 256 complete unit packets fit.

Held-out promoted configurations:

| source_run_mode | capture_source | evidence_role | physical_budget_bpw | action_family | representation | selector | selection_category | selection_regime | n | p10 | median | p90 | median_physical_pages | median_physical_bytes | median_logical_actions | median_logical_bytes | median_page_amplification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.0625 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.1829 | 0.4898 | 0.8627 | 48 | 24576 | 16 | 24576 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.1250 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.2949 | 0.5750 | 0.8921 | 96 | 49152 | 32 | 49152 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.2500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.4551 | 0.7239 | 0.9273 | 192 | 98304 | 64 | 98304 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.3750 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.6211 | 0.8003 | 0.9560 | 288 | 147456 | 96 | 147456 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.5000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.6891 | 0.8676 | 0.9713 | 384 | 196608 | 128 | 196608 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 0.7500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.8227 | 0.9349 | 0.9908 | 576 | 294912 | 192 | 294912 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 1.0000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.8989 | 0.9664 | 0.9980 | 768 | 393216 | 256 | 393216 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 1.2500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.9589 | 0.9850 | 0.9999 | 960 | 491520 | 320 | 491520 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 1.5000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 0.9828 | 0.9951 | 1.0000 | 1152 | 589824 | 384 | 589824 | 1.0000 |
| full | cross_reference | secondary_cross_reference_sensitivity | 2.0000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 142 | 1.0000 | 1.0000 | 1.0000 | 1536 | 786432 | 512 | 786432 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.0625 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.2141 | 0.5180 | 0.8417 | 48 | 24576 | 16 | 24576 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.1250 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.4029 | 0.6448 | 0.8757 | 96 | 49152 | 32 | 49152 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.2500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.5298 | 0.7619 | 0.9387 | 192 | 98304 | 64 | 98304 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.3750 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.6707 | 0.8414 | 0.9731 | 288 | 147456 | 96 | 147456 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.5000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.7474 | 0.8722 | 0.9842 | 384 | 196608 | 128 | 196608 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 0.7500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.8601 | 0.9298 | 0.9993 | 576 | 294912 | 192 | 294912 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 1.0000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.9268 | 0.9661 | 1.0000 | 768 | 393216 | 256 | 393216 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 1.2500 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.9697 | 0.9864 | 1.0000 | 960 | 491520 | 320 | 491520 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 1.5000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 0.9887 | 0.9957 | 1.0000 | 1152 | 589824 | 384 | 589824 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | 2.0000 | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 85 | 1.0000 | 1.0000 | 1.0000 | 1536 | 786432 | 512 | 786432 | 1.0000 |

The table reports exact sequential complete-expert recovery separately from isolated projection rows in the CSV and plots. Each physical budget is a separate row; recovery is never pooled across bpw. Logical actions are semantic applications, not traffic. Physical traffic is the fetched 512-byte page count and its byte equivalent.

Fresh exact-checkpoint validation candidates at one physical correction bpw:

| action_family | representation | selector | selection_category | selection_regime | n | p10 | median | p90 | passes_p10 | passes_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neuron_major | direct_q2_q4_complete_unit_packet_3pages | weight_aware_down_delta | deployable_h0_proxy | h0_deployable_metadata_late | 55 | 0.7078 | 0.8588 | 1.0000 | False | False |
| neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 55 | 0.9150 | 0.9583 | 1.0000 | True | True |
| neuron_major | direct_q2_q4_complete_unit_packet_3pages | first_order_gate_up_down_sensitivity | h0_oracle | h0_oracle_full_suffix | 55 | 0.8934 | 0.9537 | 1.0000 | True | True |
| neuron_major | progressive_q3_q4_coherent_unit_packet_2plus1_pages | exact_progressive_neuron_page_greedy | h0_oracle | h0_oracle_full_suffix | 55 | 0.8796 | 0.9295 | 0.9669 | True | True |
| neuron_major | direct_q2_q4_complete_unit_packet_3pages | exact_unit_output_marginal_fixed_greedy | h0_oracle | h0_oracle_full_suffix | 55 | 0.8648 | 0.9223 | 0.9640 | False | False |
| neuron_major | direct_q2_q4_complete_unit_packet_3pages | q2_hidden_magnitude | h0_proxy | h0_resident_proxy_late | 55 | 0.7261 | 0.8378 | 1.0000 | False | False |

## Activation shortlist containment and exact H0 rescoring

Candidate prefetch pages are charged even when a candidate is not applied. The primary gate uses `fraction_exact_over_diagonal_gain_retained_matched_fetched_pages`: exact, diagonal, and constrained recovery are all repriced to the same fetched-page count. Planned-prefix containment remains secondary. Exact rescoring within a fetched shortlist remains an H0 oracle unless an independently deployable mechanism supplies both the candidate set and a representation of the omitted target residual.

The frozen shortlist unit is `input_coordinates`. Both refinement planes are aggregated by `sum_stage_energy` and mapped through `canonical_paired_planes_four_coordinates_per_projection_page`. Thus a configured K means K input coordinates, not K plane-actions, and candidate traffic is repriced to the canonical physical page union.

Held-out promoted configurations:

No shortlist configuration passed the predeclared utility, candidate-count, p10-minimum-overfetch, and p90-maximum-overfetch gates.

Raw support recall is secondary to retained exact-over-diagonal utility at matched fetched-page budgets. To prevent an underfilled candidate set from passing this matched-page metric trivially, promotion additionally requires p10 candidate overfetch to be at least **1.00**; the bandwidth ceiling requires p90 to be at most **1.25**. P10, median, p90, and maximum overfetch remain visible in the frozen decision and summaries.

The isolated-projection plots use **fresh pilot test sanity only; n=4 invocations**. Because this family stopped before full expansion, pilot sanity curves are implementation evidence only and are not a broad held-out result.

Best fresh exact-checkpoint validation candidate per projection and selection category at one physical correction bpw (shown even when the family stopped):

| projection | score_method | shortlist_size | refresh_block | selection_category | selection_regime | n | p10 | median | p10_candidate_overfetch | p90_candidate_overfetch | passes_p10 | passes_median | passes_candidates | passes_min_overfetch | passes_max_overfetch | passes_overfetch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate | activation_weighted_delta_norm | 384 | 1 | h0_oracle | h0_oracle_full_target_residual | 55 | 0.6329 | 0.7686 | 1.1055 | 1.1547 | False | False | True | True | True | True |
| up | abs_activation | 512 | 1 | h0_oracle | h0_oracle_full_target_residual | 55 | 0.7597 | 0.8402 | 1.3414 | 1.3992 | False | False | True | True | False | False |

## Page-aligned tiles and hybrids

The frozen pilot shape is `32×32`. One 32×32 paired gate/up Q2→Q4 tile is exactly one 512-byte page. Tile selection is nonlinear and is evaluated through actual SwiGLU reconstruction. The ≥3-point validation gate is measured against the locked PR #6 all-layer one-bpw H0-hybrid median **0.9237**. The separately recomputed `input_coordinate_baseline` curve is used only for the alternative 25% matched-recovery page-reduction gate. No test row chooses a shape.

**Frozen-gate disclosure.** The one-bpw input-coordinate validation comparator has median recovery **-14.4413**. It is below the zero-correction recovery of 0. The frozen payload therefore reports a mechanically computed page reduction for a target that even the no-correction baseline already exceeds. That number—including any apparent 75% reduction—is invalid evidence for page savings and is not counted as a success. The frozen promotion bytes are preserved for auditability. Tile promotion is interpreted **solely through the independent recovery-gain gate versus the locked PR #6 H0-hybrid median**, not through this defective alternative diagnostic.

Fresh exact-checkpoint validation tile candidates at one physical correction bpw:

| action_family | tile_shape | selector | selection_category | selection_regime | n | p10 | median | p90 | recovery_point_gain_vs_locked_pr6_h0_hybrid | page_reduction_at_matched_recovery | page_reduction_evidence_valid | passes_recovery_gain | interpreted_passes_page_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_up_tile_plus_down_column | 64x16 | exact_dynamic_tile_marginal_refresh_1 | h0_oracle | h0_oracle_full_suffix | 55 | 0.9352 | 0.9660 | 0.9819 | 0.0423 | 0.7500 | False | True | False |
| gate_up_tile_plus_down_column | 16x64 | exact_dynamic_tile_marginal_refresh_1 | h0_oracle | h0_oracle_full_suffix | 55 | 0.9375 | 0.9659 | 0.9840 | 0.0422 | 0.7500 | False | True | False |
| gate_up_tile_plus_down_column | 32x32 | exact_dynamic_tile_marginal_refresh_1 | h0_oracle | h0_oracle_full_suffix | 55 | 0.9394 | 0.9650 | 0.9828 | 0.0413 | 0.7500 | False | True | False |
| gate_up_tile_plus_down_column | 16x64 | activation_energy_x_weight | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6404 | 0.8352 | 0.9498 | -0.0885 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 64x16 | cartesian_hidden_input_blocks | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6788 | 0.8336 | 0.9359 | -0.0901 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 16x64 | cartesian_hidden_input_blocks | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6622 | 0.8329 | 0.9459 | -0.0908 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 64x16 | activation_energy_x_weight | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6661 | 0.8281 | 0.9463 | -0.0956 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 32x32 | static_wina | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6173 | 0.8259 | 0.9519 | -0.0978 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 32x32 | cartesian_hidden_input_blocks | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | 0.6123 | 0.8073 | 0.9421 | -0.1164 | 0.7500 | False | False | False |
| gate_up_tile_plus_down_column | 32x32 | static_first_order | h0_oracle | h0_oracle_suffix_metadata_heavy | 55 | -30.9617 | -17.3755 | -8.2669 | -18.2992 | 0.7500 | False | False | False |

The deterministic validation ordering selected `64x16` over `16x64` by **0.000098 absolute recovery** (0.0098 percentage point), a scientific near-tie. The selected shape is not claimed to be materially superior on that difference alone.

Held-out one-bpw promoted configurations:

| source_run_mode | capture_source | evidence_role | action_family | tile_shape | selector | selection_category | selection_regime | n | p10 | median | p90 | median_physical_pages | median_physical_bytes | median_logical_actions | median_logical_bytes | median_page_amplification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | cross_reference | secondary_cross_reference_sensitivity | gate_up_tile_plus_down_column | 64x16 | exact_dynamic_tile_marginal_refresh_1 | h0_oracle | h0_oracle_full_suffix | 142 | 0.9306 | 0.9625 | 0.9873 | 768 | 393216 | 768 | 393216 | 1.0000 |
| full | exact_checkpoint | primary_fresh_held_out_test | gate_up_tile_plus_down_column | 64x16 | exact_dynamic_tile_marginal_refresh_1 | h0_oracle | h0_oracle_full_suffix | 85 | 0.9375 | 0.9683 | 0.9899 | 768 | 393216 | 768 | 393216 | 1.0000 |

## Physical accounting

`sparse_streaming_compute_storage_accounting.json` distinguishes logical actions, applied pages, fetched candidate pages, selector bytes, and external representation storage. Complete endpoint storage is accounted from the locked suffix representation; action counts never stand in for physical performance. The configured PR #6 one-bpw controls are preserved in the configuration: `{"cohort": "85 exact-checkpoint held-out invocations over layers 0,4,20,39", "current_diagonal_layout": {"median": 0.8915, "p10": 0.825}, "exact_training_layout_separate_planes": {"median": 0.8973, "p10": 0.8385}, "h0_hybrid_representation": {"median": 0.9237, "p10": 0.8724}, "metric": "exact sequential complete-expert qenergy recovery; not task accuracy", "paired_packets": {"median": 0.92, "p10": 0.8553}}`.

One physical correction bpw is 768 fetched pages or 393,216 bytes for this 3×2048×512-weight expert. Every report table names physical pages/bytes and logical actions/bytes explicitly; none infers bandwidth from the action count.

`selector_runtime_ms` is measured for one complete selector path/frontier construction per invocation. `selector_bytes_read` is a reproducible **unique tensor-footprint lower bound**, not a hardware traffic counter: it excludes cache-dependent repeated marginal scans and Gram-column reads. The same values are repeated on each budget snapshot row; rows must not be summed or read as incremental per-budget costs. Regimes `h0_oracle_full_suffix`, `h0_oracle_full_target_residual`, and `h0_oracle_suffix_metadata_heavy` inspect the full decoded suffix or target residual. Exact/dynamic and static tile selectors in those regimes are H0 oracles, not deployable selectors. Late resident-Q2 proxies are separately labeled and may require a second pass before suffix selection.

## Support stability

`support_stability_summary.csv` separates same-invocation gate/up overlap (`token_gap=0`) from true adjacent-token pairs (`token_gap=1`). Adjacent claims use only the fresh exact-checkpoint capture; the stride-16 broader capture is deliberately excluded from adjacency claims.

The support-stability plot uses **fresh pilot test sanity only; n=4 invocations**. When this is the pilot sanity cohort, it is not treated as evidence of broad temporal predictability.

## Interpretation rule

Promotion failure is a negative result, not permission to tune on test. A positive H0 oracle shows headroom only. A deployable-H0 claim additionally requires a `deployable_h0_proxy` row to pass the same validation gate and retain its result on the frozen held-out evaluation while respecting fetched pages, selector compute, and storage. H4 prefetch feasibility remains a separate future question.
