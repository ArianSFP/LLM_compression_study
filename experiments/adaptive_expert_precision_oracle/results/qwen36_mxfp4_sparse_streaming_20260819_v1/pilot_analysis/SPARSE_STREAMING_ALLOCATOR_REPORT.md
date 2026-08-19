# Activation-Dependent Sparse Weight Streaming Study

Run: `qwen36_mxfp4_sparse_streaming_20260819_v1`

## Scope and claim boundary

This study measures **expert-output qenergy recovery**, not task accuracy. It does not measure logits, routing quality, token quality, perplexity, or downstream benchmarks. The exact checkpoint, embedded Q2→Q3→Q4 codec, selected trees, and train/validation/test request split remain locked.

Every selector retains a precise `selection_regime`; a derived category groups it as `h0_oracle`, `h0_proxy`, or `deployable_h0_proxy` without erasing whether it reads the full suffix, runs late after resident Q2, or needs deployable metadata. H0 choices may use the current activation or exact suffix corrections and are not H4 prefetch claims. No H4 predictor is trained here. The broader cross-reference capture is a secondary sensitivity check; primary claims use the fresh exact-checkpoint capture.

## Promotion protocol

`sparse_streaming_promotions.json` is frozen from validation rows only. Test rows were joined only after configuration IDs had been selected. A changed decision cannot overwrite the frozen JSON. The exact action labels remain evidence, not training data for any held-out decision in this analysis.

Every candidate must cover layers `[0, 4, 20, 39]` with at least **4 invocations and 2 distinct requests per layer**. `promotion_validation_coverage.csv` records the actual count for every candidate/layer, and each promoted validation record reports its total `n`.

Promotion gates use the frozen pooled p10/median rules, but a broad-effect claim requires the separate layer audit. `neuron_major_by_layer_summary.csv`, `activation_shortlist_by_layer_summary.csv`, and `tile_streaming_by_layer_summary.csv` report n/p10/median/p90 and physical/logical accounting for every layer, split, capture source, and selector identity. Coverage alone is never presented as evidence of consistent effect size.

## Completed inputs and pilot/full deduplication

Only run directories with `completed=true` and matching expected/observed unique-invocation counts are accepted. Pilot and full rows are keyed by invocation plus scientific configuration. Duplicate recovery, support, and action evidence must agree within the frozen tolerance; the full row is then retained and the pilot copy dropped. Pre/post counts are: `{"concentration": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 472}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 472, "pre_rows": 472, "pre_rows_by_run_mode": {"pilot": 472}, "preference": "full_over_pilot", "rows_removed": 0}, "labels": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 483328}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 483328, "pre_rows": 483328, "pre_rows_by_run_mode": {"pilot": 483328}, "preference": "full_over_pilot", "rows_removed": 0}, "neuron": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 3304}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 3304, "pre_rows": 3304, "pre_rows_by_run_mode": {"pilot": 3304}, "preference": "full_over_pilot", "rows_removed": 0}, "shortlist": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 9912}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 9912, "pre_rows": 9912, "pre_rows_by_run_mode": {"pilot": 9912}, "preference": "full_over_pilot", "rows_removed": 0}, "stability": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 2794}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 2794, "pre_rows": 2794, "pre_rows_by_run_mode": {"pilot": 2794}, "preference": "full_over_pilot", "rows_removed": 0}, "tile": {"duplicate_scientific_key_groups": 0, "kept_rows_by_run_mode": {"pilot": 3726}, "numeric_evidence_tolerance": {"atol": 1e-08, "rtol": 1e-06}, "post_rows": 3726, "pre_rows": 3726, "pre_rows_by_run_mode": {"pilot": 3726}, "preference": "full_over_pilot", "rows_removed": 0}}`. Duplicate invocations are never silently reweighted.

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

| source_run_mode | capture_source | action_family | representation | selector | selection_category | selection_regime | n | p10 | median | p90 | pages | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | exact_checkpoint | neuron_major | direct_q2_q4_complete_unit_packet_3pages | independent_unit_correction_norm | h0_oracle | h0_oracle_full_suffix | 40.0000 | 0.6616 | 0.9720216511773649 | 1.0000 | 480.0 | 160.0 |

The table reports exact sequential complete-expert recovery separately from isolated projection rows in the CSV and plots. Logical actions are not treated as physical traffic.

## Activation shortlist containment and exact H0 rescoring

Candidate prefetch pages are charged even when a candidate is not applied. The primary gate uses `fraction_exact_over_diagonal_gain_retained_matched_fetched_pages`: exact, diagonal, and constrained recovery are all repriced to the same fetched-page count. Planned-prefix containment remains secondary. Exact rescoring within a fetched shortlist remains an H0 oracle unless an independently deployable mechanism supplies both the candidate set and a representation of the omitted target residual.

The frozen shortlist unit is `input_coordinates`. Both refinement planes are aggregated by `sum_stage_energy` and mapped through `canonical_paired_planes_four_coordinates_per_projection_page`. Thus a configured K means K input coordinates, not K plane-actions, and candidate traffic is repriced to the canonical physical page union.

Held-out promoted configurations:

No shortlist configuration passed the predeclared utility, candidate-count, p10-minimum-overfetch, and p90-maximum-overfetch gates.

Raw support recall is secondary to retained exact-over-diagonal utility at matched fetched-page budgets. To prevent an underfilled candidate set from passing this matched-page metric trivially, promotion additionally requires p10 candidate overfetch to be at least **1.00**; the bandwidth ceiling requires p90 to be at most **1.25**. P10, median, p90, and maximum overfetch remain visible in the frozen decision and summaries.

## Page-aligned tiles and hybrids

The frozen pilot shape is `32×32`. One 32×32 paired gate/up Q2→Q4 tile is exactly one 512-byte page. Tile selection is nonlinear and is evaluated through actual SwiGLU reconstruction. The ≥3-point validation gate is measured against the locked PR #6 all-layer one-bpw H0-hybrid median **0.9237**. The separately recomputed `input_coordinate_baseline` curve is used only for the alternative 25% matched-recovery page-reduction gate. No test row chooses a shape.

Held-out one-bpw promoted configurations:

No unrestricted gate/up-tile plus down-column path passed the validation recovery-gain or matched-recovery page-reduction gate.

## Physical accounting

`sparse_streaming_compute_storage_accounting.json` distinguishes logical actions, applied pages, fetched candidate pages, selector bytes, and external representation storage. Complete endpoint storage is accounted from the locked suffix representation; action counts never stand in for physical performance. The configured PR #6 one-bpw controls are preserved in the configuration: `{"cohort": "85 exact-checkpoint held-out invocations over layers 0,4,20,39", "current_diagonal_layout": {"median": 0.8915, "p10": 0.825}, "exact_training_layout_separate_planes": {"median": 0.8973, "p10": 0.8385}, "h0_hybrid_representation": {"median": 0.9237, "p10": 0.8724}, "metric": "exact sequential complete-expert qenergy recovery; not task accuracy", "paired_packets": {"median": 0.92, "p10": 0.8553}}`.

`selector_runtime_ms` is measured for one complete selector path/frontier construction per invocation. `selector_bytes_read` is a reproducible **unique tensor-footprint lower bound**, not a hardware traffic counter: it excludes cache-dependent repeated marginal scans and Gram-column reads. The same values are repeated on each budget snapshot row; rows must not be summed or read as incremental per-budget costs. Regimes `h0_oracle_full_suffix`, `h0_oracle_full_target_residual`, and `h0_oracle_suffix_metadata_heavy` inspect the full decoded suffix or target residual. Exact/dynamic and static tile selectors in those regimes are H0 oracles, not deployable selectors. Late resident-Q2 proxies are separately labeled and may require a second pass before suffix selection.

## Support stability

`support_stability_summary.csv` separates same-invocation gate/up overlap (`token_gap=0`) from true adjacent-token pairs (`token_gap=1`). Adjacent claims use only the fresh exact-checkpoint capture; the stride-16 broader capture is deliberately excluded from adjacency claims.

## Interpretation rule

Promotion failure is a negative result, not permission to tune on test. A positive H0 oracle shows headroom only. A deployable-H0 claim additionally requires a `deployable_h0_proxy` row to pass the same validation gate and retain its result on the frozen held-out evaluation while respecting fetched pages, selector compute, and storage. H4 prefetch feasibility remains a separate future question.
