# Next Oracle Experiment: Corrected Page Allocation and Output-Side Down Factors

## 1. Executive conclusion

**Run A finds real low-rate allocation and layout headroom, but not a higher asymptotic representation ceiling. Run B rejects a layer-shared output-side low-rank down correction at the tested ranks; a per-expert basis upper bound shows that expert heterogeneity, not coefficient quantization, is the main obstacle. Broader all-layer scaling is not yet justified.**

On 39 held-out, stratified expert invocations across layers 0/4/20/39, the corrected Q2 per-atom/page allocator improves matched complete-expert recovery over the reproduced whole-projection policy by a median **+15.83 points at 0.5 physical bpw, +10.60 at 1.0, +5.86 at 1.5, and +3.31 at 2.0**. The paired mean gain at 2 bpw is +3.01 points (1,000-request-cluster 95% CI +0.57 to +5.52). It improves the matched p10 by 9.35 points overall and 10.60 points on difficult layers at 2 bpw. However, it saturates at 84.62% median recovery and becomes worse than the current policy above about 2.5 bpw because the present nested atom code cannot spend later bytes effectively.

True co-selection packing helps: at 2 bpw it reduces median 4-KiB amplification from 3.64× (naïve layout) to 2.65×. Moving from 4-KiB pages to 512-byte sectors reduces amplification further to **1.40×** and reaches the same terminal 84.62% recovery at 1.70 actual bpw. Two/four H0 replica layouts and gate/up bundling add less than 0.1 recovery point. One layout is therefore the best use of storage.

The validation-selected deployable output-side down design is worse than the current Q2 baseline: at a 2-bpw cap it reaches 71.43% median overall and 68.20% on layers 4/20/39, versus 79.66% and 75.77% for the reproduced current policy. A non-deployable per-expert output-basis upper bound reaches 86.51% overall and 82.14% on difficult layers at the same cap. Down-only rank-512 per-expert Q4 coefficients recover about 98.8% at 1.094 physical down bpw, while no shared basis reaches 90% median across the four layers. The shared-to-per-expert gap is decisive.

The best deployable result below 2 correction bpw is therefore the corrected Run A policy on its smaller matched pilot, not Run B. It does **not** reach near-reference behavior: difficult-layer median is 81.36% at 2 bpw and only 3.7% of difficult invocations ever reach 90% within the tested corrected frontier. The broad streaming hypothesis remains partially supported as a bandwidth/allocator idea, but the current global atom representation fails the scaling criterion.

## 2. Authoritative model and data

The model is `Qwen3.6-35B-A3B`, HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. The production reference `W_ref` is the mixed-format GGUF `Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, SHA-256 `e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6`; it is not described as uniformly Q4.

| Fact | Discovered value | Deviation from expected |
|---|---:|---|
| Routed layers | 40 | none |
| Experts/layer | 256 | none |
| Top-k | 8 | none |
| Hidden / intermediate | 2,048 / 512 | none |
| Gate/up / down shapes | 512×2,048 / 2,048×512 | none |
| Weights/expert | 3,145,728 | none |

The four-layer extraction has 5,036 rows, 1,259 per layer, from six audited segments at stride 16. Splits are 53/12/16 complete train/validation/test requests and 3,292/760/984 rows. Capture NPZ SHA-256 is `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`. Run A uses 39 held-out invocations spanning 39 sampled experts and 10 request clusters; Run B uses 133 test invocations and 124 validation invocations. All bases, scales, clusters, and layouts use training data only. Run B method/rank/fit/precision choices are made on validation before test evaluation.

The resident Q2 base is the deterministic 2.250488-effective-bpw implementation from the prior study. The exact W1 results are retained as historical references and were not expensively rerun.

## 3. Baseline reproduction and audit

The deployable Q2 generalized-gate/generalized-up/native-down sequential baseline reproduced over all four layers with 14,364 rows in 375.44 seconds.

| Physical cap | p10 | Median | p90 | Median actual bpw | Page amplification |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 37.40% | 59.03% | 82.90% | 0.479 | 3.73× |
| 1.0 | 48.24% | 68.73% | 92.26% | 0.990 | 3.77× |
| 1.5 | 57.50% | 76.36% | 95.80% | 1.490 | 3.40× |
| 2.0 | 64.27% | 79.66% | 97.72% | 1.990 | 3.15× |
| 3.0 | 74.49% | 86.47% | 98.81% | 2.990 | 2.81× |
| 4.0 | 78.15% | 89.84% | 99.15% | 3.979 | 2.76× |

The suspected legacy physical-selector bug is real: a middle-ranked atom could be skipped for not fitting while a later atom fit, but reconstruction used the first returned `count`. The new allocator makes exact accepted IDs the source of truth. `test_exact_selected_ids_and_pages.py` constructs precisely this case and verifies charged pages, IDs, stored atoms, and arithmetic agree.

## 4. Run A: corrected per-atom and page-aware allocator

### 4.1 Held-out frontier

The primary Run A table is the smaller 39-invocation stratified pilot. Paired differences below use the exact same invocations from the reproduced baseline.

| Cap (bpw) | p10 | Median | p90 | Matched baseline median | Paired median gain | Paired mean gain (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 56.91% | 76.43% | 85.45% | 59.61% | +15.83 pt | +14.79 pt (+10.91, +17.69) |
| 1.0 | 65.81% | 80.24% | 90.34% | 69.12% | +10.60 pt | +10.85 pt (+7.75, +13.90) |
| 1.5 | 71.86% | 83.08% | 91.86% | 77.55% | +5.86 pt | +6.60 pt (+4.16, +9.73) |
| 2.0 | 74.64% | 84.51% | 91.90% | 82.15% | +3.31 pt | +3.01 pt (+0.57, +5.52) |
| 3.0 | 75.85% | 84.62% | 91.90% | 88.68% | −1.56 pt | −2.60 pt (−4.36, −0.62) |
| 4.0 | 75.85% | 84.62% | 91.90% | 90.49% | −5.15 pt | −5.52 pt (−7.71, −3.75) |

For layers 4/20/39, median recovery is 73.41/77.67/78.79/81.36/82.49/82.51% at 0.5/1/1.5/2/3/4 bpw. At 2 bpw, difficult-layer p10 is 73.40%, 10.60 points above the matched baseline p10. This satisfies the Run A “p10 improves by at least ten points” material-success criterion, but not the near-reference scaling criterion.

Inverse rates are heavily censored. Overall, 69.2% reach 80% recovery, 15.4% reach 90%, and none reach 95% or 99%. On difficult layers the fractions are 59.3%, 3.7%, 0%, and 0%. The unconditional median b80 is 0.75 bpw overall and 0.995 on difficult layers; b90/b95/b99 are beyond the corrected frontier.

### 4.2 Page and layout attribution

| Layout/page control at cap | Median recovery | Median actual bpw | Amplification |
|---|---:|---:|---:|
| Naïve, 4 KiB, 0.5 | 72.38% | 0.500 | 7.89× |
| Hypergraph, 4 KiB, 0.5 | 76.43% | 0.500 | 5.40× |
| 512 B, 0.5 | 81.31% | 0.499 | 1.77× |
| Naïve, 4 KiB, 1.0 | 78.33% | 0.990 | 6.70× |
| Hypergraph, 4 KiB, 1.0 | 80.24% | 1.000 | 4.32× |
| 512 B, 1.0 | 84.30% | 0.999 | 1.56× |
| Naïve, 4 KiB, 2.0 | 84.24% | 2.000 | 3.64× |
| Hypergraph, 4 KiB, 2.0 | 84.51% | 2.000 | 2.65× |
| 512 B, 2.0 | 84.62% | 1.699 | 1.40× |

Thus measured 4-KiB rounding costs at least 4.89 recovery points at a 0.5-bpw cap and 4.06 points at 1 bpw relative to the 512-byte control; the unmeasured ideal-byte loss can only be larger. At 2 bpw the quality gap has mostly closed, but 512-byte transfer reaches terminal quality with 15% fewer actual bytes.

Pairwise and average-importance layouts fall between naïve and hypergraph. Two/four partial replica H0 repricing changes median recovery by less than 0.1 point and reduces 2-bpw amplification by only about 1.5%. Gate/up bundling is similarly neutral once unwanted payload is counted. A single full layout costs at most 4.558× reference; four layouts cost up to 4.913×. The one-layout configuration is the best storage choice.

The replica rows are best-replica H0 repricing upper bounds for a fixed mathematical sequence, not a validated H4 replica-choice policy. They do not justify storing replicas.

### 4.3 Precision and projection allocation

Across the full chosen action sequences, final action precisions are 2/4/8/16 bits in 4,616/7,582/3,905/1,765 actions. **Down consumes 86.61% of charged physical increment bytes**, versus 5.58% gate and 7.81% up. Down is the dominant bandwidth consumer.

The independent-quantizer control exposes a serious nested-prefix loss:

| Projection | Bits | Nested relative atom MSE | Independent relative atom MSE |
|---|---:|---:|---:|
| Gate | 2 | 1.041 | 0.716 |
| Gate | 4 | 0.0632 | 0.0209 |
| Up | 2 | 0.915 | 0.708 |
| Up | 4 | 0.0568 | 0.0184 |
| Down | 2 | **2.781** | **0.895** |
| Down | 4 | **0.171** | **0.0561** |

At 8 bits both are near lossless, but independent MSE remains about four times lower. Multi-bit path evaluation fixes the allocator’s inability to cross a temporarily harmful 2-bit prefix; it does not fix the poor stored prefix itself. A complete-expert independent-quantizer control was not run, so atom-MSE differences are not mislabelled as exact recovery-point gains.

The 48-action full-candidate control is shallower than the 200–600 actions selected by the scalable policy and reaches only 50.6% median. It does not provide an equal-depth regret estimate. Beam/local-swap attribution remains unavailable.

## 5. Run B: output-side low-rank down correction

Run B spectrum files retain 512 singular values and normalize cumulative energy within that stored top-512 spectrum; they are useful for comparing concentration inside the tested ranks but are not a claim that rank 512 captures all residual energy.

### 5.1 Deployable validation-selected result

| Cap (bpw) | Overall p10 / median / p90 | Difficult p10 / median / p90 | Median actual bpw | Median extra FLOPs |
|---:|---:|---:|---:|---:|
| 0.5 | 17.54 / 50.58 / 92.07% | 16.55 / 47.40 / 78.49% | 0.448 | 0.343M |
| 1.0 | 25.06 / 60.88 / 95.51% | 22.38 / 57.49 / 80.02% | 0.927 | 2.483M |
| 1.5 | 32.34 / 67.32 / 96.49% | 32.20 / 63.33 / 83.71% | 1.365 | 2.483M |
| 2.0 | 41.17 / 71.43 / 97.36% | 45.71 / 68.20 / 85.66% | 1.865 | 2.483M |
| 3.0 | 48.96 / 74.60 / 97.62% | 53.37 / 72.96 / 87.81% | 2.656 | 2.483M |
| 4.0 | 51.68 / 76.26 / 97.77% | 56.00 / 73.87 / 88.17% | 2.938 | 2.483M |

At 2 bpw Run B loses a paired mean 10.40 points versus the current Q2 baseline (95% CI −17.89 to −7.07). Output-side low rank is therefore not a deployable improvement in its shared form.

Validation almost always selects rank 512 and Q4 coefficients once the budget permits; activation-weighted least squares rarely beats direct projection. This is evidence against a low-rank shared down residual, not merely against one coefficient precision.

### 5.2 Shared, clustered, and per-expert down

For authoritative-input down at rank 512/Q8 on difficult layers:

| Output basis | Median down recovery | p10 | Median down bpw | Status |
|---|---:|---:|---:|---|
| Raw shared | 47.77% | 20.47% | 2.094 | deployable |
| Action-PCA shared | **60.10%** | 17.07% | 2.094 | deployable |
| Proxy shared | 59.83% | 16.92% | 2.094 | deployable |
| 4 clusters | 50.26% | 7.55% | 0.844 median | deployable, undersampled |
| 8 clusters | 45.93% | 7.81% | 0.688 median | deployable, undersampled |
| Per expert | **99.996%** | 99.994% | 2.094 | non-deployable upper bound |

No shared or clustered method reaches 90% median across the four layers at ranks up to 512. Cluster-8 crosses 90% on layers 0 and 39 but fails badly on layers 4 and 20, so cluster conditioning does not recover most of the per-expert gain.

The per-expert rank-512 upper bound reaches about 98.8% median down recovery with Q4 coefficients at 1.094 down bpw and 99.996% with Q8 at 2.094. Thus per-expert b90 and b95 are 1.094 bpw; b99 is 2.094. It also improves complete-expert median at a 2-bpw cap to 86.51% overall / 82.14% difficult, but requires a separate resident output basis. At BF16, rank 512 costs about 2 MiB per expert, or 512 MiB/layer for 256 experts (roughly 20 GiB over 40 layers), so this is not the proposed deployment.

Because useful quality requires full input rank and a per-expert output basis, the trigger for a bounded shared two-sided `US_eV^T` extension was not met. The result points to local expert-conditioned representations rather than another global factorization.

## 6. Storage and compute accounting

Run A external storage is 4.119–4.913× reference depending on layer and the four-replica accounting; a single full view is at most 4.558×. Run B spans 2.908–3.679× and no row exceeds 5×. The extra capacity above one layout does not materially improve quality.

After hypothetical H4 support prediction, Run A at 2 bpw uses a median 34 unique generalized gate/up analysis rows, 1.204M extra FLOPs (19.14% of a 6.291M-FLOP reference expert GEMV), and 1.303M p90 FLOPs. Median estimated arithmetic intensity is 2.61 FLOP per resident-transform-plus-logical-correction byte. The full H0 2,048×2,048 transform costs 8.389M FLOPs and is oracle-selection overhead, not deployment compute.

Run B rank-512 factor application costs 0.524M FLOPs for `Ch` and 2.097M for `Uq`, total 2.621M or 41.67% of a complete reference expert GEMV. Shared/4-cluster/8-cluster BF16 output bases require about 2/8/16 MiB resident per layer. These are analytical GB10/Spark costs; measured RTX PRO 6000 timings are not presented as Spark results.

At Run A’s 2-bpw operating point the nominal bandwidth reduction versus 4-bpw full expert streaming is 2×. With 512-byte sectors, the same terminal quality uses 1.699 actual bpw, a 2.35× reduction. Since recovery is only 84.6% overall and 81.4% difficult, this is not yet a usable near-reference point.

## 7. Direct replay

Replay identity did not pass. The captures come from a complete BF16 Transformers graph, while `W_ref` and all corrections use the mixed-format production GGUF. The compact extraction also lacks a production-runtime arbitrary-MoE-output injection boundary and final logits. No hidden drift, H1–H4 router recall/Jaccard/flip, logit KL, or token-agreement number is inferred. The exact blocker and required production capture hook are in `REPLAY_IDENTITY_AUDIT_NEXT_20260818.md`.

## 8. Explicit attribution answers

1. **Atom quantization loss:** not identifiable in recovery points without a full independent-atom run; atom MSE is about 3× worse at 4 bits and catastrophic for 2-bit down (2.781 versus 0.895).
2. **Physical page loss:** 4 KiB versus 512 B costs 4.89 recovery points at 0.5 bpw, 4.06 at 1 bpw, and 0.12 at 2 bpw, plus large read amplification.
3. **Current prefix allocator loss:** corrected minus current paired median is +15.83/+10.60/+5.86/+3.31 points at 0.5/1/1.5/2 bpw; the corrected representation then saturates.
4. **True per-atom precision gain:** the combined corrected allocator gain above is measured; a per-atom-only orthogonal ablation is not available and is not fabricated.
5. **Co-selection packing:** at 2 bpw amplification falls 3.64×→2.65× versus naïve (27.3%); at 0.5 it falls 7.89×→5.40× (31.5%).
6. **Replicas:** one layout is best. Two/four copies recover <0.1 point and four copies raise maximum storage 4.558×→4.913×.
7. **Gate/up bundling:** neutral (<0.1 point) after unwanted payload; do not prioritize it.
8. **Dominant Run A limit:** at low rates allocation/layout mattered; above 2 bpw nested quantization and representation saturation dominate.
9. **Low-rank versus native down:** deployable shared/clustered low rank loses at equal complete-expert bytes; only the non-deployable per-expert basis wins.
10. **Rank for down 90/95/99:** no shared rank ≤512 crosses aggregate 90%; per-expert rank 512 needs Q4/Q4/Q8 and 1.094/1.094/2.094 down bpw.
11. **Loss from sharing:** difficult down median falls from 99.996% per-expert to 60.10% best shared at rank-512 Q8, about 39.9 points.
12. **Cluster conditioning:** no; 4/8 clusters are worse in aggregate and do not approach the per-expert upper bound.
13. **Best resident/stream point:** Q2 + corrected Run A is best at low rates; 84.51% median at 2 bpw on the bounded sample, with 19.1% extra reference-expert FLOPs.
14. **Near-reference below 2 bpw:** no. Neither deployable design reaches 90% difficult-layer median.
15. **Failure concentration:** layer 0 remains easier for the historical policy, but Run A’s high-rate saturation and Run B’s shared-basis gap also occur on layers 4/20/39; the failure is not one isolated layer.

## 9. Decision and next experiment

Run A is materially successful as an allocator/page study: it improves the difficult p10 by more than ten points at 2 bpw and 512-byte sectors reduce amplification below 1.5×. Run B is not materially successful in a deployable shared form. The combined design fails the broader scaling criterion because difficult-layer median is below 90% at 2 bpw, tails remain large, and replay is unvalidated.

**Do not scale either representation across all 40 layers.** The next experiment should be a Q2-residual, activation-conditioned complete-expert VQ/local-affine correction atlas:

1. Fit small expert-cluster or expert-local centroids/affine patches to the *complete sequential expert output*, not separate matrix columns.
2. Store contiguous Q4/Q8 packets and constrain the atlas to 1×–5× reference storage.
3. Use validation to choose cluster/expert fallback and target 512-byte/1-KiB transfers; retain full-Q4 fallback for tail invocations.
4. Replace the current two’s-complement prefix with a storage-counted independent or centroid/refinement code and rerun the 39-invocation paired control first.
5. Add the production-GGUF injection hook and pass replay identity before broader layer sampling.

More seeds of a global orthogonal basis, more physical replicas, or an H4 predictor should not be funded before this representation test.

## 10. Reproducible artifacts

- [Current vs corrected allocator](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/01_current_vs_corrected_allocator.png)
- [Layout waterfall](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/02_allocator_waterfall.png)
- [Page amplification](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/04_page_amplification_vs_page_size.png)
- [Required-rate CDF](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/10_required_rate_cdf.png)
- [Down rank curve](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/13_down_recovery_vs_rank.png)
- [Shared/cluster/per-expert comparison](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/15_shared_cluster_per_expert.png)
- [Combined frontier](results/qwen36_q2_next_oracles_analysis_20260818_v1/plots/20_run_a_run_b_frontier.png)
- [Consolidated metrics](results/qwen36_q2_next_oracles_analysis_20260818_v1/metrics.parquet)
- [Headline summary](results/qwen36_q2_next_oracles_analysis_20260818_v1/summary.csv)
- [Paired statistics](results/qwen36_q2_next_oracles_analysis_20260818_v1/paired_differences.csv)
- [Exact H0 atom/page labels](results/qwen36_q2_corrected_allocator_20260818_v7/selected_increment_sequence.parquet)
- [Run B full metrics](results/qwen36_q2_output_side_down_20260818_v3/low_rank_complete_expert_metrics.parquet)
- [Capture manifest](results/qwen36_q2_next_oracles_analysis_20260818_v1/capture_manifest.json)
- [Execution ledger](EXECUTION_LEDGER_NEXT_20260818.md)

Every figure has both PNG and SVG output and is regenerated by the one command in `README.md`.
