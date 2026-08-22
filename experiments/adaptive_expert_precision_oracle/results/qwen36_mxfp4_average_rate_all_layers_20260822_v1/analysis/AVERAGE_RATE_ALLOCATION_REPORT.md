# Average-rate frontier closure over true top-8 expert groups

## Outcome

Continuation status: **continue_to_predicted_h4_average_rate**.

This remains an exact-H4 interaction-geometry ceiling over 1,280 validation token/layer groups (10,240 routed expert invocations). The extension isolates selected-column repair, quantizes the rank-4 exact-proxy factor, and brackets the finite-frontier global group objective. No row is deployable or promotable.

## Accuracy by overall average bpw

| Overall average bpw | Policy | Burst cap | qenergy p10 | qenergy median | qenergy p90 | Paired median vs uniform | Actual bpw median |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | Exact combined local oracle, refined | 1536 | 96.631% | 98.414% | 99.525% | +1.313 pp | 0.522990 |
| 0.523478 | Exact combined local oracle, coarse | 1536 | 96.520% | 98.358% | 99.507% | +1.246 pp | 0.522664 |
| 0.523478 | Column-generated router-squared | 1536 | 96.572% | 98.384% | 99.517% | +1.267 pp | 0.523153 |
| 0.523478 | Coarse pooled router-squared | 1536 | 96.460% | 98.331% | 99.500% | +1.223 pp | 0.522827 |
| 0.523478 | Uniform per expert | 384 | 94.158% | 96.781% | 98.803% | +0.000 pp | 0.523478 |
| 0.773478 | Exact combined local oracle, refined | 1536 | 98.769% | 99.443% | 99.841% | +0.513 pp | 0.772990 |
| 0.773478 | Exact combined local oracle, coarse | 1536 | 98.731% | 99.424% | 99.834% | +0.492 pp | 0.772664 |
| 0.773478 | Column-generated router-squared | 1536 | 98.746% | 99.429% | 99.838% | +0.496 pp | 0.773153 |
| 0.773478 | Coarse pooled router-squared | 1536 | 98.703% | 99.414% | 99.833% | +0.475 pp | 0.772990 |
| 0.773478 | Uniform per expert | 576 | 97.798% | 98.826% | 99.580% | +0.000 pp | 0.773478 |
| 0.998739 | Exact combined local oracle, refined | 1536 | 99.550% | 99.795% | 99.944% | +0.193 pp | 0.998413 |
| 0.998739 | Bounded 3/4-exchange upper solution | 1536 | 99.550% | 99.795% | 99.944% | +0.193 pp | 0.998413 |
| 0.998739 | Exact combined local oracle, coarse | 1536 | 99.530% | 99.789% | 99.941% | +0.183 pp | 0.998088 |
| 0.998739 | Column-generated router-squared | 1536 | 99.543% | 99.792% | 99.942% | +0.187 pp | 0.998576 |
| 0.998739 | Coarse pooled router-squared | 1536 | 99.522% | 99.785% | 99.940% | +0.179 pp | 0.998250 |
| 0.998739 | Uniform per expert | 749 | 99.180% | 99.556% | 99.842% | +0.000 pp | 0.998739 |
| 1.023478 | Exact combined local oracle, refined | 1536 | 99.600% | 99.819% | 99.950% | +0.171 pp | 1.023153 |
| 1.023478 | Exact combined local oracle, coarse | 1536 | 99.582% | 99.812% | 99.949% | +0.163 pp | 1.022827 |
| 1.023478 | Column-generated router-squared | 1536 | 99.588% | 99.814% | 99.950% | +0.167 pp | 1.023315 |
| 1.023478 | Coarse pooled router-squared | 1536 | 99.575% | 99.810% | 99.946% | +0.158 pp | 1.022990 |
| 1.023478 | Uniform per expert | 768 | 99.271% | 99.601% | 99.858% | +0.000 pp | 1.023478 |

Recovery is exact combined top-8 qenergy recovery, not token accuracy or model quality. Overall bpw includes factor+A/B/C metadata and the allowed average correction pages.

## Selected-column frontier repair

| Mean pages | p10 gain vs coarse | Median gain vs coarse | p90 gain vs coarse | Page-vector change fraction |
| ---: | ---: | ---: | ---: | ---: |
| 384 | +0.002 pp | +0.039 pp | +0.128 pp | 78.4% |
| 576 | +0.001 pp | +0.015 pp | +0.049 pp | 72.0% |
| 749 | +0.000 pp | +0.006 pp | +0.019 pp | 73.1% |
| 768 | +0.000 pp | +0.005 pp | +0.017 pp | 72.2% |

Only allocator-selected page points receive full hard-budget local repair. Two prices around each selected marginal slope are inserted for the affected frontier section, and MCKP is rerun until the selected state vectors stabilize or the frozen three-round cap is hit.

## Rank-4 factor encodings at strict all-in rate

| Factor | Overall bpw | p10 | Median | Charged group MACs |
| :--- | ---: | ---: | ---: | ---: |
| exact_proxy_rank4_fp32 | 0.998728 | 99.484% | 99.770% | 39,845,888 |
| exact_proxy_rank4_fp16 | 0.998739 | 99.530% | 99.793% | 39,694,080 |
| exact_proxy_rank4_int4_per_row_hadamard | 0.998739 | 99.543% | 99.795% | 39,918,848 |
| exact_proxy_rank4_int8_per_row | 0.998739 | 99.542% | 99.798% | 41,192,704 |
| exact_proxy_rank4_int8_plus_euclidean_tail4_int4 | 0.998739 | 99.525% | 99.787% | 60,638,208 |

The controls include FP32, self-safe FP16, per-row INT8, Hadamard-rotated per-row INT4, and a mixed exact-proxy INT8 plus Euclidean-tail INT4 factor. Each uses the maximum integral correction pages that keep metadata+corrections within one total bpw.

## Global finite-frontier bound

- Certified-to-tolerance group fraction: 0.0%
- Relative optimality gap median/p90/max: 33.95021% / 36.67513% / 63.13659%
- Median bounded-exchange candidate evaluations: 126

The upper solution adds deterministic three/four-expert exchanges. The lower value is globally valid over every combination of the available refined frontier columns: tangent convexity plus an exact linear MCKP solve produces each dual bound. It is not a bound over states absent from those per-expert frontiers.

## Frozen primary decision

- Mean correction pages/expert: 749
- Overall average bpw: 0.998739
- Column-generated recovery p10/median: 99.543% / 99.792%
- Paired p10/median gain over uniform: +0.055 pp / +0.187 pp
- Additional selected-frontier p10/median gain: +0.000 pp / +0.006 pp
- Median-gain gate: True
- Tail-nonregression gate: True

## Q3 decision

Q3 status: **deferred_until_average_rate_baseline_is_frozen**. Q3 remains separated from this attribution experiment. A later Q3 run must include:
- ideal_logical_256_byte_plane_ceiling
- physical_512_byte_training_only_coselection_layout
- physical_512_byte_fixed_gate_up_pairing_control

The physical study must charge unique 512-byte page IDs; the ideal 256-byte bitplane ceiling alone is not a deployable Q3 result.

## Scientific boundary

- Fit: train split only; all 256 experts in each locked layer.
- Selection and summaries: validation split only.
- Test scientific tensors/values: not admitted or used.
- No H4 causal replay, latency, routing, logit, token-quality, or end-to-end model claim is made.
