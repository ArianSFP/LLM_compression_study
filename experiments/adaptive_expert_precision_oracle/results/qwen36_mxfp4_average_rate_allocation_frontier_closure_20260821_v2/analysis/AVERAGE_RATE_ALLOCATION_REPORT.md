# Average-rate frontier closure over true top-8 expert groups

## Outcome

Continuation status: **continue_to_predicted_h4_average_rate**.

This remains an exact-H4 interaction-geometry ceiling over 128 validation token/layer groups (1,024 routed expert invocations). The extension isolates selected-column repair, quantizes the rank-4 exact-proxy factor, and brackets the finite-frontier global group objective. No row is deployable or promotable.

## Accuracy by overall average bpw

| Overall average bpw | Policy | Burst cap | qenergy p10 | qenergy median | qenergy p90 | Paired median vs uniform | Actual bpw median |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | Exact combined local oracle, refined | 1536 | 96.887% | 99.066% | 99.889% | +0.769 pp | 0.522990 |
| 0.523478 | Exact combined local oracle, coarse | 1536 | 96.779% | 99.017% | 99.882% | +0.718 pp | 0.522827 |
| 0.523478 | Column-generated router-squared | 1536 | 96.787% | 99.042% | 99.886% | +0.704 pp | 0.523153 |
| 0.523478 | Coarse pooled router-squared | 1536 | 96.744% | 99.011% | 99.881% | +0.693 pp | 0.522827 |
| 0.523478 | Uniform per expert | 384 | 95.175% | 97.823% | 99.698% | +0.000 pp | 0.523478 |
| 0.773478 | Exact combined local oracle, refined | 1536 | 98.895% | 99.683% | 99.988% | +0.304 pp | 0.773153 |
| 0.773478 | Exact combined local oracle, coarse | 1536 | 98.833% | 99.666% | 99.987% | +0.286 pp | 0.772827 |
| 0.773478 | Column-generated router-squared | 1536 | 98.886% | 99.674% | 99.988% | +0.292 pp | 0.773153 |
| 0.773478 | Coarse pooled router-squared | 1536 | 98.830% | 99.658% | 99.987% | +0.273 pp | 0.772990 |
| 0.773478 | Uniform per expert | 576 | 98.123% | 99.215% | 99.943% | +0.000 pp | 0.773478 |
| 0.998739 | Exact combined local oracle, refined | 1536 | 99.598% | 99.886% | 100.000% | +0.116 pp | 0.998413 |
| 0.998739 | Bounded 3/4-exchange upper solution | 1536 | 99.598% | 99.886% | 100.000% | +0.116 pp | 0.998413 |
| 0.998739 | Exact combined local oracle, coarse | 1536 | 99.588% | 99.885% | 99.999% | +0.109 pp | 0.998250 |
| 0.998739 | Column-generated router-squared | 1536 | 99.589% | 99.885% | 99.999% | +0.112 pp | 0.998576 |
| 0.998739 | Coarse pooled router-squared | 1536 | 99.576% | 99.881% | 99.999% | +0.104 pp | 0.998413 |
| 0.998739 | Uniform per expert | 749 | 99.337% | 99.710% | 99.991% | +0.000 pp | 0.998576 |
| 1.023478 | Exact combined local oracle, refined | 1536 | 99.646% | 99.899% | 100.000% | +0.105 pp | 1.023153 |
| 1.023478 | Exact combined local oracle, coarse | 1536 | 99.637% | 99.898% | 100.000% | +0.096 pp | 1.022990 |
| 1.023478 | Column-generated router-squared | 1536 | 99.628% | 99.897% | 100.000% | +0.100 pp | 1.023315 |
| 1.023478 | Coarse pooled router-squared | 1536 | 99.623% | 99.895% | 100.000% | +0.092 pp | 1.023153 |
| 1.023478 | Uniform per expert | 768 | 99.387% | 99.736% | 99.993% | +0.000 pp | 1.023234 |

Recovery is exact combined top-8 qenergy recovery, not token accuracy or model quality. Overall bpw includes factor+A/B/C metadata and the allowed average correction pages.

## Selected-column frontier repair

| Mean pages | p10 gain vs coarse | Median gain vs coarse | p90 gain vs coarse | Page-vector change fraction |
| ---: | ---: | ---: | ---: | ---: |
| 384 | +0.001 pp | +0.023 pp | +0.118 pp | 83.6% |
| 576 | +0.000 pp | +0.011 pp | +0.047 pp | 78.9% |
| 749 | +0.000 pp | +0.003 pp | +0.016 pp | 71.1% |
| 768 | +0.000 pp | +0.002 pp | +0.015 pp | 77.3% |

Only allocator-selected page points receive full hard-budget local repair. Two prices around each selected marginal slope are inserted for the affected frontier section, and MCKP is rerun until the selected state vectors stabilize or the frozen three-round cap is hit.

## Rank-4 factor encodings at strict all-in rate

| Factor | Overall bpw | p10 | Median | Charged group MACs |
| :--- | ---: | ---: | ---: | ---: |
| exact_proxy_rank4_fp32 | 0.998728 | 99.532% | 99.868% | 39,845,888 |
| exact_proxy_rank4_fp16 | 0.998739 | 99.566% | 99.882% | 39,694,080 |
| exact_proxy_rank4_int4_per_row_hadamard | 0.998739 | 99.590% | 99.887% | 39,918,848 |
| exact_proxy_rank4_int8_per_row | 0.998739 | 99.573% | 99.884% | 41,192,704 |
| exact_proxy_rank4_int8_plus_euclidean_tail4_int4 | 0.998739 | 99.561% | 99.880% | 60,638,208 |

The controls include FP32, self-safe FP16, per-row INT8, Hadamard-rotated per-row INT4, and a mixed exact-proxy INT8 plus Euclidean-tail INT4 factor. Each uses the maximum integral correction pages that keep metadata+corrections within one total bpw.

## Global finite-frontier bound

- Certified-to-tolerance group fraction: 0.0%
- Relative optimality gap median/p90/max: 34.74829% / 45.48945% / 63.13659%
- Median bounded-exchange candidate evaluations: 126

The upper solution adds deterministic three/four-expert exchanges. The lower value is globally valid over every combination of the available refined frontier columns: tangent convexity plus an exact linear MCKP solve produces each dual bound. It is not a bound over states absent from those per-expert frontiers.

## Frozen primary decision

- Mean correction pages/expert: 749
- Overall average bpw: 0.998739
- Column-generated recovery p10/median: 99.589% / 99.885%
- Paired p10/median gain over uniform: +0.007 pp / +0.112 pp
- Additional selected-frontier p10/median gain: +0.000 pp / +0.003 pp
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
