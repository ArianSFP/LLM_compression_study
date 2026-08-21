# Average-rate allocation over true top-8 expert groups

## Outcome

Continuation status: **continue_to_predicted_h4_average_rate**.

This is an exact-H4 interaction-geometry ceiling over 128 captured validation token/layer groups (1,024 routed expert invocations). No row is deployable or promotable: exact Q4 activation-dependent responses and, for the oracle control, exact cross-expert residual vectors are used. The actionable question is whether pooled bandwidth allocation is strong enough to justify a future predicted-H4 allocator.

## Accuracy by overall average bpw

| Overall average bpw | Policy | Burst cap | qenergy p10 | qenergy median | qenergy p90 | Paired median vs uniform | Actual bpw median |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | Exact combined-MoE oracle | 1536 | 96.779% | 99.017% | 99.882% | +0.718 pp | 0.522827 |
| 0.523478 | Pooled router-squared | 1536 | 96.744% | 99.011% | 99.881% | +0.693 pp | 0.522827 |
| 0.523478 | Uniform per expert | 384 | 95.175% | 97.823% | 99.698% | +0.000 pp | 0.523478 |
| 0.773478 | Exact combined-MoE oracle | 1536 | 98.833% | 99.666% | 99.987% | +0.286 pp | 0.772827 |
| 0.773478 | Pooled router-squared | 1536 | 98.830% | 99.658% | 99.987% | +0.273 pp | 0.772990 |
| 0.773478 | Uniform per expert | 576 | 98.123% | 99.215% | 99.943% | +0.000 pp | 0.773478 |
| 0.998739 | Exact combined-MoE oracle | 1536 | 99.588% | 99.885% | 99.999% | +0.109 pp | 0.998250 |
| 0.998739 | Pooled router-squared | 1536 | 99.576% | 99.881% | 99.999% | +0.104 pp | 0.998413 |
| 0.998739 | Uniform per expert | 749 | 99.337% | 99.710% | 99.991% | +0.000 pp | 0.998576 |
| 1.023478 | Exact combined-MoE oracle | 1536 | 99.637% | 99.898% | 100.000% | +0.096 pp | 1.022990 |
| 1.023478 | Pooled router-squared | 1536 | 99.623% | 99.895% | 100.000% | +0.092 pp | 1.023153 |
| 1.023478 | Uniform per expert | 768 | 99.387% | 99.736% | 99.993% | +0.000 pp | 1.023234 |

The table reports exact combined top-8 qenergy recovery, not token accuracy or model quality. Overall bpw includes the 9,232-byte rank-8 INT4+A/B/C resident sidecar and the allowed mean correction pages. Actual bpw can be lower when the frontier leaves budget unused.

## Frozen primary decision

- Mean correction pages/expert: 749
- Overall average bpw: 0.998739
- Router-squared pooled recovery p10/median: 99.576% / 99.881%
- Paired p10/median gain over uniform: +0.006 pp / +0.104 pp
- Median-gain gate: True
- Tail-nonregression gate: True

## Runtime and optimization

Each expert frontier uses one vectorized exact-self DP table shared across all hard budgets and page prices. Hard anchors receive bounded 1/2/3-unit local repair; the descending price path uses two incremental coordinate basins without repeatedly rebuilding the DP or repairing discarded price points. CSV evidence reports actual wall time, coordinate sweeps, local passes, DP state evaluations, and charged MACs.

## Q3 decision

Q3 status: **deferred_until_average_rate_baseline_is_frozen**. Q3 was deliberately not mixed into this attribution experiment. A later Q3 run must include:
- ideal_logical_256_byte_plane_ceiling
- physical_512_byte_training_only_coselection_layout
- physical_512_byte_fixed_gate_up_pairing_control

The physical study must charge unique 512-byte page IDs; the ideal 256-byte bitplane ceiling alone is not a deployable Q3 result.

## Scientific boundary

- Fit: train split only; all 256 experts in each locked layer.
- Selection and summaries: validation split only.
- Test scientific tensors/values: not admitted or used.
- No H4 causal replay, latency, routing, logit, token-quality, or end-to-end model claim is made.
