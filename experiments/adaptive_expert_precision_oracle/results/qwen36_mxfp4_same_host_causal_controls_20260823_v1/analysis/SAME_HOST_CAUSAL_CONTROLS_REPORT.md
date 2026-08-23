# Same-host PR #13 causal controls

## Outcome

The same-host control passes. True routed-MoE replacement before shared-expert
composition and the decoder residual add **does not remove the downstream
floor** seen in the first pilot. Downstream discrete expert selection is the
largest identified amplifier: for layers 0--23, live routing produces roughly
6--13 times the final hidden MSE of fully frozen routing at alpha=1.

This is an isolated seven-layer sentinel study, not an end-to-end streamed
quality curve.

## Hard gates

- Same-process repeated hidden/routed/router/logit captures: exactly zero
  difference for all three requests.
- Zero-dose hidden/router/logit replay: exactly zero in live, frozen-set and
  fully frozen modes.
- Historical layer-0 allocation rows: 1,024;
  selected expert IDs, pages and 512-entry state vectors are exact.
- Historical reconstructed deltas: 128;
  bit-identical with max absolute difference zero.
- Same-host max group qenergy reconstruction error across layers:
  1.388e-17.
- Same-host max all-Q4 float64 reconstruction roundoff:
  2.665e-15.
- Rows: 46,800 propagation, 1,680
  quality and 63 zero-dose.
- Test rows admitted or used: false.

## Alpha=1 rate comparison

| pages | bpw | energy | live_kl | frozen_kl | ratio | top1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 0.523478 | 7.193e-03 | 2.251e-03 | 6.304e-04 | 3.57 | 99.30% |
| 576 | 0.773478 | 2.683e-03 | 1.766e-03 | 5.621e-04 | 3.14 | 100.00% |
| 749 | 0.998739 | 1.091e-03 | 1.884e-03 | 5.846e-04 | 3.22 | 99.65% |
| 768 | 1.023478 | 9.886e-04 | 1.980e-03 | 5.830e-04 | 3.40 | 99.30% |

Moving from 384 to 749 pages reduces mean realized layer-output energy by
6.59x, but live
logit KL by only
1.20x. With
fully frozen downstream expert IDs, the remaining KL is much smaller, while
frozen-set/live-weight results closely track fully frozen results. This
attributes most of the additional live disturbance to discrete expert-set
changes rather than router-weight drift.

## Depth and rate

| layer | energy_reduction_x | live_kl_384 | live_kl_749 | live_kl_lower_pct | live_over_frozen_hidden_384 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 37.80 | 2.480e-03 | 2.013e-03 | +18.8 | 11.35 |
| 1 | 7.38 | 2.721e-03 | 2.694e-03 | +1.0 | 11.53 |
| 4 | 7.23 | 3.434e-03 | 3.181e-03 | +7.4 | 9.47 |
| 6 | 7.25 | 2.504e-03 | 2.081e-03 | +16.9 | 9.47 |
| 12 | 6.90 | 2.543e-03 | 1.477e-03 | +41.9 | 10.66 |
| 23 | 6.52 | 1.507e-03 | 1.606e-03 | -6.6 | 6.14 |
| 39 | 6.53 | 5.666e-04 | 1.338e-04 | +76.4 | 1.00 |

Layer 39 behaves conventionally: 384-to-749 pages gives
6.53x lower realized energy and
4.23x
lower logit KL. Earlier layers do not. At alpha=1, every request injected
before layer 39 crosses a downstream top-8 membership boundary; the median
first crossing is one layer downstream and the latest is three layers.

## Injection implementation control

| injection_mode | mean_budget_pages_per_expert | logit_kl |
| ---: | ---: | ---: |
| post_xplus_bf16_delta | 384 | 2.034e-03 |
| post_xplus_bf16_delta | 749 | 1.846e-03 |
| post_xplus_fp32_once | 384 | 2.276e-03 |
| post_xplus_fp32_once | 749 | 2.085e-03 |
| pre_residual_routed_replacement | 384 | 2.251e-03 |
| pre_residual_routed_replacement | 749 | 1.884e-03 |

The two post-xplus controls remain within about 15% of true pre-residual
replacement after aggregation. The old injection location was therefore not
the primary cause of the floor, although pre-residual replacement is now the
scientific path.

## Dose response

At 384 pages, increasing alpha from 0.125 to
4 changes live KL from
1.674e-03 to 3.569e-03. At 749
pages the corresponding values are 1.736e-03 and
2.104e-03. The shallow-layer response is discontinuous
and non-monotone, while layer 39 is smooth (energy-to-KL Spearman approximately
0.94 across rates and doses).

## Router-weight representation

The installed backend normalizes top-8 probabilities in FP32 and casts them to
BF16 before expert execution. This study keeps both quantities:

- normalized selector weights for PR #13 allocation and qenergy accounting;
- raw BF16 execution weights for the reconstructed routed-output delta.

The maximum observed BF16 execution-weight sum error was
0.002441;
the maximum per-weight selector adjustment was
0.001275.

## Scientific boundary

- Cohort: three validation sequences, 44 input tokens, 41 labels and 32
  injected positions per layer.
- Sentinels: layers 0, 1, 4, 6, 12, 23, 39.
- Rates: all four PR #13 operating points; alpha in
  0.125, 0.25, 0.5, 1.0, 2.0, 4.0.
- Modes: live routes, exact expert set with live weights, and fully frozen
  routes; plus the two alpha=1 post-xplus injection controls.
- NLL is descriptive only on three sequences. Non-negative forward logit KL is
  the primary quality signal.
- Not measured: simultaneous all-layer streaming, held-out population
  perplexity, live re-allocation at every perturbed layer, repair-one-layer
  importance, H1 predictor error, or physical wire bpw.

## Decision

The warning signal survives both methodological controls. The next large run
should retain true pre-residual replacement and report routing controls, then
move to genuine sequential all-layer streaming. A routing-boundary-risk term
and between-layer bit allocator remain justified; local qenergy alone is not a
sufficient downstream objective.
