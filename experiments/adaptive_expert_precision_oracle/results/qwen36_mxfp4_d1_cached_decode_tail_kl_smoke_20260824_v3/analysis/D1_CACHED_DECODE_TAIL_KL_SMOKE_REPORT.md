# Cached-decode D1 downstream-tail KL smoke report

## Scientific boundary

This is an exact-prefix, one-current-token, one-injected-layer smoke study.
Every candidate receives a private clone of the complete native prefix cache.
Only the current token's terminal logits are scored, and its post-token
attention K/V, DeltaNet convolution, and recurrent states are retained and
audited. No sequence-shaped perturbation or prompt-position coupling is used.

The three retained requests provide 6960 logical terminal observations.
They diagnose mechanism and implementation only. At least
64 independent requests are
required before a terminal-quality claim.

## Live terminal result

| rate_pages_per_expert | policy | mean_logit_kl | mean_delta_nll | crossing_rate | mean_local_qenergy |
| --- | --- | --- | --- | --- | --- |
| 360 | calibration_selected_fixed_d1 | 0.0018697966 | -0.013058711 | 0.95977011 | 0.002134361 |
| 360 | exact_combined_local | 0.0019755212 | -0.015658909 | 0.94827586 | 0.0020135565 |
| 360 | exact_token_oracle | 0.0019390585 | -0.016098751 | 0.95402299 | 0.0020149089 |
| 360 | frontier_companion_d1 | 0.0020827693 | -0.0029246256 | 0.95402299 | 0.002072216 |
| 360 | pr13_router_square | 0.002303478 | -0.0087915234 | 0.96551724 | 0.0020464501 |
| 384 | calibration_selected_fixed_d1 | 0.0020439694 | -0.013790868 | 0.96551724 | 0.0018974236 |
| 384 | exact_combined_local | 0.0021600752 | -0.010171209 | 0.97701149 | 0.0017764146 |
| 384 | exact_token_oracle | 0.0020655671 | -0.0099656313 | 0.97126437 | 0.0017764344 |
| 384 | frontier_companion_d1 | 0.0021450368 | -0.01373859 | 0.92528736 | 0.0018303653 |
| 384 | pr13_router_square | 0.0019276649 | -0.0080807017 | 0.97126437 | 0.0018018553 |
| 725 | calibration_selected_fixed_d1 | 0.0015781157 | -0.0084022325 | 0.94252874 | 0.00029695932 |
| 725 | exact_combined_local | 0.0016397207 | -0.0077398132 | 0.92528736 | 0.00028424151 |
| 725 | exact_token_oracle | 0.0016295089 | -0.0078562762 | 0.92528736 | 0.00028475776 |
| 725 | frontier_companion_d1 | 0.0020278554 | -8.8613734e-05 | 0.93103448 | 0.00035125437 |
| 725 | pr13_router_square | 0.0021388147 | -0.0084874014 | 0.93678161 | 0.00028906549 |
| 749 | calibration_selected_fixed_d1 | 0.0019796412 | -0.010740551 | 0.93103448 | 0.00025302378 |
| 749 | exact_combined_local | 0.0019571012 | -0.0085217997 | 0.94827586 | 0.00024669477 |
| 749 | exact_token_oracle | 0.0019637533 | -0.0085863798 | 0.95402299 | 0.00024692178 |
| 749 | frontier_companion_d1 | 0.0019954574 | -0.01557661 | 0.93678161 | 0.00028100236 |
| 749 | pr13_router_square | 0.001494269 | -0.008900039 | 0.93103448 | 0.00025178449 |

## Held-out paired KL relative to PR #13

Negative values improve on PR #13 at identical layer, rate, token, and routing
mode.

| rate_pages_per_expert | layer_cohort | policy | route_mode | tokens | mean_logit_kl_minus_pr13 | improved_token_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 360 | held_out | calibration_selected_fixed_d1 | live | 87 | -0.00032883662 | 0.51724138 |
| 360 | held_out | exact_combined_local | live | 87 | -0.00020378668 | 0.44827586 |
| 360 | held_out | exact_token_oracle | live | 87 | -0.00028922135 | 0.42528736 |
| 360 | held_out | frontier_companion_d1 | live | 87 | -0.00036628759 | 0.49425287 |
| 360 | held_out | pr13_router_square | live | 87 | 0 | 0 |
| 384 | held_out | calibration_selected_fixed_d1 | live | 87 | 0.00017458613 | 0.49425287 |
| 384 | held_out | exact_combined_local | live | 87 | -0.00054215488 | 0.44827586 |
| 384 | held_out | exact_token_oracle | live | 87 | -0.00040110218 | 0.43678161 |
| 384 | held_out | frontier_companion_d1 | live | 87 | -1.9419779e-05 | 0.55172414 |
| 384 | held_out | pr13_router_square | live | 87 | 0 | 0 |
| 725 | held_out | calibration_selected_fixed_d1 | live | 87 | -0.0013032249 | 0.51724138 |
| 725 | held_out | exact_combined_local | live | 87 | -0.0011319342 | 0.4137931 |
| 725 | held_out | exact_token_oracle | live | 87 | -0.0011269037 | 0.42528736 |
| 725 | held_out | frontier_companion_d1 | live | 87 | -0.00062413584 | 0.4137931 |
| 725 | held_out | pr13_router_square | live | 87 | 0 | 0 |
| 749 | held_out | calibration_selected_fixed_d1 | live | 87 | 0.0010238098 | 0.36781609 |
| 749 | held_out | exact_combined_local | live | 87 | 0.00086330943 | 0.34482759 |
| 749 | held_out | exact_token_oracle | live | 87 | 0.00087462756 | 0.34482759 |
| 749 | held_out | frontier_companion_d1 | live | 87 | 0.00065473948 | 0.4137931 |
| 749 | held_out | pr13_router_square | live | 87 | 0 | 0 |

## Immediate D1 and amplification result

All deltas below are paired to PR #13 at the same rate, layer, and token.
A deployable D1 policy was required to lower live KL, live-minus-frozen KL,
and immediate crossing rate at the same rate.

| rate_pages_per_expert | policy | mean_logit_kl_minus_pr13 | live_minus_frozen_kl_minus_pr13 | d1_crossing_rate | d1_crossing_rate_minus_pr13 | mean_local_qenergy |
| --- | --- | --- | --- | --- | --- | --- |
| 360 | calibration_selected_fixed_d1 | -0.00043368143 | -0.0003696597 | 0.11494253 | -0.0057471264 | 0.002134361 |
| 360 | exact_combined_local | -0.0003279568 | -0.00029809434 | 0.074712644 | -0.045977011 | 0.0020135565 |
| 360 | exact_token_oracle | -0.00036441959 | -0.00032811926 | 0.08045977 | -0.040229885 | 0.0020149089 |
| 360 | frontier_companion_d1 | -0.00022070875 | -0.00018848966 | 0.11494253 | -0.0057471264 | 0.002072216 |
| 360 | pr13_router_square | 0 | 0 | 0.12068966 | 0 | 0.0020464501 |
| 384 | calibration_selected_fixed_d1 | 0.00011630456 | 0.00012914501 | 0.12068966 | 0.028735632 | 0.0018974236 |
| 384 | exact_combined_local | 0.00023241033 | 0.00019362787 | 0.13793103 | 0.045977011 | 0.0017764146 |
| 384 | exact_token_oracle | 0.00013790225 | 9.8843735e-05 | 0.13218391 | 0.040229885 | 0.0017764344 |
| 384 | frontier_companion_d1 | 0.0002173719 | 0.00018114199 | 0.1091954 | 0.017241379 | 0.0018303653 |
| 384 | pr13_router_square | 0 | 0 | 0.091954023 | 0 | 0.0018018553 |
| 725 | calibration_selected_fixed_d1 | -0.00056069897 | -0.00063291379 | 0.051724138 | 0.0057471264 | 0.00029695932 |
| 725 | exact_combined_local | -0.00049909404 | -0.00041487608 | 0.057471264 | 0.011494253 | 0.00028424151 |
| 725 | exact_token_oracle | -0.00050930581 | -0.00043211833 | 0.057471264 | 0.011494253 | 0.00028475776 |
| 725 | frontier_companion_d1 | -0.00011095933 | -8.5597025e-05 | 0.074712644 | 0.028735632 | 0.00035125437 |
| 725 | pr13_router_square | 0 | 0 | 0.045977011 | 0 | 0.00028906549 |
| 749 | calibration_selected_fixed_d1 | 0.00048537222 | 0.00045508836 | 0.057471264 | 0.0057471264 | 0.00025302378 |
| 749 | exact_combined_local | 0.00046283221 | 0.00047824424 | 0.045977011 | -0.0057471264 | 0.00024669477 |
| 749 | exact_token_oracle | 0.0004694843 | 0.00048721445 | 0.045977011 | -0.0057471264 | 0.00024692178 |
| 749 | frontier_companion_d1 | 0.00050118843 | 0.00052784795 | 0.068965517 | 0.017241379 | 0.00028100236 |
| 749 | pr13_router_square | 0 | 0 | 0.051724138 | 0 | 0.00025178449 |

## Conditional crossing evidence

For the frozen fixed-D1 policy, preventing a crossing is directionally
favorable while introducing one is harmful. This supports the amplifier
mechanism, but it does not rescue a policy that is inconsistent by rate.

| rate_pages_per_expert | policy | crossing_event | tokens | mean_logit_kl_minus_pr13 | improved_token_fraction |
| --- | --- | --- | --- | --- | --- |
| 360 | calibration_selected_fixed_d1 | introduced | 6 | 0.001373598 | 0.16666667 |
| 360 | calibration_selected_fixed_d1 | prevented | 7 | -0.0037597516 | 0.71428571 |
| 360 | calibration_selected_fixed_d1 | same | 161 | -0.00035642171 | 0.50310559 |
| 384 | calibration_selected_fixed_d1 | introduced | 12 | 0.0015325901 | 0.41666667 |
| 384 | calibration_selected_fixed_d1 | prevented | 7 | -0.00038787437 | 0.57142857 |
| 384 | calibration_selected_fixed_d1 | same | 155 | 2.9426012e-05 | 0.49677419 |
| 725 | calibration_selected_fixed_d1 | introduced | 6 | 2.2166321e-05 | 0.5 |
| 725 | calibration_selected_fixed_d1 | prevented | 5 | -0.0011545511 | 0.8 |
| 725 | calibration_selected_fixed_d1 | same | 163 | -0.00056393781 | 0.49079755 |
| 749 | calibration_selected_fixed_d1 | introduced | 4 | 0.00016687929 | 0 |
| 749 | calibration_selected_fixed_d1 | prevented | 3 | 0.0011114677 | 0.33333333 |
| 749 | calibration_selected_fixed_d1 | same | 167 | 0.00048175358 | 0.4491018 |

## Exact full-model label diagnostic over executed candidates

This post-hoc diagnostic uses exact full-model D1 crossing, then router-mass
churn, local qenergy, pages, and policy name to choose among the five already
executed allocations. It is not a runtime policy and never selects on terminal
KL.

| rate_pages_per_expert | tokens | mean_logit_kl | pr13_mean_logit_kl | mean_logit_kl_minus_pr13 | improved_token_fraction | d1_crossing_rate | mean_d1_router_mass_churn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 360 | 174 | 0.0019172278 | 0.002303478 | -0.00038625024 | 0.40229885 | 0.045977011 | 0.0061566712 |
| 384 | 174 | 0.0021233059 | 0.0019276649 | 0.00019564102 | 0.36781609 | 0.040229885 | 0.0058545042 |
| 725 | 174 | 0.0015176762 | 0.0021388147 | -0.00062113853 | 0.33908046 | 0.017241379 | 0.0026499595 |
| 749 | 174 | 0.0016793486 | 0.001494269 | 0.00018507957 | 0.37931034 | 0.022988506 | 0.0030595533 |

## Experiment B promotion decision

Status: **not_promoted_pause_before_experiment_b**

Only these page rates pass all three fixed-policy terms:
[360].

| rate_pages_per_expert | mean_logit_kl_minus_pr13 | live_minus_frozen_kl_minus_pr13 | d1_crossing_rate_minus_pr13 | passes_same_rate_gate |
| --- | --- | --- | --- | --- |
| 360 | -0.00043368143 | -0.0003696597 | -0.0057471264 | True |
| 384 | 0.00011630456 | 0.00012914501 | 0.028735632 | False |
| 725 | -0.00056069897 | -0.00063291379 | 0.0057471264 | False |
| 749 | 0.00048537222 | 0.00045508836 | 0.0057471264 | False |

Experiment B did not start. The directional crossing evidence is promising,
but fixed D1 and the source-slice token oracle are non-monotonic across rates,
and exact-combined local is at least as competitive. A same-host full-model D1
rerank or repair must separate route-direction benefit from combined-local
benefit before promotion.

## Validation gates

- Zero-dose logical rows: 1392
- Zero-dose rows exact in hidden states, routers, terminal logits, and full cache:
  1392/1392
- Current query length: one token
- Candidate cache reuse: forbidden
- D1 labels used at runtime: no; the exact token oracle is a diagnostic policy
- Joint all-layer compression and generated rollout: out of scope
