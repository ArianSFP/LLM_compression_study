# D1 Experiment A allocator diagnosis

## Status

This is the canonical result report for the PR #13 D1 allocator diagnosis. The
study remains **Experiment A**. It explains the existing three-request cached-
decode smoke and a deliberately enriched 24-token/layer diagnostic panel. It
does not start or report Experiment B.

The narrow result is:

> Discrete downstream expert membership remains the dominant identified
> amplifier, but globally minimizing the binary next-router top-8 event is not
> a sufficient terminal-KL objective. A tested hard-saturating repair of a
> PR #13 incumbent is a lower-variance alternative to global replacement.
> Route-safe nested local completion is the next hypothesis, not a result.

The evidence directly supports only the incumbent-hold and hard-saturation
parts of the target architecture. The repair ablation uses exact full-model D1
outcomes over five candidates that were already executed. It does not generate
a nested page path, impose an additive local guardrail, or implement a runtime
predictor. Its oracle labels cannot be runtime inputs; a runtime implementation
still needs token-dependent signed page effects and calibrated uncertainty.

## Scientific boundary

D1 means only the same decode token's next-layer router:

\[
(t,\ell)\rightarrow(t,\ell+1).
\]

There is no D2--D4 objective, predictor, VJP, loss, or allocator lookahead in
this study. Downstream layers are evaluated only as descriptive causal
outcomes after one D1-scoped allocation is injected; they never become a
multi-layer target or policy input. Terminal KL is likewise an outcome only.
It is never an allocator or remedy-selector input.

The decode contract assumes exact prefill. Each current token is processed
behind its own deep-cloned exact prefix cache. Only that token's routed MoE
output is replaced at one layer; the token then continues through the native
cached downstream tail. Candidate caches never advance the next token.

The evidence consists of:

- the complete three-request smoke: six injection layers, four rates, 29
  isolated cached-decode positions and five policies;
- a 24-token/layer case-control panel with 12 quiet controls and four cases
  each of adjacent-rate reversal, a prevented crossing and an introduced
  crossing; and
- exact live, frozen-set/live-weight and fully-frozen downstream execution on
  an NVIDIA RTX PRO 6000.

The panel was selected partly from finalized KL outcomes to enrich explanatory
cases. It is not a prevalence estimate, an unbiased validation cohort, or a
model-quality result. Only three teacher-forced requests are represented, not
free-running generated decode.

The full methodology is frozen in
[D1_EXPERIMENT_A_ALLOCATOR_DIAGNOSIS_METHODS_20260825.md](../../../D1_EXPERIMENT_A_ALLOCATOR_DIAGNOSIS_METHODS_20260825.md).
The immutable [panel facts](../d1_experiment_a_diagnostic_panel_facts.json),
[run facts](../d1_experiment_a_diagnostic_run_facts.json), and
[analysis facts](d1_experiment_a_diagnostic_analysis_facts.json) are the
numeric authorities.

## Executive findings

1. The quoted `22/28` result is not a contradiction. When fixed D1 prevented a
   PR #13 crossing, KL moved in the favorable direction on average; when it
   introduced one, KL moved in the unfavorable direction. That is the expected
   signature of a causal amplifier.
2. It is nevertheless incomplete as an allocation objective. In 646 of 696
   fixed-versus-PR13 comparisons, the immediate D1 crossing state was the same.
   Continuous perturbation direction, routing mass, later thresholds and a few
   high-sensitivity tokens can dominate the mean within this majority.
3. The route-mode intervention independently confirms the mechanism. Across
   the 480 realizable policy rows in the enriched panel, mean live KL was
   `0.0038351`; the downstream membership-execution contrast was `0.0032122`,
   the router-weight contrast was `-0.0000159`, and the fully-frozen component
   was `0.0006387`.
4. Adjacent rates are not successive refinements. On 98.85% of fixed-D1 token
   allocations, the higher cap removed pages selected at the lower cap, then
   added hundreds elsewhere. The stored/source policy-bank perturbation
   rotated while its norm fell; exact live controls separately confirm that
   direction can dominate the high-rate reversal.
5. Non-monotonicity survives a continuous vector interpolation. Every one of
   the 48 adjacent-rate paths had at least one terminal-KL turning point.
   Downstream route burden varied on all 48; 45/48 still turned while binary
   D1 status remained constant. For 725-to-749, D1 status was constant on all
   24 paths, yet every path still turned.
6. BF16 coordinate collapse is common, especially for small high-rate
   residuals, but effective vector norms remain close to planned norms and the
   recorded BF16 diagnostics have weak association with terminal KL. BF16 is a
   thresholding detail, not the principal explanation.
7. Strict incumbent repair is more conservative and rate-consistent than
   global replacement, not better in pooled mean KL. It introduced no same-rate
   crossings, retained PR #13 on 92.5--97.1% of tokens, improved three of four
   rate means, and sharply reduced variance and worst regression. The global
   selector has the more favorable pooled mean but much heavier tails.
8. The arithmetic envelope remains plausible: a scalar-adjoint repair scorer
   is 1.34% of the current PR #13 selector MAC count; a one-adjoint scorer is
   1.82%. Neither scorer has yet been shown to recover the exact-label repair.
   This is operation accounting, not measured decode latency.

## 1. Resolving the quoted 22/28 behavior

The complete smoke contains 696 fixed-D1 versus same-rate PR #13 pairs:

| Immediate D1 event | Tokens | Mean fixed-minus-PR13 KL | Median | Token improvement fraction |
| --- | ---: | ---: | ---: | ---: |
| Prevented | 22 | -0.001431 | -0.000121 | 63.64% |
| Introduced | 28 | +0.000980 | +0.000164 | 32.14% |
| Both crossed | 32 | -0.000892 | -0.000002 | 50.00% |
| Both safe | 614 | -0.000058 | +0.000004 | 48.37% |

Among the 22 pairs where fixed D1 prevented a PR #13 crossing, mean fixed-minus-
PR13 KL was `-0.001431`; among the 28 pairs where it introduced a crossing,
the mean was `+0.000980`. The signs agree with the membership-amplification
hypothesis. This is a conditional comparison, not a causal decomposition or a
randomized effect of a swap: each pair also changes the continuous injected
vector, local damage and routed mass.

The apparent paradox comes from using this conditional fact as a complete
ranking objective. The immediate D1 crossing bit is identical in 646/696
comparisons. Their weighted mean fixed-minus-PR13 KL is approximately
`-0.000100`, but their distribution is heavy-tailed and close to zero at the
median. At 384 pages the largest fixed regression is `+0.039651`; at 749 it is
`+0.051261`. The top three positive regressions account for 53.96% and 51.91%
of total positive damage at those two rates. A few threshold-sensitive token
trajectories can therefore reverse an aggregate mean without contradicting the
conditional 22/28 direction.

The fixed policy's complete-smoke result is correspondingly rate-dependent:

| Pages/expert | Prevented | Introduced | Mean KL delta vs PR13 | Median KL delta | Mean live-minus-frozen delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 360 | 7 | 6 | -0.000434 | +0.0000005 | -0.000370 |
| 384 | 7 | 12 | +0.000116 | +0.0000031 | +0.000129 |
| 725 | 5 | 6 | -0.000561 | -0.0000005 | -0.000633 |
| 749 | 3 | 4 | +0.000485 | +0.0000246 | +0.000455 |

The row-level evidence and deterministic summaries are in
[d1_experiment_a_crossing_conditional_rows.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_crossing_conditional_rows.parquet),
[d1_experiment_a_crossing_conditional_summary.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_crossing_conditional_summary.parquet), and
[d1_experiment_a_heavy_tail_summary.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_heavy_tail_summary.parquet).

## 2. What the route-mode intervention establishes

The focused panel executes every candidate in three downstream modes. Live
routing uses candidate IDs and weights; frozen-set/live-weight routing forces
the exact-Q4 expert IDs but retains candidate-derived weights on that set;
fully frozen routing forces both exact-Q4 IDs and execution weights.

Across the 480 realizable allocation-policy rows:

| Quantity | Mean terminal KL or ordered contrast |
| --- | ---: |
| Live execution | 0.0038351 |
| Live minus frozen-set/live-weight: membership-execution contrast | +0.0032122 |
| Frozen-set/live-weight minus fully frozen: router-weight contrast | -0.0000159 |
| Fully frozen: continuous component | 0.0006387 |
| Frozen-set/live-weight KL | 0.0006229 |

The terms close arithmetically to better than `1e-15`. The membership contrast
is 83.76% of mean live KL in magnitude, whereas ordinary weight drift is near
zero after aggregation. Because KL is nonlinear, these are ordered
intervention contrasts, not Shapley values or generally additive causal shares.

This result strongly confirms that discrete expert membership execution is the
largest identified downstream amplifier. It does **not** establish that the
binary immediate D1 crossing is a sufficient objective. The live-versus-
frozen-set contrast freezes membership at every downstream router, whereas
the allocator observes only the next router. Immediate D1 can be unchanged
while a continuous state difference later reaches a different boundary.

That delayed-threshold distinction is common in the diagnostic grid. Of 768
live candidate/token rows, 85 cross immediately at D1, whereas 692 cross at
some observed downstream layer. Thus 607 rows are D1-safe but cross later.
These are enriched-panel counts, not deployment prevalences, but they show why
full-tail membership execution can dominate KL while a binary D1 objective
still leaves most trajectory variation unresolved.

The enriched panel's immediate D1 associations are consistent with that
interpretation but are descriptive only:

| Immediate metric | Pearson with terminal KL | Spearman with terminal KL |
| --- | ---: | ---: |
| Boundary violation depth | 0.467 | 0.295 |
| Membership pairs changed | 0.344 | 0.299 |
| Changed routing mass | 0.327 | 0.289 |
| Binary membership changed | 0.320 | 0.297 |
| Live local qenergy | -0.061 | -0.128 |
| Planned perturbation L2 | -0.043 | -0.125 |

The panel was outcome-enriched, so these coefficients are mechanism
descriptors rather than population estimates or training results. The exact
route-mode rows are in
[d1_diagnostic_route_mode_decomposition_rows.parquet](d1_diagnostic_route_mode_decomposition_rows.parquet),
and the immediate metrics are in
[d1_diagnostic_immediate_d1_severity_correlations.parquet](d1_diagnostic_immediate_d1_severity_correlations.parquet).

A concrete panel case makes the distinction visible. For layer 1, group 7
(`mxfp4-confirm-006`, position 8) at 384 pages, fixed D1 and PR #13 are both
immediate-D1 safe, yet fixed-minus-PR13 terminal KL is `+0.032544`. Its ordered
contrast is `+0.032326` membership execution, `+0.000459` router weight and
`-0.000241` fully frozen. Immediate membership agrees, but the continuous
state later reaches a different downstream membership trajectory. Conversely,
for layer 6, group 19 at 725 pages both policies cross immediately and fixed D1
improves KL by `0.048392`; at 749 pages fixed D1 introduces a crossing and
regresses by `0.051261`. These are explanatory selected cases, not frequency
estimates. They are retained in
[d1_diagnostic_cohort_fixed_vs_pr13_rows.parquet](d1_diagnostic_cohort_fixed_vs_pr13_rows.parquet).

## 3. Why more pages can increase KL

### 3.1 Local error improves while terminal KL reverses

For the complete smoke's fixed-D1 policy:

| Adjacent caps | Tokens | Local qenergy improved | Mean local-qenergy delta | Mean terminal-KL delta | Median KL delta | KL worsened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 360 to 384 | 174 | 98.28% | -0.000237 | +0.000174 | +0.000012 | 52.30% |
| 725 to 749 | 174 | 100.00% | -0.000044 | +0.000402 | -0.000001 | 49.43% |

The larger cap almost always lowers the metric it was optimized to improve,
yet terminal KL is thresholded and heavy-tailed. The 725-to-749 median is
slightly favorable even while a few positive tails make the mean unfavorable.
This is not an arithmetic failure in local qenergy; it is a mismatch between a
smooth local norm and a routed nonlinear tail.

Freezing every downstream expert ID and weight isolates that distinction. For
the same fixed-D1 paths, mean fully-frozen KL changes by `-0.000015` from 360
to 384 and by `-0.000044` from 725 to 749, while live KL changes by
`+0.000174` and `+0.000402`. The continuous fixed-route component therefore
moves in the favorable direction on average; live routed amplification turns
the aggregate sign.

### 3.2 Higher-rate allocations are not nested

The independently optimized adjacent-rate allocations are wholesale
reallocations, not a stream of extra pages:

| Adjacent caps | Tokens with any removed page bit | Mean removed bits/group | Mean added bits/group | Net additions | Mean delta cosine | P10 cosine | Mean high/low norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 360 to 384 | 98.85% | 122.14 | 311.00 | 188.86 | 0.8633 | 0.8016 | 0.9401 |
| 725 to 749 | 98.85% | 134.30 | 332.76 | 198.47 | 0.7957 | 0.6883 | 0.8846 |

Thus an extra nominal 24 pages/expert removes roughly 122--134 previously
selected projection pages per routed group and adds 311--333 elsewhere. In the
stored/source policy bank, the resultant perturbation usually shrinks but
rotates, especially at 725-to-749. A monotone page count therefore does not
imply a monotone source perturbation path. Exact live same-host rotation is
tested separately by the interpolation and matched-norm controls below. The
source state and vector geometry is recorded in
[d1_experiment_a_source_adjacent_expert_geometry.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_source_adjacent_expert_geometry.parquet) and
[d1_experiment_a_source_adjacent_delta_geometry.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_source_adjacent_delta_geometry.parquet).

### 3.3 Continuous interpolation remains non-monotonic

The panel linearly interpolates between the live low- and high-rate fixed-D1
vectors at `lambda = 0, 0.25, 0.5, 0.75, 1`. These are explanatory vectors,
not realizable page allocations.

| Path | Mean high-minus-low KL | Median | High worse | Paths with KL turning point | D1 crossing transition | D1 choice equals observed KL minimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 360 to 384 | +0.002398 | +0.000051 | 58.33% | 100% | 12.50% | 37.50% |
| 725 to 749 | +0.003640 | +0.000037 | 54.17% | 100% | 0.00% | 25.00% |

For 360-to-384, 11/24 paths have an interior point above both endpoints and
15/24 have one below both. For 725-to-749 the counts are 18/24 and 14/24.
Some paths contain both. Here downstream route burden means cumulative changed-
layer count or cumulative router-mass churn. It changes along every one of the
48 paths, and 45/48 paths are terminal-KL non-monotonic while binary D1 status
remains constant. Among those 45, changed-layer count varies on 41 and first-
crossing distance varies on 33; cumulative route mass varies on all 45. Most
importantly, all 24 725-to-749 paths turn even though immediate D1 crossing
status is constant along every one. These observations are consistent with
later threshold and mass cascades, but they do not claim that a discrete later
crossing explains every reversal. The non-monotonicity is not explained by a
single D1 membership flip or by the discrete page-search algorithm alone.

The interpolation endpoint parity gate is exact for KL, NLL, final-hidden MSE,
full-cache MSE, injected MSE and effective BF16 delta MSE. Results are in
[d1_diagnostic_interpolation_points.parquet](d1_diagnostic_interpolation_points.parquet),
[d1_diagnostic_interpolation_turning_points.parquet](d1_diagnostic_interpolation_turning_points.parquet), and
[d1_diagnostic_interpolation_summary.parquet](d1_diagnostic_interpolation_summary.parquet).
The complete descriptive tail joins are in
[d1_diagnostic_downstream_outcome_rows.parquet](d1_diagnostic_downstream_outcome_rows.parquet)
and [d1_diagnostic_downstream_path_transition_summary.parquet](d1_diagnostic_downstream_path_transition_summary.parquet).

### 3.4 Matched-norm controls separate shrinkage from rotation

The high-rate vector was rescaled to the low-rate planned L2 norm. This gives
an ordered, non-unique decomposition of high-minus-low KL into a direction
contrast at matched norm and a norm contrast along the high direction:

| Path | Pages added/group | Cosine | Mean angle | High/low norm | High-minus-low KL | Direction at matched norm | Norm along high direction | Direction-dominant tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 360 to 384 | 199.92 | 0.8751 | 28.64 deg | 0.9397 | +0.002398 | +0.000457 | +0.001941 | 45.83% |
| 725 to 749 | 198.75 | 0.8062 | 35.33 deg | 0.8858 | +0.003640 | +0.003904 | -0.000265 | 58.33% |

Across all 48 panel paths, direction is the larger absolute component for
52.08% and norm for 47.92%. There is no universal direction-only explanation.
At 725-to-749, however, the direction contrast is larger than the observed
regression while the reduced norm is mildly beneficial. This is direct
mechanistic evidence that refinement-page reallocation can rotate the residual
into a much worse routed-tail direction even as it reduces local energy.

Planned norms match in the rescaled control to a maximum relative error below
`5.4e-8`. The row and summary tables are
[d1_diagnostic_page_direction_vs_norm_rows.parquet](d1_diagnostic_page_direction_vs_norm_rows.parquet) and
[d1_diagnostic_page_direction_vs_norm_summary.parquet](d1_diagnostic_page_direction_vs_norm_summary.parquet).

### 3.5 BF16 explains granularity, not the main reversal

The scientific hook adds the planned FP32 delta to the native routed output and
rounds once to BF16. Every 2048-coordinate panel vector loses at least one
planned nonzero coordinate, but whole-vector execution remains faithful:

| Candidate family/rate | Mean collapsed nonzero fraction | P95 | Maximum | Mean effective/planned norm | Mean quantization-error MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allocation policies, 360 | 2.39% | 6.24% | 9.57% | 1.0004 | 2.22e-10 |
| Allocation policies, 384 | 2.57% | 7.71% | 10.25% | 1.0004 | 2.23e-10 |
| Allocation policies, 725 | 9.95% | 43.47% | 70.90% | 1.0028 | 2.24e-10 |
| Allocation policies, 749 | 11.47% | 52.04% | 79.64% | 0.9988 | 2.21e-10 |

The maximum quantization error stays at approximately half a BF16 ULP. In this
enriched panel, Pearson association with terminal KL is `0.0039` for collapsed
fraction, `0.0495` for effective/planned norm ratio, `0.0193` for quantization-
error MSE and `0.0398` for P95 ULP error. These are outcome-only descriptive
associations, not evidence of absence in a larger cohort.

BF16 makes the map piecewise and can decide very small boundary cases. It does
not explain why mean KL reverses when effective norms remain near their planned
values, why the matched-norm direction contrast is large, or why downstream
membership execution dominates the route-mode decomposition.

See [d1_diagnostic_bf16_ulp_rows.parquet](d1_diagnostic_bf16_ulp_rows.parquet),
[d1_diagnostic_bf16_ulp_summary.parquet](d1_diagnostic_bf16_ulp_summary.parquet), and
[d1_diagnostic_bf16_ulp_terminal_association.parquet](d1_diagnostic_bf16_ulp_terminal_association.parquet).

## 4. Remedy target ablations

The remedy analysis is GPU-free and uses the complete smoke, not the enriched
panel. It chooses only among the five candidates already executed for each
rate/token. Exact full-model D1 membership and mass plus live execution-
weighted local qenergy are selector inputs; terminal KL is joined only after
the choice.

### 4.1 Same-rate policies

| Selector | Rate | PR13 retained | Prevented | Introduced | Mean KL delta | Mean live-minus-frozen delta | Mean local-qenergy delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict incumbent repair | 360 | 92.53% | 13 | 0 | -0.000148 | -0.000152 | +1.31e-6 |
| Strict incumbent repair | 384 | 94.83% | 9 | 0 | -0.000018 | -0.000016 | +3.07e-6 |
| Strict incumbent repair | 725 | 97.13% | 5 | 0 | -0.000029 | -0.000008 | +0.39e-6 |
| Strict incumbent repair | 749 | 97.13% | 5 | 0 | +0.000012 | +0.000011 | +0.03e-6 |
| Global hard-D1/local | 360 | 18.39% | 13 | 0 | -0.000386 | -0.000310 | +14.06e-6 |
| Global hard-D1/local | 384 | 20.11% | 9 | 0 | +0.000196 | +0.000145 | +24.43e-6 |
| Global hard-D1/local | 725 | 21.84% | 5 | 0 | -0.000621 | -0.000622 | +16.23e-6 |
| Global hard-D1/local | 749 | 20.11% | 5 | 0 | +0.000185 | +0.000192 | +5.62e-6 |

Strict repair holds PR #13 exactly whenever its immediate D1 set is safe. If
PR #13 crosses, another candidate is admissible only if it has strictly fewer
exact full-model D1 crossings; ties use mass churn, local qenergy, absolute
difference in total selected-group pages and policy name. It changes only
5--13 of 174 tokens per rate.

The global selector prevents the same number of PR #13 crossings and
introduces none because PR #13 remains in its candidate pool. Nevertheless it
changes roughly 80% of allocations and reproduces the alternating rate
failures. The key risk-control result is that most globally rewarded changes
occur after the hard D1 goal is already tied or saturated, where small mass or
local differences cause a much wider outcome distribution after rotating away
from a safe incumbent.

Across all 696 same-rate rows, strict repair has mean KL delta `-0.000046`,
standard deviation `0.000758` and maximum regression `+0.002839`; global
replacement has the more favorable mean `-0.000157`, but standard deviation
`0.003401` and maximum regression `+0.037256`. Strict repair also sacrifices
about `1.2e-6` mean local qenergy versus about `15e-6` for global replacement.
It is more conservative and improves three of four rate means, but it is not
universally or pooled-mean superior. The evidence supports incumbent bias and
saturation as a tail-risk prior, not a final quality claim.

### 4.2 A `<= budget` rule is useful but not sufficient

The adjacent-cap selector considers all low- and high-rate executed candidates
that use no more pages than high-rate PR #13:

| High-rate reference | Lower-rate chosen | Pages saved/group | Prevented | Introduced | Mean KL delta vs high PR13 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 48.85% | 99.26 | 11 | 0 | +0.000005 |
| 749 | 53.45% | 156.61 | 6 | 0 | +0.000232 |

Making extra pages optional prevents crossings and saves traffic, but the same
hard-D1 ranking still fails to improve KL at the high-rate references. The
matched-metadata low-rate policies similarly do not establish a robust quality
win. Early stopping should be retained as an architectural option, not
promoted as the remedy by itself.

### 4.3 Metadata-matched low-rate policies

The 360- and 725-page banks pay for the proposed uncertainty sidecar at the
same charged bpw as PR #13 at 384 and 749 pages. Applying the two selectors to
those low-rate banks gives:

| Selector | Low vs reference | Mean KL delta | Mean live-minus-frozen delta | Prevented | Introduced | Mean local-qenergy delta | Mean pages/group delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict incumbent repair | 360 vs 384 | +0.000228 | +0.000157 | 11 | 3 | +2.459e-4 | -192.06 |
| Global hard-D1/local | 360 vs 384 | -0.000010 | -0.000001 | 11 | 3 | +2.587e-4 | -195.17 |
| Strict incumbent repair | 725 vs 749 | +0.000615 | +0.000634 | 6 | 0 | +3.768e-5 | -192.31 |
| Global hard-D1/local | 725 vs 749 | +0.000023 | +0.000020 | 6 | 0 | +5.351e-5 | -272.27 |

Same-rate zero-introduction does not carry over automatically: both 360-page
selectors introduce three crossings relative to the 384-page PR #13 route.
None of these metadata-matched rows establishes a robust KL win.

The immutable remedy evidence is
[d1_experiment_a_deployable_target_remedy_rows.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_deployable_target_remedy_rows.parquet),
[d1_experiment_a_deployable_target_remedy_summary.parquet](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_deployable_target_remedy_summary.parquet), and
[d1_experiment_a_diagnosis_facts.json](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_diagnosis_facts.json).

## 5. Recommended minimization target

### 5.1 Tested target ablation

The tested five-candidate rule is a **hard-saturating PR #13-incumbent repair**,
not a fresh global crossing minimizer. Let `s0` denote the PR #13 incumbent and
let `R_D1(s)` denote its exact full-model binary D1 crossing count. The
executed admissible set is:

\[
\mathcal A_{tested}(s_0)=
\begin{cases}
\{s_0\}, & R_{D1}(s_0)=0,\\
\{s_0\}\cup
\{s: R_{D1}(s)<R_{D1}(s_0)\},
& R_{D1}(s_0)>0.
\end{cases}
\]

Within the unsafe admissible set, the executed ordering is fewer exact D1
crossings, lower exact D1 routing-mass churn, lower live local qenergy, smaller
absolute difference in total selected-group pages, and deterministic policy
name. Terminal KL is attached only after selection. No signed boundary score,
local guardrail, centered-logit term, functional severity, page-state distance
or nested candidate generator was used in this ablation.

### 5.2 Proposed next oracle ablations

The next Experiment A oracle should preserve the tested hard saturation and
separately ablate the following additions rather than treating them as already
validated:

1. an additive, live-execution-weighted local guardrail,
   `J_local(s) <= J_local(s0) + eta J_Q2`;
2. robust boundary-violation depth before routing-mass churn among candidates
   with the same crossing count;
3. centered D1 router-logit fidelity and then functional swap severity;
4. true page-state/Hamming distance from the incumbent, rather than only total
   page-count difference; and
5. calibrated uncertainty around the D1 label.

The guardrail is intended to stop a crossing repair from buying a fragile
cancellation with large local damage. Boundary depth is motivated by the exact
panel association, but neither it nor functional severity has yet been replayed
as a selector on an independent cohort.

### 5.3 Nested, route-safe local completion hypothesis

Independent optimization at every cap caused 98.85% of adjacent token
allocations to remove earlier pages. The future page path should instead be
nested:

1. retain already committed refinements;
2. reserve a small repair window only while the incumbent D1 boundary is
   unsafe or uncertain;
3. stop spending route-risk bandwidth once safety is saturated; and
4. spend remaining pages on local qenergy only when the addition remains inside
   the calibrated route-safe set.

This is the proposed realization of “more pages” as additional refinements,
not a new globally optimized state. It has not been executed. A nested path
would remove the large de-refine/re-refine direction jump, but cannot guarantee
monotone KL: the interpolation experiment shows that the nonlinear tail can
still turn along a continuous path. Route-safe certification and the local
guardrail therefore require their own oracle ablations.

### 5.4 Future runtime predictor, not current evidence

At runtime exact Q4 routes, exact full-model D1 outcomes and exact page effects
are unavailable. The oracle target above must later be approximated with:

- token-dependent responses `d_p(x)`, not static page vectors;
- a one- or scalar-adjoint signed D1 margin effect;
- calibrated uncertainty that covers source-to-live transfer, BF16 and model
  approximation error; and
- a hard rule that a predicted-safe incumbent is not globally replaced merely
  to improve a soft D1 score.

The separate
[corrected source-to-live transfer audit](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/transfer_diagnosis/D1_SOURCE_LIVE_TRANSFER_DIAGNOSIS.md)
is authoritative. The exact-Q4 baseline D1 margin transfers well from source
to the PRO trajectory (Pearson `0.961537`, Spearman `0.900124`, MAE
`0.016433`), and fixed-policy stored/live delta cosine averages `0.990959`.
Candidate risk still needs calibration: its ROC AUC is `0.802699`, but the
margin-zero threshold has only 20.0% recall and the source crossing label misses
93.33% of actual fixed-policy crossings. Continuous reconstruction fidelity and
risk ranking therefore do not make zero a route certificate. These are the
corrected final facts; no preliminary transfer correlation is used here.


A runtime controller should eventually replace exact `R_D1` with a calibrated
upper confidence risk `U_D1`. A switch is allowed only if the candidate's upper
risk is strictly below the incumbent by a declared margin; otherwise the
incumbent remains. False-certificate rate, confidence coverage and signed page-
effect calibration are future measurements. They have not been run here.

No D2--D4 model is proposed. Any offline tail-susceptibility feature must be
distilled into D1-available token/layer features and validated separately; it
must not silently become a multi-layer runtime lookahead.

## 6. Runtime cost boundary

### 6.1 Selector arithmetic

The checked-in accounting uses eight routed experts, 512 units, three page-
bearing projections, hidden width 2048, 16 candidate logits and six refreshes:

| Per routed group | MACs | Relative to current PR13 selector |
| --- | ---: | ---: |
| Current PR13 primary selector | 51,343,360 | 100.00% |
| Dense D1 candidate scorer, including synthesis | 9,781,248 | 19.05% |
| One-adjoint D1 scorer, including synthesis | 933,888 | 1.82% |
| Scalar-adjoint repair scorer | 688,128 | 1.34% |
| PR13 plus one-adjoint repair overlay | 52,277,248 | 101.82% total |

Hard saturation matters for systems as well as accuracy. Most tokens retain
PR #13 in the exact-label oracle repair, so a future calibrated scalar risk
gate could avoid a dense global D1 search on the common safe path. No tested
scalar or one-adjoint predictor yet recovers the oracle repair choices. The
table is analytic MAC accounting. It excludes launches, transfer stalls, cache
traffic, kernel fusion and overlap; no measured latency or no-bottleneck claim
is made.

As a descriptive screen only, a PR #13 rank-8/rank-9 margin threshold of
`4/64` flags 393/696 rows and catches all 54 observed PR #13 D1 crossings.
If that same coverage transferred, invoking the scalar overlay only on flagged
rows would add 0.76% of PR #13 selector MACs on average (1.03% for the
one-adjoint overlay). This threshold was measured on the same three requests;
it is not calibrated, held out, or a deployable certificate. The immutable
operation and screening arithmetic is in
[d1_experiment_a_repair_accounting.json](d1_experiment_a_repair_accounting.json).

A separate invalidated next-layer pre-MoE pass is not part of the recommended
first implementation. At context 32 it is estimated at 33.824 million MACs and
optimistic traffic lower bounds of 31.52 MB packed or 68.31 MB BF16, roughly
54.4% of estimated normal layer work before launch and transfer costs. That is
large enough to become the bottleneck the architecture is meant to avoid.

### 6.2 Metadata and matched traffic

| Representation | KiB/eight-expert group | bpw | 40-layer model MiB |
| --- | ---: | ---: | ---: |
| Existing PR13 factor and ABC metadata | 72.125 | 0.023478 | 90.156 |
| Joint token-response metadata | 155.125 | 0.050496 | 193.906 |
| Joint response plus uncertainty | 167.625 | 0.054565 | 209.531 |

The uncertainty-inclusive representation leaves 360 and 725 integral 512-byte
pages/expert at the two charged operating points:

| Allocated pages/expert | Traffic-equivalent PR13 pages | Charged bpw |
| ---: | ---: | ---: |
| 360 | 384 | 0.5234781901 |
| 725 | 749 | 0.9987386068 |

Routing-mass severity requires no additional stored metadata. A rank-8 FP16
functional embedding would add only 16 bytes/expert, but functional-swap
selection was not executed and is not promoted here. The accounting
implementation is [d1_accounting.py](../../../src/oracle_study/d1_accounting.py)
and its focused checks are
[test_d1_accounting.py](../../../tests/test_d1_accounting.py).

## 7. Validation and provenance

The exact panel ran on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
with 94.97 GiB, PyTorch 2.8.0+cu128 and CUDA 12.8. Model loading took 364.24 s.
The frozen grid contains:

- 24 unique token/layer identities;
- six layers: 0, 1, 4, 6, 12 and 23;
- four page caps: 360, 384, 725 and 749;
- five realizable policies and 12 non-realizable interpolation controls;
- three downstream route modes;
- 2,304 quality rows, 74,784 detailed route rows and 72 zero-dose rows.

Hard gates passed:

- all zero-dose paths were exact;
- repeated native cached baselines and complete post-token caches passed the
  runner's parity gates;
- interpolation endpoints reproduced their realizable fixed-D1 executions
  exactly;
- live/frozen-set/fully-frozen KL decomposition closure was below `1e-15`;
- matched-norm direction/norm closure was below `1e-15`; and
- final facts state that no Experiment B component and no D2--D4 objective or
  model is present; full-tail route observations are marked as descriptive
  outcomes only.

The GPU run consumed exact same-host, token-dependent expert reconstruction and
actual BF16 execution weights. The 12 source/live route-mismatch identities
were regenerated on the target trajectory before import. Source-slice labels
remain screening approximations rather than certificates.

Primary provenance:

- [immutable diagnostic config](../../../configs/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1.json)
- [panel table](../d1_experiment_a_diagnostic_panel.parquet)
- [quality table](../d1_experiment_a_diagnostic_quality.parquet)
- [detailed route table](../d1_experiment_a_diagnostic_routes.parquet)
- [zero-dose table](../d1_experiment_a_diagnostic_zero_gates.parquet)
- [host facts](../d1_experiment_a_diagnostic_host_facts.json)
- [final run facts](../d1_experiment_a_diagnostic_run_facts.json)
- [analysis facts](d1_experiment_a_diagnostic_analysis_facts.json)
- [parent cached-decode smoke report](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/analysis/D1_CACHED_DECODE_TAIL_KL_SMOKE_REPORT.md)
- [offline diagnosis facts](../../qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/diagnosis/d1_experiment_a_diagnosis_facts.json)

## 8. Decision and trajectory

### What is established

- Membership execution is an important causal amplifier.
- Preventing versus introducing a D1 crossing has the expected conditional KL
  direction.
- Binary immediate D1 membership, local qenergy and perturbation norm are each
  insufficient as global allocation rankings.
- Independent full-budget reoptimization creates large non-nested rotation;
  exact controls show that direction can dominate the high-rate reversal.
- A hard-saturating incumbent repair has substantially lower variance and
  worst-case regression than global hard-D1 replacement in the current smoke,
  although the global selector has the better pooled mean.
- Scalar/one-adjoint arithmetic is small relative to the current PR13 selector,
  while an additional full pre-MoE pass is not; predictor accuracy is untested.

### What is not established

- The strict repair is not a deployable controller; it uses exact full-model
  D1 labels over five executed candidates.
- No source margin threshold is a certified runtime stopping rule.
- No functional-swap cost has been selected or replayed.
- No metadata-matched policy has a robust quality win.
- No generated rollout, temporal cache-drift sequence, joint all-layer
  compression, independent-request quality result, or Experiment B result
  exists.

### Required next step before Experiment B

Remain in Experiment A and implement the target in two separable stages:

1. **Oracle validation:** construct hard-saturating PR13-incumbent repair with a
   nested candidate path and additive local guardrail. Use exact same-host full-
   model D1 labels only as oracle labels. Evaluate the four rates on at least 64
   independent exact-prefix decode requests; 128 is preferable. Terminal KL is
   scored after allocation.
2. **Predictor validation:** distill the oracle to token-dependent signed page
   effects plus calibrated uncertainty. Report signed-effect correlation, sign
   accuracy, top-page ranking, corrected-logit error, interval coverage and
   false certificates. Keep this a predictor study until accuracy and cost are
   independently acceptable.

Only a robust oracle win followed by a calibrated, efficient predictor should
promote a sequential controller to Experiment B.

## 9. Limitations

1. The complete smoke has only three requests and 174 token/layer identities
   per rate.
2. The 24-case panel is deliberately outcome-enriched; aggregate panel means
   cannot estimate deployment quality.
3. Tokens are isolated teacher-forced decode positions behind exact prefix
   caches, not generated rollouts.
4. Only one token and one injection layer are perturbed at a time. Cache drift
   across subsequent generated tokens is not measured.
5. The six sentinel injection layers do not establish all-layer behavior.
6. Interpolation and norm-rescaled vectors are causal controls, not legal page
   allocations and have no bpw interpretation.
7. Route-mode and direction/norm decompositions depend on intervention order
   and are not unique nonlinear attributions.
8. Exact-token and remedy labels are scientific oracles, not runtime features.
9. Mean KL is tail-sensitive; medians, request strata and individual events
   must remain visible.
10. Compute and metadata values are arithmetic accounting, not latency,
    bandwidth-overlap or throughput measurements.

## 10. Reproduction

Run from `experiments/adaptive_expert_precision_oracle` with the raw `/workspace`
inputs retained by the facts.

~~~bash
export PYTHONPATH=src:scripts
export D1_DIAG_CONFIG=configs/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1.json
export D1_DIAG_OUT=/workspace/pr13_d1_experiment_a_diagnostic_panel_20260825_v1
export D1_V3_ROOT=/workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3
~~~

Reproduce the complete-smoke offline diagnosis and remedies:

~~~bash
python scripts/analyze_d1_experiment_a_diagnosis.py \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json \
  --input "$D1_V3_ROOT" \
  --mechanism-results /workspace/pr13_d1_decode_slice_pilot_20260824_v2/results \
  --matched-results /workspace/pr13_d1_decode_matched_rates_20260824_v2/results \
  --patch-root "$D1_V3_ROOT/same_host_allocation_patch" \
  --output "$D1_V3_ROOT/diagnosis"
~~~

Freeze, run, finalize and analyze the exact panel:

~~~bash
python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase panel --config "$D1_DIAG_CONFIG" --output "$D1_DIAG_OUT"

python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase run --config "$D1_DIAG_CONFIG" \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint \
  --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
  --fit-dir /workspace/pr13_average_rate_all_layers_20260822_v1/results/fit \
  --reconstruction-workers 8 --output "$D1_DIAG_OUT"

python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase finalize --config "$D1_DIAG_CONFIG" --output "$D1_DIAG_OUT"

python scripts/analyze_d1_experiment_a_diagnostic_panel.py \
  --config "$D1_DIAG_CONFIG" --input "$D1_DIAG_OUT" \
  --output "$D1_DIAG_OUT/analysis"
~~~

Focused verification:

~~~bash
pytest -q \
  tests/test_d1_decode_tail.py \
  tests/test_d1_route_objective.py \
  tests/test_d1_experiment_a_diagnosis.py \
  tests/test_d1_source_live_transfer.py \
  tests/test_d1_accounting.py \
  tests/test_d1_downstream_outcome_diagnosis.py \
  tests/test_d1_experiment_a_diagnostic_panel.py \
  tests/test_d1_experiment_a_diagnostic_panel_analysis.py
~~~

All interpretation should remain within the boundary stated at the start of
this report: three-request smoke evidence, a 24-case explanatory panel,
terminal KL as outcome only, D1 only, and no Experiment B.
