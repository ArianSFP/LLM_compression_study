# D1 route-objective and runtime-controller protocol

## Decision boundary

D1 is only the same token's next-layer router:

\[
(t,\ell)\rightarrow(t,\ell+1).
\]

There are no D2--D4 objectives, predictors, or routing evaluations. The old
H1--H4 captures are future-token, same-layer quantities and are not inputs to
this work. Terminal final-logit KL remains an outcome measure.

The work is intentionally split into two experiments. Experiment A tests the
scientific objective without any deployment machinery. Experiment B is
allowed to begin only if Experiment A improves exact D1 routing and final KL
at matched all-in bandwidth.

## Experiment A: exact D1 objective oracle

For each token/layer group, reconstruct every complete per-expert frontier
option from the current token activation. The exact option delta is

\[
d_{e,o}(x)=\widetilde y_{e,o}(x)-y_e^{(4)}(x).
\]

Combine it with BF16 execution weights. Exact Q4 D1 routes, exact margins,
exact token-dependent option deltas, and exact full-VJP sensitivities are
oracle labels. They are never runtime features.

The core API is `oracle_study.d1_route_objective`:

- `token_option_deltas` reconstructs exact `d_{e,o}(x)` from complete options;
- `build_d1_boundary_problem` creates exact-Q4 selected/outsider pairs from
  unique candidate-logit sensitivities;
- `d1_group_option_allocate` performs deterministic coordinate and pair
  search under the pooled page cap and additive local-qenergy guardrail;
- `run_d1_oracle_policies` selects a final allocation and invokes the supplied
  exact downstream replay exactly once per D1 policy, returning both D1 router
  logits and terminal logits for KL;
- `adaptive_outsider_ranks` selects train-calibrated near-boundary outsiders,
  capped at eight, and `additive_local_damage_limit` freezes the requested
  `J_local* + eta J_Q2` constraint.

The primary comparison is at the four frozen PR #13 all-in rates and identical
burst cap. It includes router-square, exact-combined local qenergy, strict D1,
routing-mass-weighted D1, and functional-swap-weighted D1. Strict D1 is the
first mechanistic policy; the two severity forms are ablations.

There is no initial `S0`, omitted-tail prediction, speculative next-layer
execution, or certification in Experiment A. Layer 39 is a local-only negative
control.

## Experiment B: sequential target-free controller

The controller starts from an explicitly charged initial state only after
Experiment A passes. The first control is a deterministic, train-only,
physically contiguous prefix. It is not claimed as the eventual solution.
The eventual initial state should maximize immediate D1 certificate coverage
at fixed base traffic rather than average local-qenergy recovery alone.

### Token-dependent latent

The runtime approximation is

\[
d_p(x)\approx U_\ell c_p(x),
\]

not a static `c_p`. Static sidecar metadata contains output geometry and page
structure. Dynamic token information contains current Q2 responses, predicted
Q4 gate/up responses, execution weights, router sensitivity, and uncertainty.

`TokenLatentExpert` stores static down2/down4 coefficients but receives a
dynamic `[unit,4]` table of `h22,h24,h42,h44`. Its legal transition effects
therefore change with the token and with the source gate/up/down state. The
runtime never treats all pages as independent fixed vectors.

### Certification

The crude route is not a target. The controller predicts candidate logits at
the Q4 endpoint, allows their order to flip, and constructs a joint interval
certificate. The candidate pool expands while any excluded expert can cross
the boundary. Stopping requires both:

1. a certified predicted Q4 target set; and
2. a fetched state certified to execute the same set.

An exact final execution verifies the prediction. A false certificate must be
recorded and charged through the fallback path; overriding expert IDs is not
allowed.

### Required predictor evaluation

Before sequential claims, report on held-out validation groups:

- Pearson and Spearman signed-effect correlation;
- signed-effect sign accuracy;
- top-page overlap at fixed shortlist sizes;
- corrected-logit MAE, RMSE, maximum error, and correlation;
- interval coverage and confidence--coverage curves;
- candidate-pool misses.

`signed_effect_metrics`, `corrected_logit_metrics`, and `certify_topk` implement
the frozen metric definitions.

### Systems accounting

The controller must report actual fetched pages, physical wire bytes,
transaction amplification, rejected prefetch, transfer overlap, provisional
pass invalidation, and p50/p90 latency. Selector metadata is charged against
the total bpw before correction pages are assigned.

`d1_accounting.controller_accounting` records both the dense candidate-logit
upper estimate and the one-adjoint scan estimate. `provisional_decode_accounting`
separately records the hybrid 29-linear/10-full next-layer cost. These are
arithmetic and traffic estimates, not latency measurements.

The frozen rank-8, 16-candidate, six-refresh estimate is:

| Quantity | PR #13 primary | D1 dense | D1 one-adjoint |
| --- | ---: | ---: | ---: |
| Selector MACs/group | 51,343,360 | 9,781,248 | 933,888 |
| Fraction of PR #13 | 100% | 19.05% | 1.82% |

The D1 totals include 81,920 response-synthesis MACs/group. The dense estimate
charges all six rescans; the one-adjoint estimate collapses the current
directional loss before each page scan. These figures exclude the provisional
next-layer execution, transfer stalls, and kernel-launch latency.

Joint token-response metadata increases the PR #13 payload from 72.125 to
155.125 KiB per routed eight-expert group, or from 0.023478 to 0.050496 bpw.
With the uncertainty sidecar it is 167.625 KiB and 0.054565 bpw. At the four
unchanged all-in rates, the corresponding correction caps fall from
`384/576/749/768` pages per expert to `360/552/725/744`. Thus metadata is never
free and the scientific comparison remains bandwidth-matched.

The vectorized 128-page NumPy shortlist measured 5.92 ms median on the supplied
RTX 3090 host CPU. The device-resident tensor path measured 0.831 ms median and
0.910 ms p90 per refresh on that GPU. Six serialized
refreshes would therefore still cost roughly 5 ms before transfer or
provisional execution; batching, overlap, CUDA graphs/fusion, and actual model
latency must be measured before a non-bottleneck claim.

## RTX 3090 layer-slice result (2026-08-24)

The original three-layer result below was subsequently expanded to layers 0,
1, 4, 6, 12, and 23 at both 384 and 749 pages/expert. In the balanced
six-layer result, PR #13 changes 20/192 and 11/192 exact D1 top-8 sets; the
exact request-coupled repair changes 1/192 and 0/192. Layer 6 exercises a
full-attention next-layer path. The complete sampling, implementation, parity,
fixed-policy results, runtime accounting, artifacts, and limitations are in
[the stratified expansion methodology and evidence ledger](D1_STRATIFIED_LAYER_SLICE_EXPANSION_20260824.md).

The following paragraphs document the earlier promotion sequence and remain
useful provenance for the expansion.

The implemented paired slice runner evaluated complete column-generated PR #13
frontiers at layers 0, 12, and 23. It used all 32 validation groups per layer,
true pre-residual replacement, BF16 execution weights, exact full-VJP labels,
and exact execution through the next router. The promoted screen covered the
384-page rate on all three layers and the 749-page rate on layers 12 and 23.

At 384 pages, historical PR #13 changed the exact paired top-8 set in 6/96
groups. At 749 pages it changed 3/64. The exact-combined local oracle still
changed 5/96 and 3/64, respectively. The final exact request-coupled D1
reranker changed 0 groups at both rates while its mean local-qenergy damage was
0.981479 and 0.981571 times PR #13. Thus the sampled result did not purchase
route agreement by worsening aggregate local qenergy.

Layer 0 exposed an important prefill boundary. At an exact BF16 rank-8/rank-9
tie, the current-token and causal-prefix perturbations were separately safe but
crossed when replayed together. A global per-request choice among the D1
policies left 1/96 crossings at 384 pages. A deterministic exact repair over
mixtures of complete finalists removed it. That repair is an oracle-only
upper-bound and identifies causal-prefix coupling as a required training/eval
feature; it is not deployable controller work.

On the runs that emitted a regenerated router-square comparator, its selected
delta was bit-identical to the preserved historical PR #13 delta. Repeated
paired 3090 zero-dose slices were also bit-identical. Cross-device BF16 drift
against the PRO 6000 captures was measured separately and never counted as an
allocator-induced switch.

This result establishes exact D1 membership control only on the sampled
layer/rate slices. It contains no final-logit KL, D2--D4, all-layer, or runtime
certificate evidence.

## Later latent next-layer update ablation

After the initial controller is measured, fit or precompute projections such
as `W_Q U_l`, `W_K U_l`, `W_V U_l`, and analogous DeltaNet response factors.
Test whether a latent change can correct provisional router estimates or lower
the frequency of full reruns. This approximation must not be presented as
exact final attention or recurrent-state execution unless exact state parity
is separately demonstrated.

## Current implementation boundary

The repository contains the objective/search, token-dependent latent,
certificate, prediction metrics, exhaustive/vectorized/device-resident
transition ranking, safe batch application, exact-once replay API, and
compute/byte accounting. `run_d1_slice_oracle_pilot.py` now supplies the Qwen
model-host exact VJP and paired pre-residual D1 replay path;
`run_d1_exact_request_repair.py` performs the explicitly oracle-only
causal-prefix finalist repair. The deployable signed-effect predictor,
uncertainty calibration, sequential certificate study, and exact downstream
KL replay remain future gates.
