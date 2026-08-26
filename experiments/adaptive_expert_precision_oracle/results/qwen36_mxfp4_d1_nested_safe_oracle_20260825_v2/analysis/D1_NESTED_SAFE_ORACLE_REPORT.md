# D1 nested-safe Experiment A report

## Status

This is the canonical result report for the held-out nested D1-safe allocator
study. **Experiment A failed its preregistered promotion gate. Experiment B was
not started and should remain paused.**

The result is scientifically useful but architecturally negative:

> Exact D1 repair prevented immediate expert-membership crossings without ever
> introducing one, yet did not reliably improve terminal KL. Literal physical
> nesting also did not make terminal KL monotonic in refinement pages. D1
> membership is a causal safety signal, not a sufficient continuous allocator
> value function.

The appropriate next allocator is:

\[
\text{PR13 incumbent}
\rightarrow
\text{bounded D1 safety repair}
\rightarrow
\text{add-only tail-aware completion}
\rightarrow
\text{early stop},
\]

where the completion score is a cheap, token-dependent continuous tail proxy,
not another D2--D4 routing objective.

The full predeclared methodology is
[D1_NESTED_SAFE_ORACLE_METHODS_20260825.md](../../../D1_NESTED_SAFE_ORACLE_METHODS_20260825.md).
The immutable numeric decision is in
[d1_nested_experiment_a_promotion.json](d1_nested_experiment_a_promotion.json).

## Scientific boundary

D1 means only the same decode token's next-layer router:

\[
(t,\ell)\rightarrow(t,\ell+1).
\]

Prefill is exact. Each request contributes one isolated teacher-forced decode
token behind a complete native hybrid prefix cache. Candidate caches are
private. The allocator uses no terminal logit, final hidden state, downstream
route after D1, or post-token cache outcome. Complete selected states were
checksum-sealed before downstream execution.

This is an oracle allocator experiment, not a deployable controller. Exact Q4
D1 routes and exact candidate replay are labels here; they are not runtime
inputs. No D2--D4 objective, VJP, model, or certificate was introduced.

## Protocol

The study used:

- 32 calibration and 96 untouched evaluation requests, deterministically split
  across eight domains;
- one model-output-independent position per request,
  `min(33, token_count - 2)`;
- injection layers 0, 1, 4, 6, 12 and 23;
- rates 360, 384, 725 and 749 pages per expert;
- live, frozen-set/live-weight and fully-frozen downstream route modes;
- five arms: independent PR13, strict frozen-candidate repair, nested local,
  nested D1-safe, and nested D1-safe with violation/mass severity;
- exact physical bit containment for every nested low-to-high chain;
- request-level inference after equal averaging over the six layers.

Calibration froze one parameter pair for each traffic pair without terminal
outcomes:

| Traffic chain | Repair window | Additive local eta |
| --- | ---: | ---: |
| 360 to 384 | 16 group pages | 0.001 |
| 725 to 749 | 16 group pages | 0.0005 |

The outcome phase executed 96 requests on an NVIDIA RTX PRO 6000 Blackwell
Server Edition. It finalized 34,560 quality rows, 1,082,880 propagation rows,
34,560 cache rows, 1,728 zero-dose rows and 11,520 route-mode contrasts. The
analysis used 10,000 paired request-cluster bootstrap resamples and 100,000
paired sign-flip draws where exact enumeration was not smaller. Holm adjustment
applied to the four primary sign-flip tests.

## Validation and reproducibility

All engineering gates passed:

- the sealed allocation manifest authenticated before any state was exposed;
- 3,456/3,456 declared physical nesting chains were literal subsets;
- all common-core and PR13-high reference identities matched;
- every local guardrail passed;
- no safe nested-local endpoint was made unsafe;
- all zero-dose routes, terminal outputs and hybrid cache commits were exact;
- stored route-mode contrasts reproduced exactly;
- a second complete analysis reproduced all 17 output files byte-for-byte.

The complete allocation manifest is preserved losslessly as
[d1_nested_evaluation_allocation_manifest.json.zst](../sealed/d1_nested_evaluation_allocation_manifest.json.zst).
Decompression must reproduce SHA-256
`1bdd2d1c047416ff4649d65306733af5a760ed7aa4295d293cd06c034679ca59`.
It contains every selected `[8,512]` physical state and complete move chain.
The original 424,570,119-byte JSON and multi-gigabyte candidate banks remain at
`/workspace/pr13_d1_nested_safe_oracle_20260825_v2` on network storage.

Two outcome invocations failed closed before any outcome cell was written. The
first found an over-strict hardware-dictionary comparison; the second found a
CPU/CUDA device mismatch in the first zero-dose proof. Both are disclosed and
pinned in the checked-in compatibility protocol. The final allocation code
identity remains `d301feff...fae08`; separately versioned outcome code is
`27068ba4...759b`. The compatibility change permits only four exact,
semantics-inert CPU-worker metadata fields and preserves every capture execution
field.

## Primary outcome

All four primary contrasts failed the joint nominal-bootstrap and Holm-adjusted
sign-flip gate:

| Candidate minus reference, live terminal KL | Mean | Nominal 95% CI | p95 regression | Maximum regression |
| --- | ---: | ---: | ---: | ---: |
| D1-safe 360 minus nested-local 360 | -0.0000435 | [-0.0002059, +0.0000598] | +0.0000121 | +0.001431 |
| D1-safe 725 minus nested-local 725 | +0.0000005 | [-0.0000924, +0.0000811] | +0.0000844 | +0.002025 |
| D1-safe 360 minus PR13 384, all-in matched | +0.0006062 | [+0.0001270, +0.0010995] | +0.005602 | +0.005602 |
| D1-safe 725 minus PR13 749, all-in matched | -0.0002304 | [-0.0006026, +0.0001673] | +0.002022 | +0.008712 |

The 360-page product comparison is significantly unfavorable, not merely
inconclusive. The 725-page product comparison has a favorable mean and median,
but its confidence interval crosses zero and its tail regressions violate the
predeclared limits. Holm-adjusted one-sided p-values were 1.0, 1.0, 1.0 and
0.496 respectively.

Therefore:

- the inferential gate failed;
- the p95-regression gate failed;
- the product single-request maximum-regression gate failed;
- Experiment B must not start.

## What D1 repair accomplished

Against the identical nested-local incumbent, the D1-safe arm changed only 67
of 2,304 rate/layer/request cells:

| Rate | Local crossings | D1-safe crossings | Prevented | Introduced | Changed-cell mean KL delta | Changed-cell favorable fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 360 | 63 | 50 | 13 | 0 | -0.001927 | 46.2% |
| 384 | 52 | 35 | 17 | 0 | -0.001928 | 70.6% |
| 725 | 48 | 29 | 19 | 0 | +0.000015 | 52.6% |
| 749 | 33 | 15 | 18 | 0 | -0.000362 | 66.7% |

This cleanly confirms D1 as a repair mechanism. It prevented 67 immediate
crossings, introduced none, never changed a safe incumbent into an unsafe one,
and obeyed every local guardrail.

It also shows why binary D1 is insufficient. At 360 pages the prevented-event
mean is favorable, but fewer than half of changed cells improve KL and the
maximum cell regression is `+0.00858`. At 725 pages the mean is neutral even
though all 19 changes prevent crossings. The ordered changed-cell decomposition
at 725 is:

- membership execution: `-0.000293` mean, favorable;
- router-weight term: `+0.000403`, unfavorable;
- fully-frozen continuous term: `-0.000094`, favorable;
- live total: `+0.000015`, neutral-to-unfavorable.

At the request level, D1-safe 725 reduced downstream membership-change fraction
by `0.000804` relative to nested local but changed mean KL by only
`+0.0000005`. Conversely, D1-safe 725 versus PR13 749 increased downstream
membership churn by `0.02138` while its mean KL was `-0.000230`. Route churn
and KL can therefore move in opposite directions.

The violation-depth/mass severity arm was byte-identical in terminal behavior
to the strict D1-safe arm at every rate. The available unsafe ties did not
produce a distinct final state, so this study supplies no evidence that the
severity ordering improves KL.

## Physical nesting and page non-monotonicity

Literal nesting removed wholesale de-refine/re-refine behavior, but did not
make KL monotonic:

| Arm | Rate increase | Mean high-minus-low KL | More-pages favorable | p95 regression | Maximum regression |
| --- | --- | ---: | ---: | ---: | ---: |
| Independent PR13 | 360 to 384 | +0.000382 | 30.2% | +0.002228 | +0.009881 |
| Independent PR13 | 725 to 749 | +0.000457 | 36.5% | +0.003780 | +0.010822 |
| Nested local | 360 to 384 | -0.000926 | 50.0% | +0.001756 | +0.012553 |
| Nested local | 725 to 749 | +0.000468 | 39.6% | +0.002875 | +0.019186 |
| Nested D1-safe | 360 to 384 | -0.000939 | 52.1% | +0.001102 | +0.012553 |
| Nested D1-safe | 725 to 749 | +0.000456 | 37.5% | +0.002875 | +0.017741 |

Nesting helped the lower-rate chain in its mean and p95, but the higher-rate
chain remained non-monotonic. This refines the earlier diagnosis:

> Non-nested global reallocation was a structural source of gratuitous residual
> rotation, but add-only pages can still rotate the residual, cross later
> thresholds and change continuous tail sensitivity. Nesting reduces one source
> of instability; it cannot guarantee terminal monotonicity.

The nested-local heuristic itself is not a quality solution. Relative to
same-cap independent PR13, its mean KL deltas were `+0.001032`, `-0.000276`,
`+0.000226` and `+0.000237` at 360, 384, 725 and 749 pages. Strict incumbent
repair remained conservative and nearly neutral, but did not establish a
product-level win.

## Attention-type localization

The 360-page product failure localizes strongly to the single full-attention
sentinel layer:

| Product contrast | Full-attention mean | Five linear-attention layers mean |
| --- | ---: | ---: |
| D1-safe 360 minus PR13 384 | +0.003755 | -0.0000235 |
| D1-safe 725 minus PR13 749 | +0.000716 | -0.000420 |

Because the six layers are equally weighted per request, the full-attention
`+0.003755` contribution explains almost all of the pooled 360 regression. This
is descriptive, not a multiplicity-adjusted subgroup claim, but it provides a
specific design target: the next continuous tail proxy must model full-attention
response rather than assuming a single layer-agnostic local geometry.

## Recommended remedy

Do not resume the sequential runtime controller with this objective. First run
a new Experiment A oracle with the architecture:

1. retain PR13, or a safely pruned PR13 subset, as the incumbent;
2. invoke bounded D1 repair only when the incumbent is robustly unsafe;
3. freeze the repaired state and permit additions only;
4. rank legal additions by local gain and a token-dependent continuous tail
   proxy under the hard D1 safety envelope;
5. stop early if no page improves the continuous proxy safely.

A suitable first proxy is a small fitted operator

\[
J_{\mathrm{tail}}(\delta_\ell)
=\|B_{\ell,a}\,\delta_\ell\|_2^2,
\]

where `a` denotes attention type. It should be low rank, calibrated only on the
calibration split, and frozen before held-out tails. The dynamic page effect
must remain activation-dependent:

\[
d_p(x)\approx U_\ell c_p(x),
\qquad
J_{\mathrm{tail}}\text{ sees }c_p(x),
\]

with static basis/geometry metadata separated from token-dependent gate/up/down
response synthesis. D1 remains a hard safety constraint and repair trigger,
not a global utility. This stays within the D1-only architecture: the new term
models continuous terminal sensitivity, not explicit D2--D4 router targets.

The next oracle should compare nested local, nested D1-safe, and nested
D1-safe-plus-tail at both product traffic points, with the same request split
and failure-tail gates. Only if it improves paired KL at both product points
without p95/worst-case regression should Experiment B resume.

## Runtime and cost boundary

The exact oracle is intentionally not deployable. Calibration consumed about
2.89 summed layer-hours and held-out allocation about 3.21 summed layer-hours
because it constructs candidate banks and executes exact D1 replay. This is
scientific label generation, not proposed decode work.

The pod's cgroup exposed 27.2 effective CPU cores despite a larger host core
count. A worker audit measured:

| Workers | Throughput, million iterations/s | Effective cores | Throttled aggregate seconds |
| ---: | ---: | ---: | ---: |
| 24 | 219.3 | 23.89 | 0 |
| 27 | 240.7 | 26.84 | 0.0009 |
| 32 | 239.9 | 27.13 | 14.09 |

Thus 27 workers was the measured optimum; further tuning was marginal and
32 workers only induced throttling.

Prior arithmetic estimates place a future scalar/one-adjoint D1 overlay at
roughly 1.34--1.82% of PR13 selector MACs, excluding launch latency, memory
traffic and overlap. No measured controller latency or predictor accuracy exists
because Experiment B remains paused. The present negative oracle result means
there is no cost-benefit case for implementing that controller yet.

## Artifact map

- [Promotion decision](d1_nested_experiment_a_promotion.json)
- [Primary inference](d1_nested_primary_inference.parquet)
- [Allocation side summary](d1_nested_allocation_side_summary.parquet)
- [Changed-cell decomposition](d1_nested_changed_cell_failure_decomposition.parquet)
- [Adjacent-rate monotonicity](d1_nested_adjacent_rate_monotonicity.parquet)
- [Cross-metric failure summary](d1_nested_primary_metric_failure_summary.parquet)
- [Attention decomposition](d1_nested_product_attention_decomposition.parquet)
- [Post-hoc reproducer](reproduce_d1_nested_failure_diagnostics.py)
- [Outcome run facts](../outcomes/d1_nested_outcome_run_facts.json)
- [Allocation seal](../sealed/d1_nested_evaluation_allocation_seal.json)
- [Complete compressed allocation manifest](../sealed/d1_nested_evaluation_allocation_manifest.json.zst)

The post-hoc failure tables are descriptive only and their facts file records
that they were not used for allocation, primary inference or promotion.
