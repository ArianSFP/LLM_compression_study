# D1 cached-decode downstream-tail KL methodology

## Decision and replacement of the old tail stage

This is the product-facing successor to the no-cache full-sequence tail design.
The old implementation is retained as
D1_FULL_SEQUENCE_TAIL_KL_LEGACY_METHODS_20260824.md and its script and config
are explicitly named legacy. It must not be cited as decode evidence because
it injects all admitted positions together, has no prefix cache, and averages
quality across an entire request.

The replacement asks:

> At identical refinement traffic, does a D1-aware allocation for one current
> decode token reduce downstream route amplification and terminal-logit damage
> relative to PR #13?

D1 means only the same generated token's next-layer router:
(t, l) -> (t, l+1). No D2-D4 objective, model, or evaluation is present.

## Exact-prefix, isolated-token contract

The product supplies prefill by a separate exact mechanism. The smoke study
models that contract with native exact-Q4 cached execution:

1. Execute token zero exactly to create the complete hybrid cache.
2. For each admitted absolute position, retain the exact cache before the
   current token.
3. Deep-clone that full cache for every baseline and candidate.
4. Execute the current token exactly through layers 0 through l-1.
5. At layer l, replace only that token's routed-expert output before shared
   expert composition and the decoder residual addition.
6. Continue the same token natively through layers l+1 through 39.
7. Score only that token's terminal logits against its one observed next-token
   label.
8. Retain the private post-token cache and audit all attention K/V, DeltaNet
   convolution, and DeltaNet recurrent-state commits.
9. Use the exact baseline post-token cache, never a candidate cache, to advance
   to the next absolute position.

No helper in the new policy module constructs a sequence-shaped delta. The
29 admitted positions are logically independent candidates, even when
execution is scheduled sequentially for efficiency.

## Token-dependent execution of selected refinements

A stored slice vector is not treated as a static page effect. Refinement
effects depend on the live token activation:

d_p(x) approximately equals U_l c_p(x).

The 29-token packages preserve expert IDs and complete 512-state allocations.
On the PRO 6000, the runner decodes the selected experts once per layer and
reevaluates every complete selected state on the current full-model cached
activation. It uses the actual BF16 execution weights emitted by the live
router. The injected delta is therefore:

delta_l(x) = selected approximate routed output at x
             minus exact Q4 routed output at x.

The stored 3090 slice delta is retained only as a transfer diagnostic. Each
quality row reports stored-to-live MSE and cosine similarity. The full cached
baseline top-8 set must equal the allocation package's top-8 set; otherwise
the selected states are not defined for the new route and execution stops
rather than silently assigning states to a different expert.

The reconstructed all-Q4 routed output is compared with the native BF16
experts output. The immutable tolerance is 0.125 maximum absolute error,
covering only BF16/dequantization execution-order differences. The actual
replacement is formed from the native repeated baseline plus the reconstructed
approximate-minus-Q4 correction.

## Frozen allocation inputs

The immutable config is
configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json.

It freezes:

- injection layers 0, 1, 4, 6, 12, and 23;
- page caps 360, 384, 725, and 749 pages per expert;
- requests mxfp4-confirm-006, 007, and 008;
- 10, 8, and 11 isolated positions, respectively;
- one current query token with at least one exact prefix token;
- live and fully-frozen downstream execution;
- five allocation policies;
- complete native cache retention;
- exact zero-dose gates; and
- a single dequantized-BF16 RTX PRO 6000 96 GB execution path.

The 384 and 749 cells come from the promoted exact-prefill decode package.
The 360 and 725 cells were generated on the same RTX 3090 path with the same
six layers, three requests, 29-position mask, temperature 0.0625, six eta
values, column-generated complete frontiers, and no historical-table claim.
Their immutable network root is:

/workspace/pr13_d1_decode_matched_rates_20260824_v2

Its 132-file manifest SHA-256 is:

650fcffbd1e07f859764ab76003fc22d655f338e595f97da04097bcf1c3dc3a2

## V3 route-transfer and same-host repair

The original allocation packages were produced on an RTX 3090. Before using
them for PRO 6000 tail replay, v3 audits the routed expert identity for all 174
layer/token groups. Twelve groups in layers 1, 6, 12, and 23 changed top-8
membership on the target host. Their complete allocation banks were therefore
regenerated on the same PRO 6000 trajectory; no allocation state vector was
silently reassigned to a different expert.

The repair capture advances every request token sequentially through the exact
full model and stores current-layer hidden state, post-mixer activation,
residual, native routed output, current router, and D1 router. Recomposition of
the current hidden state is bit-identical for every repaired group.

The next-layer slice uses the exact cached prefix and exact current hidden.
Raw slice logits can differ from the full-model capture by BF16 execution-order
rounding (observed maximum 0.03125, frozen gate 0.0625), while the ordered top-8
route is exact. Candidate logits are therefore anchored as:

    exact full-model baseline logits
    + (same-host slice candidate logits - same-host slice baseline logits).

This retains the exact full-model baseline and uses only the candidate change
from the differentiable slice. It is a bounded VJP approximation, not an exact
full-model VJP. Exact complete-option replay then chooses the diagnostic token
oracle among the frozen eta candidates.

Only four layer-0 cells are reused from v2. The route-transfer audit has zero
set and order mismatches at layer 0, those cells completed before the first v2
failure, and v3 verifies each cell-facts hash before copying. Every other cell
is executed anew. The immutable v3 config freezes the audit, capture, patch,
and reused-cell hashes.

The complete raw evidence and code snapshot remain on network storage at:

    /workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3
    /workspace/pr13_d1_runtime_controller_20260824_v1

The checked-in compact package contains the finalized tables, analysis,
transfer audit, same-host repair tables and deltas, capture tensors, and all
hash manifests needed to audit them.

## Page accounting

| Allocated pages/expert | All-in traffic equivalent | Charged bpw | Role |
| ---: | ---: | ---: | --- |
| 360 | 384 | 0.5234781901041666 | metadata-matched D1 point |
| 384 | 384 | 0.5234781901041666 | unadjusted oracle mechanism point |
| 725 | 749 | 0.9987386067708334 | metadata-matched D1 point |
| 749 | 749 | 0.9987386067708334 | unadjusted oracle mechanism point |

The 24-page reserve is the measured conservative joint signed-effect and
uncertainty sidecar charge. PR #13 at 360/725 uses the same router-square
algorithm but is explicitly labelled regenerated and nonhistorical.

## Five policy tiers

Every layer/rate/token cell carries:

1. pr13_router_square.
2. exact_combined_local.
3. calibration_selected_fixed_d1.
4. frontier_companion_d1.
5. exact_token_oracle.

The fixed eta is selected only on layers 0, 12, and 23. Layers 1, 4, and 6
remain held out. The frozen mappings are:

| Pages | Fixed eta policy | Companion eta policy | Companion purpose |
| ---: | --- | --- | --- |
| 360 | 0.001 | 0.0005 | adjacent, lower local damage |
| 384 | 0.001 | 0.0005 | adjacent, lower local damage |
| 725 | 0.0001 | 0.0005 | zero-crossing six-layer frontier |
| 749 | 0.00005 | 0.00025 | zero-crossing six-layer frontier |

The exact token oracle chooses among complete eta allocations with exact 3090
slice D1 labels. It is a transfer diagnostic and immediate-route upper bound
on that source execution; it is not necessarily a terminal-KL upper bound and
is not a runtime input.

## Matched-rate mechanism evidence

Calibration on layers 0/12/23 selected eta 0.001 at 360 pages: 2/87 D1
crossings. Across all six layers it changed 6/174 routes, versus 7/174 for
eta 0.0005.

Calibration selected eta 0.0001 at 725 pages: 0/87 crossings. Across all six
layers it changed 1/174 routes. The companion eta 0.0005 changed 0/174 while
accepting higher local qenergy.

These are mechanism results only. They do not establish terminal quality.

## Live and fully-frozen paths

Live mode recomputes all downstream routes normally. Fully-frozen mode retains
the live router logits for observation but replaces executed expert IDs and
BF16 weights at layers l+1 through 39 with the paired exact baseline values.

For otherwise identical token/layer/rate/policy rows:

live-minus-frozen KL = live KL minus fully-frozen KL.

A lower live KL and a lower live-minus-frozen KL at matched traffic are the
decisive signs that the D1 allocation reduces discrete routing amplification.

## Metrics

Each terminal row records:

- Q4-to-candidate current-token categorical KL;
- current-token delta NLL and perplexity ratio;
- final hidden MSE;
- realized injected-layer hidden MSE;
- source and live local qenergy;
- source stored-delta versus live reconstructed-delta MSE and cosine;
- first downstream membership-change layer;
- full post-token cache MSE and maximum error by state class; and
- paired live-minus-frozen KL.

Every downstream layer records ordered-route agreement, top-8 membership
change, top-1 change, routed probability-mass churn, hidden MSE, and that
layer's post-token cache error.

## Logical grid and offline execution cost

The five-policy logical quality grid is:

6 layers x 4 rates x 29 positions x 5 policies x 2 route modes = 6,960 rows.

A four-policy grid would be 5,568 rows, not 6,960. The fifth exact-token-oracle
diagnostic is what brings this experiment to 6,960.

Variable downstream tails produce 218,080 propagation rows. The immutable
logical zero-dose grid has 1,392 rows.

The current conservative implementation uses 9,816 full-model current-token
forwards: 6,960 policy executions, 1,392 zero-dose executions, and 1,464
prefix/baseline/repeat executions. Expert tensors are decoded once per layer
and reused across four rates. This is an offline oracle cost, not the proposed
runtime controller cost. Rate-deduplicating zero-dose and repeated baselines
can reduce study cost later without changing its logical grid.

The deployed allocator remains a separate Experiment B. It must predict
signed page effects and uncertainty cheaply, batch metadata scoring, and stop
when route risk is certified. No runtime-latency claim is made from this tail
runner.

## Hard gates

Finalization rejects:

- any non-bit-identical repeated cached baseline;
- any nonzero zero-dose current hidden, downstream router, terminal-logit, or
  full-cache difference;
- a candidate cache reused as another candidate's prefix;
- a sequence-shaped or cross-position perturbation;
- a source/live top-8 set mismatch at the injection layer;
- an invalid or missing 512-state vector;
- a changed artifact hash, config, checkpoint, or selected-tree file;
- a live all-Q4 reconstruction exceeding its frozen BF16 tolerance;
- an incomplete five-policy, four-rate, two-route-mode grid; or
- a null or non-finite final metric.

## Interpretation

- Fixed D1 and the exact token oracle lower live KL: expand to at least 64
  independent requests before any quality claim.
- Token oracle lowers KL but fixed D1 does not: prediction or fixed-policy
  selection is the bottleneck.
- Exact local matches D1: combined local error, not router direction, explains
  most of the gain.
- No D1 policy lowers KL: strict membership is insufficient; test routing-mass
  and functional-swap severity.
- A companion eta beats the crossing-minimal eta: terminal KL prefers a
  different route/local-error frontier point.

The three current requests are a smoke cohort only. A generated multi-token
rollout, cache drift across future decode steps, joint all-layer allocation,
and the sequential runtime certificate remain later experiments.

## Completed smoke result and promotion decision

The finalized v3 grid contains 6,960 quality rows, 218,080 propagation rows,
and 1,392 logical zero-dose rows. All 1,392 zero-dose rows are exact in current
hidden state, every downstream router, terminal logits, and the complete
post-token hybrid cache.

For the calibration-selected fixed D1 policy, paired mean live-KL changes
relative to PR #13 are:

| Pages/expert | Mean live KL change | Live-minus-frozen KL change | D1 crossing-rate change | All three improve |
| ---: | ---: | ---: | ---: | --- |
| 360 | -0.0004336814 | -0.0003696597 | -0.0057471264 | yes |
| 384 | +0.0001163046 | +0.0001291450 | +0.0287356322 | no |
| 725 | -0.0005606990 | -0.0006329138 | +0.0057471264 | no |
| 749 | +0.0004853722 | +0.0004550884 | +0.0057471264 | no |

The causal direction is nevertheless visible conditionally. Among fixed-policy
tokens, 22 prevented PR13 D1 crossings have mean KL change -0.0014305334,
whereas 28 newly introduced crossings have mean KL change +0.0009797566.
This supports discrete membership as an amplifier and identifies policy
selection/transfer as a bottleneck.

It does not establish the proposed objective as a reliable allocator. Fixed
D1 is non-monotonic by rate, the source-slice exact-token diagnostic is also
non-monotonic, and exact-combined local is at least as competitive. Even a
post-hoc exact full-model D1 selector over the five executed allocations
improves KL at 360 and 725 pages but regresses at 384 and 749 pages.

The immutable promotion status is therefore:

> **Experiment B not promoted; paused after Experiment A.**

Before promotion, a same-host full-model D1 rerank or repair must lower both
live KL and live-minus-frozen KL at matched rates and separate route-direction
benefit from combined-local benefit. The three requests remain smoke evidence;
no terminal-quality claim is made.

## Reproduction

Validation needs no full model and succeeds on the current 3090:

~~~bash
PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase validate \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json
~~~

The quality phase requires the PRO 6000:

~~~bash
PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase run \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint \
  --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
  --fit-dir /workspace/pr13_average_rate_all_layers_20260822_v1/results/fit
~~~

After all 24 cells complete:

~~~bash
PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase finalize \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json

PYTHONPATH=src python scripts/analyze_d1_downstream_tail_kl.py \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json \
  --input /workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3 \
  --output /workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3/analysis
~~~

The 3090 is sufficient for allocation generation, artifact validation, and
unit tests. It cannot host the full dequantized BF16 model plus native hybrid
caches and live expert reconstruction; the terminal-quality phase therefore
remains intentionally blocked on a PRO 6000.
