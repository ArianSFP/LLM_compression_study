# D1 single-injection downstream-tail KL methodology

## Question and scientific boundary

This experiment asks one narrow question:

> At the same refinement-page budget, do D1-aware allocations reduce exact
> next-layer route changes and terminal-logit damage relative to PR #13?

D1 is only the same token's next-layer router,
`(t, l) -> (t, l+1)`. One routed-MoE output is replaced at a time and the
candidate is then propagated through the complete native downstream tail. The
study does not apply D1 allocations jointly at all layers, train a runtime
predictor, start from an `S0` prefix, or certify a route during execution.

The frozen smoke cohort has three requests and six injection layers. It can
separate mechanisms and catch implementation errors, but it cannot support a
general terminal-quality claim. The immutable gate requires at least 64
independent requests, with 128 preferred, before such a claim.

## Frozen experiment contracts

The promoted layer-slice mechanism result is described exactly by
[the immutable expansion config](configs/qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1.json),
whose SHA-256 is
`de00bd6ada308a7693fc7816fdee3e1e6de1b2d6206a26a7f1dd61a094d0adfd`.
It freezes:

- layers 0, 1, 4, 6, 12, and 23;
- page caps 384 and 749 pages/expert;
- requests `mxfp4-confirm-006`, `007`, and `008`;
- the final-four-position exclusion, leaving 11, 9, and 12 positions;
- selected ranks 6--8 against outsider ranks 9--16;
- softplus temperature 0.0625;
- eta values 0, 0.00005, 0.0001, 0.00025, 0.0005, and 0.001;
- eight coordinate sweeps and eight pair passes;
- exact-repair limits of two coordinate sweeps, two pair passes, and 10,000
  pair evaluations; and
- the RTX 3090 software and model-slice execution path.

The generic all-layer config remains a design envelope and is not evidence for
the completed expansion. The standalone validator checks the complete 12-cell
raw grid, sidecar hashes, request order and masks, candidate width, eta,
temperature, frontier mode, and GPU identity.

[The downstream-tail config](configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json)
cryptographically binds that expansion config and freezes the tail comparison,
quality gate, hardware path, and artifact locations. Its SHA-256 is
`87a272351f6b87ca4e838730bfcf1103432755cc08865d874fa03b581ce621e0`.

## Budget points and accounting

The mechanism comparison retains the original 384- and 749-page caps. Two
additional points charge the measured D1 sidecar against refinement traffic:

| Allocated refinement pages/expert | All-in traffic equivalent | Charged bpw | Meaning |
| ---: | ---: | ---: | --- |
| 360 | 384 | 0.5234781901041666 | joint signed-effect and uncertainty metadata charged |
| 384 | 384 | 0.5234781901041666 | oracle mechanism point; metadata unadjusted |
| 725 | 749 | 0.9987386067708334 | joint signed-effect and uncertainty metadata charged |
| 749 | 749 | 0.9987386067708334 | oracle mechanism point; metadata unadjusted |

The 24-page reserve is the conservative uncertainty-bearing sidecar cost from
the measured runtime accounting. At 360 and 725, the PR #13 comparator is
regenerated with the unchanged router-square algorithm because no historical
PR #13 table exists at those non-principal caps. It must be labelled
`regenerated_same_algorithm_nonhistorical_cap`, not historical PR #13.

## Compared allocations

Every layer/rate cell contains the same five policy tiers:

1. `pr13_router_square`: preserved historical PR #13 at 384/749, or the same
   algorithm regenerated at 360/725.
2. `exact_combined_local`: the complete-option oracle minimizing the combined
   eight-expert local qenergy.
3. `calibration_selected_fixed_d1`: one complete eta policy per rate, selected
   on layers 0, 12, and 23 by crossings, changed membership pairs, lost routing
   mass, local qenergy, pages, and deterministic name, in that order.
4. `exact_request_reranked_d1`: the best complete eta policy selected
   independently for each exact request. This is an oracle upper bound.
5. `exact_request_repaired_d1`: request-level coordinate/pair repair over
   complete D1 finalist allocations. This is a stronger oracle upper bound and
   is never a runtime input.

Layers 1, 4, and 6 form the held-out architecture slice for the fixed-eta
policy. Request reranking and repair use exact labels and therefore are not
held-out deployable policies.

## Exact tail execution

The full dequantized BF16 checkpoint is loaded on one RTX PRO 6000 96 GB GPU
with eager attention and eager experts. For each request, two complete Q4
baseline captures must be bit-identical. Fresh residual, pre-MoE input, routed
output, input IDs, and attention mask must also exactly match the retained
same-host source capture.

For injection layer `l`, the stored policy delta is

```text
delta_l = approximate routed output - exact Q4 routed output.
```

The runner replaces the true routed-MoE output before shared-expert
composition and the decoder residual addition, using alpha 1. It executes
native layers `l+1..39`, final normalization, and the LM head. Each policy is
run in two downstream modes:

- `live`: all downstream routes are recomputed normally;
- `fully_frozen`: downstream expert IDs and BF16 execution weights are taken
  from the exact baseline.

A zero-delta replay is performed for every request, layer, and mode. Hidden
states, router logits, and terminal logits must all reproduce Q4 with maximum
absolute difference zero.

Allocation reconstruction, exact D1 VJPs, reranking, and repair were performed
on the frozen RTX 3090 layer-slice path. The PRO 6000 tail measures D1 again on
the execution host. Cross-device policy-transfer loss is therefore observable
in the result and is not mistaken for a freshly PRO-calibrated oracle.

## Metrics

For reference distribution `p_Q4` and candidate distribution `p_s`, terminal
KL is the mean next-token categorical divergence

```text
D_KL(p_Q4 || p_s).
```

`delta_nll` is candidate NLL minus Q4 NLL on the observed next-token labels;
`ppl_ratio = exp(delta_nll)`. Final hidden MSE is evaluated over every token
and hidden coordinate at the output of layer 39.

Each tail row also retains the source allocation's exact local-qenergy damage,
the mean squared stored routed-output delta, and the realized injected-position
layer-output MSE after the scientific BF16 composition. Thus a lower-KL policy
cannot silently evade the predeclared local guardrail in the final tables.

At every downstream layer the analysis records ordered-route agreement,
top-8 membership change fraction, top-1 change fraction, normalized routed
probability-mass churn, and hidden MSE. D1 is also evaluated only on the 32
admitted source positions. The first changed layer is the earliest downstream
layer with nonzero admitted-position membership churn.

For each otherwise identical row,

```text
live-minus-frozen KL = live KL - fully-frozen KL.
```

This paired difference is the operational measure of amplification associated
with allowing discrete downstream routes to change. It is descriptive rather
than a formal causal decomposition of all nonlinear differences.

## Expected artifact grid and hard gates

The four rates, six layers, three requests, five policies, and two route modes
produce 720 terminal-quality rows. The variable-length tails produce 22,560
per-layer propagation rows. There are 144 zero-dose control replays in cell
sidecars. Each of the 24 layer/rate cells is written atomically with hashes and
can be resumed only if its complete sidecar and both tables agree.

Finalization rejects:

- a missing or extra allocation cell;
- an allocation or repair hash mismatch;
- a changed immutable expansion config;
- a changed checkpoint index;
- non-identical repeated baselines or source captures;
- any nonzero zero-dose hidden/router/logit difference;
- incomplete policy, request, route-mode, or layer grids; and
- non-finite quality or propagation metrics.

No test rows are admitted to calibration or policy selection.

## Decisive smoke interpretation

The predeclared reading is:

- repair and fixed D1 both lower live KL: expand requests before an all-layer
  study;
- repair lowers live KL but fixed D1 does not: the mechanism is positive but
  deployable policy prediction or selection is the bottleneck;
- neither lowers live KL: strict immediate membership is not a sufficient
  quality surrogate; test routing-mass and functional swap severity; or
- exact local matches D1: combined local error may explain the gain more than
  router direction.

The same comparisons are reported separately on calibration layers 0/12/23
and held-out layers 1/4/6. Means over this three-request cohort are smoke
descriptors, not confidence intervals.

## Reproduction

After the matched 360/725 allocation and repair cells exist on network storage:

```bash
PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase validate \
  --config configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json

PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase run \
  --config configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint

PYTHONPATH=src python scripts/run_d1_downstream_tail_kl.py \
  --phase finalize \
  --config configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json

PYTHONPATH=src python scripts/analyze_d1_downstream_tail_kl.py \
  --config configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json \
  --input /workspace/pr13_d1_downstream_tail_kl_smoke_20260824_v1 \
  --output /workspace/pr13_d1_downstream_tail_kl_smoke_20260824_v1/analysis
```

The raw root is
`/workspace/pr13_d1_downstream_tail_kl_smoke_20260824_v1`. The repository
package should contain the compact tables, reports, facts, hashes, configs, and
an exact code snapshot; complete raw deltas and per-cell material remain on
network storage.
