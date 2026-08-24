# Exact-prefill, decode-only D1 protocol

## Product boundary

Prefill is supplied by a separate exact mechanism. Compression begins only
when processing a generated token. The optimization target remains D1:

\[
(t,\ell)\rightarrow(t,\ell+1),
\]

the same token's next-layer router. D2--D4 objectives remain excluded. A
future token can inherit a changed attention or DeltaNet cache, but that is a
temporal decode outcome to measure later, not another lookahead objective.

This boundary supersedes running the v1 no-cache full-sequence downstream-tail
smoke as the next product-facing experiment. Its implementation and artifacts
remain immutable provenance; they do not describe the target decode path.

## Experiment A2: cached single-token mechanism pilot

The first decode experiment separates objective validity from a deployable
controller. There is no `S0`, speculative pass, omitted-tail predictor, or
certificate loop.

For each admitted token and injected layer:

1. Reconstruct the all-Q4 current-layer output for every earlier token.
2. Run only that exact prefix through the native pre-MoE portion of layer
   `l+1` with `use_cache=True`.
3. Deep-clone the resulting cache for every baseline, gradient, and candidate.
4. Run one current token through layer `l+1` from the cloned cache.
5. Compute cache-conditioned D1 VJPs and choose complete frontier options.
6. Replay each final allocation once from an independent clone and compare its
   current-token router result with the paired cached all-Q4 baseline.

DeltaNet convolution and recurrent tensors are mutated during cached decode;
full-attention key/value tensors are appended. Cache cloning and non-aliasing
are therefore correctness requirements, not performance conveniences.
Repeated cached baselines, selected IDs, and final cache tensors must be
bit-identical. Cache differences between all-Q4 and a candidate are measured
as causal outputs. A candidate is never allowed to alter the exact prefix.

The native single-token DeltaNet forward also commits cache tensors in place
before reverse-mode autograd reads them. The VJP path retains private clones
through PyTorch saved-tensor hooks. This leaves the native cached forward
unchanged while preserving its pre-commit values for backward; it does not use
a functional or finite-difference approximation.

The source data are existing teacher-forced validation tokens. Each position
is isolated behind an exact reconstructed prefix and receives only its own
delta. This tests cached decode architecture cheaply, but it is not yet a
free-running generated-token rollout.

Complete coarse option geometries are independent of the page cap. Expansion
runs may reuse a pickle cache only after exact equality of layer, ordered
request/position groups, capture hashes, tree hash, factor hash, PR #13 config
hash, and base frontier-runner hash. Column generation remains rate-specific.
The cache changes neither allocation inputs nor reported bandwidth; it avoids
reconstructing the same oracle options twice.

## Reference endpoint

The available frontier states and captures define all-Q4 as the pilot's exact
endpoint. If the production prefill mechanism is BF16, its caches are a
different reference state. A later full-model decode run must capture or
export the actual handoff state and re-establish parity for:

- attention K/V tensors;
- DeltaNet convolution and recurrent tensors;
- absolute position, cache position, masks, position IDs, and RoPE;
- dtype, attention backend, and expert backend; and
- current hidden state, router logits, and terminal logits.

No all-Q4-vs-BF16 discrepancy may be attributed to the allocator.

## Frozen pilot matrix

The immutable configuration is
[qwen36_mxfp4_d1_exact_prefill_decode_slice_pilot_20260824_v2.json](configs/qwen36_mxfp4_d1_exact_prefill_decode_slice_pilot_20260824_v2.json).
The canary uses layers 0 and 6, which exercise cached DeltaNet and cached full
attention respectively. Promotion may expand to layers 0, 1, 4, 6, 12, and
23. The mechanism rates are 384 and 749 pages per expert; runtime-metadata
matched points remain 360 and 725 for a later predictor/controller study.

For the three existing requests, positions zero are excluded because they
have no prefix. The remaining mask contains 29 isolated groups per layer.
Strict selected ranks 6--8 are compared with outsider ranks 9--16. The exact
one-sided D1 loss uses temperature 0.0625 and the additive local guard grid
`0, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3`.

The required comparisons at a fixed page cap are historical PR #13,
exact-combined local qenergy, fixed strict-D1 allocations, and the per-token
exact D1 rerank upper bound. Strict membership is the primary mechanism;
routing-mass and functional-swap severity remain later ablations.

## Accuracy gates

The pilot is viable only when:

- repeated cached Q4 baselines and their cache tensors are bit-identical;
- exact-prefix cache clones are initially equal and storage-disjoint;
- all policies use the same prefix state and a one-token query;
- historical states, page counts, and reconstructed deltas retain their
  existing validation gates; and
- strict D1 improves exact cached next-router membership at matched page cap
  without violating its declared local-qenergy guard.

Cached and no-cache full-sequence router values are both reported, but their
difference is diagnostic. Cached decode is the decision baseline because it
is the product execution path.

## Compute and bandwidth boundary

Experiment A2 is an oracle and may use exact per-token VJPs; its wall time is
not a runtime claim. It preserves the page traffic of each PR #13 mechanism
point and adds no runtime metadata to those points.

The later controller must remain within the existing matched accounting. The
current one-adjoint design estimate is 933,888 selector MACs per routed
eight-expert group, 1.82% of PR #13's 51,343,360 selector MACs. Its joint
token-response and uncertainty sidecar is 167.625 KiB/group (0.054565 bpw),
leaving 360 and 725 correction pages per expert at the all-in traffic of the
384 and 749 PR #13 points. The measured unfused device shortlist was about
0.831 ms per refresh on the 3090; six serialized refreshes would still be a
bottleneck. Predictor fusion, fewer refreshes, transfer overlap, and actual
decode latency remain mandatory gates.

Exact prefill removes compressed-prefill page traffic and selector compute
from the online objective. It does not make decode metadata free, nor remove
the cost of one current-token next-layer calculation. The sequential runtime
controller is attempted only after this cached oracle establishes the D1
mechanism.

## Promotion sequence

1. Run the two-layer, one-rate canary and audit cache parity.
2. Add the second mechanism rate if the policy behavior is sound.
3. Expand the cached slice to the six sentinel layers.
4. Evaluate on independent generated decode requests with a true exact-prefill
   cache handoff.
5. Propagate isolated injections through the cached downstream tail and report
   current-token KL, hidden error, and route churn.
6. Run multi-token generation to measure temporal cache drift.
7. Only then train and benchmark the token-dependent signed-effect predictor,
   uncertainty calibration, and sequential route certificate.

The central success criterion remains lower live route churn and final KL at
the same all-in decode bandwidth, with local qenergy retained as a guardrail.

## Completed expansion

The promoted six-layer/two-rate slice is complete. A calibration-selected
strict D1 policy reduced held-out exact next-router top-8 crossings from 9/87
to 2/87 at 384 pages/expert and from 5/87 to 1/87 at 749 pages/expert. The
complete methodology, hard gates, compute/bandwidth accounting, raw cell
artifacts, and limitations are in the
[exact-prefill decode result package](results/qwen36_mxfp4_d1_exact_prefill_decode_slice_expansion_20260824_v2/README.md).
Terminal KL remains the next full-model hardware gate.
