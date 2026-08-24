# Exact-prefill, decode-only D1 route-objective expansion

Date: 2026-08-24

## Outcome

The exact-prefix cached slice result supports the D1 mechanism strongly enough
to proceed to downstream-tail KL on full-model hardware.

On held-out layers 1, 4, and 6, a fixed strict-D1 policy calibrated only on
layers 0, 12, and 23 changes the paired all-Q4 next-router top-8 set in 2/87
groups at 384 pages per expert, versus 9/87 for historical PR #13. At 749
pages it changes 1/87, versus 5/87 for PR #13. These are reductions of 77.8%
and 80.0%, respectively. Exact-combined local-qenergy optimization changes
10/87 and 3/87, so combined local error alone does not explain the D1 result.

The result does not establish lower terminal KL, lower perplexity, free-running
generation quality, an all-layer allocation, or a non-bottleneck runtime
controller. The three source sequences provide mechanism evidence only.

## Scientific boundary

The target product performs prefill with a separate exact mechanism.
Compression begins at the first generated decode token. The only allocator
lookahead objective is

\[
(t,\ell)\rightarrow(t,\ell+1),
\]

the same token's next-layer router. D2--D4 objectives, future-token margin
proxies, and compressed prefill are excluded. Cache drift into token \(t+1\)
is a later temporal outcome, not another optimization horizon.

This experiment uses existing all-Q4 captures as an exact-prefix surrogate.
For every admitted prompt position, earlier positions are reconstructed at the
all-Q4 endpoint and used only to populate the next layer's native cache. The
current position is then treated as one isolated decode token. A delta at one
position never affects another position. This preserves cheap use of the
existing three requests while testing the correct cached execution
architecture; it is not a generated-token rollout or a BF16 exact-prefill
handoff.

## Frozen experiment

The
[immutable configuration](../../../configs/qwen36_mxfp4_d1_exact_prefill_decode_slice_pilot_20260824_v2.json)
fully describes the executed choices.
The completed matrix is:

| Dimension | Frozen value |
| --- | --- |
| Injection layers | 0, 1, 4, 6, 12, 23 |
| Next-layer mixer paths | DeltaNet at 1, 2, 5, 13, 24; full attention at 7 |
| Mechanism page caps | 384 and 749 pages/expert |
| Requests | `mxfp4-confirm-006`, `-007`, `-008` |
| Positions | 29/layer; position 0 and final four positions excluded |
| D1 selected ranks | 6--8 |
| D1 outsiders | 9--16 |
| Boundary pairs | 24/group |
| Softplus temperature | 0.0625 |
| Local-guard eta grid | 0, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3 |
| Frontier search | Complete options, rate-specific column generation, 8 coordinate and 8 pair passes |
| Hardware | NVIDIA GeForce RTX 3090, CUDA 12.8, PyTorch 2.8.0 |

There are 12 cells and 348 unique layer/rate/group contexts. Exact policy
replay contains 3,132 rows; group allocations contain 2,784 rows; expert
allocations contain 22,272 rows. No test-split row was admitted.

The main comparison policies are:

- historical PR #13 router-square allocation;
- a regenerated PR #13 allocation from the same complete frontiers;
- exact-combined local-qenergy allocation;
- six strict one-sided D1 allocations, one for each eta;
- a per-token oracle that selects the best member of the six-eta D1 grid after
  exact replay.

The per-token oracle is an upper-bound diagnostic, not a runtime policy. It
sorts candidates by exact crossing, changed membership pairs, lost routing
mass, local qenergy, pages, and deterministic policy name.

## Cached decode execution

For each request and injection layer \(\ell\), the slice first reconstructs the
all-Q4 output of layer \(\ell\): historical routed output plus the exact shared
expert composition and decoder residual. It then runs prefix positions
\([0,t)\) through the native pre-MoE portion of layer \(\ell+1\) with a
Transformers `DynamicCache`.

The cache includes attention key/value tensors for a full-attention layer and
convolution/recurrent tensors for DeltaNet. Every baseline, VJP, and candidate
receives a deep copy whose tensor inventory and values equal the prefix
snapshot and whose tensor storage is disjoint. A one-token native cached
forward is run at absolute position \(t\). Candidate caches are outputs; no
candidate may mutate the shared exact prefix.

The initial layer-0 canary exposed a real implementation issue: native cached
DeltaNet commits its recurrent state in place before reverse-mode autograd
reads saved pre-commit values. The failed trace is retained in
[the evidence directory](../evidence/rate_384_layer00_attempt1_inplace_failure.log).
The final VJP path
uses PyTorch saved-tensor hooks to clone tensors at save time. This preserves
the native forward and its cache commits; it is neither a functional rewrite
nor a finite-difference approximation. A focused hardware diagnostic and all
348 finalized VJPs passed.

## D1 objective

For a complete allocation \(s\), the current routed-output perturbation uses
actual BF16 execution weights:

\[
\delta_\ell(s)=\sum_{e=1}^{8} a_e^{\mathrm{exec}}
\left[\widetilde y_e(s_e)-y_e^{(4)}\right].
\]

Each option delta is exact for the current token activation and its complete
512-entry expert state vector. Gate/up interactions therefore remain inside
the option; the oracle does not assume independent page effects.

For selected next-router expert \(i\), outsider \(j\), exact cached Q4 margin
\(m_{ij}\), and cache-conditioned sensitivity

\[
g_{ij}=\nabla_{h_\ell^+}(z_{\ell+1,i}-z_{\ell+1,j}),
\]

the predicted margin is

\[
\widetilde m_{ij}(s)=m_{ij}+g_{ij}^{T}\delta_\ell(s).
\]

The strict route loss is the one-sided directional objective

\[
L_{\mathrm{D1}}(s)=\sum_{i,j}
\operatorname{softplus}\left(-\widetilde m_{ij}(s)/0.0625\right).
\]

The allocator minimizes this combined 24-boundary loss subject to the exact
page cap and additive local guard

\[
J_{\mathrm{local}}(s)\leq
J_{\mathrm{local}}^\star+\eta J_{\mathrm{Q2}}.
\]

Local qenergy uses the historical FP32-normalized selector weights for PR #13
parity; causal D1 effects use BF16 execution weights. Coordinate and pair
updates maintain the combined signed boundary-effect vector, so cross-expert
cancellation and reinforcement are included.

## Fixed-policy calibration

Eta was not selected on the held-out layer results. Calibration used layers 0,
12, and 23, independently at each page cap. The deterministic selection order
was fewest exact D1 crossings, fewest changed membership pairs, least total
routing mass lost, lowest mean local qenergy, fewest pages, then policy name.
Layers 1, 4, and 6 were held out.

| Pages/expert | Selected eta | Calibration crossings | Held-out crossings |
| ---: | ---: | ---: | ---: |
| 384 | 0.001 | 2/87 | 2/87 |
| 749 | 0.00005 | 0/87 | 1/87 |

This is a layer-held-out check, not a request-held-out check: the same three
requests occur on both sides. A deployable eta still requires independent
generated requests.

## Exact D1 results

### Held-out layers

| Pages/expert | Policy | Crossings | Crossing rate | Lost routing mass, sum | Mean local qenergy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 384 | PR #13 | 9/87 | 10.345% | 0.130673 | 0.001181 |
| 384 | Exact-combined local | 10/87 | 11.494% | 0.144289 | 0.001160 |
| 384 | Fixed D1, eta 0.001 | **2/87** | **2.299%** | **0.023193** | 0.001270 |
| 749 | PR #13 | 5/87 | 5.747% | 0.066409 | 0.000149 |
| 749 | Exact-combined local | 3/87 | 3.448% | 0.036046 | 0.000146 |
| 749 | Fixed D1, eta 0.00005 | **1/87** | **1.149%** | **0.013019** | 0.000151 |

At 384 pages, fixed D1 accepts about 7.5% higher mean local qenergy than PR
#13 in exchange for a 77.8% reduction in crossings and an 82.3% reduction in
summed lost routing mass. At 749 pages, the local-qenergy increase is about
1.3%, while crossings fall 80.0% and lost routing mass falls 80.4%. This is
the intended trade: local damage remains bounded rather than serving as the
sole objective.

The fixed policy's held-out crossings by layer are:

| Pages/expert | Layer 1 | Layer 4 | Layer 6 |
| ---: | ---: | ---: | ---: |
| 384 | 1/29 | 1/29 | 0/29 |
| 749 | 0/29 | 0/29 | 1/29 |

### Six-layer eta sweep

| Pages/expert | Policy | Exact crossings | Mean local qenergy |
| ---: | --- | ---: | ---: |
| 384 | PR #13 | 13/174 | 0.001755 |
| 384 | Exact-combined local | 14/174 | 0.001723 |
| 384 | D1 eta 0.00005 | 9/174 | 0.001725 |
| 384 | D1 eta 0.00025 | 7/174 | 0.001749 |
| 384 | D1 eta 0.0005 | 5/174 | 0.001782 |
| 384 | D1 eta 0.001 | **4/174** | 0.001849 |
| 749 | PR #13 | 7/174 | 0.000226 |
| 749 | Exact-combined local | 4/174 | 0.000221 |
| 749 | D1 eta 0.00005 | 1/174 | 0.000227 |
| 749 | D1 eta 0.00025 | **0/174** | 0.000254 |

Eta 0.0005 and 0.001 also produce zero six-layer crossings at 749 with larger
local damage. The per-token six-eta oracle produces 3/174 crossings at 384 and
0/174 at 749. Every observed crossing in this run changes exactly one
membership pair.

These results answer the mechanism question: page choice can target the signed
next-router boundary more effectively than PR #13's independent router-square
score or exact-combined local qenergy at the same refinement-page cap.

## Hard validation gates

All aggregate checks are machine-recorded in
`d1_decode_analysis_facts.json`.

- All 12 cells contain exactly 29 groups and the frozen policy set.
- Repeated cached all-Q4 logits, top-8 IDs, and committed cache states are
  bit-identical for every group.
- All VJPs use the native cached forward with saved-tensor cloning.
- All allocations remain within the page budget.
- Every D1 local guard passes; maximum numerical excess is
  \(7.81\times10^{-18}\).
- The 2,784 regenerated same-host PR #13 expert allocations match the
  authoritative per-layer tables exactly in router rank, expert ID, selected
  pages, full 512-state vector, selector weight, and BF16 execution weight.
- Regenerated PR #13 combined deltas are bit-identical in 10/12 cells. Layer 0
  and layer 4 at 749 pages differ only by floating reconstruction roundoff;
  the global maximum absolute difference is \(4.55\times10^{-13}\).
- All source files and outputs are SHA-256 recorded; no test row is used.

The last distinction is intentional. The authoritative discrete allocations
are exact, while two independently reconstructed combined float32 deltas are
not falsely described as byte-identical.

## Cached versus no-cache diagnostic

Repeated cached execution is exact, but it is not interchangeable with the
old full-sequence no-cache slice:

| Injection layer | Next mixer | Cached/full top-8 set agreement | Maximum router-logit absolute difference |
| ---: | --- | ---: | ---: |
| 0 | DeltaNet | 100.00% | 0.0625 |
| 1 | DeltaNet | 93.10% | 0.0625 |
| 4 | DeltaNet | 100.00% | 0.0625 |
| 6 | Full attention | 89.66% | 0.0625 |
| 12 | DeltaNet | 93.10% | 0.0625 |
| 23 | DeltaNet | 100.00% | 0.0625 |

Rows are duplicated across rates; the unique position disagreements are 2/29
at layer 1, 3/29 at layer 6, and 2/29 at layer 12. This validates the product
pivot: a no-cache full-sequence router is not a safe decision baseline for
decode.

## Oracle and offline cost

The measured exact full-VJP medians are 45.7 ms/group for the layer-6
full-attention path and 66.2--67.3 ms/group for the DeltaNet paths. These are
offline label-generation costs and must never be interpreted as controller
latency.

Rate-independent complete-option geometry was cached with exact layer,
request/position, capture, tree, factor, configuration, and base-runner hash
identity. Each cache is about 4 GiB; five cached layers occupy 20 GiB on
network storage. Column generation remains rate-specific. Cache-created cells
have a 334.4 s median wall time and cache-loaded cells a 157.0 s median, a
roughly 2.1x offline speedup. The cache is not runtime metadata or traffic.

Three finalized cells predate this cache-only runner change: 384/layer 0,
384/layer 6, and 749/layer 0. Their runner SHA differs, while the decode core,
slice core, objective core, inputs, and scientific outputs retain their own
hashes. The exact pre-cache code snapshot is preserved on network storage.

## Projected runtime compute and bandwidth

Experiment A2 uses exact labels and adds no production sidecar to its reported
384/749 mechanism points. A future controller must pay for its predictor.
Existing accounting gives:

| Selector | MACs/routed eight-expert group | Relative to PR #13 |
| --- | ---: | ---: |
| PR #13 primary | 51,343,360 | 100% |
| Dense D1 candidate scorer | 9,781,248 | 19.05% |
| One-adjoint D1 scorer | 933,888 | 1.82% |

The one-adjoint estimate includes 81,920 MACs for token-dependent response
synthesis. It is the viable direction; the exact VJP is oracle-only.

Static and dynamic state must remain separate. Static sidecars may store page
geometry, low-rank bases, and projection structure, but the coefficient is
token-dependent:

\[
d_p(x)\approx U_\ell c_p(x),\qquad
\widehat s_{p,ij}(x)=\widehat{g_{ij}^{T}d_p(x)}.
\]

The runtime predictor must synthesize \(c_p(x)\) from the current layer input,
gate/up intermediate response, current D1 direction, and any learned
correction. This experiment uses exact complete-option deltas and does not
claim that predictor exists.

Metadata increases from PR #13's 72.125 KiB/group (0.023478 bpw) to 155.125
KiB/group for joint token response, or 167.625 KiB/group (0.054565 bpw) with
uncertainty. Across 40 routed layers that is approximately 2.82 MiB versus
6.55 MiB of selector metadata per generated token if all metadata is read
once. To keep total charged bandwidth fixed, the production comparison points
are 360 rather than 384 and 725 rather than 749 correction pages per expert.
The present 384/749 oracle points isolate allocation quality at equal
refinement-page traffic; they are not yet all-in metadata-matched controller
points.

The unfused device shortlist measured 0.831 ms median and 0.910 ms p90 per
refresh on the 3090. Six serialized refreshes would cost roughly 5 ms before
transfer stalls or provisional next-layer execution, which is a bottleneck.
The controller therefore needs one-adjoint scoring, few refreshes, batching or
fusion, and overlap with refinement transfer. Exact prefill removes selector
compute and page traffic for prompt tokens, but does not reduce any of these
per-generated-token costs.

## Accuracy and product implications of exact prefill

Exact prefill improves the scientific and product boundary in three ways:

1. prompt-token compression error is absent, so the decode allocator is not
   asked to compensate for a corrupted starting cache;
2. the D1 target is conditioned on the actual attention/DeltaNet prefix state;
3. online selector work and refinement traffic scale with generated tokens,
   not prompt plus generated tokens.

It also raises the implementation bar. A production handoff must export or
reproduce every attention K/V tensor, DeltaNet convolution and recurrent
state, absolute/cache position, mask and RoPE convention, dtype, and backend.
The first compressed token must match the exact-prefill model in current
hidden state, router logits, and terminal logits before any delta is applied.
If the production endpoint is BF16 rather than all-Q4, this experiment must be
rebased; an all-Q4/BF16 discrepancy cannot be assigned to D1.

## Limitations

- Only three requests and 29 isolated positions per layer are used.
- Positions are teacher-forced prompt tokens behind reconstructed caches, not
  free-running generated tokens.
- Only six injection layers and two mechanism page caps are evaluated.
- The paired endpoint is all-Q4, not a demonstrated production BF16 prefill
  handoff.
- No exact downstream-tail final-logit KL, NLL, final hidden MSE, or
  live-minus-frozen KL is measured here.
- No multi-token rollout measures temporal cache drift.
- No token-dependent signed-effect predictor, uncertainty model, initial
  \(S_0\), sequential certificate, or latency pipeline is evaluated.
- Swap severity is not used; strict membership is isolated first.
- The metadata-matched 360/725 points have not yet been run.

## Decision and next experiment

Promote the strict D1 mechanism to an exact-prefill cached downstream-tail
smoke, while keeping the runtime controller deferred.

The next run should use a full model on a PRO 6000-class host because the 3090
slice validates only two adjacent layers and cannot hold the full BF16 model
needed for an exact downstream tail. For each existing layer/rate cell it
should:

1. establish a true full-model exact-prefix cache and paired zero-dose parity;
2. inject one current token before shared-expert composition/residual addition;
3. propagate that token through layers \(\ell+1\ldots39\) with private cache
   clones;
4. compare PR #13, exact-combined local, calibration-fixed D1, and per-token D1
   oracle in live and fully frozen routing;
5. report current-token final-logit KL, delta NLL, final hidden MSE, downstream
   route churn, and live-minus-frozen KL;
6. add 360/725 only when controller metadata is charged.

Three requests are adequate for this smoke decision. Before a quality claim,
use at least 64 independent generated requests, preferably 128 or more. If
fixed D1 lowers final KL, proceed to temporal multi-token rollout and only then
train the token-dependent predictor. Predictor evaluation must report signed
effect correlation, sign accuracy, top-page ranking accuracy, corrected-logit
error, and uncertainty calibration before route certification is attempted.
