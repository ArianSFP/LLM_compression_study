# D1 Experiment A allocator-diagnosis methodology

## 1. Status and scientific boundary

This document freezes the methodology for diagnosing the paradoxical behavior
of the PR #13 D1-aware allocator before any sequential runtime-controller work
begins. The study remains **Experiment A only**.

D1 has exactly one meaning throughout:

\[
(t,\ell)\rightarrow(t,\ell+1),
\]

the same decode token's next-layer router. There is no D2--D4 allocation
target, predictor, loss, VJP, or controller in this study. Downstream layers
are executed only to observe the causal tail and terminal distribution after a
single D1-scoped allocation has been injected.

Terminal logit KL is an outcome. It is never an allocation score, remedy
selector input, or runtime feature. The only exception is deliberate
case-control enrichment of the 24-case diagnostic panel: already-finalized KL
outcomes are used to locate strong paradox cases for explanation. That use
does not choose a refinement allocation and makes the panel unsuitable for
estimating population prevalence or mean quality.

The study contains:

- no initial `S0` refinement state;
- no omitted-tail signed-effect predictor;
- no uncertainty certificate or stopping loop;
- no provisional or speculative production pass;
- no generated multi-token rollout;
- no joint all-layer compression run; and
- no Experiment B result or claim.

Only three requests are available. Every quality statement is therefore a
smoke or mechanism statement, not a model-quality conclusion.

The numeric panel findings are intentionally not recorded in this methods
file. After finalization, they belong in:

`results/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1/analysis/D1_EXPERIMENT_A_ALLOCATOR_DIAGNOSIS_REPORT.md`

## 2. Questions

The experiment is designed to answer six narrow questions.

1. Why can removing a D1 top-8 crossing help terminal KL while a globally
   crossing-minimal allocation still underperform PR #13?
2. Why can a larger page cap reduce local qenergy yet increase terminal KL?
3. How much terminal damage is associated with discrete membership execution,
   ordinary router-weight drift inside a fixed expert set, and the continuous
   fully-frozen path?
4. Along the adjacent-rate fixed-D1 path, are reversals explained mainly by
   perturbation norm, perturbation direction, BF16 quantization, or a change in
   the next-router boundary?
5. Does a conservative repair of an unsafe PR #13 incumbent behave better
   than globally replacing PR #13 with the crossing-minimal candidate?
6. Can the resulting target retain a selector-compute and metadata budget that
   is small relative to the current PR #13 allocator?

The study does not assume that all top-8 swaps have equal functional cost. It
first identifies what strict membership, routed mass, margin movement, local
damage, and error-vector direction can and cannot explain. Functional swap
severity remains a later target ablation unless an explicitly executed table
says otherwise.

## 3. Provenance and immutable inputs

### 3.1 Model and parent studies

The allocation foundation is the all-40-layer PR #13 result at commit
`089deb4bd41eb71864779ced0cbb3db41e9dcb63`. The same-host causal-control
foundation is commit `64e94ace22f42734941bb3b4fd6702b8ff380295`.

The diagnostic panel consumes the finalized exact-prefix cached-decode tail
package, not the legacy no-cache full-sequence experiment:

`/workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3`

The corresponding checked-in compact package is:

`results/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3/`

The frozen parent artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| Base v3 config | `4fb569a6c0a0b7cde5a3267bb8571bedd0e82e5a690e8bac452bfb51e41a6e4a` |
| Final quality table | `ae626e499721639a2cd426ae452eeb2e3634787d1ae84c2a58ec2ab9d7e8b39d` |
| Final propagation table | `1058848b2985af307dfa6a0a54b1fca85568752aa648fb15260ec32aea03a984` |
| Final run facts | `7e68e32975e91bc8516b944294548716f35c1e4fc3cd60fbe5cc53cea37178c3` |
| Policy-bank manifest | `4a976a341aad73e5c0f18a2970246c912f8f17793ad5980b46bebf18d24b4fb4` |

The checkpoint index SHA-256 is
`842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
The selected-tree SHA-256 is
`da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.

### 3.2 Diagnostic configuration

The immutable diagnostic config is:

`configs/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1.json`

Its pre-execution SHA-256 is
`f1cec1bbe14497fdc400a0764bafcb4e67414fbd22bb1aa1be37d916834a66fa`.
The runner verifies this config's hashes before selecting the panel or loading
the full model. Per-layer and final facts bind the actual executed config,
panel, inputs, and output tables again, so the facts remain authoritative if a
later code-only branch changes documentation.

The config freezes:

- injection layers `0, 1, 4, 6, 12, 23`;
- page caps `360, 384, 725, 749`;
- the same three requests and 29 admitted isolated positions used by v3;
- five existing allocation policies;
- three route modes;
- a 24-token/layer enriched panel;
- two adjacent-rate interpolation paths and five interpolation coefficients;
- the high-rate, low-norm control; and
- an RTX PRO 6000 with at least 90 GiB as the full-model execution path.

### 3.3 Raw allocation inputs

Mechanism-rate source cells are retained under:

`/workspace/pr13_d1_decode_slice_pilot_20260824_v2/results`

Metadata-matched source cells are retained under:

`/workspace/pr13_d1_decode_matched_rates_20260824_v2/results`

The same-host allocation patch and its exact capture live under:

`/workspace/pr13_d1_runtime_controller_20260824_v1/same_host_allocation_patch`

Every source cell preserves ordered expert IDs, router information, selected
group pages, and all eight complete 512-entry projection-state vectors. A
state vector is never reassigned to a different expert merely because another
host selected a nearby route.

## 4. Decode-only execution contract

Prefill is assumed exact and supplied separately from compressed decode. The
study models that handoff with native exact-Q4 cached execution.

For one request, current absolute position, and injection layer `l`:

1. Execute token zero exactly to create the native hybrid cache.
2. Advance all earlier positions sequentially with exact Q4 execution.
3. Retain the exact cache immediately before the current token.
4. Deep-clone the complete cache for every baseline, zero-dose control, and
   candidate.
5. Execute the current token exactly through layers `0` to `l-1`.
6. At layer `l`, replace only the current token's routed-expert output before
   shared-expert composition and the decoder residual addition.
7. Continue that same token natively through layers `l+1` to `39` using its
   private cache.
8. Score only the terminal distribution emitted by that current token.
9. Audit the private post-token attention K/V, DeltaNet convolution, and
   DeltaNet recurrent-state commits.
10. Discard every candidate cache. Only the exact baseline cache is permitted
    to advance to the next absolute position.

The runner never constructs a sequence-shaped delta and never injects two
positions together. Scheduling several isolated positions sequentially is an
execution optimization, not cross-position causal coupling.

The repeated native baseline is required to be bit-identical in every hidden
state, router tensor, routed output, terminal logit, and complete post-token
cache before any candidate result is admitted.

## 5. Token-dependent reconstruction and injection

A page is not assigned a static output effect. Its response depends on the
current activation:

\[
d_p(x)\approx U_\ell c_p(x).
\]

For each layer, expert tensors are decoded once and reused across its four
rates. For each current token, complete selected precision states are
reevaluated on the live cached activation. Gate/up refinements therefore pass
through the token's actual SiLU and multiplicative intermediate response, and
down refinements use the actual current intermediate activation.

Let `a_e` denote the actual BF16 execution weight and let
`y_e^(4)(x)` be the exact-Q4 expert output. The injected perturbation is:

\[
\delta_\ell(x)=
\sum_{e\in S_\ell} a_e
\left[\widetilde y_e(s_e,x)-y_e^{(4)}(x)\right].
\]

The sign is approximate minus exact Q4. The stored source-slice delta is a
transfer diagnostic only. It is not injected. Each live row records the
stored-to-live MSE and cosine, the live local qenergy, and the native all-Q4
reconstruction discrepancy.

The all-Q4 reconstructed routed output must remain within `0.125` maximum
absolute error of the native BF16 routed output. This is a BF16 execution-order
tolerance, not permission to change the selected expert set.

## 6. Source-to-live transfer audit

The original allocation packages were created on an RTX 3090, while the exact
tail is executed on an RTX PRO 6000. The target host therefore audits all 174
layer/token identities before importing allocations.

The route-transfer audit found 12 source/live top-8 set mismatches in layers
1, 6, 12, and 23. Those 12 complete five-policy, four-rate banks were
regenerated on the target trajectory. Their patch facts SHA-256 is
`217e4241a743767300ef3c61793735f1d1c523bb348c7e18ef71d1fa03809f9d`.
Recomposed current hidden states are bit-identical, and patched ordered slice
routes equal the captured full-model routes.

The differentiable same-host slice can still differ from full-model router
logits by BF16 execution ordering. Candidate logits for the patch are anchored
as:

\[
z_{\mathrm{candidate,anchored}}
=z_{\mathrm{baseline,full}}
+\left(z_{\mathrm{candidate,slice}}-z_{\mathrm{baseline,slice}}\right).
\]

The frozen raw-slice discrepancy gate is `0.0625`; the observed maximum in the
parent patch was `0.03125`. This is a bounded source-label approximation, not
an exact full-model VJP.

The separate offline transfer analysis compares source margins and crossing
labels with the finalized full-model D1 outcomes. It must distinguish ranking
from certification: a useful margin-risk ROC ranking does not make margin zero
a valid full-model stopping threshold. The source margin may screen cases; it
must not be described as a calibrated route certificate.

The transfer audit writes its own row tables, threshold-calibration tables,
delta-fidelity tables, patch-stratum tables, and immutable facts. It selects no
allocation or threshold and starts no Experiment B work.

## 7. Deterministic 24-case panel

### 7.1 Why an enriched panel

The full v3 smoke established that crossing direction is informative but not
sufficient: preventing a PR #13 D1 crossing tends to move KL in the favorable
direction, while newly introducing one tends to move it unfavorably, yet the
aggregate fixed-D1 policy remains rate-dependent and non-monotonic. A focused
case-control panel is used to expose the competing mechanisms at much lower
GPU cost than rerunning every source cell.

The panel is diagnostic, not representative. Its rates, layers, and outcomes
must never be pooled as an unbiased estimate of expected decode quality.

### 7.2 Frozen selection algorithm

Panel selection consumes only the already-finalized v3 quality and
propagation tables. It produces 24 unique `(layer, group, request, position)`
identities.

First, two quiet controls are reserved in each of the six layers. A quiet
candidate has no immediate D1 membership crossing for either PR #13 or the
fixed-D1 policy at any of the four rates. Within a layer, candidates are
ordered by the smallest maximum absolute fixed-minus-PR13 terminal-KL
difference across rates and then by stable group identity.

The remaining 12 identities are selected without replacement:

- four adjacent-rate reversals where the high rate lowers live local qenergy
  but raises fixed-D1 terminal KL;
- four cases where fixed D1 prevents an immediate PR #13 crossing; and
- four cases where fixed D1 introduces a crossing that PR #13 avoided.

Adjacent reversals are ordered by the high-minus-low KL regression, then rate,
layer, and group. Prevented cases are ordered by PR13-minus-fixed KL;
introduced cases are ordered by fixed-minus-PR13 KL. Stable sorting and
explicit de-duplication make the panel byte-reproducible under input-row
shuffling.

Terminal KL is used here only to enrich explanatory cases. It is not used to
choose any candidate allocation replayed within a case.

The panel facts record input hashes, output hash, class counts, two quiet
controls per layer, and the explicit case-control boundary.

## 8. Candidate grid

Every selected token/layer identity evaluates the same 20 realizable
allocation candidates:

| Family | Count per token/layer |
| --- | ---: |
| Four rates times five existing policies | 20 |
| Two adjacent-rate paths times five linear coefficients | 10 |
| Two high-direction, low-norm controls | 2 |
| Total | 32 |

The five realizable policies are:

1. `pr13_router_square`;
2. `exact_combined_local`;
3. `calibration_selected_fixed_d1`;
4. `frontier_companion_d1`; and
5. `exact_token_oracle`.

The exact-token oracle remains a source/same-host slice diagnostic. It is not
an exact full-model D1 oracle, a terminal-KL oracle, or a runtime input.

The 12 interpolation candidates are explicitly non-realizable vector
controls. They have no page allocation and make no bandwidth claim.

With three route modes, the frozen design contains:

\[
24\times32\times3=2{,}304
\]

candidate executions, plus `24 x 3 = 72` exact zero-dose executions. Prefix,
baseline, and repeat forwards are separately measured in per-layer facts.
This is offline scientific cost, not runtime-controller latency.

## 9. Three downstream route modes

The candidate's router logits are always retained for observation before any
route-execution override.

### 9.1 Live

Every downstream router executes its candidate-selected expert IDs and live
weights. This is the causal quality path.

### 9.2 Frozen set with live weights

At layers `l+1` through `39`, executed expert IDs are replaced by the exact-Q4
baseline IDs. Candidate router logits remain live. Probabilities for the
baseline IDs are gathered from the candidate logits and renormalized over
those eight experts.

This mode removes discrete membership changes while retaining ordinary weight
drift over the baseline set.

### 9.3 Fully frozen

At layers `l+1` through `39`, both executed IDs and BF16 router weights are
replaced by their exact baseline values. Candidate live logits are still
captured for diagnostics.

For paired rows, the analyzer reports the exact arithmetic decomposition:

\[
K_{\mathrm{live}}
=K_{\mathrm{fully\ frozen}}
+(K_{\mathrm{frozen\ set}}-K_{\mathrm{fully\ frozen}})
+(K_{\mathrm{live}}-K_{\mathrm{frozen\ set}}).
\]

The three terms are labelled continuous fully-frozen, router-weight, and
membership-execution components. They close arithmetically, but they are not a
Shapley decomposition and KL is nonlinear. Their interpretation is tied to
this intervention ordering.

## 10. Detailed D1 and tail observations

At every downstream layer the runner retains:

- baseline and candidate ordered top-8 expert IDs;
- baseline and candidate executed top-8 weights;
- their selected-expert logits;
- rank-8/rank-9 margins;
- baseline-set margin under candidate logits;
- entered and left expert IDs;
- membership-pair count, order change, and top-1 change;
- left, entered, and total sparse routing-mass churn;
- centered router-logit MSE and maximum absolute error;
- hidden-state MSE; and
- the layer-specific post-token cache discrepancy.

The D1 analyzer uses only distance one for allocator-severity associations. It
reports membership crossing, changed routing mass, margin violation depth,
centered-logit RMSE, D1 hidden MSE, local qenergy, perturbation norm, and BF16
collapse against terminal KL as an outcome.

The downstream route rows are retained to explain causal propagation and the
three route modes. The deterministic analyzer also reports, per live
candidate, cumulative changed-layer count, cumulative routing-mass churn, and
first crossing distance. Along each five-point path it reports route-burden
transitions and descriptive associations with terminal KL. These executed-tail
quantities are explanatory outcomes only: they neither score candidates nor
define a D2--D4 target or model.

Terminal rows retain Q4-to-candidate categorical KL,

\[
D_{\mathrm{KL}}\!\left(p_{\mathrm{Q4}}\Vert p_{\mathrm{candidate}}\right),
\]

delta NLL, terminal top-1 agreement, final-hidden MSE, injected-layer MSE,
first membership-change layer, and full post-token cache metrics. Terminal KL
is attached only after any D1-only remedy choice.

## 11. Adjacent-rate interpolation and matched-norm control

For each token/layer and adjacent pair `(360,384)` or `(725,749)`, let
`d_low` and `d_high` be the live reconstructed fixed-D1 perturbations. The
runner evaluates:

\[
d(\lambda)=(1-\lambda)d_{\mathrm{low}}+\lambda d_{\mathrm{high}},
\qquad
\lambda\in\{0,0.25,0.5,0.75,1\}.
\]

The endpoint replays must reproduce the corresponding realizable fixed-D1
rows exactly. Finalization rejects any endpoint discrepancy in terminal KL,
delta NLL, final-hidden MSE, full-cache MSE, live injected MSE, or effective
BF16 delta MSE.

The matched-norm direction control is:

\[
d_{\mathrm{high,rescaled}}
=d_{\mathrm{high}}
\frac{\lVert d_{\mathrm{low}}\rVert_2}
{\lVert d_{\mathrm{high}}\rVert_2}.
\]

It keeps the high-rate direction but matches the low-rate planned FP32 norm.
The comparison `rescaled high minus low` is the direction-at-matched-norm
contrast. The comparison `high minus rescaled high` is the norm-along-the-high-
direction contrast. Their KL differences close arithmetically to `high minus
low`, but this ordered contrast is not a unique nonlinear causal attribution.

The midpoint norm also recovers low/high vector geometry without storing the
full vectors in the compact table:

\[
d_{\mathrm{low}}^T d_{\mathrm{high}}
=2\left\lVert d(0.5)\right\rVert_2^2
-\frac{1}{2}\left(
\lVert d_{\mathrm{low}}\rVert_2^2+
\lVert d_{\mathrm{high}}\rVert_2^2
\right).
\]

The analyzer reports cosine, angle, increment norm, selected-page difference,
local-qenergy difference, route changes, and both KL contrasts. These controls
test whether more pages changed the perturbation direction rather than merely
shrinking its norm. They do not represent a streamable page sequence.

## 12. BF16 ULP accounting

The scientific injection follows the actual hook:

\[
y_{\mathrm{injected}}
=\operatorname{BF16}
\left(\operatorname{FP32}(y_{\mathrm{baseline}})+\delta\right).
\]

For every candidate the runner independently computes the effective BF16
delta, its L2 norm and MSE, planned-to-effective norm ratio, quantization-error
MSE and maximum, and error measured in BF16 ULPs. It counts:

- planned nonzero coordinates;
- coordinates that change after BF16 addition;
- planned nonzero coordinates that collapse to zero;
- the collapsed-nonzero fraction; and
- the fraction of planned coordinates below half an ULP.

For a normal finite value `x`, the accounting uses BF16 spacing

\[
\operatorname{ULP}_{\mathrm{BF16}}(x)
=2^{\lfloor\log_2|x|\rfloor-7},
\]

with the BF16 subnormal spacing for values below the normal range. Planned
norm matching and effective BF16 norm matching are reported separately; the
former does not silently stand in for the latter.

## 13. Remedy target ablations

The remedy study is performed offline over the five candidates already
executed in the complete v3 smoke. It runs no model and fetches no pages.
Exact full-model D1 crossing and mass plus live local qenergy are oracle labels
for target comparison, not deployable features.

### 13.1 Strict PR #13 incumbent repair

PR #13 remains selected whenever its immediate D1 set is safe. If the
incumbent crosses, a replacement is allowed only when it has strictly fewer
exact full-model D1 crossings. Eligible repairs are ordered by:

1. fewer membership changes;
2. lower D1 routing-mass churn;
3. lower live local qenergy;
4. smaller absolute difference in total selected-group page count from PR #13;
   and
5. deterministic policy name.

This tests a conservative overlay: preserve the existing allocator unless an
identified D1 failure can actually be repaired.

### 13.2 Global hard-D1/local selector

All five same-rate candidates, including PR #13, are ordered globally by the
same lexicographic criteria. A safe PR #13 incumbent is not privileged. This
separates the value of the D1 target from the risk of unnecessary global
reallocation.

### 13.3 Adjacent-rate early stop under a cap

For each high-rate cap, the candidate pool contains all five executed policies
at the adjacent low and high rates. Candidates with more selected group pages
than high-rate PR #13 are excluded. The remaining candidates use the same D1,
mass, local, page-change, lower-rate, and policy tie-break order.

This tests a `<= budget` rule: additional pages are optional when the lower-
rate perturbation is safer. It is not yet a nested page-stream implementation.

### 13.4 Comparisons and leakage boundary

The analyzer reports same-numeric-rate, metadata-matched low-versus-high, and
adjacent-cap comparisons. It stratifies by layer, request, and layer/request.
Only after the D1-only choice is frozen are terminal KL and live-minus-frozen
KL attached.

Routing-mass severity needs no additional stored metadata because live router
values already exist. A rank-8 FP16 functional expert embedding and a full
256-way FP16 pair row are costed as possible later severity representations,
but the current remedy tables must not be described as having executed a
functional-swap selector unless such a policy is added and replayed explicitly.

## 14. Compute and bandwidth accounting

### 14.1 Scientific panel execution

The 2,304 candidate executions and 72 zero-dose executions are deliberately
redundant controls. Complete exact caches are cloned per candidate, endpoints
are replayed twice for parity, and every downstream layer is observed. The
per-layer facts record measured baseline and candidate seconds, file bytes,
and row counts. Those measurements quantify the offline experiment only.

The interpolation vectors do not correspond to pages. They are excluded from
bandwidth comparisons.

### 14.2 Frozen all-in rates

| Allocated pages/expert | Traffic-equivalent PR #13 pages | Charged bpw | Purpose |
| ---: | ---: | ---: | --- |
| 360 | 384 | 0.5234781901041666 | D1 response plus uncertainty metadata matched |
| 384 | 384 | 0.5234781901041666 | Unadjusted oracle mechanism point |
| 725 | 749 | 0.9987386067708334 | D1 response plus uncertainty metadata matched |
| 749 | 749 | 0.9987386067708334 | Unadjusted oracle mechanism point |

The lower page caps reserve traffic for the proposed signed-effect and
uncertainty sidecar. The high page caps isolate the objective without charging
runtime metadata. Results from the two roles must retain their labels.

### 14.3 Selector arithmetic relative to PR #13

The checked-in arithmetic model uses rank 8, 16 candidate logits, six
refreshes, eight routed experts, 512 units, three page-bearing projections, and
hidden width 2048.

| Quantity per routed group | MACs | Fraction of PR #13 selector |
| --- | ---: | ---: |
| PR #13 primary selector | 51,343,360 | 100% |
| Dense D1 candidate scorer, including synthesis | 9,781,248 | 19.05% |
| One-adjoint D1 scorer, including synthesis | 933,888 | 1.82% |
| Scalar-adjoint repair scorer | 688,128 | 1.34% |
| PR #13 plus one-adjoint repair overlay | 52,277,248 | 101.82% total |

The D1 one-adjoint totals include 81,920 response-synthesis MACs. These are
operation counts, not latency. They exclude transfer stalls, launch overhead,
provisional next-layer execution, and cache traffic.

### 14.4 Metadata

| Representation | KiB/eight-expert group | bpw | 40-layer model MiB |
| --- | ---: | ---: | ---: |
| Current PR #13 factor and ABC metadata | 72.125 | 0.023478 | 90.156 |
| Joint token-response metadata | 155.125 | 0.050496 | 193.906 |
| Joint response plus uncertainty | 167.625 | 0.054565 | 209.531 |

At charged bpw `0.5234781901041666` and `0.9987386067708334`, the
uncertainty-inclusive representation leaves exactly 360 and 725 integral
512-byte correction pages per expert.

Routing-mass severity adds zero stored bytes. A rank-8 FP16 functional
embedding adds 16 bytes/expert, `0.0000406901` bpw, and `0.15625` MiB over 40
layers; it still leaves 360 and 725 pages at these two rates. A full 256-way
FP16 pair row adds 512 bytes/expert, `0.0013020833` bpw, and 5 MiB. The compact
embedding is the only plausible functional-cost sidecar among these two, but
neither is promoted by this experiment.

### 14.5 Provisional next-layer cost remains outside Experiment A

The arithmetic model estimates that a separate invalidated next-layer pre-MoE
pass at context 32 would average about 33.824 million MACs and optimistic
traffic lower bounds of 31.52 MB packed or 68.31 MB BF16. It is roughly 54.4%
of the estimated normal layer work before transfer and launch costs. This is
why the present study evaluates the target before building a sequential
controller. It is not evidence that a provisional pass is affordable.

## 15. Hard gates

### 15.1 Input and environment gates

- Diagnostic config, base config, source tables, run facts, policy manifest,
  checkpoint index, and selected trees must match their frozen SHA-256 values.
- Full-model execution requires an RTX PRO 6000 with at least 90 GiB.
- The dequantized model uses BF16, eager attention, and eager experts.
- Every source policy bank must have the expected token identity and complete
  eight-expert 512-state allocations.

### 15.2 Execution gates

- Repeated same-process cached baselines are bit-identical.
- Every zero-dose mode is exact in current hidden state, all downstream
  routers, terminal logits, and the complete hybrid cache.
- Candidate caches are private and never reused as prefixes.
- No sequence-shaped or cross-position delta is constructed.
- Injection-layer source and live expert sets are identical after the declared
  same-host patch.
- The replacement hook observes a bit-repeatable native routed output and is
  exercised exactly once per candidate.
- Live execution weights are positive and have the expected normalized sum.
- Native all-Q4 reconstruction remains within the BF16 tolerance.

### 15.3 Panel and interpolation gates

- The panel contains exactly 24 unique token/layer identities.
- Every sentinel layer has exactly two quiet controls.
- Event quotas are four adjacent reversals, four prevented crossings, and four
  introduced crossings.
- Every identity has 20 realizable candidates and 12 interpolation controls in
  all three route modes.
- Linear interpolation endpoints reproduce fixed-D1 executions exactly.
- The rescaled-high control matches the low planned norm, with the remaining
  BF16 effective-norm error reported rather than hidden.

### 15.4 Finalization and analysis gates

- Per-layer resume accepts only completed files matching child-facts hashes.
- Final tables have their expected row counts and contain no nulls.
- All zero-dose rows remain exact after concatenation.
- The live/frozen-set/fully-frozen KL decomposition closes within `1e-15`.
- The ordered direction/norm KL contrast closes within `1e-15`.
- Analysis inputs match final run-facts hashes.
- Analysis facts state that terminal KL is outcome-only, no report prose is
  generated by the table builder, and no Experiment B component is present.

## 16. Output tables

The GPU runner writes:

- `d1_experiment_a_diagnostic_panel.parquet`;
- `d1_experiment_a_diagnostic_quality.parquet`;
- `d1_experiment_a_diagnostic_routes.parquet`;
- `d1_experiment_a_diagnostic_zero_gates.parquet`;
- per-layer resumable copies and facts; and
- panel, host, and final run facts.

The deterministic analyzer writes 17 Parquet tables:

- row and summary tables for the three-route-mode decomposition;
- row, correlation, and summary tables for immediate D1 severity;
- interpolation points, turning points, and summaries;
- row and summary tables for page direction versus norm;
- row, summary, and terminal-outcome association tables for BF16 ULP effects;
- row and summary tables for fixed-D1 versus PR #13 within the four panel
  cohorts; and
- per-candidate downstream-outcome rows and per-path transition/association
  summaries, both explicitly outcome-only.

The broader offline v3 diagnosis separately preserves crossing-condition,
adjacent-rate, heavy-tail, D1-only remedy, and source state/delta geometry
tables. It selects no policy from routes after D1.

## 17. Limitations

1. The panel is selected from only three teacher-forced requests. It is an
   enriched case-control sample, not an independent validation cohort.
2. The tokens are isolated decode surrogates behind exact prefixes, not
   free-running generated tokens.
3. Only layers 0, 1, 4, 6, 12, and 23 are injected. Layer 39 and all-layer
   simultaneous compression are outside this run.
4. Only one layer and one current token are perturbed per candidate. Temporal
   cache drift across subsequent generated tokens is not measured.
5. Exact Q4 is the scientific endpoint. A different exact-prefill product
   representation must re-establish the full cache handoff contract.
6. Twelve banks are same-host regenerated; all others retain source-generated
   allocations after route-identity audit. Source VJP margins remain screening
   labels, not certificates.
7. Linear interpolation and norm rescaling are vector controls, not legal page
   allocations. Their outcomes cannot be assigned a bpw.
8. Differences among three route modes are intervention-order contrasts, not
   a unique attribution of nonlinear KL.
9. The remedy selectors use exact full-model D1 outcomes over five already
   executed candidates. They test minimization targets, not a deployable
   predictor or search algorithm.
10. Mean KL is vulnerable to a small number of threshold-sensitive tokens.
    Medians, tails, per-request rows, and event cohorts must remain visible.
11. Compute and metadata values are analytic accounting. No end-to-end decode
    latency, overlap, kernel-fusion, or bandwidth-bottleneck claim is made.
12. No Experiment B promotion follows automatically from a favorable enriched
    panel. A broader independent-request Experiment A validation is required.

## 18. Exact reproduction

Run from `experiments/adaptive_expert_precision_oracle` in the repository
checkout containing the frozen config and implementation.

### 18.1 Environment and paths

~~~bash
export PYTHONPATH=src:scripts
export D1_DIAG_CONFIG=configs/qwen36_mxfp4_d1_experiment_a_diagnostic_panel_20260825_v1.json
export D1_DIAG_OUT=/workspace/pr13_d1_experiment_a_diagnostic_panel_20260825_v1
export D1_V3_ROOT=/workspace/pr13_d1_cached_decode_tail_kl_smoke_20260824_v3
export D1_CHECKPOINT=/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint
export D1_TREES=/workspace/codebook_granularity_study/locked/selected_trees.json
export D1_FIT=/workspace/pr13_average_rate_all_layers_20260822_v1/results/fit
~~~

### 18.2 Focused tests

~~~bash
pytest -q \
  tests/test_d1_decode_tail.py \
  tests/test_d1_experiment_a_diagnosis.py \
  tests/test_d1_source_live_transfer.py \
  tests/test_d1_accounting.py \
  tests/test_d1_experiment_a_diagnostic_panel.py \
  tests/test_d1_experiment_a_diagnostic_panel_analysis.py \
  tests/test_d1_downstream_outcome_diagnosis.py
~~~

### 18.3 Reproduce the full-v3 offline cause and remedy tables

~~~bash
python scripts/analyze_d1_experiment_a_diagnosis.py \
  --config configs/qwen36_mxfp4_d1_cached_decode_tail_kl_smoke_20260824_v3.json \
  --input "$D1_V3_ROOT" \
  --mechanism-results /workspace/pr13_d1_decode_slice_pilot_20260824_v2/results \
  --matched-results /workspace/pr13_d1_decode_matched_rates_20260824_v2/results \
  --patch-root "$D1_V3_ROOT/same_host_allocation_patch" \
  --output "$D1_V3_ROOT/diagnosis"
~~~

### 18.4 Reproduce the source-to-live transfer audit

~~~bash
python scripts/analyze_d1_source_live_transfer.py \
  --input "$D1_V3_ROOT" \
  --output "$D1_V3_ROOT/transfer_diagnosis"
~~~

This command requires the raw source-cell paths recorded in the v3 policy-bank
manifest to remain available.

### 18.5 Freeze the 24-case panel

No full model is loaded in this phase.

~~~bash
python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase panel \
  --config "$D1_DIAG_CONFIG" \
  --output "$D1_DIAG_OUT"
~~~

### 18.6 Run the exact-prefix panel on the PRO 6000

The most efficient invocation loads the model once and completes every
unfinished layer:

~~~bash
python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase run \
  --config "$D1_DIAG_CONFIG" \
  --checkpoint "$D1_CHECKPOINT" \
  --trees "$D1_TREES" \
  --fit-dir "$D1_FIT" \
  --reconstruction-workers 8 \
  --output "$D1_DIAG_OUT"
~~~

For recovery, rerun one layer without invalidating completed layer facts:

~~~bash
python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase run \
  --config "$D1_DIAG_CONFIG" \
  --checkpoint "$D1_CHECKPOINT" \
  --trees "$D1_TREES" \
  --fit-dir "$D1_FIT" \
  --reconstruction-workers 8 \
  --layer 6 \
  --output "$D1_DIAG_OUT"
~~~

### 18.7 Finalize and analyze

~~~bash
python scripts/run_d1_experiment_a_diagnostic_panel.py \
  --phase finalize \
  --config "$D1_DIAG_CONFIG" \
  --output "$D1_DIAG_OUT"

python scripts/analyze_d1_experiment_a_diagnostic_panel.py \
  --config "$D1_DIAG_CONFIG" \
  --input "$D1_DIAG_OUT" \
  --output "$D1_DIAG_OUT/analysis"
~~~

### 18.8 Verify the compact package

After the result package and its report have been copied into the repository,
verify its package-level manifest from that result directory:

~~~bash
sha256sum -c artifact_hashes.sha256
~~~

The final report must cite the final run facts, analysis facts, panel facts,
source-transfer facts, broader v3 diagnosis facts, exact code snapshot, and
package manifest. If any hash or hard gate differs, the report must describe a
new run rather than silently updating this experiment.
