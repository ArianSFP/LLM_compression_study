# D1 nested-safe Experiment A oracle methodology

## Status and question

This protocol remains Experiment A. It tests a candidate allocator architecture,
not a runtime controller:

\[
\text{PR13 incumbent}
\rightarrow
\text{D1-blind physical subset core}
\rightarrow
\text{bounded D1 repair}
\rightarrow
\text{frozen add-only local completion}
\rightarrow
\text{early stop}.
\]

The decisive question is whether D1 adds quality value **after physical nesting
is held fixed**. The primary scientific contrast is therefore nested D1-safe
completion versus nested local-only completion, not merely D1 versus an
independently reoptimized PR13 allocation.

The protocol is frozen by
[`qwen36_mxfp4_d1_nested_safe_allocation_20260825_v1.json`](configs/qwen36_mxfp4_d1_nested_safe_allocation_20260825_v1.json).
No evaluation outcome may change its request split, page window, guardrail
grid, search width, tie ordering, contrasts, or promotion gates.

## Scientific boundary

D1 means only the same decode token's next-layer router:

\[
(t,\ell)\rightarrow(t,\ell+1).
\]

There is no D2--D4 objective, gradient, predictor, or selector input. The
allocator may use exact Q4 D1 routes, exact prefix-conditioned D1 sensitivities
and exact D1 finalist replay because this is an oracle experiment. It may not
use terminal logits, terminal KL, final hidden state, any router after D1, or
post-token cache error.

Allocation and outcome execution are separate physical phases. The allocation
phase writes complete states and a checksum-sealed manifest without terminal
fields. Only then may the outcome phase execute the downstream tail. This
prevents accidental terminal-label leakage in code as well as in policy.

Prefill is exact. Each request contributes exactly one isolated teacher-forced
decode token behind its complete native hybrid prefix cache. Candidate caches
are private and never advance another token.

## Independent request cohort

The source is a pinned 256-request, eight-domain prompt-token manifest. Its
tokenizer hash equals the MXFP4 checkpoint tokenizer hash. Previous BF16
states, routes and caches are not reused; only request token IDs are inputs.

Requests are deterministically hash-ranked within each of eight domains. Four
per domain form a 32-request calibration set and the next twelve form a
96-request untouched evaluation set. The three prior development prompts are
excluded by SHA-256. Every prompt, token, layer, rate and cache derived from a
request remains in that request's split.

Source request IDs may be non-empty strings or nonnegative JSON integers. An
integer ID is converted losslessly to its base-10 decimal string before
duplicate checks, deterministic selection hashing, split assignment and output
serialization; booleans, negative integers and all other JSON types are
rejected. This normalization is pinned in the immutable experiment config.

One position is selected without model outputs:

\[
p=\min(33,\;n_{\mathrm{tokens}}-2).
\]

This leaves a next-token label and caps exact-prefix work. It also gives the
study 96 independent evaluation units instead of treating positions or layers
within one request as independent observations.

## Physical state and traffic

Each routed expert has 512 units and three independently addressable Q4 page
bits: down `1`, up `2`, and gate `4`. A unit state lies in `[0,7]`, and its page
cost is the three-bit population count. A group is an aligned `[8,512]` state
array indexed by actual expert ID.

For states `A` and `B`, physical nesting is exact bit containment:

\[
A\subseteq B
\quad\Longleftrightarrow\quad
A\;\&\;\neg B=0.
\]

The two product traffic comparisons are:

- D1 metadata plus at most 360 pages/expert versus PR13 at 384 pages/expert;
- D1 metadata plus at most 725 pages/expert versus PR13 at 749 pages/expert.

The page cap is an upper bound. Early stopping may leave bandwidth unused.
Budgets are pooled across the eight experts and retain the PR13 group/burst
contract; no per-expert 360/725 constraint is introduced.

## Common D1-blind core

Arms 3--5 begin from the identical core. Starting from the corresponding
384- or 749-page PR13 reference, reverse greedy pruning clears one physical bit
at a time until the cost is at most `B-W`, where `B` is 360 or 725 pages per
expert pooled over the group and `W` is a reserved group-level repair window.

Pruning sees only exact combined local qenergy under actual BF16 execution
weights. It never sees D1. This is essential: a route-aware core would confound
the nesting-only control with the D1 treatment. Every core must be a literal
subset of its declared PR13-high reference.

`W` is calibrated from `{16,32,64}` physical group pages. The core need not be
the globally optimal subset; the search is a deterministic bounded heuristic.
The report must use the word “heuristic” unless a wider calibration audit
establishes its candidate recall and regret.

## Five arms

1. **Independent PR13.** Execute independently optimized PR13 at 360, 384,
   725 and 749 pages. The high endpoints are product all-in references; the
   low endpoints are same-numeric-cap mechanism controls.
2. **Strict frozen-candidate incumbent repair.** Freeze the complete five-state
   candidate bank before exact D1 replay. A safe PR13 incumbent is immutable.
   An unsafe incumbent changes only to a candidate with strictly fewer exact
   D1 membership changes. The bank is pinned to independent PR13, independent
   exact-combined local, and the three D1-blind nested-local endpoints produced
   with windows 16, 32 and 64 at the same numeric cap. Thus every candidate is
   fixed without an exact D1 label; terminal outcomes never generate or prune
   the bank. This arm reproduces the conservative nonnested **selection
   semantic**, rather than claiming byte identity with the earlier pilot's
   five global-policy candidates.
3. **Nested local-only.** From the common core, add physical pages using exact
   combined local qenergy alone. It is the nesting control and never reads D1.
4. **Nested D1-safe.** Use the same core, candidate universe and search bounds.
   If Arm 3's endpoint is exactly D1-safe, retain it byte-for-byte. Otherwise,
   choose only a nested candidate with strictly fewer exact D1 membership
   changes and within the additive local guardrail. Once safe, D1 hard-
   saturates and local completion resumes subject to safety.
5. **Nested D1-safe with violation/mass severity.** Identical to Arm 4, except
   equal crossing-count unsafe candidates tie-break first by exact boundary-
   violation depth and then routing-mass churn. Severity has no utility after
   safety; it cannot globally reshape a safe allocation.

The same-rate incumbent rule is applied independently at both endpoints. If
Arm 3 high is exactly D1-safe, Arm 4/5 high must be byte-identical to Arm 3
high. If Arm 3 high is unsafe, any changed high state must strictly reduce its
exact membership-change count. A high state must also be no worse than its
selected low state, so neither same-rate nor cross-rate monotonicity is traded
away silently.

The completed state may add bits that were not present in the original PR13-
high reference. The guaranteed chain is common core subset completed low
subset completed high for each nested arm. The original reference-high and
new completed-high states are named and hashed separately.

## Local guardrail and completion

For a route-aware candidate `S`, require

\[
J_{\mathrm{local}}(S)
\leq
J_{\mathrm{local}}(S_{\mathrm{nested\ local}})
+\eta J_{\mathrm{Q2}},
\]

where all terms are computed for the same token/layer group using the exact
combined eight-expert qmetric and actual execution weights. The calibration
grid is `eta in {0, 1e-4, 2.5e-4, 5e-4, 1e-3}`.

After a state is frozen, only one-bit additions are legal. Gate/up/down effects
are recomputed from the current complete unit state, so nonlinear interactions
are not treated as independent static page deltas. Completion stops when the
cap is reached or when no positive local gain survives the route/guardrail
screen. A page whose BF16-once effective output change is zero is not useful
completion.

For each Arm 4/5 high endpoint, the allocator constructs the complete bounded
add-only local path from the selected low state to the high cap, then scans its
checkpoints from highest page count downward (lowest local damage breaks a
same-page tie). Intermediate states need not be published or executed as
outcomes. A checkpoint is eligible for exact D1 replay only when it satisfies
the strict high-endpoint guard relative to Arm 3 high. It is admitted only if
its exact crossing count is no worse than selected low and it also obeys the
same-rate Arm 3-high incumbent rule. The literal Arm 3-high identity is also an
eligible endpoint when selected low is its physical subset.

## D1 screening and exact gate

The cheap oracle screen uses exact token-dependent signed effects for selected
ranks 6--8 against outsiders 9--16, refreshed after small page batches. It is
only a shortlist mechanism. It is never called a certificate.

For each request/layer, capture stores the exact full-model layer output. The
singleton current hidden and cloned prefix cache must reproduce that stored
current hidden and cache bit-for-bit. The raw slice router-logit delta from the
captured full-model baseline is bounded by `0.0625`; its constant captured-
baseline logit anchor is fixed before, and is independent of, every allocation
and VJP. After anchoring, zero delta must reproduce the captured router logits
exactly. Every finalist is then replayed through that anchored same-host next
router, and all 256 router logits participate in the final top-8 check.

The unanchored slice route is diagnostic, not a target gate. In particular,
bounded slice drift can reorder an exact tie or move a near-boundary expert
before the fixed anchor is applied. Raw ordered-route and set agreement are
therefore serialized separately. The hard route gate is that the anchored
zero-delta logits are bit-identical to the captured full-model logits and that
the authoritative CUDA softmax/top-k operation reproduces the captured ordered
top-8 exactly.

Each repair round screens at most 32 signed-margin page moves and replays at
most 8 exact finalists. This compute cap is distinct from the configured
`W in {16, 32, 64}` group-level repair-window grid. A safe incumbent is
immutable. An unsafe replacement must strictly reduce exact membership change.
Arm 4 then tie-breaks by local qenergy and physical edit distance. Arm 5 inserts
exact violation depth and exact routing-mass churn before those terms while the
state remains unsafe.

If no high checkpoint or bounded matched-swap repair satisfies all of these
conditions, the allocator fails closed: it discards the attempted repaired
pair and reverts the entire Arm 4/5 low/high pair byte-for-byte to the Arm 3
nested-local incumbents. This preserves the fixed allocation grid, literal
core-to-low-to-high nesting and both endpoint guardrails; it deliberately
forfeits the low repair rather than publishing an invalid or non-extendable
pair. The per-pair fallback flag, number of guard-qualified high checkpoints
and number of bounded repair attempts are serialized. Fallback frequency is
reported once per policy pair, not once per endpoint row. Evaluation results
cannot expand the frozen window.

## Calibration and sealing

Calibration uses only exact D1 outcomes, local damage, traffic and physical
state invariants. It selects one shared `W` and `eta` per traffic pair from the
metadata-matched low endpoint (360 for the 360/384 chain and 725 for the
725/749 chain). The choice is then frozen for both endpoints, so calibration
cannot destroy literal nesting by independently choosing two common cores.
Candidates are ordered by:

1. zero safe-to-unsafe events;
2. fewest unresolved exact D1 crossings;
3. smallest repair window;
4. smallest eta;
5. lowest local damage;
6. deterministic parameter hash.

No calibration terminal KL or downstream route is read. Frozen parameters are
written before evaluation allocation. Evaluation selected states, full move
chains and exact D1 labels are then serialized and checksum-sealed before the
tail runner starts.

## Outcomes and inference

The outcome phase executes live, frozen-set/live-weight and fully-frozen tails.
For every arm/rate/layer/request it records current-token terminal KL and NLL,
final-hidden MSE, complete downstream route outcomes, and post-token cache
differences. These quantities are outcomes only.

Inference first averages equally over the six predeclared layers within each
request, then treats requests as independent. Paired request-cluster bootstrap
confidence intervals use 10,000 resamples; paired sign-flip inference uses
100,000 draws. Means, medians, win fractions, p90, p95, maxima and domain/layer
strata are all reported. Token-level crossing cohorts are descriptive and do
not replace request-cluster inference.

The primary contrasts are Arm 4 versus Arm 3 at 360 and 725 pages, followed by
the two product-matched comparisons against PR13-high. Holm adjustment applies
within this four-contrast family. Arm 2 replication, nesting-only effects and
Arm 5 severity are secondary and cannot select a winning evaluation policy.

## Promotion and failure interpretation

Experiment B remains paused. Promotion requires all engineering gates, fewer
exact D1 crossings than Arm 3, no safe Arm-3 endpoint made unsafe, no local-
guard violations, favorable Holm-adjusted paired mean KL at both same-rate and
product-matched contrasts, nonpositive paired p95 regression, and maximum
single-request regression no larger than the prior strict-repair diagnostic
benchmark `0.002839`.

Failure has a specific interpretation:

- Arm 3 helps but Arm 4 does not: nesting/direction control, not D1, is the
  useful remedy.
- Arm 4 lowers D1 crossings without lowering KL or live-minus-frozen KL: D1 is
  a mechanism marker but remains an insufficient allocator surrogate.
- Wider calibration search repairs cases that the frozen shortlist misses:
  search is the bottleneck, not necessarily the target.
- Failures concentrate at `W` or the guardrail: coverage is the bottleneck;
  evaluation data may not retune it.
- Arm 5 helps only unsafe cases: retain severity only in repair. If it changes
  safe states, the implementation violates the protocol.
- Neither nested arm improves product KL: do not start Experiment B; evaluate a
  low-rank continuous tail-sensitivity proxy inside Experiment A instead.

## Runtime boundary

This study does not implement a deployable certificate. Existing analytic
accounting suggests a scalar/one-adjoint overlay can be about 1.34--1.82% of
the PR13 selector MAC count, but that excludes launch latency, memory traffic,
top-k work and overlap. Predictor validation—signed-effect correlation, sign
accuracy, page ranking, corrected-logit error, interval coverage and false-safe
rate—begins only after the oracle passes. A full provisional next-layer rerun
remains outside the intended architecture.
