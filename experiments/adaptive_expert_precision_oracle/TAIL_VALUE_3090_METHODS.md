# Tail-value feasibility on the RTX 3090

Parent: `8cd2230a21bbe4aca2910ec7fb6b5426bd43eda7`. This is a new,
privileged offline diagnostic. The prior Experiment A decision remains failed;
Experiment B is paused. No old immutable configuration or artifact is changed.

## Inputs and independence

Audit the authenticated old request manifest using the actual consumed tokens,
not whole-prompt identities. Exclude all 128 old complete prompts from the pinned
256-request source. Freeze `p=min(63,n-2)`, require at least eight prefix tokens,
and reject duplicate effective inputs or overlap with old effective inputs.
Within each domain, sort remaining prompts by SHA256 of seed 20260909, domain,
and prompt hash: four development, four validation, eight reserved requests.
The 64 reserved requests receive no model outcomes in this study. Input-only
inspection is allowed. These are unused in this experiment, not certified unused
across every historical project. Shared template/document provenance is not
provided by the source and remains a limitation.

## Execution and gates

Use the existing MXFP4 checkpoint expanded to BF16. Keep routed expert banks in
CPU RAM, execute native eager arithmetic on CUDA with live selected expert
weights transferred on demand. No additional reference quantization is allowed.
Use exact native full-prefix prefill, then one-token decode with private complete
hybrid caches. This full-prefix prefill is a newly versioned execution choice;
the old token-at-a-time prefill results are not assumed bit-equivalent.

Before labels, prove repeated reference logits/caches are bit-identical and
compare transferred expert arithmetic with native GPU-resident eager arithmetic.
Preserve full model and numerical stack identities. Check zero injection per
request. Hardware benchmarks are offline measurements, not runtime-controller
latency claims. Abort quality interpretation if these gates fail.

## Frozen bank

Start with a development pilot and expand on successful correctness/resource
checks. Planned development grid: 32 requests, layers 0,1,4,6,12,23, payload
rates 360 and 725 pooled across eight experts. PR13 384/749 are external
all-in comparators for the historical D1 metadata allowance. Additional learned
metadata is not free: until repriced, this remains a headroom diagnostic.

Reuse the PR13 factors, full column-generation settings and Q2/Q4 tree layout.
For each cell, generate PR13-low, exact-local-low, a common core obtained by
pruning PR13-high to low cap minus TWO GROUP pages, and local completion.
Include four locally ranked single additions, up to four paired additions
(including related gate/up bits), and four matched-cost swaps around PR13-low.
These search limits are deliberately small and differ from old W=16 repair.
All states and realized BF16 injection identities are sealed before any candidate
tail labels. Deduplicate exact physical states and realized injected outputs.
The Q4 zero-dose reference is not a feasible compressed allocation.

Score live terminal KL against the same-host Q4 reference and record D1 next
router membership, NLL and actual pages. Compare hindsight best within the full
bank with its subset that does not worsen PR13-low D1 membership count. Also
report a D1-first/local-tiebreak selector. This is a new bank ablation, not a
replication of the old bounded-VJP D1 repair policy. Candidate generation must
not be filtered by terminal quality or D1 labels. Swaps are separate diagnostics;
no cross-rate nesting claim is made for independently selected endpoints.

Negative results only concern this bank. An oracle/no-op curve is monotonic by
construction and is not evidence of deployability. Report gains against both
same-payload and external all-in references, harmed-request fractions, p95 and
maximum request-average differences, and layer/domain strata. Development
results are descriptive. No promotion to Experiment B is possible here.

## Later gated work

If meaningful headroom exists, fit a small candidate comparison model on
development only, validate on held-out requests, and explicitly separate exact
residual/page features from available-token features. Do not execute reserved
confirmation requests or call a privileged feature runtime available. First test
post-injection cache persistence on a short teacher-forced continuation with
private evolving caches; all-layer compression needs fresh live allocations and
cannot reuse isolated exact-prefix deltas. Record the measured bottleneck and
remaining large-VRAM work before stopping the pod with runpodctl.

## Pre-label performance amendments

The reference loader now keeps up to 8 GiB of immutable selected expert weights
in a GPU LRU cache. Every saved current-token activation, expert output, route
ID and route weight from uncached capture must be bit-identical when replayed
with this cache before that request receives labels. Earlier performance
versions can prepare banks concurrently: full-bank states, deltas and candidate
records must match the original implementation exactly. The optimizations omit
an unused high-rate completion, vectorize unit move effects, and cache unchanged
per-expert residuals while preserving the original expert summation order.
Amendment JSON files preserve script hashes and the absence of terminal labels
at each change. These are execution optimizations, not a changed search budget.

## Small metric follow-up, specified before development labels

If the bank shows useful same-payload hindsight headroom, fit one small positive
semidefinite metric per layer using development candidate comparisons. Use the
existing local damage plus the four squared projections onto the already frozen
PR13 proxy directions. Constrain all learned coefficients nonnegative and keep
a small positive local-metric floor. Thus the fitted correction cannot create a
negative quadratic cost or remove the local penalty. Fit differences relative
to each cell's PR13 allocation; exclude external higher-payload comparators.
Do not tune ranks, features or regularization against validation outcomes.
Freeze coefficients before validation labels. This tests an exact-residual
metric only: constructing its residual still requires privileged Q4 information.
Report recovered oracle headroom, harmed requests, extremes and D1 ablations.
A validation failure is a reason to stop this particular metric, not to reject
all tail-sensitive objectives or all candidate neighbourhoods.

The metric fit uses nonnegative least squares on five features: existing local
damage and four squared proxy projections. It pools both payload rates within
each layer. Pair differences are scaled by their training RMS. Ridge strength
is fixed at 0.001 times the number of pairs. The local floor is 5% of the
nonnegative through-origin absolute-damage slope (with a positive numerical
floor); the remaining coefficients are nonnegative. These constants are fixed
before terminal labels, rather than selected on validation. Prediction has a
separate code path that does not open label files.

The continuation probe uses the first request in each domain at the already
fixed decode position, with up to four available future teacher-forced steps.
A request with no such tokens is explicitly omitted from the continuation
probe; its single-token cell remains in the main study.
