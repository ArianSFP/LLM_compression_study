# Low-rank interaction-field reproducibility protocol

## Question and boundary

This study asks whether the static signed interaction geometry used by the
successful PR #10 exact hybrid oracle can be compressed into a tiny resident
field. It is stacked on PR #10 commit
`54ebf5b64ec04c4605222a8f96d04d0f61e1dcdd` and reuses its immutable
69-invocation, 12-cell exact-checkpoint validation cohort and three frozen
baselines.

The experiment intentionally supplies exact validation-time `h4` and exact
per-unit A/B/C self terms. It therefore isolates geometry compression and
discrete optimization. It is not a candidate predictor, H4 prefetch, latency,
downstream-routing, logit, token-quality, or model-accuracy result. Test
scientific tensors and values are never admitted or evaluated; split/request
metadata in the monolithic capture remains available only to enforce the
existing separation audit.

## Algebraic contract

For static Q2/Q4 down columns, form

```text
Q = [[D4' M D4, D4' M D2],
     [D2' M D4, D2' M D2]] ~= L L'.
```

Rows of `L4` and `L2` are signed unit signatures. For hidden responses `h2`
and `h4`, the four remaining-error signatures are

```text
00: h4 L4 - h2 L2
 G: h4 (L4 - L2)
 D: (h4 - h2) L4
GD: 0.
```

The approximate damage is

```text
||sum_i rho_i(s_i)||^2 + sum_i [q_i(s_i) - ||rho_i(s_i)||^2],
```

where each `q_i` is reconstructed from exact A/B/C. Consequently every
single-unit state energy is exact; only cross-unit interactions are
compressed. Full teacher corrections or Grams are never consulted by the
compressed forward, coordinate, relaxation, or local-repair solvers. They are
used only through the already frozen PR #10 aggregate baselines and by direct
post-selection qenergy evaluation.

Coordinate descent removes the current unit contribution and scores all four
states from two latent dot products, `u_-i' L4_i` and `u_-i' L2_i`. Its charged
interaction cost is exactly `2 * 512 * rank` MACs per complete sweep, plus
the four state-score scalar combinations.

Two solver families are frozen before validation:

1. the primary compute candidate uses only all-00 and coherent independent
   A/B/C seeds, followed by both coordinate solves and bounded local repair;
2. the broad geometry diagnostic additionally constructs compressed
   residual-forward and projected-gradient-relaxation seeds, runs four
   coordinate solves, and applies the same local repair.

Every row separately records field construction, independent-seed scoring,
forward, relaxation setup, relaxation iterations, coordinate, and local MACs,
plus the candidate-evaluation, iteration, sweep, and evaluated-pass counters.
The analyzer recomputes every component and the total from those counters.

The relaxation uses a charged Frobenius-safe step bound; it never calls an
uncharged spectral-norm/SVD setup. Exact-page dynamic-programming rounding,
simplex projection, sorting, state additions, top-k, and control flow are
disclosed as non-MAC operations. Broad recovery cannot be assigned to the
cheap path, and cheap accounting cannot be assigned to the broad path.

## Frozen factor grid

The factor construction is static per sampled expert and uses only locked
weights plus the existing train-derived rank-4 proxy metric.

- joint truncated eigendecomposition: tail ranks 4, 8, 16, 32;
- joint pivoted Cholesky: ranks 4, 8, 16, 32;
- exact rank-4 proxy coordinates plus Euclidean eigentail ranks 0, 4, 8, 16,
  32;
- joint-eigen rank 8/16/32 row-INT8;
- joint-eigen rank 8/16/32 Walsh-Hadamard-rotated packed row-INT4;
- joint-eigen rank-16 unrotated packed row-INT4 control.

Quantized artifacts contain actual signed packed codes, one stored FP16 scale
per Q4/Q2 factor row, and one stored FP32 global shrink. The shrink is rounded
conservatively and verified to keep every unit's residual 2x2 A/B/C block PSD.
Payload accounting is taken from these arrays, not inferred from decoded FP32
references. Combined metadata adds the established 3,084-byte PR #10 A/B/C
payload.

The budgets are exactly 384, 576, and 768 pages (0.5, 0.75, and 1.0 physical
correction bpw under the inherited convention). The full factor/solver grid is
declared in `configs/qwen36_mxfp4_interaction_field.json` before execution.

## Gates and interpretation

At one physical bpw, geometry compression passes only if:

- invocation p10 set-gain retention versus the PR #10 exact hybrid teacher is
  at least 97%; and
- median recovery is within 0.5 percentage points of that teacher.

Metadata must remain at or below 0.35 bpw. The inherited selector-compute gate
is strict `< 1,572,864` MACs. The first two gates answer whether the compressed
statistic is sufficient. The compute gate answers whether the cheap two-seed
implementation is arithmetically viable; the broad four-seed diagnostic is
reported separately. Neither outcome advances to a sealed holdout because
exact H4 is supplied.

## Parallel execution contract

The rented host exposes 256 logical CPUs through affinity, but its cgroup
`cpu.max` is `2720000 100000`, an effective quota of 27.2 CPU cores. The
canonical runner therefore uses 24 forked invocation workers and freezes
OpenMP, OpenBLAS, and MKL to one thread per worker. Static factor arrays are
constructed once in the parent and inherited read-only through `fork`; only
the parent writes Parquet checkpoints and facts, in the original deterministic
cell and observation order.

The canonical 69-invocation evaluation completed in about 15.5 minutes. The
serial attempt had required 28.5 minutes for its first seven-invocation cell
and projected to roughly four hours, so invocation parallelism provided about
a 15-fold wall-clock improvement. Raising the pool to 64 or 256 would only
time-slice against the 27.2-core quota; 24 leaves headroom for the parent and
checkpoint I/O.

The serial attempt and a subsequent pre-Frobenius-step attempt were
TERM-stopped and preserved separately as performance-only quarantines. Neither
is canonical evidence and neither informed scientific selection. The canonical
facts record the physical CPU quota, worker/thread environment, 69 tasks, and
24 effective workers.

## Commands

From `experiments/adaptive_expert_precision_oracle`, with the same external
checkpoint, trees, exact capture, and canonical PR #10 validation directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts python \
  scripts/run_interaction_field_study.py \
  --config configs/qwen36_mxfp4_interaction_field.json \
  --pr10-config configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/locked-checkpoint \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/exact-captures.npz \
  --pr10-validation-dir /path/to/pr10/validation_exact_checkpoint \
  --output /path/to/qwen36_mxfp4_interaction_field_20260820_v1 \
  --device cuda:0

PYTHONPATH=src:scripts python scripts/analyze_interaction_field_study.py \
  --config configs/qwen36_mxfp4_interaction_field.json \
  --evidence-dir /path/to/qwen36_mxfp4_interaction_field_20260820_v1 \
  --output /path/to/qwen36_mxfp4_interaction_field_20260820_v1/analysis
```

Focused verification:

```bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_interaction_field.py \
  tests/test_interaction_field_runner.py \
  tests/test_interaction_field_analysis.py
```

## Canonical outcome

The canonical run used 24 single-thread workers on the 27.2-core quota and
closed all 69 invocations, 12 layer/expert cells, 20 factor configurations,
and 10,143 frontier rows with no failure.

At one physical bpw:

- Best FP32 geometry was exact-proxy plus rank-32 Euclidean tail: p10/median
  recovery 96.2668%/98.1596% and p10 set-gain retention 99.9693%.
- The best configuration passing geometry, metadata, and strict compute gates
  was joint-eigen rank-8 row-INT8: p10/median recovery 96.1354%/98.0678%,
  p10 retention 99.9313%, 0.03389 metadata bpw, and 834,560 MACs.
- Hadamard-rotated rank-8 packed INT4 also passed every gate: p10/median
  recovery 96.1068%/98.0219%, p10 retention 99.8039%,
  0.02348 metadata bpw, and 834,560 MACs.
- Exact existing rank-4 proxy coordinates without a Euclidean tail already
  achieved 96.0501%/97.9515% p10/median recovery and 99.8535% p10 retention.

Every tested rank and quantized point passed the frozen geometry criterion;
rank 16 and 32 missed only the compute gate. At rank 8, the field build plus
coordinate work is small, while bounded 1/2/3-unit repair contributes 595,968
of the 834,560 conservative MAC bound. Thus interaction geometry is highly
compressible here, and local repair—not factor rank—is the first compute target.

The continuous relaxation alone was weak under the frozen 128-iteration
Frobenius-safe step, and compressed residual-forward alone was much weaker than
coordinate plus repair. Four expensive seed basins did not materially improve
the final repaired result over all-00 plus independent A/B/C. These are solver
negatives, not evidence against the interaction field.

### Frozen hashes

- Config: `32c8c3178cf277ed2b066f31f6993f2c9cd7204c0003258e1b6883e04f10ec95`
- Runner: `177ed93a2ce1c8859c3e9d12fe68d189f3aca229511e975c9f990865b27151da`
- Interaction core: `0c88acc60264466baa48b01e99882c89adedfad2c6599b81e95a817972a34eab`
- Run facts: `835ac98e495566cb1f2aab181d5be3871174c859dcdf77de74470a751872396e`
- Factor manifest: `d9c2974927a5bf26b7258ddc493217c472cf3ebbe0b23b9e77571e406a4ff280`
- Frontier: `2f251a9d92a12de448712d96a0de87d59f2e596733a04df5070e8b5326f71b47`
- Analyzer core: `00ca76c33272e4c8df31846e4fe1e33c21a618e5f0dbcbd14948afdee606e074`
- Analyzer wrapper: `339ffc8d75a4272942f98635c4bbdb10767b6a336acc5cdb27b13839f287b093`
- Conclusion: `c8b76e699b1c05681698a609a88872b489640d98c2f719c84d205663e678af70`
- Analysis manifest: `8affc34627fb312346a02deddcbdb8a2583a25e11e5c473a40c9edbd439e78d4`

Verification: focused local and remote suites passed 15 tests; the final full
local suite passed 297 tests with 1 skip and one inherited NumExpr warning.

## Hardware decision

A single 24 GB RTX 3090 is sufficient for this entire exact-H4 interaction
study. The largest dense static object is the 1024x1024 joint Gram (8 MiB in
FP64 or 4 MiB in FP32), and runtime fields are at most tens of kilobytes.
System RAM and CPU time, not VRAM, dominate the broad NumPy reference grid.

A 48/96 GB professional GPU becomes useful only for the later finite-horizon
extension that loads the full model and performs many JVP/VJP or downstream
replay probes. It is not needed to decide rank, factor method, quantization, or
the four-state algebraic solver in this PR.
