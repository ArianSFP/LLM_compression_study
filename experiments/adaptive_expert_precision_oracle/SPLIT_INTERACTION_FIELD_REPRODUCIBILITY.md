# Split interaction-field reproducibility

## Result and scope

This experiment is stacked on PR #11 commit
`b276da7e53d09bdb9450021c6eb4ea7b704debd2`. It reuses the frozen PR #11
interaction factors and the immutable PR #10 validation cohort; no factor is
refit. Gate, up, and down suffix pages are independently selectable. The
result remains an exact mixed-H4 geometry ceiling, not a deployable prefetch,
latency, routing, logit, token, or downstream-accuracy claim.

The attribution control requested after the first PR #12 run is now present.
The final evidence distinguishes:

1. the legacy PR #11 four-state solver;
2. compressed four-state search with the exact-self DP/all-00 seeds, the new
   coordinate implementation, the same local repair, and the same factor;
3. the eight-state solver with that identical seed/solver/factor path.

This separates seed/optimizer headroom from action-space gain. The frozen
continuation decision remains `continue_to_predicted_mixed_hidden_interface`.
No test scientific tensor or value is admitted.

## Algebra

For one unit, the four hidden scalars are

\[
h_{22}=\operatorname{SiLU}(G_2x)(U_2x),\quad
h_{24}=\operatorname{SiLU}(G_2x)(U_4x),
\]
\[
h_{42}=\operatorname{SiLU}(G_4x)(U_2x),\quad
h_{44}=\operatorname{SiLU}(G_4x)(U_4x).
\]

The state integer is `4g+2u+d`; its page cost is `g+u+d`.

| state | gate | up | down | pages | hidden |
|---:|---|---|---|---:|---|
| 0 | Q2 | Q2 | Q2 | 0 | h22 |
| 1 | Q2 | Q2 | Q4 | 1 | h22 |
| 2 | Q2 | Q4 | Q2 | 1 | h24 |
| 3 | Q2 | Q4 | Q4 | 2 | h24 |
| 4 | Q4 | Q2 | Q2 | 1 | h42 |
| 5 | Q4 | Q2 | Q4 | 2 | h42 |
| 6 | Q4 | Q4 | Q2 | 2 | h44 |
| 7 | Q4 | Q4 | Q4 | 3 | h44 |

The PR #11 states `00/G/D/GD` map to `0/6/1/7`. The eight-state feasible set
therefore contains the four-state set at every page budget.

For hidden scalar h with Q2 down, exact self damage is

\[
q_{D2}(h)=h_{44}^{2}A-2h_{44}hB+h^{2}C.
\]

For Q4 down it is

\[
q_{D4}(h)=(h_{44}-h)^2 A.
\]

The signed residual signatures are

\[
\rho_{D2}(h)=h_{44}L_4-hL_2,\qquad
\rho_{D4}(h)=(h_{44}-h)L_4.
\]

Thus no new down Gram or factor metadata is required. All eight A/B/C self
terms are exact; only cross-unit interactions use the serialized L4/L2 factor.

## Solvers, attribution, and incremental updates

Both the compressed four-state control and the primary eight-state path use:

- all-state-0 and exact-self multiple-choice-DP seeds;
- coordinate descent under the same page budget;
- bounded 1/2/3-unit local repair over the same factor;
- the same sweep, shortlist, bundle, and pass caps.

Only their allowed state sets differ: `[0,6,1,7]` versus `[0..7]`. The legacy
PR #11 result is a separate control, not the action-space denominator.

Coordinate descent maintains the rank-r residual field incrementally. For a
unit transition `source -> destination`, it updates damage by the difference
of the two already scored state objectives, mutates the state/residual, and
performs no full objective reconstruction inside the move loop. A regression
proves only initial and terminal `field.damage` calls occur. Terminal exact
recalculation is a fail-closed parity assertion.

The charged complete-field upper bound uses actual evaluated sweeps and local
passes. For an allowed state count S, coordinate work is

`(2*512*r + 2*S*512) * evaluated_sweeps`.

Local work is conservatively charged from the actual maximum shortlist and
actual evaluated passes. Rank-8 eight-state rows remain at most 1,329,152
MACs; the rank-4 exact-proxy control remains at most 736,256 MACs.

The exact three-Gram controls repeat the same restricted/full action-space
comparison using training-only exact interaction geometry. The inherited PR
#11 warm-start path remains a charged, nonpromotable diagnostic.

## Strict all-in budget

One physical total bpw is 393,216 bytes per expert. Each correction page is
512 bytes. The strict all-in rows reserve encoded A/B/C plus the factor first,
then take the largest integral page count that still fits. Adding one page
would exceed the byte cap.

| factor | metadata bytes | correction-page cap | slack bytes | total bpw |
|---|---:|---:|---:|---:|
| exact-proxy rank 4 FP32 | 19,468 | 729 | 500 | 0.998728 |
| joint rank 8 FP32 | 35,852 | 697 | 500 | 0.998728 |
| joint rank 8 INT8 | 13,328 | 741 | 496 | 0.998739 |
| joint rank 8 Hadamard INT4 | 9,232 | 749 | 496 | 0.998739 |

The table reports reserved caps. INT4 uses a median 749 actual pages: 62/69
invocations use 749 and seven use 748. Strict all-in rows are diagnostics and
do not alter the original fixed one-correction-bpw promotion gate. Their predeclared recommendation rule is
maximum median recovery, then minimum metadata bytes, then stable factor ID.

## Frozen inputs and sources

| object | SHA-256 |
|---|---|
| split config | `e61aae7b38a76b95c858f00344d0d9cc9e6d59fa4c686d8f28dbac170f877557` |
| split runner | `24735d86459d3c4d0aa7fb10c69259210efb2abf79ad759665054b406999173b` |
| split core | `311365d169dd8c5c1d041ab023130d4d5e08980f0ab50b12afe331c5a16e2c1e` |
| analysis wrapper | `c0bbeff1be0cd1bea2f29d4f3cbfaff86b1dcc7e12fa91a8172e5e61eb126455` |
| analysis core | `1fb68813b8ed43f107a09d423ba41069236e30cd09e3a0e17517fcea8d0da29e` |
| PR #11 run facts | `835ac98e495566cb1f2aab181d5be3871174c859dcdf77de74470a751872396e` |
| PR #11 factor manifest | `d9c2974927a5bf26b7258ddc493217c472cf3ebbe0b23b9e77571e406a4ff280` |
| PR #11 frontier | `2f251a9d92a12de448712d96a0de87d59f2e596733a04df5070e8b5326f71b47` |
| exact capture | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| selected trees | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| checkpoint config | `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` |
| checkpoint index | `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb` |

Final test hashes:

| test | SHA-256 |
|---|---|
| split core | `4b660f5906cf75998698ab8fac162eb13812295fd80875109dd26e352d804708` |
| runner | `fca738ab5f0f72f9173d2e5e357afb5610c316c3e84b8eea1415b901aa5196b4` |
| analyzer/E2E | `00ea432b48f025320d86bfe382b41f00ccc86583658aecfdb0b54cd2c83bdf2e` |

## Hardware and parallelism

The rented host exposed 256 logical CPUs: two 64-core AMD EPYC 7H12 sockets
with SMT. Its cgroup quota was only 2,720,000 microseconds per 100,000-
microsecond period, or 27.2 effective CPU cores. The run therefore used 24
fork workers with OMP, OpenBLAS, and MKL fixed to one thread per worker. All 24
workers were observed CPU-active. Raising the pool toward 64 or 256 would
oversubscribe the quota rather than increase available CPU time.

The GPU was one NVIDIA GeForce RTX 3090 (24 GiB). Runtime provenance was
Python 3.12.3, NumPy 2.1.2, PyTorch 2.8.0+cu128, and CUDA 12.8. A Pro 6000 is
not required. Launch-to-completion was approximately 9 minutes 14 seconds,
from the empty-log creation time to the recorded completion timestamp.

## Reproduction

External model/capture paths are intentionally not committed. With identical
mounted inputs, run from the experiment directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts \
python scripts/run_split_interaction_field_study.py \
  --config configs/qwen36_mxfp4_split_interaction_field.json \
  --pr11-config configs/qwen36_mxfp4_interaction_field.json \
  --pr10-config configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/qwen36_mxfp4_candidate \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/qwen36_exact_confirm_captures.npz \
  --pr11-dir /path/to/qwen36_mxfp4_interaction_field_20260820_v1 \
  --output /path/to/qwen36_mxfp4_split_interaction_field_20260820_v1 \
  --device cuda:0
```

Generate canonical analysis from repository-relative paths:

```bash
PYTHONPATH=src:scripts python scripts/analyze_split_interaction_field_study.py \
  --config configs/qwen36_mxfp4_split_interaction_field.json \
  --validation-dir results/qwen36_mxfp4_split_interaction_field_20260820_v1 \
  --output results/qwen36_mxfp4_split_interaction_field_20260820_v1/analysis
```

Focused verification:

```bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_split_interaction_field.py \
  tests/test_split_interaction_field_runner.py \
  tests/test_split_interaction_field_analysis.py
```

The final focused suite passes 20 tests. The final full repository suite passes
317 tests with one CUDA skip in 167.75 seconds; its only warning is the
inherited pandas/numexpr version warning. The prelaunch locked-pod suite passed
19 tests before the final analyzer-only tamper regression was added. A second
clean canonical analyzer invocation reproduces all 12 analysis-directory files
byte-for-byte.

## Immutable evidence

The run closed 69 validation invocations over all 12 locked layer/expert cells
and emitted exactly 4,278 rows: 3 fixed correction budgets across two exact
controls and four factors times four compressed controls, plus four
factor-specific strict all-in budgets across the same-solver four/eight-state
pair. It contains no test scientific use.

| artifact | SHA-256 |
|---|---|
| run facts | `56a838c271cf3c3bd54584fbf42d1f60fae0e1b1f1926eabc001b08c374a2e77` |
| frontier Parquet | `9b38c9056551de2ba14b462230c432f612b9ac5b0ebdd820b9118ffbfa9a2b24` |
| analysis manifest | `b7f727806ecbccdca3d2030c455f3f9c97b492748a92bdc9e9b571ebee7c7b9a` |
| promotion decision | `40b70815729b15fb34ddd07820765b671ecbba5dd4f1c71942f3df0bd6a1466e` |
| Markdown report | `f1955aeff5ce85d6f68a385cd9a11513011f6e3d974099c11bb210785f525a65` |
| analysis JSON | `265b4518a212719737f4afbc6a77c3420aebb768014b780a4704213fe1f1b33e` |
| all-in summary | `9b38bc83b23971384e3f4883c2b75fc9dbe98732dd24c52f4b876175ecd5214c` |
| solver diagnostics | `3c82d749d821de3eba5e70c90c5eafc94e034f6787d0f4ab4b40719008cb1aff` |

The canonical directory contains 14 files and 3,494,385 bytes. The analysis
manifest hashes all three inputs, all four source files, and all 11 generated
artifacts. All four JSON files are strict: no NaN or Infinity. The manifest
stores repository-relative inputs and sources.

The previous 2,898-row protocol is preserved on the pod under an explicit
`invalid_pre_same_solver_all_in_20260821` name and was not reused. One operator
restart while reconfirming the final timing-source hash was preserved as
`invalid_pre_field_build_walltime_20260821`; it committed zero work units and
contains no accepted evidence. Neither quarantine is in the canonical package.

## Attribution result at fixed one-correction-bpw

| factor/control | p10 recovery | median recovery | DP/solver median gain vs PR11 | action median gain, same solver | total median gain vs PR11 | max MACs |
|---|---:|---:|---:|---:|---:|---:|
| rank-8 INT4 Hadamard | 99.065% | 99.476% | +0.5753 pp | +0.8443 pp | +1.4304 pp | 1,329,152 |
| rank-8 INT8 | 99.094% | 99.488% | +0.4990 pp | +0.7756 pp | +1.3859 pp | 1,329,152 |
| rank-8 FP32 ceiling | 99.108% | 99.497% | +0.5516 pp | +0.8119 pp | +1.3707 pp | 1,329,152 |
| exact-proxy rank 4 | 99.084% | 99.477% | +0.5490 pp | +0.8173 pp | +1.4600 pp | 736,256 |
| exact full Gram | 99.168% | 99.512% | +0.7219 pp | +0.7353 pp | +1.5409 pp | nondeployable |

The three columns are paired distributions summarized independently, so their
medians need not add exactly. Rowwise, however, seed/optimizer gain plus
same-solver action gain equals total gain versus PR #11. The clean compressed
same-solver action-space effect is positive for every factor; the exact-Gram
control independently gives +0.7353 median points.

For INT4 specifically, p10/median recovery is 96.107%/98.022% with the legacy
PR #11 solver, 97.270%/98.504% with compressed four-state DP/same-solver
search, and 99.065%/99.476% with eight states.

## Strict all-in result

| factor | page cap | p10 recovery | median recovery | same-solver action gain | max MACs |
|---|---:|---:|---:|---:|---:|
| rank-8 INT4 Hadamard | 749 | 98.916% | 99.408% | +0.9355 pp | 1,329,152 |
| rank-8 INT8 | 741 | 98.887% | 99.388% | +0.8808 pp | 1,329,152 |
| exact-proxy rank 4 | 729 | 98.822% | 99.337% | +1.0074 pp | 736,256 |
| rank-8 FP32 ceiling | 697 | 98.572% | 99.237% | +1.0977 pp | 1,329,152 |

INT8 minus INT4 paired recovery is -0.0590/-0.0129/+0.0138 percentage points
at p10/median/p90. Thus the tiny fixed-correction INT8 median advantage
reverses under the all-in budget: INT4 has eight more pages, lower metadata,
and +0.0129 paired median points. The frozen all-in recommendation is INT4.
The rank-4 exact-proxy factor is within 0.071 median points of INT4 while using
about 55% of its charged MACs, making it the compute-oriented control.

## Actual solver work and wall time

| all-in factor | runtime p10/median/p90 | coordinate sweeps median/p90 | local passes median/p90 | accepted local bundles median |
|---|---:|---:|---:|---:|
| rank-8 INT4 | 3.581/4.214/4.304 s | 12/16 | 12/12 | 12 |
| rank-8 INT8 | 3.662/4.164/4.274 s | 12/15 | 12/12 | 12 |
| exact-proxy rank 4 | 3.641/4.148/4.239 s | 13/15 | 12/12 | 12 |
| rank-8 FP32 | 3.678/4.090/4.160 s | 12/16 | 12/12 | 12 |

These are actual wall times for the Python reference selector while 24 workers
share a 27.2-core quota; they are not single-request or optimized-kernel
latency. Field construction itself is only about 0.5 ms median for the
recommended INT4 row. Local repair exhausts the frozen 12-pass cap and accepts
a median 12 bundles, so PR #12 establishes a bounded-search result rather than
convergence. Further local passes were not added post hoc because they would
change both the frozen solver and its compute budget.

## Interpretation

The algebraic sufficiency claim survives the cleaner experiment. The large
old headline gain is a combination of meaningful four-state solver headroom
and a separately strong eight-state action-space gain. Under strict all-in
accounting, Hadamard INT4 is the better default than INT8, while rank-4 exact
proxy is the useful compute control. The remaining decisive problem is still
prediction of `h42/h24/h44` before suffix fetch and maintenance of the
approximate unfetched residual. No sealed holdout is run by this PR.
