# Split interaction-field reproducibility

## Result and scope

This experiment is stacked on PR #11 commit
b276da7e53d09bdb9450021c6eb4ea7b704debd2. It changes only the action
space: gate, up, and down suffix pages can be selected independently. It
reuses the frozen PR #11 interaction factors and the immutable PR #10
validation cohort. No factor is refit and no test scientific tensor or value
is admitted.

The frozen result is a continuation to a predicted mixed-hidden interface.
Both predeclared deployable-factor controls pass; no single winner is selected
because the protocol did not predeclare a tie-break. The result remains an
exact mixed-H4 geometry ceiling, not a candidate-prefetch, latency, routing,
logit, token, or downstream-accuracy claim.

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

The state integer is 4g+2u+d. Its page cost is g+u+d.

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

The PR #11 states 00/G/D/GD map to 0/6/1/7. Therefore the eight-state
feasible set contains the four-state set at every page budget.

For hidden scalar h with Q2 down, exact self damage is

\[
q_{D2}(h)=h_{44}^{2}A-2h_{44}hB+h^{2}C.
\]

For Q4 down it is

\[
q_{D4}(h)=(h_{44}-h)^2 A.
\]

The signed residual signatures are respectively

\[
\rho_{D2}(h)=h_{44}L_4-hL_2,\qquad
\rho_{D4}(h)=(h_{44}-h)L_4.
\]

Thus no new down Gram or factor metadata is required. The implementation
keeps all eight A/B/C self terms exact and approximates only cross-unit terms
with the already serialized L4/L2 rows.

## Solvers and controls

The promotion-eligible compressed path has two seeds:

1. all state 0;
2. an exact-self multiple-choice dynamic program under the page budget.

Each seed receives eight-state coordinate descent. The better compressed
objective receives bounded 1/2/3-unit local repair. Coordinate scoring uses
only two rank-r dot products per unit per sweep. At rank 8 the complete
conservative upper bound is 1,329,152 MACs, below the strict 1,572,864 gate.

Two controls prevent optimistic interpretation:

- the PR #11 four-state solver is recomputed with the same factor and checked
  against the immutable PR #11 frontier;
- a training-only exact three-Gram solver runs the identical optimization
  first over states 0/6/1/7, then over all eight states while retaining the
  four-state solution as a seed. Every exact eight-state row must be no worse.

An inherited-four-state warm-start diagnostic is charged with both selector
paths. It changes none of the 828 compressed primary solutions and is
nonpromotable.

## Frozen inputs and sources

| object | SHA-256 |
|---|---|
| split config | e62fdc4d3ffdc45482342736215433e443e59d9037e597f4cb72df39bbfa17f3 |
| split runner | 3528d3a865f6089ea189ab339fea5285a00fad140b3f5270912a1079c31d81f6 |
| split core | 3f977a67293b53d6919b18b6322305e03b6e7bfd1ea3102af0b5bda469932760 |
| analysis wrapper | c0af815eb3bfcbf3216e6747af0c126db60e84d174d52cc93968c18185aee957 |
| analysis core | d47f00b4c8eb7effbe9fcc478069ead7543711b0d26fc81152d5b73ce9ef5210 |
| PR #11 run facts | 835ac98e495566cb1f2aab181d5be3871174c859dcdf77de74470a751872396e |
| PR #11 factor manifest | d9c2974927a5bf26b7258ddc493217c472cf3ebbe0b23b9e77571e406a4ff280 |
| PR #11 frontier | 2f251a9d92a12de448712d96a0de87d59f2e596733a04df5070e8b5326f71b47 |
| exact capture | 52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931 |
| selected trees | da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8 |
| checkpoint config | 52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a |
| checkpoint index | 842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb |

The final tests have these hashes:

| test | SHA-256 |
|---|---|
| split core | a7aca04b117c2493be37c6703c93016280c51bfbae0e531ad81972b655185693 |
| runner | 89029b0bc34e24d01aeed206abd1b5f5ad6f24bddce88d8979ae19f56c633898 |
| analyzer/E2E | 5955d8e3b0a407b10fc42a58da3cf69aeceb56ee76a72e12dcc0f7258826a524 |

## Hardware and parallelism

The rented host exposed 256 logical CPUs: two 64-core AMD EPYC 7H12 sockets
with SMT. The container cgroup was limited to 2,720,000 microseconds per
100,000-microsecond period, or 27.2 effective CPU cores. The run therefore
used 24 fork workers and fixed OMP, OpenBLAS, and MKL to one thread per worker.
This retains 3.2 cores of quota headroom and avoids nested BLAS
oversubscription. Increasing workers above approximately 27 would not create
more licensed CPU time.

The GPU was one NVIDIA GeForce RTX 3090 (24 GiB). Runtime provenance was
Python 3.12.3, NumPy 2.1.2, PyTorch 2.8.0+cu128, and CUDA 12.8. A Pro 6000 is
not required for this experiment.

## Reproduction

External model/capture paths are intentionally not committed. With the same
mounted inputs, run from the experiment directory:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1     CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts     python scripts/run_split_interaction_field_study.py       --config configs/qwen36_mxfp4_split_interaction_field.json       --pr11-config configs/qwen36_mxfp4_interaction_field.json       --pr10-config configs/qwen36_mxfp4_set_utility_distillation.json       --checkpoint /path/to/qwen36_mxfp4_candidate       --trees /path/to/selected_trees.json       --exact-captures /path/to/qwen36_exact_confirm_captures.npz       --pr11-dir /path/to/qwen36_mxfp4_interaction_field_20260820_v1       --output /path/to/qwen36_mxfp4_split_interaction_field_20260820_v1       --device cuda:0

Generate analysis from repository-relative paths:

    PYTHONPATH=src:scripts python scripts/analyze_split_interaction_field_study.py       --config configs/qwen36_mxfp4_split_interaction_field.json       --validation-dir results/qwen36_mxfp4_split_interaction_field_20260820_v1       --output results/qwen36_mxfp4_split_interaction_field_20260820_v1/analysis

The focused local suite is:

    PYTHONPATH=src:scripts pytest -q       tests/test_split_interaction_field.py       tests/test_split_interaction_field_runner.py       tests/test_split_interaction_field_analysis.py

It passed 17 tests. The full oracle-study suite passed 314 tests with one CUDA
skip in 87.70 seconds. Its only warning was the inherited pandas/numexpr
version warning. The remote locked-environment prelaunch suite passed all 17
focused tests in 11.64 seconds.

## Immutable evidence

The run closed 69 validation invocations over all 12 locked layer/expert
cells and emitted 2,898 rows. It contains no test scientific use.

| artifact | SHA-256 |
|---|---|
| run facts | c4bbfab4cc69216e8015a6d97ff2befbcb04499646c4ade98630fc9ea526a1e1 |
| frontier Parquet | f46ee5ee8caeb35f7cc1bf97de483e26d1270c987c98a497b2f5d6631c4ebe71 |
| analysis manifest | 54532950762ea9de15824feccf85af84772bc704bd5ea0478f9f26ee0abc54d5 |
| promotion decision | 1a0a5181ddeae343c3db4fc4c285864ebc0d0f8eb1ecdb7400238029f2542966 |
| Markdown report | 3d1a9a7cce1a6b0f341f2c8cb6f7fbef5b74a428fe9ce6506af5c11ea419d7bb |

The directory contains 12 files and 1,899,743 bytes. The analysis manifest
hashes all three inputs, all four source files, and all nine generated
artifacts. Every JSON is strict: there is no NaN or Infinity. An independent
rerender reproduced all ten analysis-directory files byte-for-byte.

## Results

At one physical bpw:

| factor/control | p10 recovery | median recovery | paired median gain vs four state | p10 retention | metadata bpw | max MACs |
|---|---:|---:|---:|---:|---:|---:|
| rank-8 INT4 Hadamard | 99.065% | 99.476% | +1.430 pp | 100.010% | 0.023478 | 1,329,152 |
| rank-8 INT8 | 99.094% | 99.488% | +1.386 pp | 100.010% | 0.033895 | 1,329,152 |
| rank-8 FP32 ceiling | 99.108% | 99.497% | +1.371 pp | 100.010% | 0.091176 | 1,329,152 |
| exact proxy rank-4 | 99.084% | 99.477% | +1.460 pp | 100.010% | 0.049510 | 736,256 |
| exact full Gram | 99.168% | 99.512% | +0.735 pp | 100.010% | nondeployable | nondeployable |

The retention ratio may exceed 100% because PR #10's exact hybrid teacher used
the restricted four-state action space. The eight-state action set can
legitimately improve on that reference.

The rank-8 INT4 budget curve is:

| physical bpw | p10 recovery | median recovery | paired median gain | median split-only units |
|---:|---:|---:|---:|---:|
| 0.50 | 92.438% | 95.826% | +3.552 pp | 162 |
| 0.75 | 97.030% | 98.378% | +2.333 pp | 216 |
| 1.00 | 99.065% | 99.476% | +1.430 pp | 239 |

At one bpw its layer p10/median recoveries are:

| layer | p10 | median | paired median gain |
|---:|---:|---:|---:|
| 0 | 99.855% | 100.000% | +0.000 pp |
| 4 | 99.003% | 99.333% | +1.617 pp |
| 20 | 98.995% | 99.268% | +2.362 pp |
| 39 | 99.365% | 99.714% | +0.687 pp |

All 69 INT4 and all 69 INT8 primary invocations have nonnegative paired gain
at 0.5 and 0.75 bpw. At one bpw, 62 improve and seven tie for each encoding;
none regress. The exact-Gram control has the same 62-improve/seven-tie pattern.
The warm-start diagnostic and two-seed primary select identical states for all
828 factor/budget/invocation cases.

## Interpretation

The proposed algebraic sufficiency claim survives the real experiment.
Separating gate and up creates large gains without changing the static
interaction metadata. Most units use split-only states, especially in layers
4 and 20, so the result is not driven by a few exceptional swaps. The
remaining problem is no longer the eight-state optimizer or static down
geometry: it is prediction of h42, h24, and h44 before suffix fetch, followed
by the previously proposed approximate-unfetched-residual update.

No sealed holdout is run. The continuation is to a new deployable-interface
experiment, whose predictor and candidate contract must be frozen before
examining any new scientific data.
