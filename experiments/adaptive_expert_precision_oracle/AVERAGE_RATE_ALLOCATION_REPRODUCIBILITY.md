# Average-rate frontier-closure reproducibility

## Scope and outcome

This study is stacked directly on PR #12 commit
`32601905f45c2e7caf402fd0a6d297d3fbd2a6d4`. It tests average page-rate
allocation over the true top-8 routed experts of one captured token/layer, then
closes three specific PR #13 questions:

1. whether repairing only the frontier columns selected by the pooled allocator
   materially improves recovery;
2. whether the rank-4 exact-proxy interaction factor remains effective after
   FP16, INT8, or Hadamard INT4 quantization;
3. whether stronger three/four-expert exchanges or a global finite-frontier
   lower bound reveal remaining group-allocation headroom.

Frozen status:
`continue_to_predicted_h4_average_rate`. This is an exact-H4 geometry ceiling,
not a deployable selection result. `selected_deployable_configuration` is
null. Q3 remains deferred so it cannot confound the average-rate attribution.

Recovery means exact combined top-8 expert-output qenergy recovery. It is not
token accuracy, model quality, a latency measurement, or a routing/logit claim.

## Frozen data and scientific boundary

- Factor fit: train split only.
- Fit scope: all 256 experts in layers 0, 4, 20, and 39.
- Selection and summaries: validation split only.
- Validation groups: 32 per layer, 128 total.
- Each group contains the captured eight unique routed experts and normalized
  captured router weights.
- Routed expert invocations: 1,024.
- Test scientific tensors/values: not admitted or used.
- No bandwidth is borrowed between unrelated requests or layers.
- No predicted H4, causal replay, or online frontier kernel is evaluated.

Immutable inherited hashes:

| Input | SHA-256 |
| :--- | :--- |
| Exact capture | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| Selected trees | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| PR #12 config | `e61aae7b38a76b95c858f00344d0d9cc9e6d59fa4c686d8f28dbac170f877557` |
| PR #12 frontier | `9b38c9056551de2ba14b462230c432f612b9ac5b0ebdd820b9118ffbfa9a2b24` |
| PR #12 run facts | `56a838c271cf3c3bd54584fbf42d1f60fae0e1b1f1926eabc001b08c374a2e77` |
| PR #12 runner | `24735d86459d3c4d0aa7fb10c69259210efb2abf79ad759665054b406999173b` |
| Checkpoint config | `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` |
| Checkpoint index | `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb` |

## Factors and strict-rate accounting

Every factor retains exact FP16 A/B/C self terms, charged at 3,084 bytes per
expert. The correction page is 512 bytes. Overall average bpw includes the
factor, A/B/C, and the allowed correction pages.

| Factor | Factor bytes/expert | Strict correction pages | Overall bpw |
| :--- | ---: | ---: | ---: |
| Rank-8 Hadamard per-row INT4 | 6,148 | 749 | 0.998739 |
| Rank-4 exact-proxy FP32 | 16,384 | 729 | 0.998728 |
| Rank-4 exact-proxy FP16 | 8,196 | 745 | 0.998739 |
| Rank-4 exact-proxy per-row INT8 | 6,148 | 749 | 0.998739 |
| Rank-4 exact-proxy Hadamard per-row INT4 | 4,100 | 753 | 0.998739 |
| Rank-4 INT8 proxy + rank-4 INT4 Euclidean tail | 10,244 | 741 | 0.998739 |

Primary mean correction budgets are 384, 576, 749, and 768 pages/expert. The
primary burst-cap grid is 768, 1,152, and 1,536 pages/expert. Group budgets are
exactly eight times the mean; the endpoint is 1,536 pages/expert.

## Frontier and allocation algorithms

A single vectorized exact-self DP table is built per expert and reused across
all hard budgets and Lagrange prices. The coarse frontier contains eight hard
anchors with incremental coordinate descent plus bounded 1/2/3-unit repair, and
a descending 22-price path from two basins with coordinate descent. Pareto
pruning follows.

The allocation-aware closure procedure is deterministic:

1. solve the current router-weight-squared MCKP;
2. rerun coordinate descent and full hard-budget local repair at each of the
   eight selected page counts;
3. add two prices at 0.5x and 2x the selected local marginal slope;
4. Pareto-prune and rerun MCKP;
5. stop on identical selected state vectors or after three rounds.

The deployable-style objective is the router-weight-squared sum of compressed
per-expert damage. Equal-weight pooling and exact combined-MoE qenergy are
validation-only controls.

For the exact combined control, the refined finite frontiers are first improved
with coordinate changes and pair swaps. A bounded search then evaluates exact
three/four-expert Cartesian exchanges over deterministic eight-option
shortlists. This produces a feasible upper solution, not a global certificate.

The global lower bound applies tangent convexity to the combined residual norm.
At every Frank-Wolfe iteration, an exact linear MCKP minimizes the tangent over
all finite refined columns. The lower value is globally valid for those
columns, but says nothing about unit states absent from their frontiers.

The frozen grid contains exactly 66 policy/factor/rate rows per group: 8,448
group rows and 67,584 expert-allocation rows. The analyzer rejects missing,
extra, duplicate, nonfinite, wrong-split, over-budget, or mislabelled rows.

## Hardware, workers, and observed runtime

The rented machine exposed 256 logical / 128 physical CPUs, but cgroup v1 set
`cpu.cfs_quota_us/cpu.cfs_period_us = 2720000/100000`, or 27.2 CPU cores.
Evaluation therefore used 24 fork workers with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

One RTX 3090 with 24,576 MiB VRAM is sufficient; a Pro 6000 is not required.
The complete closure validation took approximately 93 minutes. Layer work was
heterogeneous and 32 groups/layer leave an eight-group tail after the first
24-worker wave. More than 27 CPU-bound workers oversubscribes the quota; a
32-worker tail experiment might reduce wall time slightly, but it was not
included in frozen evidence.

At strict rate for the primary column-generated frontier:

| Work counter | p10 | Median | p90 |
| :--- | ---: | ---: | ---: |
| Coordinate sweeps/group | 2,324.7 | 2,833 | 2,884.3 |
| Local passes/group | 269.1 | 389 | 410.3 |
| Frontier-refinement rounds | 3 | 3 | 3 |
| Selected hard repairs | 24 | 24 | 24 |
| Adaptive-price solves | 48 | 48 | 48 |
| Selector wall time/group | 62.28 s | 84.05 s | 86.99 s |

These are quota-contended Python reference timings, not optimized-kernel
latency. The selector remains nondeployable because exact H4 is supplied.

## Commands

External model and capture paths are deliberately placeholders.

```bash
export PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python experiments/adaptive_expert_precision_oracle/scripts/run_average_rate_allocation.py \
  --phase fit \
  --config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json \
  --pr12-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_split_interaction_field.json \
  --pr10-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/exact_captures.npz \
  --pr12-dir /path/to/pr12_results \
  --output /path/to/frontier_closure/fit \
  --device cuda:0

python experiments/adaptive_expert_precision_oracle/scripts/run_average_rate_allocation.py \
  --phase evaluate \
  --config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json \
  --pr12-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_split_interaction_field.json \
  --pr10-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/exact_captures.npz \
  --pr12-dir /path/to/pr12_results \
  --fit-dir /path/to/frontier_closure/fit \
  --output /path/to/frontier_closure/validation_exact_h4 \
  --device cuda:0
```

Evaluation workers are frozen to 24 in the config. Fit and evaluation resume
only from source-, data-, runtime-, row-, and diagnostics-identical checkpoints.

Regenerate analysis from the experiment root into a new directory:

```bash
cd experiments/adaptive_expert_precision_oracle
MPLBACKEND=Agg PYTHONPATH=src:scripts \
python scripts/analyze_average_rate_allocation.py \
  --config configs/qwen36_mxfp4_average_rate_allocation.json \
  --fit-dir results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/fit \
  --validation-dir results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/validation_exact_h4 \
  --output results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/analysis_reproduced
```

The output directory must not already exist. Independent reanalysis regenerated
the report, PNG, SVG, six CSVs, and promotion JSON byte-for-byte. Its manifest
differed only in the intentionally different generated-output path.

## Accuracy by overall average bpw

Primary rank-8 Hadamard INT4, 1,536-page burst cap:

| Overall average bpw | Uniform p10 | Uniform median | Coarse pooled p10 | Coarse pooled median | Column-generated p10 | Column-generated median | Exact combined local p10 | Exact combined local median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | 95.175% | 97.823% | 96.744% | 99.011% | 96.787% | 99.042% | 96.887% | 99.066% |
| 0.773478 | 98.123% | 99.215% | 98.830% | 99.658% | 98.886% | 99.674% | 98.895% | 99.683% |
| 0.998739 | 99.337% | 99.710% | 99.576% | 99.881% | 99.589% | 99.885% | 99.598% | 99.886% |
| 1.023478 | 99.387% | 99.736% | 99.623% | 99.895% | 99.628% | 99.897% | 99.646% | 99.899% |

Quantiles are not pairwise differences. The strict-rate paired
column-generated gains over uniform are +0.00667 pp p10 and +0.11213 pp median.

## Investigation results

### Selected-column repair

| Mean pages | p10 gain vs coarse | Median gain vs coarse | p90 gain vs coarse | Changed page vectors |
| ---: | ---: | ---: | ---: | ---: |
| 384 | +0.00098 pp | +0.02295 pp | +0.11763 pp | 83.6% |
| 576 | +0.00025 pp | +0.01087 pp | +0.04732 pp | 78.9% |
| 749 | +0.00001 pp | +0.00305 pp | +0.01622 pp | 71.1% |
| 768 | +0.00000 pp | +0.00198 pp | +0.01533 pp | 77.3% |

Repair changes many discrete page vectors but produces little strict-rate
recovery. Sparse frontier coverage is not the remaining primary bottleneck.

### Rank-4 exact-proxy encodings

Strict all-in results use the column-generated router-weighted policy:

| Factor | Pages | p10 | Median | Charged group MACs |
| :--- | ---: | ---: | ---: | ---: |
| Rank-8 Hadamard INT4 | 749 | 99.589% | 99.885% | 60.41M |
| Rank-4 FP32 | 729 | 99.532% | 99.868% | 39.85M |
| Rank-4 FP16 | 745 | 99.566% | 99.882% | 39.69M |
| Rank-4 INT8 | 749 | 99.573% | 99.884% | 41.19M |
| Rank-4 Hadamard INT4 | 753 | 99.590% | 99.887% | 39.92M |
| Rank-4 INT8 + tail-4 INT4 | 741 | 99.561% | 99.880% | 60.64M |

Rank-4 Hadamard INT4 improves p10/median by +0.00149/+0.00233 pp versus
rank-8 INT4 while reducing charged group arithmetic by 33.93%. The mixed tail
adds compute and metadata without improving recovery. Rank-4 INT4 is the best
next implementation default among these exact-H4 controls.

### Global group-allocation bound

Bounded three/four-expert exchange changes 2/128 page vectors and yields no
quantile-level recovery gain over the refined exact-combined local solution.
Median candidate evaluations are 126; p90 is 3,346.3.

The valid finite-frontier lower bound certifies 0/128 groups at the frozen
relative tolerance. Relative damage gaps are 31.19% p10, 34.75% median, 45.49%
p90, and 63.14% maximum; the absolute median gap is 9.35e-5. This is a useful
negative result: the stronger feasible search finds no improvement, but the
relaxation is too loose to claim that the discrete global optimum is closed.

## Layer and allocation sensitivity

For the strict primary column-generated policy:

| Layer | p10 | Median | Paired median gain vs uniform |
| ---: | ---: | ---: | ---: |
| 0 | 99.991% | 99.999% | +0.011 pp |
| 4 | 99.645% | 99.832% | +0.252 pp |
| 20 | 99.427% | 99.685% | +0.193 pp |
| 39 | 99.650% | 99.872% | +0.059 pp |

At strict rate, mean selected pages for router-rank 1 versus rank 8 are
808/669 in layer 0, 1,081/427 in layer 4, 1,013/573 in layer 20, and 925/552 in
layer 39. The allocator spends more on high-weight difficult-layer experts,
rather than simply assigning more pages to every low-recovery expert.

## Verification and provenance

Final verification:

- remote focused suite: 19 passed;
- local full suite before publication: 336 passed, 1 skipped;
- only warning: inherited pandas/NumExpr version warning;
- independent canonical reanalysis: all ten non-manifest artifacts byte-identical.

Frozen source and test hashes:

| File | SHA-256 |
| :--- | :--- |
| Config | `a889dbf9c5ebdf79de0e7e7b7d65165d9f39782d72d669e073c80966e5d83edf` |
| Runner | `55dd6f1b28b3dea36913943d5094b007176ddeabf8c12925dfbcf200118acaae` |
| Analyzer wrapper | `1862575a97f2af71df9560925a13d8f8331cbbd3ee84871d036a867babf33319` |
| Allocator core | `30ef9e95970533903754b2e68d299f8113a1c2c00f583a981bc3cc89498ba75b` |
| Analyzer core | `54d743b4f68ec1dbc9e624c259a85a8c9de8ee9f2c37196dba661ff115e5e05e` |
| Allocator tests | `2ff5d895218db25cfc38f97d9701f04095bb6f92ec8c89caf17b202228c32db5` |
| Runner tests | `9003ffb87d5009d0c8fba1a3a00f165bc1808f302c12aed4b86928e60fb98d08` |
| Analyzer tests | `2c86c99a7af369404efb2a3409c317acd1a47d31f494663a50dfb6b8cd792895` |

Canonical artifact hashes:

| Artifact | SHA-256 |
| :--- | :--- |
| Fit facts | `f4191af6db2041a5235a44fa9c3325baf7d3356670b440417db50e1f07f90693` |
| Factor manifest | `1b11e5fe32b27db0a7e1edcd45f57abebe76997d701c2e1b0ed80291a5b24f54` |
| Validation facts | `b791fc211f804826f2fba3661eb8cc4cb3859b4cff2e13a7e0b41c4e48a24166` |
| Group frontier | `80223f2bdf485ffd61c25056c2e524e368b1a48cb7eabffdebfae7d915974a9c` |
| Expert allocation | `f7fba5fbb509d902483c6bc70675cfa0d17ce884d5e1f5b96bf5933dcf07f891` |
| Validation accounting | `daa0664979e5589c051fac65ff514b8522dcf66718138169801ae8e24a93ad10` |
| Promotion decision | `59d55019d72a2895452c6afe679152ab7768293c75a664c3e499b5f0e080a2f2` |
| Markdown report | `ebbb3223f75be6050bc50c1f460fc9830a116f5c2035a020eb8c309b5d77a5bd` |
| Analysis manifest | `06d0c91dcc663d1c2694e7304f9f66f77cefb2bc4a086edf264233dd046f24a9` |

The canonical package contains 75,813,422 bytes (72.30 MiB); its largest file
is 25,754,504 bytes. No file approaches GitHub's 100 MB limit. The analysis
manifest records repository-relative paths and hashes every input, source, and
generated artifact.

## Next boundary

The result supports a predicted-H4, reduced-frontier implementation using the
rank-4 Hadamard INT4 factor. It does not authorize deployment or a test-set run.

Q3 must remain a separate attribution study with all three predeclared packing
controls: ideal logical 256-byte planes, training-only 512-byte co-selection,
and fixed 512-byte gate/up pairing. Physical results must charge unique page
IDs, not nominal half pages.
