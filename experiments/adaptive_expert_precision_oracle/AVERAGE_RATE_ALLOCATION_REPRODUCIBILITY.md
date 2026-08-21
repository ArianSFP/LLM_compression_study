# Average-rate allocation reproducibility

## Scope

This study is stacked directly on PR #12 commit
`32601905f45c2e7caf402fd0a6d297d3fbd2a6d4`. It tests whether the same
eight-state rank-compressed interaction geometry benefits from an average page
budget over the true top-8 routed experts of one token/layer.

It does not test a predicted H4 response, a deployable online frontier kernel,
cross-layer bandwidth borrowing, causal H4 replay, latency, routing changes,
logits, tokens, or model quality. Recovery is exact combined top-8 H0 qenergy
recovery. All result rows are nonpromotable exact-H4 geometry ceilings.

## Frozen data boundary

- Factor fit split: train only.
- Factor fit scope: all 256 experts in layers 0, 4, 20, and 39.
- Selection/summary split: validation only.
- Validation groups: 32 per layer, 128 total.
- Every group contains the captured eight unique routed experts and normalized
  captured router weights.
- Evaluated expert invocations: 1,024.
- Test scientific tensors/values: not admitted or used.
- No bandwidth is borrowed between unrelated requests.

The exact capture SHA-256 is
`52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`.
The selected-tree SHA-256 is
`da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.

## Frozen representation and allocation

Primary factor:

- matrix-free randomized joint Q2/Q4 metric subspace iteration;
- rank 8, oversample 8, two power iterations;
- per-row signed INT4 after Walsh-Hadamard rotation;
- self-safe global shrink;
- 6,148 factor bytes plus 3,084 exact A/B/C bytes per expert;
- 9,232 resident bytes/expert = 0.023478 metadata bpw.

Compute control:

- exact rank-4 future-proxy coordinates, FP32;
- 16,384 factor bytes plus 3,084 A/B/C bytes;
- strict all-in correction mean 729 pages.

The primary mean correction budgets are 384, 576, 749, and 768 pages/expert.
The 749-page point is the strict all-in one-bpw point after metadata. Primary
burst caps are 768, 1,152, and 1,536 pages/expert. The group budget is always
eight times the mean; the full endpoint is 1,536 pages/expert.

Each expert frontier uses:

1. one vectorized exact-self DP table shared across all budgets and prices;
2. eight hard anchors;
3. incremental coordinate descent;
4. bounded 1/2/3-unit repair on hard anchors;
5. a descending 22-price warm path from two seed basins;
6. Pareto pruning before group allocation.

The deployable-style control minimizes the router-weight-squared sum of
compressed damages with exact multiple-choice knapsack. Equal-weight pooling
and exact combined-MoE qenergy optimization are validation-only controls.

## Hardware and worker contract

The supplied machine had one RTX 3090 with 24,576 MiB VRAM. It exposed 256
logical CPUs, but cgroup v1 fixed
`cpu.cfs_quota_us/cpu.cfs_period_us = 2720000/100000`, or 27.2 cores.
The frozen evaluation therefore used 24 fork workers with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

A 3090 is sufficient. One real layer-0/expert-62 matrix-free factor fit took
0.099 seconds. Reusing one vectorized DP table reduced the same full frontier
from 123.738 seconds to 4.557 seconds (27.2x). The complete grouped validation
took about 19 minutes wall time under the 27.2-core quota. Increasing to 64 or
256 workers would oversubscribe the quota.

## Commands

External model/capture paths are deliberately represented as placeholders and
are not committed.

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
  --output /path/to/average_rate/fit \
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
  --fit-dir /path/to/average_rate/fit \
  --output /path/to/average_rate/validation_exact_h4 \
  --device cuda:0
```

The fit resumes only from matching clean per-layer facts/sidecars. Evaluation
checkpoints after every 32-group layer and resumes only when source, data,
runtime, row-count, and diagnostics provenance match exactly.

Regenerate analysis from the repository root with the command in the README.
The analyzer refuses missing/extra policy rows, identity changes, nonfinite
values, page-cap violations, byte/bpw inconsistencies, state/count mismatches,
source-hash drift, test-scientific admission, and deployability mislabelling.

## Results

Primary rank-8 INT4 results:

| Overall average bpw | Uniform p10 | Uniform median | Pooled router-squared p10 | Pooled median | Paired p10 gain | Paired median gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.523478 | 95.175% | 97.823% | 96.744% | 99.011% | +0.128 pp | +0.693 pp |
| 0.773478 | 98.123% | 99.215% | 98.830% | 99.658% | +0.031 pp | +0.273 pp |
| 0.998739 | 99.337% | 99.710% | 99.576% | 99.881% | +0.006 pp | +0.104 pp |
| 1.023478 | 99.387% | 99.736% | 99.623% | 99.895% | +0.005 pp | +0.092 pp |

At strict all-in 0.998739 bpw, the 1,152-page and 1,536-page pooled burst caps
are essentially tied; the wider cap changes the aggregate median by only about
0.0045 pp. The rank-4 exact-proxy control reaches 99.514% p10 / 99.865% median
at 0.998728 bpw with 33.63M charged MACs, versus 51.34M for rank 8.

The largest per-layer strict-rate paired median gains are layer 4 (+0.239 pp)
and layer 20 (+0.186 pp), consistent with their less-saturated rate-distortion
curves. Allocation also follows router importance: mean selected pages fall
from about 963 at router rank 1 to 552 at rank 8.

Frozen status: `continue_to_predicted_h4_average_rate`. This is a continuation
decision only; `selected_deployable_configuration` remains null.

## Q3 follow-up boundary

Q3 remains the next representation study only after this attribution baseline.
It must contain all three controls:

1. ideal logical 256-byte bitplane ceiling;
2. physical 512-byte training-only co-selection layout;
3. physical 512-byte fixed gate/up pairing layout.

Physical accounting must charge unique page IDs, because a single useful
256-byte plane does not save a 512-byte transfer unless its paired plane is also
useful. Down-Q3 should follow only if the 18-state gate/up-Q3 ceiling is
material.

## Verification and hashes

Local full suite:

`PYTHONPATH=... pytest -q experiments/adaptive_expert_precision_oracle/tests`

Result: 332 passed, 1 skipped, one inherited pandas/NumExpr warning.

Final source hashes:

- config: `5edbcd24b4187d22ad1e5628203764d2327befb5df67307dcc4031e15dcba1e7`
- runner: `d3dcccbe853e06d91461116d31c4e92a3e4449a0dd62786f8b7b6bb5a07120a7`
- allocator core: `ef851ad3ebdf7d12706f934077ddc6a728c29a74d6e2f071276f457ad5109781`
- split core: `8928574b12c87553d21e1bda0c80dbb7d412d50e52da91273cdff408455eca82`
- analyzer wrapper: `6de5e9d781524f03c061a7028124f403805d500861c3a97f89ce6a67d77eaf08`
- analyzer core: `a428f866a6a520159d147f61f0ef6ae74d6811b44334f133ff7fbcf8f561f8a9`

Primary artifact hashes:

- fit facts: `1aca6f6ad683f3c7880885e831b70e4f3b4ccd2029d31e8da09f4fe0d35be46a`
- factor manifest: `de41f4e97950f1cac5134381db0fead802a6b774ab432fb2d7ec409632118557`
- evaluation facts: `1cd166b78f47c3315ffb59444d8c41b49de5ea9ba48d529b6ceaba2e817eb788`
- group frontier: `7aa91f647af40ff87f132e42bf6acba5a92a00d3d677a6628f0bf09df83a6fc5`
- expert allocation: `7c28832ac5943cfd429cc2a52e2bf3710666ec1aeb12ed45d37828d8d885c4eb`
- promotion decision: `2418eab8c23ebd70016ce4e74b9df621da737aaeb923ce38f23b9925cca17191`
- analysis manifest: `b31f2b9bebd2ded2253d47fa62942e1ef5b561ae548395031699edd386dd3bab`

Every generated non-manifest artifact was regenerated byte-for-byte under the
same environment after freezing the SVG hash salt and removing SVG date
metadata. The analysis manifest records repository-relative paths.
