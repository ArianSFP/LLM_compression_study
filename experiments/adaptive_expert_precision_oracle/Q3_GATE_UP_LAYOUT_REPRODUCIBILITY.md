# Rank-4 gate/up Q3 physical-layout reproducibility

> **Superseded attribution.** This file preserves PR #14's original evidence.
> The same-solver exact-self-DP closure in
> [Q3_DP_DOMINANCE_REPRODUCIBILITY.md](Q3_DP_DOMINANCE_REPRODUCIBILITY.md)
> supersedes the `-0.231 bpw` comparison and freezes the final gate/up-Q3
> decision as `stop_embedded_gate_up_q3`.

## Scope and outcome

This study is stacked directly on PR #13 commit
`dfba3748e51d6916dc1f28cb1d3b3250188adbbe`. It asks whether embedded Q3
for gate and up can left-shift the average-rate curve while retaining the
selected rank-4 exact-proxy Hadamard INT4 interaction field.

Frozen result:

- gate/up-Q3 status: `continue_to_predictive_q3_study`;
- selected physical layout: `q3_physical_fixed_gate_up_pairing`;
- down-Q3 status: `eligible_for_separate_followup`;
- selected deployable configuration: none.

Recovery is exact combined top-8 expert-output qenergy recovery on captured
exact H4. It is not token accuracy, end-to-end model quality, measured kernel
latency, predicted-H4 performance, causal replay, or a routing/logit claim.

## Algebraic contract

For gate/up levels `a,b in {2,3,4}`,

`h_ab = SiLU(G_a x) (U_b x)`.

Down remains in `{D2,D4}`, giving 18 states per unit. With exact per-unit
streams `A_i=d4_i^T M d4_i`, `B_i=d4_i^T M d2_i`, and
`C_i=d2_i^T M d2_i`, plus existing signed rows `L4_i,L2_i`:

- Q2 down: `rho_ab,2=h44 L4-h_ab L2` and
  `q_ab,2=h44^2 A-2 h44 h_ab B+h_ab^2 C`;
- Q4 down: `rho_ab,4=(h44-h_ab)L4` and
  `q_ab,4=(h44-h_ab)^2 A`.

Gate/up Q3 therefore adds scalar state comparisons but no new down Gram,
`L3`, or A/B/C stream. Each coordinate move still needs only the two latent
dots against `L4_i` and `L2_i`.

The factor is `exact_proxy_rank4_int4_per_row_hadamard`:

- rank 4;
- factor payload 4,100 bytes/expert;
- exact FP16 A/B/C 3,084 bytes/expert;
- no learned-layout descriptor for baseline, ideal, or fixed pairing;
- 16 bytes/expert amortized descriptor for one learned layout;
- 32 bytes/expert for two replicas.

## Physical layout controls

One Q3 refinement bitplane for a 2,048-weight gate/up row is 256 bytes.
Physical pages are 512 bytes. The solver uses integer 256-byte quanta but
charges physical layouts through exact unique page unions.

The controls are:

1. `q2q4_monolithic_same_solver_reference`: inherited eight states, same
   coordinate/local solver and factor, both projection bitplanes in the PR13
   Q2-to-Q4 page;
2. `q3_ideal_logical_256_byte_plane_ceiling`: independent 256-byte actions;
3. `q3_physical_fixed_gate_up_pairing`: same-unit, same-stage gate/up planes
   share a 512-byte page;
4. `q3_physical_training_coselection_single`: one train-only pairing;
5. `q3_physical_training_coselection_replicated2`: two complete train-only
   pairings, selecting the cheaper whole-layout union.

The replicated control never mixes replicas. External duplication is charged.
Every physical state records and hashes its exact page union. The analyzer
reconstructs every 512-unit state vector and independently checks page IDs,
unique counts, bytes, metadata, and total bpw.

### Ideal-control attribution

The identifier retains the predeclared word `ceiling`, but its coordinate/local
solve is not a certified global optimum. Every physical-layout state vector is
feasible in the ideal space at equal-or-lower cost. At strict rate the heuristic
ideal solve reaches 95.9537% p10 / 98.3971% median, while the fixed physical
witness reaches 96.3975% / 98.5807%.

The ideal solver therefore has at least 0.4438 p10 and 0.1836 median percentage
points of optimization headroom. The physical result remains valid; the
heuristic ideal row must not be described as a representation bound.

## Frozen data and boundary

- Factor and layout fitting: train split only.
- Selection and summaries: exact-checkpoint validation only.
- Layers: 0, 4, 20, 39.
- Validation groups: 32/layer, 128 total.
- Routed experts/group: captured normalized top eight.
- Routed expert invocations: 1,024.
- Test scientific tensors/values: never admitted or used.
- Bandwidth pooling: one token/layer group only.
- No Q3-down signature, predicted H4, prefetch, or downstream replay.

Learned layouts use at most eight train invocations/expert and six frozen price
ratios. Validation values never fit pairings or the interaction factor.

Immutable inherited hashes:

| Input | SHA-256 |
| :--- | :--- |
| PR13 base commit | `dfba3748e51d6916dc1f28cb1d3b3250188adbbe` |
| Exact capture | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| Selected trees | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| PR13 config | `a889dbf9c5ebdf79de0e7e7b7d65165d9f39782d72d669e073c80966e5d83edf` |
| PR13 runner | `55dd6f1b28b3dea36913943d5094b007176ddeabf79ad759665054b406999173b` |
| PR13 allocator | `30ef9e95970533903754b2e68d299f8113a1c2c00f583a981bc3cc89498ba75b` |
| PR13 fit facts | `f4191af6db2041a5235a44fa9c3325baf7d3356670b440417db50e1f07f90693` |
| PR13 fit manifest | `1b11e5fe32b27db0a7e1edcd45f57abebe76997d701c2e1b0ed80291a5b24f54` |
| PR13 validation facts | `b791fc211f804826f2fba3661eb8cc4cb3859b4cff2e13a7e0b41c4e48a24166` |
| PR13 promotion | `59d55019d72a2895452c6afe679152ab7768293c75a664c3e499b5f0e080a2f2` |

## Rate and solver contract

Overall average bpw includes A/B/C, factor, amortized layout descriptor, and
allowed correction bytes. Strict total storage is 393,216 bytes/expert. Fixed
pairing has 7,184 bytes/expert resident metadata and permits 1,507 256-byte
correction quanta, giving 0.999390 total bpw.

The grid contains 43 mean budgets from 384 through 1,664 quanta. Hard frontier
anchors extend to the full Q4 endpoint of 3,072 quanta/expert. The burst cap is
3,072 quanta, or 1,536 physical pages.

Each expert frontier contains a descending 22-price path plus hard anchors.
Coordinate descent maintains its rank-4 residual field incrementally. The
physical tracker vectorizes all 18 candidate page costs. Local repair
vectorizes unit/state changes and evaluates bounded 1/2/3-unit bundles over
eight positive/negative candidates.

Selected-column repair and adaptive prices follow PR13:

1. solve router-weight-squared MCKP over eight experts;
2. hard-repair each selected rate;
3. add prices at 0.5x and 2x nearby marginal slope;
4. Pareto-prune and rerun;
5. stop on stable states or the frozen round cap.

The exact combined-MoE policy is a validation-only coordinate/pair local
control, not a global certificate.

Exact evidence grid:

- group rows: 82,560 = 128 x 5 layouts x 43 rates x 3 policies;
- expert rows: 220,160 = 128 x 5 x 43 x 8 primary allocations;
- every expert row stores an exact 512-state vector;
- missing, extra, duplicate, nonfinite, wrong-split, or over-budget rows fail.

## Accuracy by overall average bpw

Primary selected layout: `q3_physical_fixed_gate_up_pairing`.

| Allowed total bpw | Mean actual bpw | p10 | Median | p90 | Median residual ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.268270 | 0.267863 | 88.2971% | 94.9717% | 98.5207% | 1.0557 |
| 0.351603 | 0.351164 | 90.7283% | 96.2844% | 98.8701% | 0.9731 |
| 0.434937 | 0.434554 | 92.6115% | 97.0238% | 99.0446% | 0.9146 |
| 0.518270 | 0.517911 | 93.9501% | 97.4860% | 99.2216% | 0.8758 |
| 0.601603 | 0.601222 | 95.0209% | 97.7730% | 99.3336% | 0.8557 |
| 0.684937 | 0.684465 | 95.5348% | 97.9557% | 99.4136% | 0.8818 |
| 0.747437 | 0.747000 | 96.1043% | 98.1070% | 99.4453% | 0.8696 |
| 0.768270 | 0.767853 | 96.1372% | 98.1914% | 99.4587% | 0.8460 |
| 0.830770 | 0.830193 | 96.2126% | 98.3748% | 99.4949% | 0.7964 |
| 0.893270 | 0.892469 | 96.3191% | 98.4638% | 99.4966% | 0.7801 |
| 0.934937 | 0.933764 | 96.3394% | 98.5163% | 99.4983% | 0.7753 |
| 0.976603 | 0.974508 | 96.3917% | 98.5606% | 99.4980% | 0.7595 |
| 0.997437 | 0.994329 | 96.3952% | 98.5747% | 99.4979% | 0.7535 |
| 0.999390 | 0.996206 | 96.3975% | 98.5807% | 99.4980% | 0.7516 |
| 1.018270 | 1.013381 | 96.3996% | 98.6237% | 99.4979% | 0.7318 |
| 1.059937 | 1.049328 | 96.4064% | 98.6739% | 99.5062% | 0.7103 |
| 1.101603 | 1.083138 | 96.4059% | 98.6850% | 99.5141% | 0.7146 |

The canonical CSV and report contain all 43 points. None of the 99%, 99.5%,
99.8%, or 99.9% median targets, nor 99%/99.5% p10 floors, is reached for exact
combined group recovery. This differs from PR13's per-expert-style headline
because this study reports the exact combined top-8 output.

### Strict layout comparison

| Layout | p10 | Median | p90 | Median damage ratio |
| :--- | ---: | ---: | ---: | ---: |
| Eight-state same solver | 95.9911% | 98.1115% | 99.3830% | 1.0000 |
| Ideal logical heuristic | 95.9537% | 98.3971% | 99.4673% | 0.8488 |
| Fixed gate/up pairing | 96.3975% | 98.5807% | 99.4980% | 0.7516 |
| Learned co-selection, one | 96.5708% | 98.4557% | 99.5544% | 0.8177 |
| Learned co-selection, replicas | 96.5234% | 98.4701% | 99.5493% | 0.8101 |

Fixed pairing wins the median-first frozen rule. One learned layout has best
p10 but gives up median. Replication does not justify its storage.

Fixed pairing reaches both strict eight-state p10 and median at 0.768270 total
bpw. Against 0.999390, the shift is -0.231120 bpw. At strict rate the median
remaining damage ratio is 0.751550, a 24.845% reduction.

### Layer and state use at strict rate

| Layer | p10 | Median | p90 |
| ---: | ---: | ---: | ---: |
| 0 | 97.0104% | 98.9958% | 99.5825% |
| 4 | 94.9333% | 98.1325% | 99.1633% |
| 20 | 95.9883% | 98.1784% | 99.1864% |
| 39 | 97.8448% | 98.8566% | 99.7803% |

Mean selected units/expert:

- gate Q3-or-above 254.61; gate Q4 154.99;
- up Q3-or-above 274.72; up Q4 158.75;
- down Q4 266.46;
- actual use 1,502.11 of 1,507 allowed quanta.

## Hardware and optimization

The host exposed 256 logical CPUs, but cgroup quota was 27.2 cores. Evaluation
used 32 fork workers with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
Oversubscription was 1.176x, below the frozen 1.2x ceiling. A single RTX 3090
with CUDA 12.8 and Torch 2.8.0+cu128 was sufficient.

| Layout | p50 full 43-budget frontier | p90 | p99 | Median sweeps | Median local passes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Eight-state | 161.64 s | 165.68 s | 168.90 s | 2,518.5 | 671.5 |
| Ideal Q3 | 47.20 s | 48.20 s | 49.91 s | 2,772.0 | 641.0 |
| Fixed pairing | 181.92 s | 187.22 s | 195.41 s | 2,978.5 | 563.0 |
| Learned single | 179.39 s | 185.49 s | 188.90 s | 2,927.5 | 554.5 |
| Learned replicas | 273.26 s | 281.94 s | 286.72 s | 2,929.0 | 548.5 |

These are quota-contended reference timings for all 43 budgets, not kernel
latency. Vectorized page-cost evaluation was 32.27x faster than exhaustive
recomputation; local move scans were also vectorized.

Superseded runs were quarantined before canonical evidence: pre-state-evidence,
pre-vectorized-page-cost, pre-vectorized-local-scan, pre-32-worker,
pre-total-bpw-reporting, and pre-ideal-state-evidence attempts. None is in the
canonical package or promotion input.

Two post-validation analyzer attempts stopped without completion manifests:
one used an overly strict 1e-8 router-sum tolerance despite only 1.71e-7 FP32
drift; one had unresolved repository-relative manifest paths. The final
analyzer uses eight FP32 epsilons, rejects material drift, resolves then
relativizes paths, and renders deterministic SVG.

## Reproduction

External paths are placeholders.

```bash
export PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python experiments/adaptive_expert_precision_oracle/scripts/run_q3_gate_up_layout.py \
  --phase fit-layout \
  --config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --pr13-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json \
  --pr13-dir /path/to/pr13_frontier_closure \
  --pr10-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/exact_captures.npz \
  --output /path/to/q3_run/layout_fit \
  --device cuda:0

python experiments/adaptive_expert_precision_oracle/scripts/run_q3_gate_up_layout.py \
  --phase evaluate \
  --config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --pr13-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json \
  --pr13-dir /path/to/pr13_frontier_closure \
  --pr10-config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /path/to/qwen36_mxfp4_checkpoint \
  --trees /path/to/selected_trees.json \
  --exact-captures /path/to/exact_captures.npz \
  --layout-dir /path/to/q3_run/layout_fit \
  --output /path/to/q3_run/validation_exact_h4 \
  --device cuda:0
```

Analyze from the experiment root into a new directory:

```bash
cd experiments/adaptive_expert_precision_oracle
MPLBACKEND=Agg PYTHONPATH=src:scripts \
python scripts/analyze_q3_gate_up_layout.py \
  --config configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --layout-dir results/qwen36_mxfp4_q3_gate_up_layout_20260821_v1/layout_fit \
  --validation-dir results/qwen36_mxfp4_q3_gate_up_layout_20260821_v1/validation_exact_h4 \
  --output /path/to/q3_analysis_reproduced
```

Independent final analysis regenerated all 11 files byte-for-byte, including
PNG, deterministic SVG, promotion JSON, report, and manifest.

## Verification and hashes

- Focused Q3 suite: 19 passed.
- Complete suite: 355 passed, 1 skipped in 89.47 seconds.
- Only warning: inherited pandas/NumExpr version warning.
- Python compilation and `git diff --check`: pass.
- Strict JSON parse: no NaN or Infinity.
- Independent reanalysis: 11/11 byte-identical.

| Source/test | SHA-256 |
| :--- | :--- |
| Config | `f3eabd1ae59a1d6521ca95b8341b54ed03629e151a141595df662bb29aa7b2c0` |
| Runner | `dd1646325205656d4debcd8cac9d4e8ab96305366d70401edbfadb98ed6cc7bd` |
| Analyzer wrapper | `3ffe8417861a03ad1d2893b23322c69b5c1acbf0d783ed42182aafcc0a4503ef` |
| Q3 field | `35bf2a69b1350e1f85da793b23721834a8479abf9acea8d68b31ec61e5b92535` |
| Q3 allocator | `b50c98486c5b08a5644f33098189d09d30f75547dc27bcd9e98173aaa7c0b1ae` |
| Analyzer core | `d46f453201b1940008a0dcbab32bfcd0f9e17bef3444d4478ea2c535df6e55fb` |
| Field tests | `71391df9e2db6e55516de66d5c53efeb2919c5b305323ac64950504871f88625` |
| Allocator tests | `303c8e99d445b320e2b512d1723c4c864230154a0290461918087987673fcdaf` |
| Runner tests | `1173d71b33dbfecc6c3363092533eab79f6ad140f02d2b8db6dac7b63b8031fc` |
| Analyzer tests | `fcfdd618dd17d046e3e79ce039fff026204252ab0f10aaa6b67125b3d31c5221` |

| Canonical artifact | SHA-256 |
| :--- | :--- |
| Layout facts | `55430e92635d5f109e2a6dcbf84b47e42cbf0f7d57023327a696ba8a765b59e5` |
| Layout manifest | `c4edd13a53c300761afc91ef042ae28f2f90b5f854fa7cb267830a4b693698ac` |
| Validation facts | `7ad889e4b484d6a2b4963fa1f6b6bce593645d4208c19308e8521faca69505be` |
| Group frontier | `667419eef36b5c57aea7ddf25984b09774f92eb052f8c4208e57a1e4edce5153` |
| Expert allocation | `b69e8f6116ebda71e3607a4338a3f498731eca4cefc7427d10b00b56264c993a` |
| Validation accounting | `b460d8e7a6358f745f83a261322bc55f5b8e1e19fcc1938de15106508a3f5058` |
| Promotion | `98c471bd03a800000c1edc75b647e738421517a31006e008f33b74247e011b88` |
| Report | `d326548a4efa3b512536fde28bb0c9def25496c2a22e249006236fa6c9cae4bb` |
| Accuracy CSV | `f9c529434de5ba58b2f744bd728047506bea7a25ec23877ae7be3a90022648cf` |
| Analysis manifest | `2ba1f936f8637939543a772fb1aa6d8489743de6615f7f329d53337175ae361d` |

Canonical package: 25 files, approximately 41.4 MiB payload; largest file
38,508,606 bytes. Only `layout_fit/`, `validation_exact_h4/`, and `analysis/`
are canonical. No quarantine or logs belong in the PR.

## Next boundary

Gate/up Q3 passes. Next is a separate predicted-H4 study using fixed gate/up
pairing and rank-4 Hadamard INT4. Down-Q3 may proceed only as a separately
frozen 27-state experiment with `L3` and exact `Q33,Q23,Q34` local streams.

The ideal optimizer also has measurable headroom. A dominance-seeded ideal
solver or stronger discrete bound should accompany, not delay, predictive
physical Q3. This exact-H4 result authorizes no test-set or deployment claim.
