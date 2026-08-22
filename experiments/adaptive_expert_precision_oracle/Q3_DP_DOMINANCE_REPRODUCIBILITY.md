# Rank-4 gate/up-Q3 solver-closure reproducibility

## Question and answer

This study closes the attribution problem in PR #14. It combines embedded
gate/up Q3 with PR #13's successful exact-self dynamic-programming seed,
coordinate/local solver, Rank-4 exact-proxy Hadamard INT4 factor, and strict
all-in accounting.

The corrected answer is:

- the inherited eight-state control reproduces PR #13 within the frozen
  absolute tolerance;
- physical gate/up Q3 improves strict-rate recovery, but only reduces median
  residual damage by 7.57% and left-shifts matched PR #13 quality by about
  0.009 bpw;
- neither predeclared promotion gate passes, so embedded gate/up Q3 stops and
  down-Q3 remains deferred;
- ideal independently readable 256-byte planes reduce median residual damage
  by 42.71%, so Q3 retains a strong representation signal, but current
  512-byte packing does not realize enough of it.

The earlier `-0.231 bpw` headline compared Q3 with PR #14's regressed
eight-state solve. It is superseded by the same-solver comparison here.

All recovery numbers are exact combined top-8 expert-output H4 qenergy on the
sealed validation cohort. They are not token accuracy, end-to-end model
quality, predicted-H4 behavior, measured kernel latency, routing, logit,
prefetch, or downstream claims.

## Stack and frozen scope

The branch is stacked on PR #14 commit
`27b0ecc5ac13fff70f2d8118e3d69c69e9df971e`. PR #14 is itself stacked on
PR #13. This continuation changes only the Q3 solver/frontier closure and its
analysis; it does not rewrite PR #14's historical evidence.

- Fit split: train only.
- Selection and summaries: exact-checkpoint validation only.
- Test scientific tensors or values: never admitted or used.
- Layers: 0, 4, 20, 39.
- Validation groups: 32 per layer, 128 total.
- Routed experts: the captured normalized top eight per token/layer.
- Routed expert invocations: 1,024.
- Factor: `exact_proxy_rank4_int4_per_row_hadamard`.
- Gate/up precisions: Q2, Q3, Q4.
- Down precisions: Q2, Q4 only.
- Allocation: one token/layer group; no borrowing across requests.

The exact capture, checkpoint, selected trees, PR #13 inputs, source hashes,
runtime identity, row grid, and split boundaries are fail-closed in fit,
validation, and analysis facts.

## Exact-self dynamic programming

For gate/up refinement stages `a,b in {0,1,2}` and down stage
`d in {0,1}`, fixed same-unit gate/up pairing has additive cost in 256-byte
quanta:

`c_i(a,b,d) = 2 [max(a,b) + d]`.

The exact-self seed is therefore:

`DP[i,b] = min_s DP[i-1,b-c_i(s)] + q_i(s)`.

The implementation builds one DP table per expert and reuses it across hard
budgets and price-path solves. The full 18-state and inherited restricted
eight-state controls then use the same incremental residual-field coordinate
descent and bounded local repair. This isolates the action space from the
seed/optimizer change that confounded PR #14's first comparison.

At strict rate, the restored eight-state control reaches 99.5855% p10 and
99.8824% median versus PR #13's frozen 99.5903% and 99.8869%. The respective
deltas are -0.00472 and -0.00454 percentage points, inside the predeclared
0.05-point absolute reproduction tolerance.

## Constructive dominance

Every Q3 frontier explicitly contains:

1. all final, refined inherited Q2/Q4 frontier state vectors;
2. all 18-state exact-self DP solutions;
3. every price-path solution;
4. every selected-budget repaired solution.

Each physical layout includes a fully charged legacy Q2/Q4 packing replica,
so an inherited state remains physically feasible at its original page cost.
The ideal 256-byte control additionally includes every final physical state
vector after re-costing it in ideal quanta. Runner assertions verify literal
state equality, page cost, compressed-objective dominance, and allocation
fallbacks.

This is a constructive candidate-set guarantee. The ideal coordinate/local
solution is still a heuristic exact-recovery control rather than a certified
global optimum. At strict rate the ideal result exceeds the best physical
witness by 0.1494 p10 and 0.0397 median percentage points. The signed report
stores this as witness-minus-ideal, so the gaps are negative; it does not call
the ideal solver worse.

## Strict all-in result

The frozen PR #13 target is 99.5903% p10 / 99.8869% median at 0.998739 total
bpw. The dense target region is sampled every four 256-byte quanta, or
0.002604 total bpw.

| Layout | Total bpw | p10 | Median | Median residual ratio vs PR #13 |
| :--- | ---: | ---: | ---: | ---: |
| Restored Q2/Q4 eight-state | 0.998739 | 99.5855% | 99.8824% | 1.0401 |
| Ideal logical 256-byte planes | 0.998739 | 99.7775% | 99.9352% | 0.5729 |
| Fixed gate/up pairing | 0.998739 | 99.6256% | 99.8917% | 0.9577 |
| Train co-selection, one layout | 0.998779 | 99.6281% | 99.8927% | 0.9490 |
| Train co-selection, two replicas | 0.998820 | 99.6281% | 99.8955% | 0.9243 |

The two-replica layout is the best strict physical median. Its 0.9243 residual
ratio is a 7.57% median residual-damage reduction, short of the required 20%.
Its external-storage multiplier is 1.6318. The replicated layout chooses one
complete page layout per expert and never mixes pages between replicas.

## Matched PR #13 quality

| Physical layout | Previous sampled total bpw | Minimum passing total bpw | Shift vs PR #13 |
| :--- | ---: | ---: | ---: |
| Fixed gate/up pairing | 0.989624 | 0.992228 | -0.006510 |
| Train co-selection, one layout | 0.987061 | 0.989665 | -0.009074 |
| Train co-selection, two replicas | 0.987101 | 0.989705 | -0.009033 |

The best certified shift is `-0.009074 bpw`, not `-0.231 bpw`, and does not
meet the frozen `-0.1 bpw` criterion.

## Accuracy by overall average rate

The canonical primary curve is the two-replica physical layout. The report and
CSV contain all 99 evaluated average-rate points. Representative points are:

| Allowed total bpw | Mean actual bpw | p10 | Median | p90 |
| ---: | ---: | ---: | ---: | ---: |
| 0.268351 | 0.268155 | 90.4477% | 96.9175% | 99.1912% |
| 0.497518 | 0.497271 | 96.5042% | 98.9376% | 99.8756% |
| 0.747518 | 0.747232 | 98.8039% | 99.6490% | 99.9869% |
| 0.859497 | 0.859229 | 99.3175% | 99.8031% | 99.9961% |
| 0.929810 | 0.929484 | 99.5005% | 99.8551% | 99.9986% |
| 0.989705 | 0.989318 | 99.6201% | 99.8905% | 99.9996% |
| 0.998820 | 0.998446 | 99.6281% | 99.8955% | 99.9996% |
| 1.013143 | 1.012741 | 99.6523% | 99.9012% | 99.9997% |
| 1.101685 | 1.101098 | 99.7646% | 99.9332% | 99.9999% |

Minimum total bpw at selected quality thresholds:

| Layout | Median 99% | Median 99.5% | Median 99.8% | Median 99.9% | p10 99% | p10 99.5% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Restored eight-state | 0.539103 | 0.684937 | 0.885457 | 1.039103 | 0.809937 | 0.955770 |
| Ideal 256-byte | 0.476603 | 0.601603 | 0.789103 | 0.921916 | 0.705770 | 0.851603 |
| Fixed pairing | 0.518270 | 0.684937 | 0.864624 | 1.018270 | 0.789103 | 0.934937 |
| One train layout | 0.539144 | 0.684977 | 0.867269 | 1.013102 | 0.789144 | 0.929769 |
| Two train layouts | 0.518351 | 0.685018 | 0.859497 | 1.013143 | 0.789185 | 0.929810 |

The ideal row's large left shift shows that the embedded Q3 endpoint is useful
when 256-byte planes are independently readable. The much smaller physical
shift localizes the current limitation to page packing/layout granularity and
its optimizer, not to the absence of a Q3 representation signal.

## Runtime and workers

The rented host exposed 256 logical CPUs but had a cgroup quota of 27.2 cores.
The final run used 32 fork workers and one BLAS thread per worker, about 1.18x
quota oversubscription. Increasing to 64 or 256 workers would only add
contention. The RTX 3090 was sufficient; this reference solver was primarily
CPU-bound.

| Layout | p50 complete 99-rate solve | p50 frontier | p50 allocation | Median coordinate sweeps | Median local passes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Restored eight-state | 168.38 s | 133.78 s | 34.42 s | 2,401.0 | 511.5 |
| Ideal 256-byte | 285.79 s | 40.38 s | 244.26 s | 2,562.5 | 422.0 |
| Fixed pairing | 258.20 s | 177.82 s | 77.55 s | 2,925.5 | 395.5 |
| One train layout | 265.01 s | 182.88 s | 77.87 s | 2,925.5 | 552.0 |
| Two train layouts | 344.23 s | 267.71 s | 74.62 s | 2,929.5 | 550.5 |

These are quota-contended wall times for offline construction, repair, and
allocation of the complete 99-rate frontier for one validation group. They
are not one-shot selector or kernel latency.

## Exact evidence grid

- Group rows: `190,080 = 128 x 5 layouts x 99 rates x 3 policies`.
- Expert rows: `506,880` primary allocation records.
- Every expert record stores its exact 512-state vector and page union.
- All 128 groups and 1,024 routed expert invocations are present.
- Missing, extra, duplicate, nonfinite, wrong-split, or over-budget rows fail.
- Validation completed with 128 diagnostics and no failures.

Canonical validation hashes:

| Artifact | SHA-256 |
| :--- | :--- |
| Fit facts | `0fb1b06b6230c1d38cecdf10d71febd415c03dcd166d3578863794dde90d36dc` |
| Fit manifest | `29549b0d3cf7d27e739f10a7dc41abd127f3e62f1e27257528dacc5eb29e0b1a` |
| Validation facts | `c6c90405d5fa25b4136b9667568e7d73eb891d59a6d1cd18dc9451e8d5306fa1` |
| Validation accounting | `92974f8b2980a8eb7d87f017b033bd9d70feb2801cbc7045882b5ec38cb0b638` |
| Group frontier Parquet | `7f32b57480fd6cb5358e99e032b4570a96f72444dbf20fda493db876acbf9a19` |
| Expert allocation Parquet | `d141578358222f143cd4dbb80794ada28e9c4bd3cac0d422cf2fc52e79d67b22` |

Canonical analysis hashes:

| Artifact | SHA-256 |
| :--- | :--- |
| Analysis manifest | `2f403b168cd82f4ab69c6049a01b522fc59f6dcfc46cbab20e5cc83e187e9b90` |
| Promotion decision | `48174b84511075edc00d453ad328c802f4c07fa8b518fff9cf23f6e596ee3317` |
| Full report | `8ba9569d87746003a422c3a0f6b4e2f989edbc015ba7b4cd0cc7354a23b4e7e5` |
| Accuracy-by-bpw CSV | `54f1d343a486ea57cbe3f88bb57c420c4a18c8f3ccfdadd08506e39884433778` |
| Threshold CSV | `0dc1c462d743c3e275377fd4bd3814bfd9f7ba252b9288a9c743bb4a39db2863` |
| Runtime CSV | `b9f06b638b061fd0448e3aadd65a4f46094165bba7ba3d3d72d1e399d0a4a1d3` |

Frozen source hashes:

| Source | SHA-256 |
| :--- | :--- |
| Config | `d5817503e23350785cdf0f0eae0cf4bd1d2c93c72f601baa29ab1ef82436a073` |
| Runner | `738eb06c35f11ba2a631e34f78d4d02e8332375e07b89b988c2afb3d0a0f33c9` |
| Analyzer wrapper | `8c373e24634a5d0fa1bd46d86cac9ccc2a475f777297e9d4953612569aa9ea46` |
| Analyzer core | `53ccdc3bc5299215361b297612dd0726568688a922828bac6c3f107a2842c6a7` |
| Q3 field core | `0db2424dbea62d4d31dd06ae3c6e6ad0d892424f549ebaf3aaf069e486756302` |
| Rate allocator | `5ad3de7ca43eb28af22e190e8438489f10d3460ad3479f898ebec28a022c2a1f` |
| Runner tests | `d84c99f2cca103239d4c15686e804adcef0649113afac0be0cca5c749b486970` |
| Analyzer tests | `1da96b7934b5f4507e2faa96596ab7331640bebf2c753bea402be05da651eedb` |

## Verification

- Focused local and remote Q3 suites: 23 passed.
- Complete local experiment suite: 359 passed, 1 skipped, one inherited
  pandas/NumExpr warning.
- `py_compile`: pass.
- `git diff --check`: pass.
- Credential, endpoint, and private-key scan: no match.
- The two raw facts files retain the audited checkpoint label
  `/workspace/qwen36_mxfp4_candidate`; it is provenance text, is not
  dereferenced by the portable analyzer, and contains no endpoint or secret.

## Reproduction

Run from `experiments/adaptive_expert_precision_oracle` with the PR #13 fit,
exact capture, checkpoint, and selected trees at their hash-locked paths.

```bash
PYTHONPATH=src:scripts python scripts/run_q3_gate_up_layout.py \
  --phase fit-layouts \
  --config configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --output results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/layout_fit \
  [hash-locked fit/capture/checkpoint/tree arguments]

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts python scripts/run_q3_gate_up_layout.py \
  --phase validate \
  --config configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --layout-dir results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/layout_fit \
  --output results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/validation_exact_h4 \
  --workers 32 \
  [hash-locked fit/capture/checkpoint/tree arguments]

MPLBACKEND=Agg PYTHONPATH=src:scripts \
python scripts/analyze_q3_gate_up_layout.py \
  --config configs/qwen36_mxfp4_q3_gate_up_layout.json \
  --layout-dir results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/layout_fit \
  --validation-dir results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/validation_exact_h4 \
  --output results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis_reproduced
```

The canonical manifest records the complete literal commands, paths,
dependencies, runtime identity, and all generated hashes. The abbreviated
bracketed arguments above are descriptive and must not replace those locked
manifest values.

## Canonical reviewer entry points

- [Full report](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis/Q3_GATE_UP_LAYOUT_REPORT.md)
- [All 99 average-rate accuracy points](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis/q3_accuracy_by_bpw.csv)
- [Matched-quality thresholds](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis/q3_minimum_bpw_thresholds.csv)
- [Frozen stop decision](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis/q3_gate_up_promotions.json)
- [Analysis manifest](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/analysis/q3_analysis_manifest.json)
- [Exact validation facts](results/qwen36_mxfp4_q3_dp_dominance_20260822_v1/validation_exact_h4/q3_gate_up_run_facts.json)

Only canonical `layout_fit`, `validation_exact_h4`, and `analysis` directories
belong to the evidence package. Superseded or failed attempts were quarantined,
were not promotion inputs, and are not committed.
