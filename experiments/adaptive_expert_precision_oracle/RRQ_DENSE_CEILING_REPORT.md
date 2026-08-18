# Dense RRQ Ceiling for Adaptive Expert Precision Streaming

## Executive conclusion

Dense recurrent residual coding **removes PR #2's codec plateau but does not, by itself, make a dense stream competitive below 2 correction bpw**. On the 92 difficult-layer held-out invocations, the validation-selected raw heterogeneous RRQ reaches 68.21% median complete-expert recovery within a 2-bpw budget (the best discrete tuple actually reads 1.542 bpw), 87.28% within 3 bpw (2.333 actual), and 95.90% within 4 bpw (3.917 actual). The full three-stage `(3,3,3)` ceiling reaches 99.57%, proving that the recurrent code has ample fidelity; it costs 7.083 correction bpw and is not itself a bandwidth-saving deployment.

The key scientific outcome is therefore **partial support for RRQ as the action space, not support for dense RRQ as the final system**. At 3 bpw it improves the old nested-prefix control by 22.0 recovery points, and unlike that control it continues improving. However, it trails the independent direct-atom control at the 2–3 bpw transition and does not reach 90% difficult-layer median recovery until an asymmetric tuple just above 3 bpw. This makes the selective retention experiment—not more dense stages—the decisive next test.

This report is intentionally a dense-prefix ceiling. It asks whether recurrent
residual coding repairs PR #2's progressive-code plateau before any engineering
effort is spent on selective packets, H4 prediction, top-8 allocation, or a new
representation family.

## Authoritative model and data facts

The model is Qwen3.6-35B-A3B. The production reference is a mixed-format GGUF,
so this report uses `W_ref`/“production reference,” not “Q4,” for its expert
matrices. The discovered routed architecture matches the expected shape:

| Fact | Discovered value |
| --- | ---: |
| Routed layers | 40 |
| Routed experts/layer | 256 |
| Router top-k | 8 |
| Hidden dimension | 2,048 |
| Expert intermediate dimension | 512 |
| Gate/up | 512 × 2,048 each |
| Down | 2,048 × 512 |
| Weights/complete expert | 3,145,728 |

The bounded sample uses layers 0/4/20/39 and 12 training-frequency-stratified
experts per layer. Captures contain 5,036 layer-token records from 53/12/16
complete train/validation/test requests (3,292/760/984 records). Up to four test
invocations per sampled expert are used. The test comparison is exploratory:
PR #2's test outcomes motivated RRQ, so a fresh request set is required before
calling a later selective-RRQ result confirmatory.

Model, capture, source-checkpoint, repository, and serialized-package hashes
are in `results/qwen36_q2_rrq_dense_ceiling_20260818_v1/run_facts.json` and
`serialized_stage_manifest.json`.

## Representation and quantization

The resident base is the repository's deterministic activation-aware Q2
quantizer (levels ±1/±3, G64). Unlike the historical implementation, the
primary run executes the exact decode of serialized FP16 scales. Each residual
stage is constructed recurrently from that stored decode:

\[
E_0=W_{ref}-W_2,\quad S_k=Q_2(E_{k-1}),\quad E_k=E_{k-1}-\hat S_k.
\]

The bounded validation grid contains:

- zero-containing signed INT2 `{-2,-1,0,1}` with RTN or MSE scale search;
- affine INT2 with a byte-aligned per-group zero point, RTN or MSE scales;
- the existing `{-3,-1,+1,+3}` codebook as a historical control;
- G64 and G128 variants;
- a four-centroid scalar diagnostic that is not treated as the deployable
  two-bit format.

The deployable series is layer-independent but projection- and stage-specific,
selected with a width-two validation beam. The clean homogeneous series is
reported separately. FP16 scales are primary; BF16 (same storage width) and
FP32-scale mathematical controls are evaluated with complete-expert metrics.

## Exact rate and storage convention

One symmetric stage costs 2.25 bpw at G64 and 2.125 bpw at G128 after FP16
scales. The implemented affine G64 stage costs 2.375 bpw because zero points are
byte-aligned. A full down stage contributes one third of its projection rate to
the complete-expert average. Headers count toward external capacity; code,
scale, and zero-point streams count toward physical reads and are rounded as
separate contiguous streams. The primary physical granularity is 512 bytes;
1/2/4 KiB are reported as bookkeeping controls.

The Q2 base is Spark-resident and occupies 884,928 bytes per expert (2.250488
effective bpw including headers/scales). It is not counted as streamed traffic
or duplicated externally. The production-reference fallback is included in the
5× external-capacity constraint.

## Validation-selected construction

Validation selected the following layer-independent paths:

| Projection | Stage 1 | Stage 2 | Stage 3 | Validation remaining functional damage |
| --- | --- | --- | --- | ---: |
| Gate | affine-MSE G64 | affine-MSE G64 | affine-MSE G64 | 0.00380 |
| Up | midrise-MSE G64 | affine-MSE G64 | affine-MSE G64 | 0.00413 |
| Down | affine-MSE G64 | affine-MSE G64 | affine-MSE G64 | 0.00453 |

The training-only sequential-mixture calibration won for down (0.00447 validation remaining-damage ratio versus 0.00453 weight-only and 0.00526 authoritative-`h`). The clean best homogeneous three-stage control was affine-RTN G64. Its lower final validation score exposes bounded width-two beam pruning: the heterogeneous path is the specified deployable primary, while homogeneous results remain visible as a search-regret control. The four-centroid diagnostic reached 0.00046/0.00054/0.00082 remaining damage for gate/up/down, showing further scalar-codebook headroom but using more parameter bytes.

## Dense projection and sequential-expert results

Difficult-layer aggregate (layers 4/20/39, 92 held-out invocations; p10/median/p90):

| Budget cap (physical bpw) | Actual median bpw | Recovery p10 | Median | p90 | Median 95% request-bootstrap CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.750 | 0.18% | 27.06% | 47.98% | [17.86%, 31.08%] |
| 1.00 | 0.792 | 32.44% | 43.16% | 54.97% | [41.52%, 45.55%] |
| 1.50 | 0.792 | 32.44% | 43.16% | 54.97% | [41.73%, 45.55%] |
| 2.00 | 1.542 | 58.04% | 68.21% | 77.90% | [64.72%, 70.64%] |
| 2.25 | 1.542 | 58.04% | 68.21% | 77.90% | [64.72%, 70.13%] |
| 3.00 | 2.333 | 78.96% | 87.28% | 92.96% | [84.03%, 89.20%] |
| 4.00 | 3.917 | 93.36% | 95.90% | 97.62% | [95.37%, 96.06%] |

The steps are discrete because this run fetches complete dense stages. Figure: [dense rate–distortion](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/dense_rate_distortion.png).

All 64 `(gate depth, up depth, down depth)` tuples are evaluated. Down always
uses the actual sequential `h = SiLU(g_tilde) * u_tilde` in the primary expert
metric; authoritative-`h` down recovery is separately labelled as a projection
upper bound.

## Codec attribution and scale serialization

At a 3-bpw cap on difficult layers, median recovery is 87.28% for RRQ, 65.28% for the PR #2 nested-prefix control, and 91.23% for independently quantized direct atoms. At 4 bpw those values are 95.90%, 68.75%, and 95.26%. Thus RRQ recovers **22.00 points** over the defective nested codec at the first complete-stage operating point and **27.14 points** by 4 bpw, eliminating the old plateau. It does not dominate a direct independent code at every lower rate; recurrent residual construction becomes competitive as later stages arrive.

FP16 serialization of the resident Q2 scales changes mean functional base error by only 0.0011–0.0014% relative to fitted FP32 scales. FP16 and BF16 RRQ-scale complete-expert medians differ by at most 0.03 points at the reported 3/4-bpw caps. No FP16 scale became zero for down; later gate/up stages contain many exactly zero groups and 1,639/852 stage-three subnormals, but the BF16 control shows no material quality gain.

Median functional residual contraction per stage is gate 0.099/0.151/0.176, up 0.113/0.156/0.172, and down 0.129/0.176/0.199; no held-out stage increased aggregate invocation damage. RRQ works because every stage still contracts the functional residual strongly, although contraction weakens with depth. Figures: [functional contraction](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/functional_contraction.png) and [PMR correlation](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/pmr_contraction.png).

## Asymmetric allocation and down

The exact difficult-layer tuple frontier confirms that down remains the first and largest bandwidth consumer:

| Tuple `(kg,ku,kd)` | Actual average correction bpw | Recovery p10 | Median | p90 |
| --- | ---: | ---: | ---: | ---: |
| `(0,0,1)` | 0.792 | 9.40% | 32.83% | 51.48% |
| `(1,1,0)` | 1.542 | 19.33% | 48.30% | 70.82% |
| `(1,1,1)` | 2.333 | 78.18% | 87.28% | 92.96% |
| `(1,1,2)` | 3.125 | 86.41% | 91.44% | 95.20% |
| `(2,2,1)` | 3.917 | 87.54% | 93.58% | 95.91% |
| `(2,2,2)` | 4.708 | 96.12% | 98.01% | 98.81% |
| `(3,3,3)` | 7.083 | 99.25% | 99.57% | 99.81% |

A first dense down stage alone recovers 87.11% median down correction on difficult layers; a second reaches 97.69%. Gate/up first-stage medians are 90.26%/88.71%. The move from `(1,1,1)` to `(1,1,2)` adds 4.16 complete-expert recovery points, validating the user's observation that the next dense increment is usually down. Figure: [all tuple Pareto points](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/tuple_pareto.png).

## Page and external-storage accounting

The primary three-stage package is 2,785,856 serialized correction bytes per expert. With the production-reference fallback included, median external storage is **2.518× reference**, comfortably below the 5× cap; the resident 884,928-byte Q2 base is local and excluded. The 432 exact serialized packages total 133,721,088 bytes; their file hashes and sizes are consolidated in `serialized_stage_manifest.json`, while the payload files themselves are intentionally not committed.

Dense stage streams are contiguous and align exactly at all audited 512 B/1/2/4 KiB granularities, so measured page amplification is 1.000×. The page problem is therefore not responsible for dense RRQ's low-rate weakness; it reappears only when stages are split into selectable packets. Figure: [dense page amplification](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/dense_page_amplification.png).

## Compute accounting

Each dense RRQ stage adds one low-bit GEMV over its projection plus output
accumulation. A raw-space depth tuple `(kg,ku,kd)` adds
`2 × 1,048,576 × (kg+ku+kd)` arithmetic operations. The transformed full-support
gate/up control additionally evaluates one shared 2,048×2,048 analysis transform
per layer-token whenever gate or up correction is used; this control is not the
recommended dense deployment.

At the 3-bpw cap, the selected tuple reads a median 917,504 correction bytes, adds 6.29 million operations (1.00× one complete reference expert GEMV's arithmetic), and has 6.86 flop per streamed correction byte. At 4 bpw it reads 1,540,096 bytes and adds 10.49 million operations (1.67×). Under the stated GB10 headline limits, both points are analytically external-memory-bound: approximately 3.36/5.64 μs of 273-GB/s transfer versus 0.006/0.010 μs at peak sparse-FP4 arithmetic. This excludes protocol latency, base execution, unfused-launch overhead, and predictor overfetch, so it is a roofline lower bound—not a latency prediction.

NVIDIA specifies 273 GB/s unified-memory bandwidth and up to 1 PFLOP sparse FP4
for GB10. The saved roofline estimate uses those headline limits and is
analytical. The experiment ran on an RTX 3090 and does not claim measured DGX
Spark serving latency.

## Statistics and failure tails

Layer 39 remains the hardest at the 3-bpw cap (84.45% median), versus 87.36% at layer 4, 90.33% at layer 20, and 85.39% at layer 0. By 4 bpw medians converge to 95.51–96.56%, with difficult-layer p10 93.36%; the tail is therefore much healthier once the second-stage allocation becomes feasible. Only 29.35% of difficult invocations exceed 90% recovery at the 3-bpw cap, versus 100% at 4 bpw.

On the exact 39-invocation PR #2 paired subset, RRQ is 12.64 points worse at the 2-bpw cap because a complete dense stage does not fit flexibly, but is 5.56 points better at 3 bpw (95% cluster CI [1.21, 8.30]) and 11.11 points better at 4 bpw ([9.73, 14.41]). This crossover is precisely why selective RRQ packets are needed. Figures: [layer frontiers](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/layer_frontiers.png) and [failure tails](results/qwen36_q2_rrq_dense_ceiling_20260818_v1/plots/failure_tails_2p25bpw.png).

Intervals use 1,000 resamples over complete request IDs. Values are not clamped;
negative recovery remains present in Parquet and summary statistics.

## Direct replay and scope limits

Direct H1–H4 replay was not rerun. PR #2 established that the available BF16
Transformers graph does not reproduce the mixed-GGUF production-reference graph
at the required injection identity boundary. Therefore this report makes no
router-logit, top-8, logit-KL, or token-agreement claims. A production-GGUF
post-MoE injection hook remains a prerequisite.

Other limitations are the four-layer sample, 48 sampled experts, exploratory
reuse of the previous test requests, activation-diagonal rather than full
Hessian scale fitting, no fused RRQ kernel, and no joint top-8 cancellation.

## Decision and next experiment

**Proceed to the bounded selective-RRQ retention study; do not scale dense RRQ to all 40 layers.** Dense RRQ passes the codec-ceiling test: the full prefix reaches 99.57%, every stage contracts functional error, and the old 84.6% plateau is gone. It fails as a standalone bandwidth design: difficult-layer median recovery is only 68.21% within 2 physical bpw and requires 3.917 actual bpw for 95.90%.

The next study therefore reorients these stages into atom-major packets and directly measures how much of the full-support atom-RRQ benefit survives 0.5–3 bpw. It also measures whether stages two and three become sparser, whether bases must be refitted after each residual, and how much diagonal ranking loses to exact residual and complete-expert search. If selective retention remains weak, the recommended fallback is an activation-conditioned complete-expert residual VQ/local-affine atlas rather than more seeds of the same basis.

## Reproduction and artifacts

Every table and PNG/SVG plot is regenerated from saved Parquet files with the
single command in `README.md`. Exact commands, environment, failures,
corrections, timings, hashes, and pod state are in
`EXECUTION_LEDGER_RRQ_DENSE_20260818.md`.

Method context: [Recurrent Residual Quantization, arXiv:2608.04048v1](https://arxiv.org/html/2608.04048v1).
GB10 assumptions: [NVIDIA DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
