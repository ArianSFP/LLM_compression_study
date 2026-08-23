# PR #13 H1 predictor experiments at a 0.55 bpw stream cap

Dates: 2026-08-22 to 2026-08-23

## Frozen contract

- The per-request streamed refinement cap is 0.55 bpw.
- All frozen allocator results in this log use 384 pages, or 0.523478 bpw.
- The selected sparse fallback streams 17 of 64 input blocks shared between
  gate and up: 139,281 bytes/expert, 1,114,248 bytes for top-8 experts, or
  0.5313148499 bpw.
- Static Q4 experts resident in CPU RAM are reported separately. They do not
  consume the per-request stream budget.
- The resident base is the natural nested Q2 representation from the MXFP4
  checkpoint, including the original E8M0 scales: 2.25 expert bpw.
- A resident Q4 fraction `f` therefore costs `2.25 + 2f` expert bpw in CPU RAM.
- Primary target: layer-14 validation p10 qenergy recovery >= 0.90 through the
  frozen PR #13 allocator. Activation MSE and response regression are screens.

## Budget audit and rejected predictors

The gate+up refinement denominator is `2*512*2048 = 2,097,152` weights/expert.

- Two INT8 low-rank residual heads fit at rank 136: 143,360 bytes/expert or
  0.546875 bpw. Layer-14 delta gate/up cosine was only 0.38685/0.37843 and
  interaction cosine 0.16910. Rejected.
- Three direct INT8 heads fit at rank 89: 142,848 bytes/expert or 0.544922 bpw.
  Delta gate/up/interaction cosine was 0.11215/0.16345/0.09771. Rejected.
- A four-head rank-66 design, operator/interaction hybrids, and other legal
  low-rank splits stayed below 0.55 bpw but did not improve this boundary.
- A shared HARP-trained state corrector lowered its HARP tune loss to about
  0.092 but increased external PR #13 layer-14 MSE from about 0.164 to
  0.164--0.166. Rejected for domain transfer.
- Shared in-domain PR #13 MLP correctors plateaued at epochs 1--2 and were
  stopped at epochs 20--22. Best validation MSE was 0.164036 versus the
  0.167375 all-layer baseline. Rejected.
- Local Chebyshev response maps from resident Q2 projections peaked near
  0.236/0.246 delta gate/up cosine at 0.25 bpw. Rejected.
- A learned 2-bit 512x512 cross-unit translator used 0.5390625 bpw and reached
  0.4298/0.4121/0.2120 delta gate/up/interaction cosine. Rejected.
- Fixed paired-row exact refinement at 0.547943 bpw reached
  0.5023/0.4941/0.4893 delta gate/up/interaction cosine.
- Dynamic activation-and-block-norm selection of 17 input blocks was the best
  legal nonresident fallback: 0.531315 bpw and
  0.54533/0.52067/0.32477 cosine. It was retained.

## Corrected allocator budget

The inherited launcher was hard-coded to 749 pages (0.998739 bpw). A wrapper
sets `BUDGET=384` and `RATE=0.523478` before invoking the frozen evaluator:
`run_allocator_0523.py`.

At 0.523478 bpw on layer 14:

- Sparse block-17 with all-Q2 parent: utility retained 0.869293, normalized
  regret 0.130708, median qenergy 0.805101, p10 0.692071.
- An over-budget Q2 asymmetric response diagnostic: utility 0.922085, regret
  0.077915, median 0.873563, p10 0.783763. This is diagnostic only and proved
  that response compression was not the sole bottleneck.

## Local-only static residency screen

This first screen used exact Q4 responses for resident experts locally at
layer 14, but retained the all-Q2 future hidden state. It was intentionally a
cheap conservative diagnostic.

- 15.625% residence (40/256), route-slot hit rate 0.4224:
  delta gate/up/interaction cosine 0.6676/0.6564/0.5272. Allocator utility
  0.900773, median 0.849867, p10 0.730587.
- 25% residence (64/256), hit rate 0.5431:
  cosine 0.7010/0.6986/0.5887.
- 31.25% residence (80/256), hit rate 0.5862:
  cosine 0.7139/0.7138/0.6108.
- 37.5% residence (96/256), hit rate 0.6207:
  cosine 0.7233/0.7248/0.6273. Allocator utility 0.910691, median 0.865381,
  p10 0.785380.

Conclusion: increasing local resident coverage without correcting the rollout
state produced weak gains. The future hidden state was the dominant bottleneck.

## End-to-end resident rollout capture

New scripts:

- `capture_pr13_resident_rollout.py`: one resident count.
- `capture_pr13_resident_frontier.py`: multiple descending counts from one
  dequantized model load.
- `evaluate_pr13_resident_capture.py`: exact identity alignment and state-MSE
  evaluation against PR #13 H1 examples.
- `export_resident_sparse_prediction_capture.py`: feeds an aligned resident
  state into the unchanged resident+sparse response exporter.

The immutable 22 GB checkpoint was staged to pod-local disk. The first
dequantized CPU load took 4m17s, used about 83--84 GB RSS, and the 15.625%
capture completed in 7m50s total. The frozen identity-grid SHA-256 matched the
exact PR #13 capture:
`075c851bbda32ad3009d4bfe56815a03458524b304ecb4165bb4c8f119538f00`.

Layer-14 validation activation MSE:

| Q4 residents/layer | Resident fraction | CPU expert bpw | H1 MSE | Reduction vs Q2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 2.25 | 0.164271 | -- |
| 40 | 15.625% | 2.5625 | 0.091131 | 44.52% |
| 64 | 25% | 2.75 | 0.065135 | 60.35% |
| 96 | 37.5% | 3.00 | 0.044433 | 72.95% |

The 15.625% capture reduced validation state MSE by roughly 40--70% through
every currently built PR #13 predictor dataset layer (0--24). Dataset files
25--39 were absent from that artifact and were not imputed.

At 15.625% residence with corrected rollout state, layer-14 response cosine was
0.9195/0.7286/0.7134/0.5825 for base/delta-g/delta-u/interaction. The allocator
at 0.523478 bpw retained 0.929119 oracle utility with normalized regret
0.070881, median qenergy 0.884315, p10 0.797194, and median/oracle-median ratio
0.906107. This passes 90% median retention but not the 90% p10 target.

At 25% residence, corrected-state response cosine was
0.9419/0.7857/0.7773/0.6712. At 37.5% it was
0.9603/0.8293/0.8223/0.7321. Both retain the same 0.531315 streamed bpw.

## Artifact locations on network storage

- `/workspace/pr13_resident40_capture_20260822_v1`
- `/workspace/pr13_resident_frontier_capture_20260822_v1/resident064`
- `/workspace/pr13_resident_frontier_capture_20260822_v1/resident096`
- `/workspace/pr13_resident40_rollout_sparse17_allocator_l14_v1`
- `/workspace/pr13_resident_rollout_frontier_l14_v1`

The 37.5% corrected-state allocator result was still running when this entry
was first written and must be appended before final handoff.
