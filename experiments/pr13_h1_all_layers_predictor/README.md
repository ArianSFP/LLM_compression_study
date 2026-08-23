# PR #13 H1 resident-response predictor

This directory contains a bounded H1 pilot for replacing the PR #13 response
oracle under a hard 0.55 streamed-bpw cap. The strongest measured design uses:

- the natural nested Q2 representation for every expert;
- damage-ranked Q4 refinement resident for 96/256 experts per layer;
- a branch-conditional resident rollout for H1 state and route prediction;
- a fixed 136-block gate/up stream budget recycled across predicted
  nonresident routes; and
- the frozen PR #13 response-field allocator at 384 pages (0.523478 bpw).

On the layer-14 validation pilot it reaches 0.044433 H1 activation MSE,
0.904651 p10 qenergy recovery, 0.951685 median recovery, and retains 97.11% of
oracle utility. The maximum streamed payload is 0.531315 bpw. Static resident
storage is reported separately: 3.00 expert bpw including the Q2 base and
scales.

Start with:

- `FINAL_RESULT_20260823_055BPW.md` for the result and limitations;
- `EXPERIMENT_LOG_20260822_055BPW.md` for rejected architectures and full
  accounting;
- `capture_pr13_resident_frontier.py` for resident-rollout capture;
- `export_adaptive_resident_stream.py` for the passing response predictor; and
- `run_allocator_0523.py` for the corrected frozen budget.

## Scope

The allocation result is measured on layer 14 with 29 validation groups. It is
not yet an all-layer deployment result. The Python exporters are offline
evaluation tools: a production implementation must operate only on predicted
top-8 routes, fuse Q2/Q4 decoding, and reuse rollout projections.
