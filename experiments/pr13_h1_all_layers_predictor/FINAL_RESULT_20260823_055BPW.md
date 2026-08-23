# Final bounded result: 0.55 bpw H1 PR #13 pilot

This file completes `EXPERIMENT_LOG_20260822_055BPW.md` after its final
allocator run.

## Passing architecture

- Natural nested Q2 base resident for all experts, including original E8M0
  scales: 2.25 expert bpw in CPU RAM.
- Static damage-ranked Q4 refinement resident for 96/256 experts/layer
  (37.5%): total CPU expert representation 3.00 bpw.
- H1 state is the branch-conditional end-to-end resident rollout, not a local
  Q4 computation applied to the old all-Q2 hidden state.
- The resident rollout's predicted top-8 routes select stream recipients.
- Resident routes consume no stream. A fixed 136 shared gate/up input-block
  pairs are distributed across predicted nonresident routes, capped at 64 per
  expert. Exact target routes are not used for stream assignment.
- Maximum stream is 1,114,248 bytes/top-8 or 0.5313148499 bpw, below the hard
  0.55 bpw cap.
- Frozen PR #13 allocation uses 384 pages, equivalent to 0.523478 bpw.

## Layer-14 validation results

Resident H1 activation MSE is 0.044433 versus 0.164271 for the all-Q2 parent,
a 72.95% reduction. Resident-route top-8 set overlap is 0.8922 mean and 0.75
p10. Response cosine for base/delta-g/delta-u/interaction is
0.9603/0.8727/0.8718/0.8218.

Frozen allocator metrics:

- oracle utility retained: 0.9710846;
- normalized allocation regret: 0.0289154;
- qenergy recovery p10/median/p90: 0.904651 / 0.951685 / 0.978847;
- oracle qenergy p10/median: 0.958599 / 0.975949;
- median/oracle-median retention: 0.975138;
- page Jaccard: 0.614686;
- state overlap: 0.726866.

This is the first measured candidate to pass p10 >= 0.90 below the 0.55
streamed-bpw cap. It also retains 97.51% of the oracle median.

## Lower-residency boundary

The 25% resident point has activation MSE 0.065135 and was screened with the
same adaptive stream. It reached response cosine
0.9419/0.8109/0.8115/0.7428 and route overlap 0.8664 mean / 0.75 p10. This is
weaker on gate/up than the failed 37.5% fixed-block result, so another
3.5-minute allocator pass was rejected. The minimum demonstrated passing
residence is 37.5%; 31.25% remains unmeasured.

## Persistent artifacts

- Passing result: `/workspace/pr13_resident96_adaptive_stream_l14_v1`
- Resident captures:
  `/workspace/pr13_resident_frontier_capture_20260822_v1/resident096` and
  `/workspace/pr13_resident_frontier_capture_20260822_v1/resident064`
- 15.625% capture: `/workspace/pr13_resident40_capture_20260822_v1`
- Fixed-block frontier: `/workspace/pr13_resident_rollout_frontier_l14_v1`
