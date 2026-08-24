# Qwen3.6 MXFP4 D1 layer-slice pilot

This compact package contains the promoted paired RTX 3090 D1 oracle result
for layers 0, 12, and 23. D1 is only the same token's next-layer router. The
384-page screen covers all three layers (96 groups); the 749-page screen covers
layers 12 and 23 (64 groups).

Historical PR #13 changed the exact paired top-8 set in 6/96 groups at 384
pages and 3/64 at 749 pages. The final exact request-coupled D1 repair changed
0 groups at both rates. Its mean local-qenergy damage was 0.981479 and 0.981571
times the corresponding PR #13 value.

The layer-0 repair matters methodologically: an exact BF16 top-k tie was stable
under the current-token and causal-prefix perturbations separately, but crossed
under their joint replay. The repair mixes complete one-sided finalists using
exact paired D1 replay. It is an oracle-only upper bound, not deployable
controller work.

`analysis/` contains the aggregate report, summaries, paired parity/VJP
summaries, and the regenerated-PR13 audit. `repair/` retains the complete final
allocation rows, including expert IDs, execution/selector weights, selected
pages, and complete 512-entry precision-state vectors. `runtime/` contains the
measured device scorer benchmark.

Complete candidate allocations, reconstructed deltas, coarse/eta screens,
parity outputs, and the exact code snapshot are retained on network storage at:

`/workspace/pr13_d1_layer_slice_pilot_20260824_v1`

The package makes no final-logit KL, D2--D4, all-layer, predictor-accuracy,
certificate, or production-latency claim.
