# Qwen3.6 MXFP4 D1 stratified layer-slice expansion

This compact package expands the paired RTX 3090 D1 oracle study to injection
layers 0, 1, 4, 6, 12, and 23 at 384 and 749 pages/expert. Each layer/rate
cell contains 32 complete validation groups, for 384 group/rate observations.
D1 is only the same token's next-layer router. Layer 6 enters full-attention
layer 7; the other sampled next-layer mixers use linear attention.

At 384 pages, historical PR #13 changes the paired top-8 set in 20/192 groups,
the exact-combined local-qenergy oracle changes 12/192, exact request-level D1
reranking changes 4/192, and the exact causal-prefix repair changes 1/192. At
749 pages, the corresponding counts are 11/192, 7/192, 1/192, and 0/192. The
repaired mean local-qenergy ratios to PR #13 are 0.982793 and 0.983981.

The fixed-policy result is weaker than the oracle upper bound but still
positive. At 384 pages, a fixed eta of 0.001 changes 7/192 groups versus
20/192 for PR #13, at 1.055705 times PR #13 local qenergy. At 749 pages, fixed
eta 0.00025 changes 2/192 versus 11/192, at 1.128662 times PR #13 local
qenergy. The full fixed-eta sweep is retained in
`analysis/d1_fixed_policy_aggregate.parquet`.

The implementation keeps page response token dependence explicit. Static
sidecar metadata describes page geometry, low-rank bases, coefficients, and
projection structure. Dynamic inputs are the current token activation,
gate/up intermediate responses, the D1 margin sensitivity, and eventually a
learned uncertainty correction. Exact oracle labels evaluate the signed
quantity `g^T d_p(x)`; exact Q4 routes, exact page effects, and exact VJPs are
not runtime inputs.

The one-adjoint runtime design costs an estimated 933,888 MAC/group, 1.819% of
the frozen PR #13 selector estimate of 51,343,360 MAC/group. The direct
candidate alternative is 9,781,248 MAC/group, or 19.051%. The measured RTX
3090 tensor scorer is 0.852 ms median and 0.868 ms p90 per refresh. Six
serialized refreshes would be about 5.1 ms, so the current result is not a
production-latency claim; refreshes must be reduced, fused, or overlapped.

Joint D1 metadata grows from PR #13's 72.125 KiB/group to 155.125 KiB/group,
or 167.625 KiB with uncertainty. At unchanged charged bpw this is offset by
reducing the four page caps from 384/576/749/768 to 363/555/728/747, or to
360/552/725/744 with uncertainty. Thus the proposed metadata does not silently
increase matched bandwidth, but it consumes the traffic of 21 or 24 refinement
pages/expert.

`analysis/` contains the byte-reproduced report and summaries. `repair/`
retains all 12 final request-coupled allocations, including complete expert
rows and 512-entry precision-state vectors. `parity/` records deterministic
same-host slice replay and cross-device diagnostics. `runtime/` retains the
device scorer benchmark.

Complete new candidate allocations, exact deltas, parity output, and the exact
code snapshot are retained at:

`/workspace/pr13_d1_layer_slice_expansion_20260824_v1`

The previously verified layer-0/12/23 inputs incorporated by this aggregate
remain at:

`/workspace/pr13_d1_layer_slice_pilot_20260824_v1`

This package makes no final-logit KL, all-layer, D2-D4, predictor-accuracy,
route-certificate, or production-latency claim. Experiment B—the sequential
runtime controller—remains separate and must report signed-effect correlation,
sign accuracy, top-page ranking accuracy, corrected-logit error, uncertainty
calibration, false certificates, fetched pages, and latency.
