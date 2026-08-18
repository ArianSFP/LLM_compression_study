# Oracle Study: Learned Diagonal-Mask Bases for Activation-Dependent Expert Precision Streaming

## 1. Executive conclusion

**Conclusion: partial support for projection-specific correction bases, but redesign is required for a model-wide W1 streaming system.** The core result is not a failure at 0.2 bpw; it is the full physical frontier. A deployable single-view hybrid—generalized gate, generalized up, native down—recovered 78.31% median complete-expert future-proxy-weighted damage at 1.9896 actual physical bpw across layers 0/20/39. A non-deployable multi-view H0 selector raised this to 81.95% (request-cluster bootstrap median 95% interval 73.76–90.10%). The six-layer extension fell to 76.65% median at 2 bpw and only 89.65% at 4 bpw.

The exception is routed layer 0: the deployable hybrid reached 95.08% at 1.5 average physical bpw and 96.69% at 2 bpw. Layers 4, 10, 20, 30, and 39 did not reproduce this. Thus the concept is useful as a layer-conditional component or fallback policy, but the present W1 plus diagonal-mask representation does not preserve near-Q4 behavior model-wide at materially below 4 physical bpw.

A Q2 resident base is a better design point. At 2 physical correction bpw, its complete-expert recovery was 85.53% versus 81.95% for W1, with a paired mean gain of 3.50 points (95% request-cluster CI 2.08–4.58), and median relative output error fell from 0.460 to 0.267. Even Q2 did not cross the model-wide near-Q4 bar at ≤2 bpw.

These are representation and sequential-expert results under a rank-4 future-router-gradient proxy. Direct teacher-forced H1–H4 replay was not executed because the authoritative captures did not provide a validated arbitrary-MoE-output replay checkpoint; the required replay-identity test could not be passed. No hidden-state, router-recall, logit-KL, or token-agreement number is inferred or fabricated.

## 2. Authoritative model and data facts

The exact source model was `Qwen3.6-35B-A3B`, HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, model-config hash `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.

| Fact | Discovered value | Expected value | Deviation |
|---|---:|---:|---|
| routed layers | 40 (layers 0–39) | about 40 | none |
| routed experts/layer | 256 | 256 | none |
| selected experts/token | 8 | 8 | none |
| hidden dimension | 2,048 | 2,048 | none |
| expert intermediate | 512 | 512 | none |
| gate/up shape | 512 × 2,048 | 512 × 2,048 | none |
| down shape | 2,048 × 512 | 2,048 × 512 | none |
| expert weights | 3,145,728 | 3,145,728 | none |
| attention heads / KV heads | 16 / 2 | not specified | discovered |
| vocabulary | 248,320 | not specified | discovered |

The authoritative streamed reference was the production GGUF `Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, 22,182,574,368 bytes, SHA-256 `e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6`. “Q4” is therefore a role, not a uniform four-bit format: at layers 0/20, gate/up are MXFP4 at 4.25 effective bpw and down is Q5_K at 5.5 bpw; at layer 39, gate/up are Q5_K at 5.5 bpw and down is Q6_K at 6.5625 bpw. Exact dequantized production tensors were used consistently.

The network-volume export did not include repository `.git` metadata. No project repository commit can honestly be reported. Its authoritative reproduction manifest identifies the underlying llama.cpp commit as `4fc4ec5541b243957ae5099edb67372f8f3b550e`; that code was not used for expert arithmetic, only the GGUF file and Python dequantizer were used. The local experiment workspace also began without a Git repository.

The core capture used six evenly spaced audited segments and 81 complete requests: 53 train, 12 validation, 16 held-out test. It contains 3,777 sampled layer rows, 1,259 per layer at 0/20/39, split into 2,469/570/738 train/validation/test rows. The limited extension used the same requests and positions at layers 4/10/30. Prompt-manifest SHA-256 is `916315802286cdead8fc9e4d12a3bdb288fe9feb9d0b3e84c21ecd6ea032c2f7`; tokenizer hash is `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

## 3. Experimental setup

Splits are by complete request. All activation covariances, residual operators, bases, binary/Q2 scales, and page layouts use training requests only. Test requests are used for final curves. The core Phase A sample contains 400 held-out expert invocations across 12 hot/median/cold experts per sampled layer. Sequential Phase B contains 167 held-out invocations. The extension uses six experts per layer and 50 sequential invocations total.

Physical budgets were swept at 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, and 4.0 bpw for projections, and at 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, and 4.0 average bpw for complete experts. Recovery values were never clamped in saved metrics.

The future proxy is local identity energy plus a rank-4 quadratic term fitted from training-only ridge gradients of the same-layer H1–H4 top-8 router-boundary margins with respect to captured post-MoE state. Direct replay remains the missing gold standard.

## 4. Quantisation definitions

The production GGUF is the authoritative reference. W1 uses fixed signs and one FP16 scale per 64-weight, per-output-row group. The primary base uses diagonal-activation-MSE-optimal scales. Its serialized size is 163,904 bytes per projection, 491,712 bytes per complete expert, or 1.250488 effective bpw including scales and a 64-byte header.

A simple `mean(abs(W))` W1 was also run. Activation-aware scales greatly reduce uncorrected layer-0 gate damage, but they did not improve the aggregate correction-stream frontier: at 0.2 physical bpw the simple base had gate/up/down recovery of 63.21/48.62/36.67%, versus 59.57/44.25/35.90% for activation-aware W1. This illustrates that reducing base MSE can make the remaining residual less sparse.

Q2 uses deterministic four-level signed codes ±1 and ±3 with eight alternating code/scale updates under the training activation diagonal metric, one FP16 scale per 64 weights, and 2.250488 effective resident bpw. Its median unstreamed damage was 28–40% of W1 damage depending on layer/projection.

Correction atoms are nested signed prefixes of an INT16 master code at 2/4/8/16 bits, with a scale and packet metadata. The complete-expert allocator tested 4/8/16-bit packets. At a 2-bpw cap, 96% or more of chosen down packets and essentially all gate/up packets used 4 or 8 bits; 16-bit storage was rarely useful.

## 5. Basis and dictionary methods

The study evaluated native coordinates, fixed random orthogonal, PCA, regularized residual-weighted generalized, three learned spectral-bootstrap orthogonal seeds, a two-view overcomplete dictionary, and a two-view H0 oracle. All full-rank bases passed reconstruction tests. Generalized bases used a regularized full-rank inverse with recorded retained covariance ranks and a maximum regularized condition number around 100,000.

For physical BF16 atoms, generalized dominates native for gate/up while native dominates all rotations for down:

| Projection / method | 0.2 bpw | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---:|---:|---:|---:|---:|
| gate generalized | 61.47% | 68.69% | 73.44% | 78.81% | 84.45% |
| gate native | 17.38% | 27.43% | 37.00% | 48.03% | 61.80% |
| up generalized | 46.16% | 54.89% | 61.10% | 67.70% | 75.90% |
| up native | 11.48% | 19.45% | 26.92% | 38.01% | 52.90% |
| down native | 26.97% | 42.44% | 54.37% | 67.04% | 79.66% |
| down generalized | 15.21% | 25.43% | 36.53% | 49.41% | 67.51% |

Learned orthogonal bases did not beat generalized/PCA. The two-view oracle improves generalized by less than one percentage point over most of the curve, so extra views are not the limiting factor.

## 6. Storage accounting

The full nested 16-bit correction view for one basis per projection plus Q4 fallback is 4.4538× the authoritative expert bytes at layers 0/20 and 3.7532× at layer 39, satisfying the 5× cap. Q4 plus complete 2/4/8-bit views costs 1.4538/1.8824/2.7395× at layers 0/20. Since the runtime oracle almost always chose 4 or 8 bits, storage beyond roughly 2.74× was not materially helpful.

The multi-method global H0 selector is not deployable under the same 5× cap because it assumes several full basis views coexist. The single generalized/generalized/native hybrid is deployable: it stores exactly one correction basis for each projection and stays under 4.454× even with the full 16-bit prefix.

## 7. H0 local oracle results

Progressive-precision, future-proxy-weighted projection recovery over layers 0/20/39 is:

| Physical cap | Gate median | Up median | Down median | All projections median | All-projection p10 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 52.43% | 37.50% | 25.98% | 40.22% | 11.56% |
| 0.2 | 59.57% | 44.25% | 35.90% | 45.88% | 16.97% |
| 0.5 | 66.60% | 52.44% | 49.81% | 56.28% | 27.91% |
| 1.0 | 71.73% | 58.80% | 63.24% | 64.66% | 36.16% |
| 1.5 | 75.42% | 63.59% | 70.33% | — | — |
| 2.0 | 78.05% | 67.66% | 75.54% | — | — |
| 3.0 | 83.35% | 76.87% | 84.24% | — | — |
| 4.0 | 90.29% | 86.45% | 93.19% | — | — |

Aggregate inverse rates are heavily censored at 4 bpw. Only 50.25/36.50/64.25% of gate/up/down invocations reach 90% by 4 bpw. Median b90 is 4.0 for gate, >4 for up, and 4.0 for down. Median b95 is >4 for every projection. Full censored distributions and request-bootstrap intervals are in `inverse_rate_summary.csv` and `inverse_rate_per_invocation.parquet`.

Layer 0 is different: progressive median b90 is 1.0/1.0/1.5 bpw for gate/up/down; b95 is 1.5/1.5/2.5; b99 is 3.0/3.0/>4. At layers 20 and 39, median b90 for gate/up is >4, and only layer-20 down reaches a finite median b90 (4.0).

## 8. Complete-expert results

At 2 average physical bpw, complete-expert sequential results are:

| Method | Storage ×Q4 | Actual bpw | Median recovery | p10 recovery | Page amplification | Extra FLOPs/Q4 expert |
|---|---:|---:|---:|---:|---:|---:|
| all native | 4.454× | 1.9896 | 65.12% | 46.46% | 4.11× | method-dependent |
| all PCA | 4.454× | 1.9896 | 73.26% | 47.36% | 2.40× | method-dependent |
| all generalized | 4.454× | 1.9896 | 74.61% | 49.52% | 2.38× | method-dependent |
| PCA/PCA/native | 4.454× | 1.9896 | 77.45% | 51.19% | 2.79× | method-dependent |
| generalized/generalized/native | 4.454× | 1.9896 | **78.31%** | 54.08% | 2.95× | method-dependent |
| learned/learned/native | 4.454× | 1.9896 | 76.84% | 53.56% | 3.30× | method-dependent |
| multi-view H0 oracle | >5× | 1.9896 | 81.95% | 57.82% | 2.86× | 38.66% |
| Q2 multi-view H0 oracle | >5× plus Q2 resident | 1.9896 | 85.53% | 67.82% | not separately summarized | similar |

The requested downstream columns—H4 hidden drift, router recall/flip, logit KL, and top-1 agreement—are **not measured**, because direct replay identity was unavailable. They are not silently replaced by the proxy.

At 2 bpw, the deployable hybrid achieves 96.69% at layer 0, 65.72% at layer 20, and 73.22% at layer 39. The multi-view H0 upper bound is 97.45/66.37/74.21%. The limited layers 4/10/30 reach only 74.84/71.51/70.34% with the multi-view oracle.

The authoritative-input down oracle exceeds sequential-realistic recovery by a median 5.27 points at 2 bpw (7.31 mean), confirming that gate/up error changes the down input materially. The median gap remains 2.08 points even at 4 bpw.

## 9. Joint top-8 allocation results

No joint top-8 layer-token allocator was run. Phase B optimizes each selected expert independently while retaining its captured router rank/coefficient in the metric rows. Multiplying the median 782,336 physical bytes/expert at 1.9896 bpw by eight gives 6.26 MB/layer-token before cross-expert page reuse, but this is accounting—not a measured joint allocation result. Router-rank allocation showed no trustworthy deployable policy because the H0 selection was per invocation.

This is an important limitation: joint cancellation and router-coefficient weighting could improve the frontier, while union-page effects could worsen it.

## 10. Downstream propagation results

The scalable future metric explicitly includes H1–H4 router-boundary gradients, and all “future-proxy” recovery values use that quadratic metric. No direct current-token or H1–H4 replay was performed. The available trace records contain future router logits and post-MoE states but not a validated arbitrary-output injection checkpoint plus remainder-of-forward replay identity path. Per the correctness requirement, downstream claims were stopped at the proxy rather than trusting an unvalidated replay.

Consequently, hidden drift, top-8 recall/Jaccard, boundary-flip rate, logit KL, top-1 agreement, and four-token free-running exact match are unavailable. This prevents a definitive quality go/no-go even though the representation result is weak.

## 11. Page-rounded bandwidth results

Four-KiB physical reads are the principal x-axis. At 2 bpw, median physical/logical amplification is 5.44× gate, 5.64× up, and 2.79× down for the progressive projection oracle; the complete-expert hybrid amplification is 2.95×. Physical accounting is already reflected in every headline rate, so these factors are not applied again.

At 2 physical bpw, traffic is approximately 2.0× lower than a nominal 4-bpw full expert. At 1.5 bpw it is 2.67× lower. The only region reaching ≥95% complete-expert recovery below 2 bpw is layer 0 (95.08% at 1.5 bpw). Overall six-layer recovery is only 76.65% at 2 bpw and 89.65% at 4 bpw, so a model-wide W1 deployment has no demonstrated useful traffic point.

The ideal-byte, 512-byte, and 16-KiB page variants were not rerun in the final progressive study; only ideal BF16 and exact 4-KiB physical layouts are saved. The 4-KiB result is the intended operational metric, but the missing sector sensitivity is a scope limitation.

## 12. Spark compute-overhead analysis

No DGX Spark was available. Timings are RTX PRO 6000 eager-PyTorch study timings and are not presented as Spark performance. Analytical selected-support compute at 2 bpw is 1.426 million selected-row transform FLOPs plus 0.729 million correction-accumulation FLOPs at the median; conservative total is 2.432 million FLOPs, 38.66% of one complete Q4 expert’s 6.291 million matrix-vector FLOPs. This does not credit gate/up selected-row union reuse.

The full H0 gate/up transform is 8.389 million FLOPs per layer-token and is oracle-selection overhead, not an H4 deployment cost. With predicted support, only selected analysis rows are needed. Correction arithmetic is plausible on Spark, but the main failure is representation/page efficiency, not a demonstrated compute bottleneck. Roofline and fused-kernel timing remain future work.

## 13. Failure-tail analysis

At 2 bpw, W1 multi-view complete-expert median recovery is 81.95%, p10 is 57.82%, only 35.93% of invocations exceed 90%, and the p95 remaining-damage fraction is 45.60%. At 4 bpw, median is 91.52%, p10 76.10%, only 55.09% exceed 90%, and p95 remaining damage is 28.34%. Heavy tails therefore preclude a no-fallback system.

Q2 improves but does not remove the tail: at 2 bpw median/p10 are 85.53/67.82%, 39.52% exceed 90%, and p95 remaining damage is 38.51%. The 95% request-cluster interval on W1’s 2-bpw median is broad (73.76–90.10%) because there are only 16 held-out request clusters.

## 14. Layer and expert heterogeneity

The six-layer trend is the most important negative result:

| Layer | Complete-expert recovery @2 bpw | @4 bpw |
|---:|---:|---:|
| 0 | 97.45% | 99.15% |
| 4 | 74.84% | 89.96% |
| 10 | 71.51% | 83.73% |
| 20 | 66.37% | 82.42% |
| 30 | 70.34% | 84.13% |
| 39 | 74.21% | 86.29% |

Hot/median/cold stratification is saved in per-expert metrics and the expert-frequency plot. No frequency stratum removes the >4-bpw censored tail.

Down is difficult for structural and packet reasons. At layer 20 the residual-action entropy effective rank is 369 for down versus 110 gate and 243 up; 90% action covariance energy needs rank 398 down versus 232/269. At layer 0 the corresponding effective ranks are 99/41/78. The post-SwiGLU input also has a much higher participation rank than pre-MoE x at layers 0 and 20 (50.8 vs 6.2; 160.8 vs 36.6). Each down atom is four times wider in its output payload than a gate/up atom, and down error directly enters the residual stream. Native down’s lower coherence and actual recovery explain why rotating it is counterproductive.

## 15. Storage-multiplier frontier

The five-times allowance makes a high-fidelity stored prefix feasible, but it does not solve runtime sparsity. Q4 plus full 4-bit and 8-bit correction views cost 1.88× and 2.74× at layers 0/20; the full 16-bit prefix reaches 4.45×. Runtime selection overwhelmingly uses 4/8-bit atoms, and the two-view basis gain is below one point. Therefore storage above roughly 3× Q4 does **not materially help** this pilot.

## 16. Optional H4 comparison

No H4 support predictor was trained or evaluated. The study remained H0-first. Atom IDs were not retained in the compact Parquet output—only selected counts, precision, bytes, and allocation—so this run does not yet supply the intended support-label corpus. H4 bandwidth inflation and support stability are unavailable.

## 17. Limitations

- No validated direct replay; therefore no hidden/router/logit/token quality results.
- No joint top-8 allocation, cancellation, page union, or cache reuse.
- Six sampled layers, not all 40; only three layers have the full 12-expert stratification.
- Sixteen held-out requests make cluster intervals wide.
- H0 atom/basis choice on test is an exploratory oracle; static method rankings should be reselected on validation in a confirmatory run.
- “Learned” orthogonal bases are stable spectral/bootstrap relaxations, not expensive end-to-end manifold optimization.
- Overcomplete training is two-view, not a fully optimized 2d sparse analysis dictionary.
- Exact 512-byte and 16-KiB page sensitivities and naive-layout comparison are absent from the final progressive run.
- No Spark kernel timings or roofline measurement.
- No saved H0 atom-ID label stream for future H4 training.
- The source export lacked a project repository commit.

## 18. Go/no-go recommendation

**No-go for exhaustive scaling of the current W1, single-diagonal-mask design.** It is “useful” at layer 0 (≥95% at 1.5 bpw, a nominal 2.67× traffic reduction), but weak or no-benefit across five other sampled layers. The best deployable hybrid is generalized gate + generalized up + native down. Learned bases and extra views do not address the main error.

**Go for a redesigned, region-conditional Q2 study with validated replay.** Q2 produces a significant paired improvement and much lower absolute output error. A policy that keeps Q2 resident, uses generalized gate/up and native down, and falls back to full Q4 in flat-spectrum layers is more credible than forcing a universal W1 representation.

The broad hypothesis—near-Q4 behavior with materially less than 4 physical bpw—is not supported model-wide by this pilot, but it remains supported for specific layers and partially supported after moving the resident base to Q2. This conclusion is based on the full frontier, not the 0.2-bpw point.

## 19. Recommended next experiment

Build the smallest validated direct-replay experiment first:

1. Add a deterministic injection checkpoint and pass Q4 replay identity at layers 0, 4, 20, and 39.
2. Compare W1 and Q2 with the fixed generalized/generalized/native hybrid at 1, 2, 3, and 4 physical bpw.
3. Optimize one joint top-8 router-weighted packet budget and save atom/page IDs.
4. Measure H1–H4 hidden drift, router flips, logit KL, and token agreement.
5. Permit full-Q4 fallback for high-effective-rank layers and fit the fallback trigger on validation requests.

Do not invest in a new H4 predictor or an exhaustive learned dictionary until the Q2 hybrid plus fallback passes direct replay.

## Headline artifact links

- [Physical frontier](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/physical_rate_propagation.png)
- [Complete-expert hybrid curve](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/complete_expert_hybrid_curve.png)
- [Six-layer complete-expert trend](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/six_layer_complete_expert_trend.png)
- [Projection allocation](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/projection_allocation.png)
- [Required-rate CDF](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/required_rate_cdf.png)
- [Residual-action spectra](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/residual_action_spectrum.png)
- [Resident-base sensitivity](results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/resident_base_sensitivity.png)
