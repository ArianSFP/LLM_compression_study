# D1 stratified layer-slice expansion: methodology and evidence ledger

## Executive conclusion

This study asks a deliberately narrow question: at exactly the same PR #13
refinement-page budget, can selecting different complete refinement options
reduce changes in the same token's next-layer top-8 expert set?

On six exact-capture injection layers, two matched PR #13 rates, and 32
validation groups per layer/rate, the answer is yes as an oracle mechanism:

| Pages/expert | PR #13 | Exact-combined local | Exact D1 request rerank | Exact request-coupled repair |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 20/192 | 12/192 | 4/192 | **1/192** |
| 749 | 11/192 | 7/192 | 1/192 | **0/192** |

The repaired mean local-qenergy ratios to PR #13 are 0.982793 and 0.983981.
Thus the oracle result did not obtain route agreement by worsening aggregate
local qenergy. A single residual crossing remains at 384 pages on the
layer-6 to full-attention-layer-7 boundary.

This is not yet a deployable allocator result. The strongest rows use exact
Q4 route labels and exact request-level replay to choose or repair finalists.
The fixed-policy result is weaker, although still directionally positive. No
terminal-logit KL, all-layer, D2--D4, learned signed-effect predictor, route
certificate, or production-latency claim is made.

## Provenance

The study is stacked on the following immutable foundations:

- PR #13 all-layer allocation commit
  `089deb4bd41eb71864779ced0cbb3db41e9dcb63`;
- same-host causal controls commit
  `64e94ace22f42734941bb3b4fd6702b8ff380295`;
- D1 objective implementation commit `eddc6f9`;
- initial D1 slice pilot commit `dced412`;
- six-layer/full-attention expansion commit `febd7d8`.

The exact promoted experiment is frozen in
[qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1.json](configs/qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1.json),
SHA-256 `de00bd6ada308a7693fc7816fdee3e1e6de1b2d6206a26a7f1dd61a094d0adfd`.
It records the actual six layers, two rates, six eta values, requests,
position mask, candidate ranks, temperature, search passes, and native 3090
linear/full-attention execution path. The earlier
`qwen36_mxfp4_d1_objective_oracle.json` is a prospective all-layer/four-rate
design envelope, not the reproduction config for this completed expansion.

The standalone artifact validator checked all 12 raw layer/rate cells against
the immutable config, including required hashes, and retained its evidence at
`/workspace/pr13_d1_downstream_tail_kl_smoke_20260824_v1/evidence/d1_expansion_config_validation.json`.

The checkpoint index SHA-256 is
`842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
The three immutable baseline-capture hashes are retained in
`results/qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1/parity/d1_slice_parity.json`.

Complete new raw results and an exact code snapshot are retained on network
storage at:

`/workspace/pr13_d1_layer_slice_expansion_20260824_v1`

Its 595-file manifest passed in full and has SHA-256
`c8f9372ea7ada003a089638e7b3bcfc0caa6871636c7a8ff671113a6c3865367`.
Previously verified layer-0/12/23 raw inputs remain at:

`/workspace/pr13_d1_layer_slice_pilot_20260824_v1`

The repository package contains every final allocation manifest, aggregate
and per-layer evidence table, parity result, runtime benchmark, analysis
facts, and checksums. Large intermediate candidate grids, reconstructed delta
banks, model tensors, and duplicate code snapshots remain network-only to
avoid turning Git history into an unbounded binary archive.

## Scientific boundary

D1 has exactly one meaning in this work:

\[
(t,\ell)\rightarrow(t,\ell+1).
\]

It is the same token's router at the immediately following layer. It is not
the earlier H1--H4 future-token/same-layer proxy, and it is not a D2--D4
multi-layer objective.

The experiment executes only through the next router. It therefore establishes
whether allocation can preserve immediate expert membership; it does not by
itself establish lower terminal KL. Experiment A contains no crude `S0`,
omitted-tail prediction, speculative pass, or route certification. Those are
reserved for the separate Experiment B runtime controller.

## Sampling design

### Requests and groups

The validation requests are:

- `mxfp4-confirm-006`, sequence length 15;
- `mxfp4-confirm-007`, sequence length 13;
- `mxfp4-confirm-008`, sequence length 16.

The same-host protocol excludes the final four positions of every request.
This produces 11, 9, and 12 admitted positions, respectively, or exactly 32
groups per layer. Test rows are never admitted or used.

### Layers

The sampled injection layers are `0, 1, 4, 6, 12, 23`. These are all
same-host sentinel layers with exact stored pre-residual tensors and a valid
next-layer router. Sentinel layer 39 is excluded because it has no D1 MoE
router to protect.

This sample spans early, shallow, middle, and later depths. Five boundaries
enter a linear-attention layer. Layer 6 enters full-attention layer 7 and is
the architecture-diversity slice.

No arbitrary non-sentinel layer was approximated from post-xplus state.
Expanding beyond these six layers requires a new exact pre-residual capture;
the study does not silently substitute a methodologically different injection.

### Rates

The promoted stress points are 384 and 749 pages/expert, corresponding to
0.5234781901041666 and 0.9987386067708334 charged bpw in PR #13. Each routed
group has eight experts, so the pooled option budgets are 3,072 and 5,992
pages per group.

The 576 and 768 points were not run in this cost-bounded expansion. The two
chosen rates cover the low-rate switching stress case and the approximately
1-bpw operating point. Consequently, the result must not be described as a
four-rate curve.

## Exact activation and execution path

For every request/layer pair the immutable capture supplies:

- the layer input residual;
- normalized MoE input `x`;
- exact Q4 routed-expert output;
- router IDs, scores, and logits for every layer;
- input IDs and attention mask.

The scientific injection replaces the routed MoE output before shared-expert
composition and the decoder residual. With the convention
`delta = approximate - exact_Q4`, the slice computes:

\[
r'_\ell=\operatorname{BF16}
\left(\operatorname{FP32}(r^{(4)}_\ell)+\delta_\ell\right),
\]

\[
h^+_\ell=
h^{\mathrm{res}}_\ell+r'_\ell+
\sigma(g_{\mathrm{shared}}(x_\ell))f_{\mathrm{shared}}(x_\ell).
\]

The addition is performed in FP32 and rounded once to BF16, matching the
same-host causal-control scientific path. The slice then executes the native
pre-MoE portion of layer `ell+1`, its post-mixer residual, post-attention norm,
and router.

For linear-attention boundaries, the exact checkpoint mixer is invoked with
the captured attention mask and no cache. For layer 6, the implementation
loads the official full-attention module, its Q/K norms and projections,
constructs the official causal mask, creates the checkpoint's four-axis text
position IDs and rotary embeddings, and then executes the router. Only tensors
required for the two-layer slice are loaded, allowing the 35B checkpoint's
D1 path to run on a 24 GiB RTX 3090.

## Complete option reconstruction

The historical PR #13 all-layer factors, selected trees, and complete
per-expert option representation are reused. For each group, the eight routed
experts are reconstructed from that token's activation. Every option retains:

- expert ID and router rank;
- FP32-normalized selector weight;
- actual BF16 execution weight;
- selected page count;
- selected frontier option index;
- the complete 512-entry split precision-state vector;
- the exact token-dependent approximate-minus-Q4 output delta.

The token dependence is essential:

\[
d_{e,o}(x)=\widetilde y_{e,o}(x)-y_e^{(4)}(x).
\]

The same physical page can have a large signed effect for one activation and
almost none for another. The implementation therefore reconstructs whole
frontier options using current gate/up/down responses. It never treats a page
as a static output vector.

Gate/up and down refinements are not assumed to be mutually additive through
SiLU and multiplication. Column generation constructs complete rate-specific
frontiers before D1 allocation. This is an oracle-label cost, not a proposed
runtime computation.

## Compared allocation policies

### Historical PR #13

The reference policy is the preserved column-generated router-square
allocator. It scores experts additively with normalized selector weights:

\[
J_{\mathrm{PR13}}=\sum_e a_e^2D_e(o_e).
\]

Stored historical deltas are replayed directly. Where a regenerated
router-square comparator is emitted, its delta must match the preserved PR #13
delta bit-for-bit.

### Exact-combined local qenergy

This validation-only local oracle scores the signed combined routed-output
error:

\[
J_{\mathrm{local}}(o)=
\left\|
\sum_{e=1}^{8}a_e
\left[\widetilde y_{e,o_e}(x)-y_e^{(4)}(x)\right]
\right\|_2^2.
\]

It includes cross-expert cancellation and interaction, but it does not know
which directions threaten the next router.

### One-sided D1 objective

The combined perturbation that reaches the next layer is:

\[
\delta_\ell(o)=
\sum_{e=1}^{8}a_e^{\mathrm{exec}}
\left[\widetilde y_{e,o_e}(x)-y_e^{(4)}(x)\right].
\]

Execution weights are used here because they determine the BF16 vector placed
in the residual stream. Normalized selector weights remain the legacy local
qenergy accounting representation.

For each baseline Q4 route, selected ranks 6--8 are paired with the eight
closest outsiders. If selected expert `i` and outsider `j` have baseline
margin

\[
m_{ij}=z^{(4)}_i-z^{(4)}_j\geq0,
\]

the oracle forms an exact same-token full-VJP sensitivity:

\[
g_{ij}=\nabla_{h^+_\ell(t)}
\left[z_{\ell+1,i}(t)-z_{\ell+1,j}(t)\right].
\]

Each complete option's signed effect is projected as

\[
s_{e,o,ij}=a_e^{\mathrm{exec}}g_{ij}^{T}d_{e,o}(x).
\]

The strict loss is the actual one-sided softplus:

\[
L_{\mathrm{D1}}(o)=
\sum_{ij}
\operatorname{softplus}
\left(
\frac{-m_{ij}-\sum_es_{e,o_e,ij}}{\tau}
\right),
\]

with `tau = 0.0625`, unit severity, and zero safety margin. Stabilizing
movements are not penalized symmetrically like a quadratic proxy.

### Additive local guardrail

D1 allocation is constrained by:

\[
J_{\mathrm{local}}(o)
\leq J_{\mathrm{local}}^*+\eta J_{\mathrm{Q2}}.
\]

The actual promoted eta grid is:

`0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001`.

This small grid is intentionally tighter than the broader values in the
full-oracle configuration. Each policy begins from the exact-combined local
solution and uses deterministic complete-option coordinate updates followed
by two-expert pair exchanges. Candidate solutions must remain within the
pooled page cap and local guardrail. Ties prefer lower route loss, lower local
damage, fewer pages, and deterministic option order.

## Exact replay and policy tiers

Every stored policy is evaluated by an actual BF16 next-router execution, not
by the linearized margin prediction. The replay constructs the whole request's
delta table, injects every admitted position simultaneously, and measures:

- whether the exact top-8 set changed;
- top-8 agreement;
- number of membership pairs changed;
- baseline routing mass lost;
- baseline rank-8/rank-9 margin;
- candidate margin for the baseline labeled set;
- entering and leaving expert IDs;
- local qenergy and selected pages.

The report distinguishes three increasingly oracle-heavy D1 summaries.

1. **Fixed eta.** One eta is applied to all requests and layers at a rate.
   This is the cleanest objective-mechanism result, although it still uses
   exact option deltas and VJPs.
2. **Exact request reranking.** For each request/layer/rate, exact D1 replay
   selects a single eta by crossings, local qenergy, lost routing mass, pages,
   and deterministic policy name.
3. **Exact request-coupled repair.** Complete finalist policies may be mixed
   across positions within a request. Coordinate changes are followed by
   pairs drawn only from source positions at or before the earliest remaining
   crossing. Every candidate is scored by exact causal replay.

The repair exists because prefill has causal prefix coupling: approximate
outputs at earlier positions can change a later position's next router even
though the primary D1 VJP is same-token. Repair searches mixtures of already
generated complete finalists; it does not search arbitrary pages and is not a
runtime controller.

The exact repair score is lexicographic:

1. fewest crossings;
2. fewest changed membership pairs;
3. lowest summed local qenergy;
4. lowest lost baseline routing mass;
5. fewest pages;
6. deterministic policy signature.

The selected repair is replayed twice. Scores and entering-expert identities
must match exactly.

## Parity and hardware controls

All policy comparisons use a paired local RTX 3090 Q4 baseline generated in
the same process as replay. Each slice is executed twice. Current hidden
states and next-router logits must be bit-identical across the pair for every
request and layer. This hard gate passed for all 18 request/layer cells.

The immutable activations were originally captured on an RTX PRO 6000. BF16
kernel/device drift between the 3090 and PRO 6000 is therefore reported but
is not counted as allocator-caused switching. Across the parity table the
stored-router maximum absolute difference is at most 0.0625. Some top-8 sets
differ under this cross-device comparison, including the full-attention
slice; that is exactly why the paired local baseline is mandatory.

This gate demonstrates same-process determinism, not cross-GPU bit identity.
The full-attention implementation is additionally exercised by exact layer-6
VJPs and replays at both rates.

## Results

### Aggregate exact replay

| Pages/expert | Policy | Crossings | Rate | Local qenergy / PR #13 |
| ---: | --- | ---: | ---: | ---: |
| 384 | PR #13 | 20/192 | 10.4167% | 1.000000 |
| 384 | Exact-combined local | 12/192 | 6.2500% | 0.981727 |
| 384 | Exact request rerank | 4/192 | 2.0833% | 0.996085 |
| 384 | Exact request-coupled repair | **1/192** | **0.5208%** | **0.982793** |
| 749 | PR #13 | 11/192 | 5.7292% | 1.000000 |
| 749 | Exact-combined local | 7/192 | 3.6458% | 0.980045 |
| 749 | Exact request rerank | 1/192 | 0.5208% | 1.017474 |
| 749 | Exact request-coupled repair | **0/192** | **0%** | **0.983981** |

The repaired policy removes 19 of 20 low-rate PR #13 crossings and all 11
high-rate crossings. The exact-combined local oracle improves over PR #13,
confirming that combined error matters, but the D1 objective and request-aware
selection are needed for the strongest membership control.

### Repaired result by layer

| Injection layer | Next mixer | PR #13 384 | Repaired 384 | PR #13 749 | Repaired 749 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | linear attention | 2/32 | 0/32 | 0/32 | 0/32 |
| 1 | linear attention | 4/32 | 0/32 | 2/32 | 0/32 |
| 4 | linear attention | 3/32 | 0/32 | 4/32 | 0/32 |
| 6 | full attention | 7/32 | **1/32** | 2/32 | 0/32 |
| 12 | linear attention | 2/32 | 0/32 | 2/32 | 0/32 |
| 23 | linear attention | 2/32 | 0/32 | 1/32 | 0/32 |

Layer 0 at 749 is a useful negative cell: the historical policy already has
zero crossings and D1 preserves zero. The objective therefore does not create
an apparent benefit where the baseline route is already stable.

Layer 6 is the hardest case and the only full-attention boundary. At 384 pages
PR #13 changes 7/32 groups; the best token-local D1 policy changes 2/32 and
request repair leaves 1/32. At 749 pages D1 reaches 0/32. This is evidence for
a low-rate/prefix/full-attention limit, not evidence that full-attention D1
cannot be controlled.

### Fixed-policy evidence

Fixed eta results prevent the request-level oracle from being mistaken for a
deployable policy:

| Pages/expert | Fixed eta | Crossings | PR #13 crossings | Local qenergy / PR #13 |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 0.0001 | 10/192 | 20/192 | 0.986540 |
| 384 | 0.001 | **7/192** | 20/192 | 1.055705 |
| 749 | 0.0001 | 6/192 | 11/192 | 1.036896 |
| 749 | 0.00025 | **2/192** | 11/192 | 1.128662 |

No single fixed eta simultaneously reproduces the repaired result and the
best local-qenergy value. A runtime policy must therefore predict boundary
risk and uncertainty rather than hard-code the oracle's per-request choice.

## Runtime compute and bandwidth accounting

The exact frontier build, VJPs, coordinate/pair search, and repair are
oracle-label costs. They are excluded from the runtime estimate.

For rank 8, 16 candidate router logits, 12,288 legal Q2 transitions, and six
refreshes:

| Selector design | MAC/group | Fraction of PR #13 |
| --- | ---: | ---: |
| PR #13 primary selector | 51,343,360 | 100% |
| Direct D1 candidate scorer | 9,781,248 | 19.051% |
| One-adjoint D1 scorer | 933,888 | 1.819% |

The one-adjoint total includes 81,920 response-synthesis MAC/group. It first
collapses the current boundary loss into one latent direction and then scans
candidate transitions. The direct design scores all candidate logits for all
transitions.

The measured RTX 3090 shortlist benchmark is:

| Path | Samples | Median | p90 |
| --- | ---: | ---: | ---: |
| NumPy CPU | 50 | 6.092 ms | 6.279 ms |
| Device tensor | 500 | 0.852 ms | 0.868 ms |

Six serialized device refreshes would cost approximately 5.1 ms before data
transfer or provisional next-layer execution. The current controller therefore
does not satisfy a non-bottleneck latency claim. Refreshes must be reduced,
fused, graphed, or overlapped and measured with the model.

Static/dynamic D1 metadata increases the per-group payload:

| Metadata representation | KiB/group | Metadata bpw | Model total |
| --- | ---: | ---: | ---: |
| PR #13 | 72.125 | 0.023478 | 90.156 MiB |
| D1 joint response | 155.125 | 0.050496 | 193.906 MiB |
| D1 plus uncertainty | 167.625 | 0.054565 | 209.531 MiB |

To preserve the four PR #13 all-in charged rates, page caps must fall as
follows:

| PR #13 pages/expert | Joint D1 | D1 plus uncertainty |
| ---: | ---: | ---: |
| 384 | 363 | 360 |
| 576 | 555 | 552 |
| 749 | 728 | 725 |
| 768 | 747 | 744 |

Joint metadata consumes the traffic of 21 refinement pages/expert; uncertainty
consumes 24. This accounting prevents route metadata from being treated as
free bandwidth.

## Implementation map

The checked-in implementation is separated along the same oracle/runtime
boundary used in the scientific interpretation:

- [d1_route_objective.py](src/oracle_study/d1_route_objective.py) implements
  exact combined perturbations, one-sided boundary loss, the additive local
  guardrail, deterministic coordinate/pair search, and runtime scoring
  primitives;
- [d1_layer_slice.py](src/oracle_study/d1_layer_slice.py) implements the exact
  Q4 slice, linear- and full-attention next-layer execution, D1 VJPs, BF16
  injection, and paired replay metrics;
- [d1_accounting.py](src/oracle_study/d1_accounting.py) defines matched all-in
  metadata, page-cap, and selector-compute accounting;
- [d1_runtime_controller.py](src/oracle_study/d1_runtime_controller.py)
  contains the separate token-dependent signed-effect and certification
  research scaffold; it is not used to claim the Experiment A result;
- [the oracle configuration](configs/qwen36_mxfp4_d1_objective_oracle.json)
  and [runtime-controller configuration](configs/qwen36_mxfp4_d1_runtime_controller.json)
  freeze the two experiment boundaries;
- [run_d1_layer_slice_pilot.py](scripts/run_d1_layer_slice_pilot.py),
  [run_d1_slice_oracle_pilot.py](scripts/run_d1_slice_oracle_pilot.py), and
  [run_d1_exact_request_repair.py](scripts/run_d1_exact_request_repair.py)
  are the parity, cell-execution, and exact-repair entry points;
- [analyze_d1_slice_pilot.py](scripts/analyze_d1_slice_pilot.py) generates the
  compact evidence package, while
  [benchmark_d1_transition_scorer.py](scripts/benchmark_d1_transition_scorer.py)
  generates the saved runtime measurement;
- the five focused test modules are
  [accounting](tests/test_d1_accounting.py),
  [layer slice](tests/test_d1_layer_slice.py),
  [route objective](tests/test_d1_route_objective.py),
  [runtime analysis](tests/test_d1_runtime_analysis.py), and
  [runtime controller](tests/test_d1_runtime_controller.py).

## Artifact map

The compact result root is:

`results/qwen36_mxfp4_d1_layer_slice_expansion_20260824_v1/`

Reviewer entry points:

- `README.md`: compact result and scientific boundary;
- `analysis/D1_LAYER_SLICE_PILOT_REPORT.md`: generated aggregate report;
- `analysis/d1_aggregate_policy_summary.parquet`: primary aggregate table;
- `analysis/d1_layer_policy_summary.parquet`: per-layer table;
- `analysis/d1_fixed_policy_aggregate.parquet`: every fixed eta;
- `analysis/d1_exact_reranked_metrics.parquet`: exact request-reranked rows;
- `analysis/d1_exact_reranked_request_choices.parquet`: selected finalist per
  request/layer/rate;
- `analysis/d1_paired_parity_summary.parquet`: paired/cross-device controls;
- `analysis/d1_vjp_summary.parquet`: exact label timing;
- `analysis/d1_slice_analysis_facts.json`: input hashes, accounting, and output
  manifest;
- `parity/d1_slice_parity.json`: all 18 detailed parity cells;
- `runtime/d1_transition_scorer_3090.json`: CPU/device shortlist benchmark;
- `ANALYSIS_REPRODUCTION.json`: before/after byte-identity record;
- `artifact_hashes.sha256`: compact-package manifest.

The `repair/` directory contains 12 layer/rate subdirectories. Each contains:

- 256 expert rows: 32 groups times eight routed experts;
- 32 group rows;
- 32 exact D1 metric rows;
- 32 source-policy choice rows;
- run facts and per-request search counts;
- for newly executed cells, the selected 32-by-2048 delta bank.

Across all 12 directories, 3,072 expert rows and all 3,072 complete 512-entry
precision-state vectors were audited. Legal state IDs are 0--7. The compact
manifest verifies every stored file.

## Reproduction

The exact network snapshot used Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
Transformers 5.15.0, eager attention, and an NVIDIA GeForce RTX 3090.

The parity gate is run with:

```bash
PYTHONPATH=src python scripts/run_d1_layer_slice_pilot.py \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint \
  --capture-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/baseline \
  --output-dir /workspace/pr13_d1_layer_slice_expansion_20260824_v1/results \
  --layers 0 1 4 6 12 23
```

Each promoted layer/rate cell uses the following template:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python scripts/run_d1_slice_oracle_pilot.py \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint \
  --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
  --fit-dir /workspace/pr13_average_rate_all_layers_20260822_v1/results/fit \
  --capture-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/baseline \
  --same-host-layers-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/layers \
  --pr13-config configs/qwen36_mxfp4_average_rate_all_layers.json \
  --same-host-config configs/qwen36_mxfp4_same_host_causal_controls.json \
  --output-dir OUTPUT_DIRECTORY \
  --layers LAYER --rates RATE \
  --eta 0 0.00005 0.0001 0.00025 0.0005 0.001 \
  --temperature 0.0625 --workers 16 --column-generated-frontiers
```

Column-generated mode accepts one rate per invocation because its frontier is
rate-specific. Exact request repair uses:

```bash
PYTHONPATH=src python scripts/run_d1_exact_request_repair.py \
  --checkpoint /workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint \
  --capture-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/baseline \
  --layer-dir LAYER_RESULT_DIRECTORY \
  --output-dir REPAIR_OUTPUT_DIRECTORY
```

`scripts/analyze_d1_slice_pilot.py` receives the 12 layer directories, 12
repair directories, and the saved runtime benchmark. It was executed twice;
all nine generated analysis files were byte-identical.

Validation commands and results:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_d1_accounting.py \
  tests/test_d1_layer_slice.py \
  tests/test_d1_route_objective.py \
  tests/test_d1_runtime_analysis.py \
  tests/test_d1_runtime_controller.py
# 27 passed

PYTHONPATH=src python -m pytest -q
# 383 passed in 136.96s

sha256sum -c artifact_hashes.sha256
# every compact artifact OK
```

## Interpretation and limitations

The evidence supports the mechanism that combined signed error projected onto
the next router is a better refinement target than eight independent
router-square expert scores. It also shows that exact-combined local qenergy
captures some of the benefit, but not all of it.

The evidence does not yet support the following claims:

- reduced terminal logit KL;
- an all-layer quality curve;
- a useful four-rate policy;
- generalization beyond three validation sequences;
- accurate learned `g^T d_p(x)` prediction;
- calibrated uncertainty or safe route certificates;
- a solved initial `S0` policy;
- a non-bottleneck sequential implementation;
- elimination of the low-rate full-attention/prefix residual.

The exact request repair is especially important to interpret correctly. It
demonstrates that a combination of already available D1 finalists can almost
eliminate the sampled switches. It is not evidence that a runtime controller
can identify that combination without exact labels.

## Required next gates

1. Replay the final matched-rate allocations through the full downstream tail
   and measure live final-logit KL, fully frozen KL, and live-minus-frozen KL.
2. Expand beyond the three validation requests before interpreting terminal
   quality.
3. Train the separate Experiment B predictor for token-dependent signed page
   effects and report Pearson/Spearman correlation, sign accuracy, top-page
   overlap, corrected-logit error, and interval calibration.
4. Measure false certificates, confidence--coverage, fetched pages,
   provisional-pass invalidation, overlap, and end-to-end latency.
5. Optimize `S0` for immediate certificate coverage under fixed contiguous
   traffic rather than assuming that local-qenergy prefixes are optimal.
6. Treat routing-mass and functional swap severity as ablations after strict
   membership preservation has established the mechanism.

Only after the downstream-KL gate succeeds should this work be described as a
quality improvement to PR #13 rather than a successful immediate-route oracle.

That gate is frozen separately in
[qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json](configs/qwen36_mxfp4_d1_downstream_tail_kl_smoke_20260824_v1.json).
It replays the five policy tiers through the exact downstream tail in live and
fully frozen routing, reports layers 1/4/6 separately from the 0/12/23 eta
calibration layers, and includes the metadata-matched 360/725 page caps. The
three-request output remains a smoke result; at least 64 independent requests
are required before a quality claim.
