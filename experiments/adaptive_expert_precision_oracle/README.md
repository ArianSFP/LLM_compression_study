# Adaptive Expert Precision H0 Oracle Pilot

This directory contains a reproducible, cost-bounded oracle study of activation-dependent correction-atom streaming for Qwen3.6-35B-A3B. It uses authoritative BF16 activation captures and the production GGUF expert tensors read-only. The main pilot covers layers 0, 20, and 39 with request-level train/validation/test separation; a smaller follow-up covers layers 4, 10, and 30 after layer 0 proved exceptional.

The principal deployment x-axis is actual bytes read after 4 KiB page accounting, normalized as physical streamed bits per original expert weight. The locally resident W1 base is 1.250488 effective bpw including FP16 group scales and a header. Q2 sensitivity is 2.250488 effective bpw by the same accounting convention.


## PR #13 causal downstream replay pilot

This continuation reconstructs the local routed-MoE errors selected by the
PR #13 all-layer allocator and measures causal propagation through complete
downstream sequence tails for all 40 MoE layers at 0.523478 and 0.998739
charged bpw.

Reviewer entry points:

- [Canonical report](results/qwen36_mxfp4_causal_downstream_replay_20260823_v1/analysis/CAUSAL_DOWNSTREAM_REPLAY_REPORT.md),
  [analysis manifest](results/qwen36_mxfp4_causal_downstream_replay_20260823_v1/analysis/causal_replay_analysis_manifest.json),
  and [execution ledger](EXECUTION_LEDGER_CAUSAL_REPLAY_20260823.md).
- [Combined propagation table](results/qwen36_mxfp4_causal_downstream_replay_20260823_v1/impulse/impulse_propagation.parquet),
  [isolated-layer quality table](results/qwen36_mxfp4_causal_downstream_replay_20260823_v1/impulse/impulse_quality.parquet),
  and [reconstruction facts](results/qwen36_mxfp4_causal_downstream_replay_20260823_v1/reconstruction/reconstruction_run_facts.json).

Reconstruction parity passes at approximately `1e-15`, all-Q4 is exact,
selected pages match, and repeated paired tails are bit-exact. Pairing is
required because newly loaded BF16 CPU and GPU trajectories do not
bit-reproduce the historical capture; that drift remains a separate control.
This three-sequence isolated-impulse pilot is not yet an end-to-end streamed
NLL/PPL curve. A full-sequence held-out capture and sequential four-rate run
are the next gate. Predicted-H4 remains excluded; later prediction begins at
H1.

## Split gate/up/down interaction field (stacked on PR #11)

This follow-on separates the physically distinct gate and up refinement pages.
Its state encoding is `4*gate_high + 2*up_high + down_high`, with page costs
`[0,1,1,2,1,2,2,3]`. The inherited four states map exactly to `[0,6,1,7]`,
so the eight-state action set contains every PR #11 decision while reusing the
same exact A/B/C self terms and signed L4/L2 interaction factors.

Reviewer entry points:

- [Canonical report](results/qwen36_mxfp4_split_interaction_field_20260820_v1/analysis/SPLIT_INTERACTION_FIELD_REPORT.md),
  [frozen continuation decision](results/qwen36_mxfp4_split_interaction_field_20260820_v1/analysis/split_interaction_field_promotions.json),
  and [analysis manifest](results/qwen36_mxfp4_split_interaction_field_20260820_v1/analysis/analysis_manifest.json).
- [Immutable 4,278-row frontier](results/qwen36_mxfp4_split_interaction_field_20260820_v1/split_interaction_field_frontier.parquet),
  [run facts](results/qwen36_mxfp4_split_interaction_field_20260820_v1/run_facts.json),
  and [reproducibility protocol](SPLIT_INTERACTION_FIELD_REPRODUCIBILITY.md).

The attribution caveat is now resolved. At a fixed one-correction-bpw budget,
rank-8 Hadamard INT4 progresses from 96.107% p10 / 98.022% median for the
legacy PR #11 solver, to 97.270% / 98.504% for compressed four-state search
with the exact-self DP seed and the same coordinate/local solver, to 99.065% /
99.476% for eight states. The paired median components are +0.5753 percentage
points of seed/optimizer headroom and +0.8443 points from the action space.
The old headline gain versus PR #11 remains +1.4304 points, but is no longer
attributed solely to split gate/up states. The independent exact-Gram
same-solver action-space control remains +0.7353 points.

Under a strict all-in one-bpw budget including A/B/C, factor metadata, and
correction pages, INT4 is the default at a 749-page cap, 98.916% p10 / 99.408% median
recovery, versus INT8 at 741 pages and 98.887% / 99.388%. The paired median
advantage reverses in INT4's favour by +0.0129 points while INT4 also uses less
metadata. The rank-4 exact-proxy control retains 729 pages and reaches 98.822% /
99.337% with 736,256 charged MACs instead of 1,329,152.

The implementation now updates coordinate damage incrementally and performs a
full objective calculation only at initialization and for terminal parity. For
the all-in INT4 path, actual coordinate sweeps are median/p90 12/16 and local
passes are 12/12. A median 12 repair bundles are accepted, so the local search
exhausts its frozen cap; this is a bounded-search result, not a convergence
claim. The actual quota-contended Python reference wall-time is p10/median/p90
3.581/4.214/4.304 seconds per selector invocation, including the measured
field build. These timings are not optimized-kernel latency.

The run closed all 69 validation invocations and 12 cells in about 9 minutes
14 seconds with 24 single-thread workers. The rented host exposed 256 logical /
128 physical CPUs but only 27.2 cgroup cores, so more workers would
oversubscribe the available CPU time. One RTX 3090 was sufficient; a Pro 6000
is not required. The result remains an exact mixed-H4 geometry ceiling.


## Low-rank signed interaction field (stacked on PR #10)

This exact-H4 continuation compresses the static Q2/Q4 down-column interaction
Gram into signed rank-4/8/16/32 fields while retaining exact per-unit A/B/C
self damage. It evaluates four-state residual-aware coordinate descent,
continuous relaxation, and bounded 1/2/3-unit repair on the immutable PR #10
69-invocation validation cohort.

Reviewer entry points:

- [Canonical report](results/qwen36_mxfp4_interaction_field_20260820_v1/analysis/INTERACTION_FIELD_REPORT.md),
  [conclusion](results/qwen36_mxfp4_interaction_field_20260820_v1/analysis/interaction_field_conclusion.json),
  and [analysis manifest](results/qwen36_mxfp4_interaction_field_20260820_v1/analysis/analysis_manifest.json).
- [Raw frontier and factor artifacts](results/qwen36_mxfp4_interaction_field_20260820_v1/)
  and [reproducibility protocol](INTERACTION_FIELD_REPRODUCIBILITY.md).

At one physical bpw, rank-8 row-INT8 passes every frozen gate with p10/median
recovery `96.14%/98.07%`, p10 set-gain retention `99.93%`, `0.03389` metadata
bpw, and `834,560` conservative MACs. Hadamard-rotated packed rank-8 INT4 also
passes at `0.02348` metadata bpw. The full 69-invocation grid completed in about
15.5 minutes using 24 single-thread workers under a measured 27.2-core cgroup
quota.

This is a positive compressed-geometry ceiling, not yet a deployable candidate
policy: exact validation-time H4 remains supplied. The next decisive experiment
must predict the prefetch field, substitute exact fetched responses, and retain
an approximate residual for unfetched units.

## Set-utility distillation (stacked on PR #9)

This final validation-only study runs on branch
`agent/mxfp4-set-utility-distillation`, based on PR #9 commit
`56fe7764ec28c92947c83cb3d7dbd16e5630311b`. It preserves the locked codec,
checkpoint revision, selected trees, and request split.

Reviewer entry points:

- [Final report](results/qwen36_mxfp4_set_utility_distillation_20260820_v1/validation_analysis/SET_UTILITY_DISTILLATION_REPORT.md) and [analysis manifest](results/qwen36_mxfp4_set_utility_distillation_20260820_v1/validation_analysis/analysis_manifest.json).
- [Frozen STOP decision](results/qwen36_mxfp4_set_utility_distillation_20260820_v1/validation_analysis/set_utility_promotions.json), [train-only fit](results/qwen36_mxfp4_set_utility_distillation_20260820_v1/fit/), and
  [exact-checkpoint validation evidence](results/qwen36_mxfp4_set_utility_distillation_20260820_v1/validation_exact_checkpoint/).
- [Execution ledger](EXECUTION_LEDGER_SET_UTILITY_20260820.md) and
  [reproducibility protocol](SET_UTILITY_DISTILLATION_REPRODUCIBILITY.md).

The exact one-bpw hybrid local-search oracle reaches p10/median recovery
`96.05%/97.89%` and passes its continuation gate, improving by
`+2.261/+5.439` percentage points over coherent exact and by
`+0.031/+2.123` points over the PR #9 independent control. `K=128` plus 64
oracle repair units retains p10/median template utility `96.89%/98.77%` at
256 candidates, but is a regime-existence oracle; no classifier was evaluated.
The best promotable PQ2 plus independent-ABC path reaches only
`78.72%/88.63%` recovery and `90.45%` p10 set-gain retention at `0.1771`
metadata bpw, `1.184M` MACs, and `1.512x` storage. It fails the frozen recovery
and set-retention gates; all 34 predicted/independent rows fail the scientific gates, and the direct
set models are worse.

The frozen decision is **STOP**, with no sealed holdout. The strongest next
direction is distillation of the successful hybrid action/local-search policy,
with support templates and PQ retained as secondary candidate features.
No H4, latency, model-accuracy, router, logit, or token-quality claim is made.

## Neuron-selector distillation (stacked on PR #7)

This bounded continuation is stacked on [PR #7](https://github.com/ArianSFP/LLM_compression_study/pull/7)
at commit `9ef21d519a4f2cd844e9fa75d04cd601e37d5a11`. It tests factorized
gate/up-versus-down neuron states and rank-8/16/32/64/128 predictors of the
Q4 hidden response. The codec, selected trees, checkpoint revision, and
request split are unchanged.

Reviewer entry points:

- [Final report](results/qwen36_mxfp4_neuron_selector_distillation_20260819_v1/final_analysis/NEURON_SELECTOR_DISTILLATION_REPORT.md)
  and [analysis manifest](results/qwen36_mxfp4_neuron_selector_distillation_20260819_v1/final_analysis/analysis_manifest.json).
- [Frozen validation decision](results/qwen36_mxfp4_neuron_selector_distillation_20260819_v1/validation_analysis/neuron_selector_promotions.json),
  [raw validation tables](results/qwen36_mxfp4_neuron_selector_distillation_20260819_v1/validation_exact_checkpoint/),
  and [training-only response factors](results/qwen36_mxfp4_neuron_selector_distillation_20260819_v1/fit/).
- [Protocol](NEURON_SELECTOR_DISTILLATION_PROTOCOL.md) and
  [execution ledger](EXECUTION_LEDGER_NEURON_SELECTOR_20260819.md).

Both workstreams stop on exact-checkpoint validation. At one physical bpw,
factorized G/D fixed-greedy reaches p10/median `0.8698/0.9328` on the 55-row
matched cohort, versus `0.9352/0.9660` for the frozen exact 64×16 tile. The
best deployable response model (rank-64 separate gate/up, row-FP8) reaches
only `0.7560/0.8883` after fetching 256 candidates and applying 192, despite
retaining `0.9164/0.9562` p10/median independent-score utility at 0.1849
metadata bpw. No conditional expansion or held-out run was launched, so test
blindness is preserved. These are H0-late qenergy results, not accuracy,
logit, routing, token-quality, or H4-training claims.

## Sparse-streaming allocator study (stacked on PR #6)

This continuation is stacked on [PR #6](https://github.com/ArianSFP/LLM_compression_study/pull/6) at commit `1f01edf74ce754fea1615b26a97e4465a829671f`. It preserves that branch's locked embedded Q2→Q3→Q4 codec, selected trees, exact checkpoint revision, and request separation. The evidence must be read in one direction: exact-checkpoint **validation only** selected the bounded configurations and produced the immutable promotion artifact; those frozen choices were then evaluated on the fresh held-out exact-checkpoint cohort; the broader capture is a sensitivity check and never a selection source.

Reviewer entry points:

- [Final sparse-streaming report](results/qwen36_mxfp4_sparse_streaming_20260819_v1/final_analysis/SPARSE_STREAMING_ALLOCATOR_REPORT.md) and [analysis manifest](results/qwen36_mxfp4_sparse_streaming_20260819_v1/final_analysis/analysis_manifest.json).
- [Frozen validation promotions](results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_analysis/sparse_streaming_promotions.json).
- Raw evidence: [bounded pilot](results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint/), [validation-only tile-shape follow-up](results/qwen36_mxfp4_sparse_streaming_20260819_v1/validation_tile_shapes_exact_checkpoint/), [fresh held-out expansion](results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_exact_checkpoint/), and [cross-reference sensitivity run](results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_cross_reference/).
- [Execution ledger](EXECUTION_LEDGER_SPARSE_STREAMING_20260819.md) and [reproducibility guide](SPARSE_STREAMING_REPRODUCIBILITY.md).
- [PR #6 interaction-aware allocator report](results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/INTERACTION_AWARE_ALLOCATOR_REPORT.md), the controlled baseline for this stacked study.

The promoted neuron-major and exact tile selectors are **H0 oracles only**. The activation-derived shortlist and scalable/static tile proxies stopped at validation, so this study does not establish a deployable streaming selector. It makes no task-accuracy, router, logit, token-quality, or H4-prediction/training claim.

At 1.0 physical correction bpw, the frozen 64×16 tile oracle reaches fresh held-out p10/median recovery `0.9375/0.9683` over 85 invocations. On the 72 difficult-layer invocations it reaches `0.9364/0.9632`, versus PR #4's reported `0.8334/0.8935` in the prior [success-gate artifact](results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/success_gates.json). This is expert-output qenergy recovery: it establishes a substantially better sparse action space, while the final deployment verdict remains negative because no scalable proxy passed and the exact tile oracle requires 768 global refreshes at one bpw.

## Layout

- `configs/`: exact JSON configurations.
- `src/oracle_study/`: trace, quantization, basis, weight, metric, and page-accounting code.
- `scripts/`: capture extraction, projection, progressive-precision, complete-expert, diagnostics, and analysis entry points.
- `tests/`: deterministic quantization, reconstruction, layout, leakage, router, and allocation tests.
- `results/qwen36_mxfp4_varied_pilot_extended_20260817/`: principal metrics, statistics, accounting, and plots.
- `results/qwen36_mxfp4_varied_pilot_hybrid_20260817/`: raw W1 complete-expert allocation metrics.
- `results/qwen36_mxfp4_varied_pilot_q2_20260818/`: Q2 projection sensitivity.
- `results/qwen36_mxfp4_varied_pilot_hybrid_q2_20260818/`: Q2 complete-expert sensitivity.
- `results/qwen36_mxfp4_layer_extension_20260818/`: layers 4/10/30 projection extension.
- `results/qwen36_mxfp4_layer_extension_hybrid_20260818/`: layers 4/10/30 complete-expert extension.
- `ORACLE_STUDY_REPORT.md`: interpretation and recommendation.
- `EXECUTION_LEDGER_20260818.md`: commands, timings, hashes, environment, and failures/corrections.

## Corrected allocator and output-side-down continuation

The 2026-08-18 continuation adds a four-layer Q2 pilot that audits the legacy selected-count/page mismatch, saves exact atom/precision/page labels, evaluates a nonlinear complete-expert per-atom allocator at 512 B through 4 KiB transfer granularity, and compares native down packets with shared, clustered, and per-expert output-side low-rank factors.

- `results/qwen36_q2_single_view_baseline_20260818_v2/`: reproduced deployable Q2 generalized/generalized/native baseline.
- `results/qwen36_q2_corrected_allocator_20260818_v7/`: corrected Run A metrics and exact H0 label sequence.
- `results/qwen36_q2_output_side_down_20260818_v3/`: Run B validation/test controls, storage, spectra, and complete-expert metrics.
- `results/qwen36_q2_next_oracles_analysis_20260818_v1/`: leakage-free selected headlines, 1,000-resample statistics, accounting, and PNG/SVG plots.
- `NEXT_ORACLE_EXPERIMENT_REPORT.md`: scientific conclusion and next-iteration recommendation.
- `EXECUTION_LEDGER_NEXT_20260818.md`: exact commands, failures/corrections, hardware, hashes, and timings.
- `REPLAY_IDENTITY_AUDIT_NEXT_20260818.md`: why production-reference replay identity remains unavailable.

Reproduce the continuation analysis after the three immutable result directories are present:

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_next_oracles.py \
  --baseline experiments/adaptive_expert_precision_oracle/results/qwen36_q2_single_view_baseline_20260818_v2 \
  --run-a experiments/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v7 \
  --run-b experiments/adaptive_expert_precision_oracle/results/qwen36_q2_output_side_down_20260818_v3 \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_q2_next_oracles_analysis_20260818_v1 \
  --bootstrap 1000 --seed 20260818
```

## Reproduce saved figures and statistics

From the repository root:

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_pilot.py \
  --results experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817 \
  --hybrid-results experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_hybrid_20260817 \
  --bootstrap 1000 --seed 20260817
```

Run tests:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Remote GPU commands are recorded verbatim in the execution ledger. They require the external paths named in the configs; raw captures and model files are intentionally not copied into this repository.

## Dense recurrent-residual quantization ceiling

The RRQ continuation asks whether PR #2's 84.6% plateau came from the
high-bit-prefix atom codec rather than from the underlying correction.  It is a
dense-prefix ceiling, not a selective packet experiment: every recurrent stage
has independent two-bit codes and independently serialized scales, and each
later residual is formed from the exact stored-scale decode of earlier stages.

The primary run uses layers 0/4/20/39, the same 12 hot/median/cold experts per
layer, request-level splits, production-reference GGUF matrices, serialized Q2
base, and future-router-gradient proxy as PR #2.  Quantizer formats are selected
on validation only with a bounded two-path beam.  The held-out set is labelled
exploratory because the prior result informed the RRQ hypothesis; a new request
set is required for a subsequent confirmatory result.

Important rate convention: a complete symmetric two-bit stage is 2.25 bpw at
group 64 or 2.125 bpw at group 128 after FP16 scales.  The implemented affine
G64 stage is 2.375 bpw because zero points are byte-aligned.  Reported physical
rates include these parameter streams and page rounding.  The resident Q2 base
is local and excluded from streamed bpw.  The production-reference fallback is
included in the 5x external-capacity check.

Run the remote experiment (paths are machine-specific and recorded in the RRQ
ledger):

```bash
python experiments/adaptive_expert_precision_oracle/scripts/run_dense_rrq_ceiling.py \
  --config experiments/adaptive_expert_precision_oracle/configs/q2_rrq_dense_ceiling.json \
  --captures /path/to/rrq_captures_seed20260817.npz \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_dense_ceiling_20260818_v1
```

Regenerate every RRQ table and PNG/SVG figure from saved Parquet files:

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_dense_rrq.py \
  --result experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --prior-allocator experiments/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v7/allocator_metrics.parquet \
  --config experiments/adaptive_expert_precision_oracle/configs/q2_rrq_dense_ceiling.json
```

Large serialized code streams are not committed. Their exact sizes and SHA-256
hashes are retained in `serialized_stage_manifest.json`.

## Selective RRQ retention follow-up

The selective follow-up changes the storage orientation from row-major dense
matrices to atom-major RRQ packets, so every quantization group belongs to one
independently fetchable atom. It measures the fraction of the matched
full-support three-stage atom-RRQ benefit retained at 0.5–3 physical correction
bpw, exact stage-1/2/3 support crossings, full-support basis transfer, ranking
regret, and nonlinear gate/up/down allocation. Exact selected atom IDs, stage
depths, packet/page IDs, bytes, and marginal scores are retained in Parquet.

Run and reproduce it with:

```bash
python experiments/adaptive_expert_precision_oracle/scripts/run_selective_rrq_retention.py \
  --config experiments/adaptive_expert_precision_oracle/configs/q2_rrq_selective_retention.json \
  --captures /path/to/rrq_captures_seed20260817.npz \
  --dense-result experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_selective_retention_20260818_v1

python experiments/adaptive_expert_precision_oracle/scripts/run_selective_rrq_ranking_supplement.py \
  --config experiments/adaptive_expert_precision_oracle/configs/q2_rrq_selective_retention.json \
  --captures /path/to/rrq_captures_seed20260817.npz \
  --dense-result experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_selective_retention_20260818_v1/ranking_regret_supplement.parquet \
  --layers 20 39

MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_selective_rrq.py \
  --result experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_selective_retention_20260818_v1
```

The basis audit includes a stage-refitted generalized basis and two-view
overcomplete diagnostic. The butterfly result is explicitly a fixed
Hadamard-style control; it is not represented as a trained butterfly model.

The final difficult-layer frontier reaches 87.43% median proxy recovery at 2
physical bpw and 90.38% at 2.5 bpw. `RRQ_SELECTIVE_FOLLOWUP_REPORT.md` explains
why this is material but still exploratory/partial support. The result directory
contains standard `metrics.parquet`, `summary.csv`, per-layer/per-expert tables,
all exact H0 action labels, and `artifact_hashes.json`.

## Scope boundary

The pilot validates projection and sequential complete-expert representation with a rank-4 future-router-gradient proxy. It does not claim direct teacher-forced H1-H4 replay, top-8 joint layer allocation, or token/logit agreement: the available captures did not include a validated arbitrary-MoE-output replay checkpoint, and the replay identity prerequisite was therefore not satisfied. Those omissions are explicit in the report rather than filled with synthetic data.
## Grouped top-8 average-rate allocation and frontier closure

This PR #13 continuation replaces a uniform per-expert page cap with an exact
multiple-choice allocation over the true eight routed experts of each captured
validation token/layer. It preserves PR #12's eight gate/up/down states, exact
A/B/C self terms, and signed interaction field. The train-only factor fit covers
all 256 experts in layers 0/4/20/39; validation contains 128 groups and 1,024
routed expert invocations.

The implementation constructs one vectorized exact-self DP table per expert and
reuses it across every hard budget and Lagrange price. The closure run adds three
controls requested after the first PR #13 result:

- allocation-aware column generation repairs only the eight selected frontier
  points, inserts two prices around each selected marginal slope, and reruns the
  MCKP for at most three deterministic rounds;
- rank-4 exact-proxy factors are evaluated as FP16, per-row INT8, Hadamard INT4,
  and mixed proxy-INT8 plus Euclidean-tail INT4 encodings;
- a validation-only combined-MoE search adds bounded three/four-expert exchanges
  and a globally valid convex-hull lower bound over the finite refined columns.

Accuracy by overall average rate for the rank-8 Hadamard INT4 factor and the
1,536-page burst cap is:

| Overall average bpw | Uniform p10 / median | Coarse pooled p10 / median | Column-generated p10 / median | Exact combined local p10 / median |
| ---: | ---: | ---: | ---: | ---: |
| 0.523478 | 95.175% / 97.823% | 96.744% / 99.011% | 96.787% / 99.042% | 96.887% / 99.066% |
| 0.773478 | 98.123% / 99.215% | 98.830% / 99.658% | 98.886% / 99.674% | 98.895% / 99.683% |
| 0.998739 | 99.337% / 99.710% | 99.576% / 99.881% | 99.589% / 99.885% | 99.598% / 99.886% |
| 1.023478 | 99.387% / 99.736% | 99.623% / 99.895% | 99.628% / 99.897% | 99.646% / 99.899% |

Selected-column repair changes 71.1% of strict-rate page vectors but adds only
+0.00001 pp p10 and +0.00305 pp median recovery over the coarse pooled
frontier. Sparse price coverage is therefore measurable but is not the
remaining accuracy bottleneck.

At strict one-total-bpw, rank-4 Hadamard INT4 is the best factor control:
99.590% p10 / 99.887% median at 753 correction pages and 39.92M charged group
MACs. It slightly exceeds rank-8 INT4 at 749 pages (99.589% / 99.885%, 60.41M
MACs) while cutting charged arithmetic by 33.9%. Rank-4 INT8 reaches 99.573% /
99.884% at the same total rate and uses the same factor payload as rank 8;
quantization precision is not the limiting factor here.

Three/four-expert exchange changes only 2/128 selected page vectors and produces
no quantile-level accuracy gain. The finite-frontier lower bound is globally
valid, but its median relative damage gap is 34.75% (absolute gap 9.35e-5), so
this run does not certify global optimality. The defensible result is that no
improvement was found within the frozen exchange search, not that allocation is
globally solved.

These are exact combined top-8 qenergy-recovery statistics, not token accuracy.
Every row uses exact validation H4 and is nonpromotable. Column generation costs
a median/p90 2,833/2,884 coordinate sweeps, 389/410 local passes, and
83.84/86.82 seconds of quota-contended Python reference wall time per group.
The 24-worker validation took about 93 minutes under a measured 27.2-core cgroup
quota; the host exposed 256 logical CPUs, but 64 or 256 workers would
oversubscribe it. One RTX 3090 is sufficient.

Q3 remains deliberately separate. A later Q3 study must report an ideal logical
256-byte bitplane ceiling, a training-only 512-byte co-selection layout, and a
fixed 512-byte gate/up pairing control, all charged by unique physical page IDs.

Canonical artifacts are under
`results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/`.
Reviewer entry points are the [report](results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/analysis/AVERAGE_RATE_ALLOCATION_REPORT.md),
[frozen continuation decision](results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/analysis/average_rate_promotions.json),
[analysis manifest](results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/analysis/average_rate_analysis_manifest.json),
and [reproducibility contract](AVERAGE_RATE_ALLOCATION_REPRODUCIBILITY.md).
Regenerate the analysis with:

```bash
MPLBACKEND=Agg \
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts \
python experiments/adaptive_expert_precision_oracle/scripts/analyze_average_rate_allocation.py \
  --config experiments/adaptive_expert_precision_oracle/configs/qwen36_mxfp4_average_rate_allocation.json \
  --fit-dir experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/fit \
  --validation-dir experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/validation_exact_h4 \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_average_rate_allocation_frontier_closure_20260821_v2/analysis_reproduced
```

See `AVERAGE_RATE_ALLOCATION_REPRODUCIBILITY.md` for the complete execution,
hash, accounting, worker, and scientific-boundary contract.
