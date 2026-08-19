# Adaptive Expert Precision H0 Oracle Pilot

This directory contains a reproducible, cost-bounded oracle study of activation-dependent correction-atom streaming for Qwen3.6-35B-A3B. It uses authoritative BF16 activation captures and the production GGUF expert tensors read-only. The main pilot covers layers 0, 20, and 39 with request-level train/validation/test separation; a smaller follow-up covers layers 4, 10, and 30 after layer 0 proved exceptional.

The principal deployment x-axis is actual bytes read after 4 KiB page accounting, normalized as physical streamed bits per original expert weight. The locally resident W1 base is 1.250488 effective bpw including FP16 group scales and a header. Q2 sensitivity is 2.250488 effective bpw by the same accounting convention.

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
