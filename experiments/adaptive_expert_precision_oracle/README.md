# Adaptive Expert Precision H0 Oracle Pilot

This directory contains a reproducible, cost-bounded oracle study of activation-dependent correction-atom streaming for Qwen3.6-35B-A3B. It uses authoritative BF16 activation captures and the production GGUF expert tensors read-only. The main pilot covers layers 0, 20, and 39 with request-level train/validation/test separation; a smaller follow-up covers layers 4, 10, and 30 after layer 0 proved exceptional.

The principal deployment x-axis is actual bytes read after 4 KiB page accounting, normalized as physical streamed bits per original expert weight. The locally resident W1 base is 1.250488 effective bpw including FP16 group scales and a header. Q2 sensitivity is 2.250488 effective bpw by the same accounting convention.

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

## Scope boundary

The pilot validates projection and sequential complete-expert representation with a rank-4 future-router-gradient proxy. It does not claim direct teacher-forced H1-H4 replay, top-8 joint layer allocation, or token/logit agreement: the available captures did not include a validated arbitrary-MoE-output replay checkpoint, and the replay identity prerequisite was therefore not satisfied. Those omissions are explicit in the report rather than filled with synthetic data.
