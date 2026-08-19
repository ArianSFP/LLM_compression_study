# PR6 exact-checkpoint baseline reproduction

The historical PR6 physical selective allocator was reproduced on the frozen CPU-generated exact-checkpoint capture. The reproduction emitted all 85 held-out invocations and 8,330 complete-expert packet rows. Its primary parquet is byte-for-byte identical to the historical artifact.

## Frozen inputs

- Capture: `qwen36_exact_confirm_captures.npz`, SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`; 564 rows, 280/128/156 train/validation/test, 12 requests, layers 0/4/20/39.
- Checkpoint: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` revision `7eceff3a9f7e6f916c824d197266d86676bce695`.
- Checkpoint `config.json`: `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`.
- Checkpoint tensor index: `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
- Locked trees: `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.
- Copied PR6 runner: `20cbbca3a242504f2e55d38980e3991d419224f539890da884476b96939498fc`.
- Copied PR6 config: `064d0d4f5f942add6b43339de7cb6119b5bad9c5ac6e2a750b6c78211dd06e3d`.

## Exact command

Run from `/workspace/codebook_granularity_study`:

```bash
setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts \
  /workspace/codebook_env/bin/python scripts/run_interaction_physical_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures inputs/qwen36_exact_confirm_captures.npz \
  --checkpoint /workspace/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/qwen36_mxfp4_codebook_granularity_20260819_v2/baseline_reproduction/pr6_exact_checkpoint_physical_52bc \
  --source exact_checkpoint --mode full
```

The process was launched at 2026-08-19 19:46:17 UTC. Script accounting began at 19:46:23.353 UTC and completed at 20:23:31.502 UTC: 2,228.150 seconds. It was detached, reparented to PID 1, and wrote atomic per-expert parquet checkpoints.

## Validation

| Artifact | Reproduction SHA-256 | Historical SHA-256 | Result |
|---|---|---|---|
| `action_to_page_frontier.parquet` | `cd9225d1c4474875f27908d757a422d710e90f3321b3cb30aaa3d94e77f5e4ed` | same | exact |
| `layout_training_facts.csv` | `82b116c4f66e3c13b16a31117d226994e42b003a0f4989cd829ffabcc40b21b4` | same | exact |
| `paired_hybrid_frontier.parquet` | `000c340fde29c3827287b903bb7bd83bf41af23f10bd51628ba95017bf85ecf8` | same | exact |

The reproduced and historical 8,330-by-28 dataframes compare exactly. Across separate, paired and hybrid representations; both the all-layer and difficult-layer cohorts; and six streamed rates, all 36 frozen comparisons pass with maximum statistic error `4.44e-16`. The 2-bpw maximum absolute endpoint error is `2.92e-12` from numerical roundoff.

At one physical streamed bpw:

| Method | Cohort | n | p10 | Median | p90 | Median gain vs current diagonal |
|---|---|---:|---:|---:|---:|---:|
| Current diagonal separate | all four | 85 | 0.825022 | 0.891499 | 0.957836 | 0 |
| Exact-training separate | all four | 85 | 0.838531 | 0.897337 | 0.957036 | +0.005837 |
| Exact-training paired | all four | 85 | 0.855259 | 0.920027 | 0.968948 | +0.028528 |
| Exact-training hybrid H0 oracle | all four | 85 | 0.872356 | 0.923651 | 0.974856 | +0.032152 |
| Current diagonal separate | layers 4/20/39 | 72 | 0.820375 | 0.883710 | 0.956340 | 0 |
| Exact-training separate | layers 4/20/39 | 72 | 0.835098 | 0.892560 | 0.945116 | +0.008850 |
| Exact-training paired | layers 4/20/39 | 72 | 0.852224 | 0.910014 | 0.966250 | +0.026304 |
| Exact-training hybrid H0 oracle | layers 4/20/39 | 72 | 0.868238 | 0.920733 | 0.969918 | +0.037023 |

`current diagonal` here is the PR6 52bc-capture control, not the separately frozen PR4 headline computed from the unavailable 84b7 capture. The hybrid row is an H0 representation-choice ceiling that retains both suffix layouts (4 external suffix bpw); it is not a deployable fixed representation.

Two transient RunPod route/storage interruptions occurred while the detached job was running. During the first, the PID survived and briefly entered `folio_wait_bit_common`; it recovered without intervention and continued from the valid atomic checkpoint. The process completed during the second route interruption. Exact historical hashes after recovery show that neither event altered results.

Machine-readable evidence:

- `pr6_52bc_one_bpw_reproduction_summary.csv`
- `../baseline_reproduction/pr6_exact_checkpoint_physical_52bc/frozen_pr6_validation.json`
- `../baseline_reproduction/pr6_exact_checkpoint_physical_52bc/pr6_exact_checkpoint_physical_52bc.log`
