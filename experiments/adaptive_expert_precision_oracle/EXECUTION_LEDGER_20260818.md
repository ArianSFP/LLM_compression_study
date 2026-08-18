# Execution Ledger — 2026-08-18

## Source identity

- Local workspace: `/home/arian/LLM_compression_study`.
- Local repository commit/branch: unavailable; the supplied workspace was not a Git repository.
- Network-volume project commit/branch: unavailable; the export had no `.git` directory.
- Authoritative model: `Qwen3.6-35B-A3B`.
- HF model revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Model config SHA-256: `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.
- Production GGUF: `/workspace/LLM_prefetch_study/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`.
- GGUF size/SHA-256: 22,182,574,368 bytes / `e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6`.
- GGUF reproduction manifest llama.cpp commit: `4fc4ec5541b243957ae5099edb67372f8f3b550e` (not used for arithmetic).
- Capture schema: `gcrp2r_transformers_run_manifest_v1`.
- Prompt manifest SHA-256: `916315802286cdead8fc9e4d12a3bdb288fe9feb9d0b3e84c21ecd6ea032c2f7`.
- Tokenizer SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

## Data and split

Six of 39 audited capture segments were chosen at evenly spaced indices: `000000`, `000097_000112`, `000225_000240`, `000337_000352`, `000465_000480`, and `000593_000608`. Position stride was 16. Splits were deterministic by complete request: 53 train, 12 validation, 16 test; 2,469/570/738 sampled layer rows. Core layers were 0/20/39; extension layers were 4/10/30. No request crossed splits.

Raw captures, models, and checkpoints remained read-only on the network volume. No authoritative input was modified or copied into the repository.

## Hardware and environment

- RunPod ID/name: `jp3jjmka340cd8` / `silly_amber_antlion`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Driver: 595.71.05.
- Host: Ubuntu/Linux 6.8.0-124-generic, x86_64, 188 GB RAM, 32 vCPUs.
- Python: 3.12.3, GCC 13.3.0.
- PyTorch: 2.8.0+cu128; CUDA runtime 12.8.
- NumPy 2.1.2; pandas 3.0.5; pyarrow 20.0.0; safetensors 0.8.0; transformers 5.14.1.
- No DGX Spark was used. RTX timings are not Spark timings.
- Peak GPU/host memory was not instrumented as a time series. No run OOMed; a diagnostic snapshot used 781 MiB GPU memory. Peak memory is therefore unavailable rather than estimated.

The resumed paid-pod uptime was 3,907 seconds. At $2.09/hour, resumed cost was approximately $2.27. The pod was stopped with `runpodctl pod stop jp3jjmka340cd8` at completion; it was not terminated.

## Configurations and seeds

- Primary: `configs/varied_pilot_extended.json`.
- Complete expert: `configs/varied_pilot_hybrid.json`.
- Layer extension: `configs/varied_pilot_layer_extension.json` and `configs/varied_pilot_layer_extension_hybrid.json`.
- Global seed: 20260817.
- Learned-basis seeds: 11, 29, 47 in the core; seed 11 in the limited extension.
- Bootstrap: 1,000 request-cluster resamples, seeds 20260817 onward.
- Binary group size: 64.
- Page size: 4,096 bytes for operational results.
- Progressive atom precisions: 2/4/8/16 for Phase A; 4/8/16 for complete experts.

## Exact command patterns

SSH endpoint changed after resume; the active endpoint was `root@213.192.6.98:40175`. Commands below are verbatim apart from line wrapping.

```bash
PYTHONPATH=/root/adaptive_expert_precision_oracle/src:/root/adaptive_expert_precision_oracle/scripts \
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
/root/adaptive_expert_precision_oracle/scripts/run_phase_a_v2_remote.py \
--config /root/adaptive_expert_precision_oracle/configs/varied_pilot_extended.json \
--captures /root/adaptive_expert_precision_oracle/work/pilot_captures.npz \
--output /root/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817
```

```bash
PYTHONPATH=/root/adaptive_expert_precision_oracle/src:/root/adaptive_expert_precision_oracle/scripts \
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
/root/adaptive_expert_precision_oracle/scripts/run_progressive_remote.py \
--config /root/adaptive_expert_precision_oracle/configs/varied_pilot_extended.json \
--captures /root/adaptive_expert_precision_oracle/work/pilot_captures.npz \
--output /root/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817
```

```bash
PYTHONPATH=/root/adaptive_expert_precision_oracle/src:/root/adaptive_expert_precision_oracle/scripts \
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
/root/adaptive_expert_precision_oracle/scripts/run_hybrid_remote.py \
--config /root/adaptive_expert_precision_oracle/configs/varied_pilot_hybrid.json \
--captures /root/adaptive_expert_precision_oracle/work/pilot_captures.npz \
--output /root/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_hybrid_20260817
```

The same commands with extension configs/captures produced layers 4/10/30. `run_progressive_q2_remote.py`, `run_hybrid_q2_remote.py`, and `run_progressive_simple_w1_remote.py` produced resident-base sensitivities.

Local analysis:

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_pilot.py \
  --results experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817 \
  --hybrid-results experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_hybrid_20260817 \
  --bootstrap 1000 --seed 20260817
```

```bash
python experiments/adaptive_expert_precision_oracle/scripts/analyze_hybrid_statistics.py \
  --w1 experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_hybrid_20260817/hybrid_metrics.parquet \
  --q2 experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_hybrid_q2_20260818/hybrid_metrics.parquet \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817 \
  --bootstrap 1000 --seed 20260817
```

Tests:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts \
pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Result: 9 passed in 1.60 seconds.

## Wall-clock timings

| Stage | Rows | Wall seconds |
|---|---:|---:|
| core Phase A BF16/full methods | 369,600 | 548.97 |
| core progressive precision | 84,000 | 415.06 |
| core complete-expert hybrid W1 | 22,044 | 402.45 |
| structural diagnostics | 33 summary + 10,261 spectrum | 90.38 |
| extension Phase A | 101,304 | 130.88 |
| extension progressive | 28,140 | 164.55 |
| extension complete expert | 6,600 | 219.21 |
| core progressive Q2 | 84,000 | 108.97 |
| core complete-expert Q2 | 22,044 | 397.16 |
| simple-W1 progressive sensitivity | 84,000 | 31.84 |
| main local bootstrap/plots | 475,644 consolidated | 27.84 |

Capture extraction and file transfer timings were not individually instrumented. Total paid uptime includes startup, restoration after resume, smoke tests, transfers, and inspection.

## Failed attempts and corrections

1. The prior `/root` overlay was absent after stop/resume. All code and completed results had already been copied locally; the experiment was restored and captures were deterministically re-extracted from the unchanged read-only volume.
2. `/usr/bin/time` was unavailable on the pod. Runners’ internal wall clocks were used.
3. The first Phase A wrapper launch lacked `PYTHONPATH` and failed before doing work. It was rerun with explicit source/script paths.
4. The first generalized basis could be rank-deficient under empirical covariance. `run_phase_a_v2_remote.py` uses a strictly invertible regularized full-rank transform and validates reconstruction.
5. Complete-expert physical page selection was corrected to prefix/page-union accounting and tested against the uncached implementation.
6. The first local pytest command omitted `PYTHONPATH` and failed during collection; the documented command passed all nine tests.
7. The workspace patch helper intermittently failed to create a network namespace. Existing-file efficiency edits used the standard `patch` utility only after repeated helper failures; new files continued to use the patch helper.
8. Direct replay was investigated but no validated arbitrary-MoE-output injection path was present. Per the protocol, propagation results were not produced without replay identity.

## Output hashes

| File | SHA-256 |
|---|---|
| core `phase_a_metrics.parquet` | `f0fe55cd873de695a856f5cf25a2deb6e9e4a691d236e07bf342137872a80ee2` |
| core `progressive_metrics.parquet` | `c1021c71ce043e8f5f7431a768338d4881b6bc2ddecbadb7598110571864a6d0` |
| consolidated `metrics.parquet` | `483d5f786ac105e9b78d00b76f8750cf740eecb378ccc1ba69300eee69446683` |
| W1 `hybrid_metrics.parquet` | `a5a2dfcf4688f7de140b7307baa7d3336342783016e8d147dfe6184a8cc4ce90` |
| Q2 `hybrid_metrics.parquet` | `002f86490f500f8086f86ac6bbaf535611a9dcf52223742297a476971975193c` |
| extension `phase_a_metrics.parquet` | `80d88fd372eadd985be64bdc28774204f7b568657843aec32848dc3992d3e91a` |
| extension `progressive_metrics.parquet` | `41f4f1f22629b6432f5917eeee6d783607579a0dbaeab3f79584c350f25232cb` |
| extension `hybrid_metrics.parquet` | `3cbcdf679ac57d4dcaf0cec9674f0cf517a5f17e22987c9a8cda45e9eebe405f` |

## Limitations recorded at execution time

No direct replay, joint top-8 allocation, Spark timing, 512-byte/16-KiB final progressive page sweep, H4 predictor, or saved atom-ID label corpus was executed. The results directory is a bounded oracle pilot, not a claim of completed model-quality validation.
