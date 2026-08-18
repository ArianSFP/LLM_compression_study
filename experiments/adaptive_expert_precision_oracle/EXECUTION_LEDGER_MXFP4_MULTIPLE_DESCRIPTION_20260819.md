# Execution Ledger: Exact MXFP4 Multiple Descriptions

Date: 2026-08-18–19 (Europe/London)  
Repository: `ArianSFP/LLM_compression_study`  
Branch: `agent/mxfp4-multiple-description-refinement`  
Starting commit: `6b31a88ee517359847f940cc74db56898a8b8425`  
Seed: `20260818`

## Hardware and pod policy

Paid execution host:

```text
RunPod hostname: b3bcd9044643
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 97,887 MiB
Driver: 595.71.05
Pod-local root: /root/mxfp4_md
Network preservation root: /workspace/adaptive_expert_precision_oracle_md_20260818
```

The user explicitly instructed that this pod remain running. It was not stopped or terminated. All raw captures, full unsplit action tables and source/result snapshots needed for recovery were copied from NVMe to the network volume while the pod was live.

No DGX Spark was used. GPU execution times below are construction/evaluation timings and are not represented as Spark serving measurements.

## Environment

Pod virtual environment: `/root/mxfp4_md_env`

```text
Python 3.12.3
PyTorch 2.8.0+cu128
NumPy 2.1.2
Pandas 3.0.5
PyArrow 25.0.1
safetensors 0.8.0
Transformers 5.15.0
compressed-tensors 0.18.0
Accelerate 1.14.0
```

Local reporting and tests used Python 3.13. The local suite emits a non-fatal pandas warning because numexpr 2.10.1 is older than pandas' recommended 2.10.2.

## Exact checkpoint

```text
Repository: pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4
Revision: 7eceff3a9f7e6f916c824d197266d86676bce695
Pod path: /root/qwen36_mxfp4_candidate
Format: compressed-tensors mxfp4-pack-quantized
Leaves/scales: E2M1 / E8M0
Group size: 32
config.json SHA-256: 52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a
model.safetensors.index.json SHA-256: 842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb
```

The audit passed all 6,144 packed tensor entries required for 4 layers × 256 experts × 3 projections × (codes + scales). The experiment read packed codes/scales losslessly. It did not reinterpret mixed GGUF tensors and did not quantize BF16 tensors to create an endpoint.

## Fresh authoritative capture

The prior pod's capture was absent because this was a fresh container. Rather than copy a previously examined test set, 24 new requests were executed through the exact checkpoint.

Command, from `/root/mxfp4_md/experiments/adaptive_expert_precision_oracle`:

```bash
/root/mxfp4_md_env/bin/python scripts/capture_exact_mxfp4_multiple_description.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_mxfp4_md_fresh_capture.npz \
  --layers 0 4 20 39 \
  --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695
```

Capture facts:

```text
Requests: 24
Request split: 12 train / 6 validation / 6 test
Rows: 1,360
Row split: 692 train / 332 validation / 336 test
Schema: exact_mxfp4_transformers_confirm_v1
Loader: compressed-tensors dequantize=True; original packed decode, no requantization
Wall time: 39.300 s
NPZ SHA-256: 75087507e3df27735fc202ab02ea0b891daabcd4461290e58e5e55d4cd0d2698
```

Preserved raw input:

```text
/workspace/adaptive_expert_precision_oracle_md_20260818/inputs/qwen36_mxfp4_md_fresh_capture.npz
/workspace/adaptive_expert_precision_oracle_md_20260818/inputs/qwen36_mxfp4_md_fresh_capture.manifest.json
```

The raw tensor capture is not committed to GitHub. The manifest and hash are committed.

## Experiment commands

Paths below are relative to `/root/mxfp4_md/experiments/adaptive_expert_precision_oracle`. The locked base hierarchy came from the stacked PR #4 output:

```text
results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json
```

### First attempt and checkpoint

```bash
/root/mxfp4_md_env/bin/python scripts/run_mxfp4_multiple_descriptions.py \
  --config configs/qwen36_mxfp4_multiple_description_fresh.json \
  --captures work/qwen36_mxfp4_md_fresh_capture.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense \
  --output results/qwen36_mxfp4_multiple_description_fresh_20260818_aborted_perf_v0
```

This attempt evaluated all page-size diagnostics for every held-out invocation and serialized a rapidly growing action table after each expert. It was stopped after layer 0 expert 52 because the projected paid runtime was wasteful. The checkpoint is preserved on the network volume. No scientific conclusion uses its partial rows.

The bounded correction kept the primary 512-byte evaluation over all selected held-out invocations, capped each expert at eight test occurrences, and limited 1-KiB/4-KiB diagnostics to the first test occurrence of the first sampled expert per layer. The training-fit description tables from the aborted run were locked before rerunning so the test set could not alter them.

### Main A-first, bidirectional and parity run

```bash
/root/mxfp4_md_env/bin/python scripts/run_mxfp4_multiple_descriptions.py \
  --config configs/qwen36_mxfp4_multiple_description_fresh.json \
  --captures work/qwen36_mxfp4_md_fresh_capture.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense \
  --description-trees results/qwen36_mxfp4_multiple_description_fresh_20260818_aborted_perf_v0/description_trees.json \
  --output results/qwen36_mxfp4_multiple_description_fresh_20260818_v1
```

```text
Wall time: 778.367 s
Held-out expert invocations: 70
Difficult-layer invocations: 54
Metric rows: 1,470
Action-label rows: 4,410
```

### Parity shared-layout control

```bash
/root/mxfp4_md_env/bin/python scripts/run_mxfp4_multiple_descriptions.py \
  --config configs/qwen36_mxfp4_parity_shared_layout_control.json \
  --captures work/qwen36_mxfp4_md_fresh_capture.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense \
  --description-trees results/qwen36_mxfp4_multiple_description_fresh_20260818_v1/description_trees.json \
  --output results/qwen36_mxfp4_parity_shared_layout_control_20260819_v1
```

Wall time: 382.988 s. The parity codec used the exact A/B co-selection layout to remove layout as a confounder.

### Bidirectional/direct-action ablation

```bash
/root/mxfp4_md_env/bin/python scripts/run_mxfp4_multiple_descriptions.py \
  --config configs/qwen36_mxfp4_bidirectional_action_ablation.json \
  --captures work/qwen36_mxfp4_md_fresh_capture.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense \
  --description-trees results/qwen36_mxfp4_multiple_description_fresh_20260818_v1/description_trees.json \
  --output results/qwen36_mxfp4_bidirectional_action_ablation_20260819_v1
```

Wall time: 372.378 s. This run compared A-first plus direct AB against A/B-first without direct AB on the same training-only A/B layout.

### Analysis and figures

```bash
MPLBACKEND=Agg /root/mxfp4_md_env/bin/python scripts/analyze_mxfp4_multiple_descriptions.py \
  --result results/qwen36_mxfp4_multiple_description_fresh_20260818_v1 \
  --bootstrap 1000 --seed 20260818

MPLBACKEND=Agg /root/mxfp4_md_env/bin/python scripts/analyze_mxfp4_description_ablation.py \
  --main results/qwen36_mxfp4_multiple_description_fresh_20260818_v1 \
  --parity-control results/qwen36_mxfp4_parity_shared_layout_control_20260819_v1 \
  --action-ablation results/qwen36_mxfp4_bidirectional_action_ablation_20260819_v1 \
  --bootstrap 1000 --seed 20260818
```

Both PNG and SVG versions of every figure are committed.

## Result preservation and packaging

Pod-local experiment outputs were first copied to:

```text
/workspace/adaptive_expert_precision_oracle_md_20260818/results/
/workspace/adaptive_expert_precision_oracle_md_20260818/source_snapshot/
```

The network snapshot includes the original 132-MiB monolithic main action table. For GitHub, it was losslessly partitioned by layer into four Zstd Parquet files of 7.5–9.7 MiB each. The two controlled action tables are 42 MiB and 88 MiB and remain below GitHub's 100-MiB per-object limit. Every action and all statistics remain available in the branch.

The network root size before the final refreshed transfer was approximately 319 MiB. `RESULT_SHA256SUMS` records its preserved file hashes. A refreshed final snapshot is performed after report/test completion; the pod is not stopped.

## Tests

Pod smoke tests for the two new modules:

```bash
PYTHONPATH=src /root/mxfp4_md_env/bin/python -m pytest -q \
  tests/test_mxfp4_descriptions.py \
  tests/test_mxfp4_description_selective.py
```

Result: **10 passed**.

Full local suite:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Result after final edits: **49 passed**.

Correctness gates include exact reconstruction from AB/AQ/BQ, exact action/state/page identity, request split separation, deterministic layouts, physical-byte accounting and exact packed-code endpoint preservation.

## Failed attempts and corrections

1. The user supplied a fresh PRO 6000 rather than the prior retained container; the old 14-MiB capture was absent. A fresh 24-request exact-checkpoint capture was generated instead. This is scientifically cleaner and avoids another post-hoc reuse of the old three test requests.
2. The first multiple-description run was too broad operationally because page diagnostics and repeated full action-table checkpoints dominated runtime. It was stopped after one expert without stopping the pod. The corrected bounded run completed all four layers.
3. The initial parity comparison used a separately trained parity-aware page layout. It produced mixed positive and negative paired differences, which could not isolate the codec. A controlled rerun reused the A/B layout and proved exact zero parity gain.
4. The initial bidirectional policy jointly introduced B-first and direct AB. A dedicated two-policy ablation separated these effects and showed that direct AB explains the small gain.
5. The main action monolith exceeded GitHub's 100-MiB object limit. It was partitioned by layer without dropping rows; the analyzer accepts either the monolith or partition directory.

## Limitations

* No direct injection/replay identity harness was built. H1–H4 router/logit/token metrics are unavailable.
* Only six test request clusters and 70 capped held-out expert invocations are included.
* Page-size diagnostics above 512 bytes cover only four invocations.
* The allocator is diagonal within each projection; it is not unrestricted exact complete-expert search.
* CPU peak memory and experiment peak GPU memory were not instrumented. Full-model capture is known from the preceding branch to require about 84 GiB, within the 97.9-GiB device.
* No top-8 joint allocation and no H4 predictor were attempted.

## Output roots

```text
experiments/adaptive_expert_precision_oracle/MXFP4_MULTIPLE_DESCRIPTION_REPORT.md
experiments/adaptive_expert_precision_oracle/EXECUTION_LEDGER_MXFP4_MULTIPLE_DESCRIPTION_20260819.md
experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_multiple_description_fresh_20260818_v1/
experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_parity_shared_layout_control_20260819_v1/
experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_bidirectional_action_ablation_20260819_v1/
```
