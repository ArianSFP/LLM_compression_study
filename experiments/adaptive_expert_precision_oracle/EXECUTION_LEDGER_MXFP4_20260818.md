# Execution Ledger: Exact Embedded MXFP4 Hierarchy

Date: 2026-08-18 (Europe/London)
Repository: `ArianSFP/LLM_compression_study`
Branch: `agent/mxfp4-progressive-q2-q3-q4`
Starting commit: `b9e39e487436d6bd42b10360e6c2bd17878e74cb`
Seed: `20260818`

## Hardware

Paid execution host: RunPod `4324999ecfd9`

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 97,887 MiB
Driver: 595.71.05
CUDA reported by PyTorch: 12.8
```

No DGX Spark was used. No measured timing in this report is presented as a Spark timing.

## Software environments

Primary construction/evaluation environment: `/root/mxfp4_env`

```text
Python 3.12.3
PyTorch 2.8.0+cu128
NumPy 2.1.2
Pandas 3.0.5
PyArrow 25.0.1
safetensors 0.8.0
```

Exact-checkpoint capture environment: `/root/ct_env`

```text
Transformers 5.15.0
compressed-tensors 0.18.0
Accelerate 1.14.0
PyTorch 2.8.0+cu128
```

Local analysis/test environment used Python 3.13. The full test suite emitted one non-fatal pandas/numexpr version warning.

## Inputs

### Exact checkpoint

```text
Repository: pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4
Revision: 7eceff3a9f7e6f916c824d197266d86676bce695
Pod-local path: /root/qwen36_mxfp4_candidate
Size: approximately 22 GB across 26 safetensors shards
config.json SHA-256: 52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a
model.safetensors.index.json SHA-256: 842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb
Audited expert-stream aggregate SHA-256: 28ea957daf4248338116620c0be810b70852f85cfcbb098d2f6712940a54e45e
```

The checkpoint was downloaded to pod-local writable storage. Network-volume files were treated as read-only and were not modified. The checkpoint itself is not committed.

### Existing cross-reference captures

```text
Pod path: /root/mxfp4_progressive_oracle/work/captures_seed20260817.npz
Rows: 5,036
Request split: 53 train / 12 validation / 16 test
Layers: 0, 4, 20, 39
Source model revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
```

These activations are from aligned Qwen3.6 BF16 execution and are labeled cross-reference exploratory throughout.

### Fresh exact-checkpoint capture

```text
Pod path: /root/mxfp4_progressive_oracle/work/qwen36_exact_confirm_captures.npz
NPZ SHA-256: 84b7e727d774f6393d7e4e4cb5304c7df283b817002b48813a30ff55e6262e61
Rows: 564
Request split: 6 train / 3 validation / 3 test
Wall time: 32.147 s
```

The raw 14-MiB NPZ is not committed. Its manifest, schema, hash and all derived metrics are committed.

## Commands

Commands are shown from the pod experiment root `/root/mxfp4_progressive_oracle` unless noted.

### Full exact-code audit

```bash
/root/mxfp4_env/bin/python scripts/audit_full_mxfp4_checkpoint.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --layers 0 4 20 39 \
  --output work/qwen36_checkpoint_audit
```

The audit opened all 3,072 expert/projection packed-code/scale pairs and losslessly unpacked/repacked every leaf stream.

### Cross-reference dense hierarchy

```bash
/root/mxfp4_env/bin/python scripts/run_mxfp4_hierarchy_dense.py \
  --config configs/qwen36_mxfp4_progressive_hierarchy.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_dense_v1
```

Wall time: 44.551 s. Output: 24 experts, 285 test invocations, 12,933 dense rows plus projection metrics.

### Cross-reference selective pages

```bash
/root/mxfp4_env/bin/python scripts/run_mxfp4_selective_pages.py \
  --config configs/qwen36_mxfp4_progressive_selective_pilot.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output work/qwen36_dense_v1 \
  --output work/qwen36_selective_v1
```

Wall time: 624.868 s. Output: 12 experts, 142 test invocations, 994 primary rows, 1,512 layout rows and 2,982 compact action-label rows representing nearly 1.5 million exact selected actions.

### Exact checkpoint capture

```bash
/root/ct_env/bin/python scripts/capture_exact_mxfp4_confirm.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_exact_confirm_captures.npz \
  --layers 0 4 20 39 --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695
```

### Locked dense confirmation

```bash
/root/mxfp4_env/bin/python scripts/run_mxfp4_hierarchy_dense.py \
  --config configs/qwen36_mxfp4_exact_capture_confirm.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees work/qwen36_dense_v1/selected_trees.json \
  --output work/qwen36_exact_confirm_dense_v1
```

Wall time: 8.971 s. Output: 12 experts, 85 test invocations, 4,158 dense rows. Tree selection was locked before the new capture was evaluated.

### Selective exact-checkpoint confirmation

```bash
/root/mxfp4_env/bin/python scripts/run_mxfp4_selective_pages.py \
  --config configs/qwen36_mxfp4_exact_capture_selective_confirm.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output work/qwen36_exact_confirm_dense_v1 \
  --output work/qwen36_exact_confirm_selective_v1
```

Wall time: 509.303 s. Output: 85 test invocations, 595 primary rows, 1,512 layout rows and exact action labels.

### Bounded search supplement

```bash
/root/mxfp4_env/bin/python scripts/run_mxfp4_search_supplement.py \
  --config configs/qwen36_mxfp4_progressive_selective_pilot.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output work/qwen36_dense_v1 \
  --output work/qwen36_search_supplement.parquet

/root/mxfp4_env/bin/python scripts/run_mxfp4_search_supplement.py \
  --config configs/qwen36_mxfp4_exact_capture_selective_confirm.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --dense-output work/qwen36_exact_confirm_dense_v1 \
  --output work/qwen36_exact_confirm_search_supplement.parquet
```

Each run produced 1,152 rows covering two planes, 80/90/95/99 crossings, up to eight local swaps, and widths 16/64/256 over a top-32 depth-12 beam.

### Analysis and figures

From `experiments/adaptive_expert_precision_oracle/`:

```bash
MPLBACKEND=Agg python scripts/analyze_mxfp4_hierarchy.py \
  --result results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1 \
  --bootstrap 1000 --seed 20260818

python scripts/analyze_mxfp4_system_controls.py \
  --result results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1

MPLBACKEND=Agg python scripts/analyze_mxfp4_hierarchy.py \
  --result results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/exact_checkpoint_confirmation \
  --bootstrap 1000 --seed 20260818

python scripts/analyze_mxfp4_system_controls.py \
  --result results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/exact_checkpoint_confirmation
```

### Tests

The first local invocation omitted `PYTHONPATH` and failed collection with `ModuleNotFoundError: oracle_study`. The corrected command was:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Result: **39 passed**, one non-fatal numexpr warning, 11.28 seconds.

## Failed attempts and corrections

1. A candidate `olka-fi/Qwen3.5-35B-A3B-MXFP4` checkpoint was initially used for bounded method development. Its model-version mismatch with the requested Qwen3.6 study was discovered before scientific interpretation. That run is marked invalidated and excluded. The small artifacts remain for audit transparency.
2. The exact-capture virtual environment initially pulled an incompatible prerelease Torch build while installing compressed-tensors, breaking torchvision imports. The environment-local Torch copy was removed and the system `torch 2.8.0+cu128` was used successfully.
3. The first Transformers load supplied the text sub-config without the checkpoint quantization marker and failed closed with `pre_quantized=False`. It was corrected to load `Qwen3_5MoeForConditionalGeneration` from the full checkpoint config while passing the checkpoint-derived compressed-tensors configuration with `dequantize=True`.
4. The next forward reached the router but the hook assumed a tensor output. Qwen3.6's router returns `(logits, weights, ids)`. The hook was corrected to capture element zero and restore the batch dimension. The subsequent 12-request capture completed.
5. The first local full-test command omitted the experiment source path. It was rerun with the documented `PYTHONPATH` and all tests passed.
6. The first bounded-search supplement used `torch.flatnonzero`, which is not present in the installed Torch API. It was corrected to `torch.nonzero(...).flatten()` and both exploratory and exact-confirmation supplements completed.
7. The local patch helper intermittently failed because its sandbox could not create the loopback interface (`RTM_NEWADDR`). Narrow updates were applied only after the patch attempt failed; every modified Python/JSON file was compiled or parsed afterward.

## Correctness gates

* Checkpoint audit: passed, 3,072/3,072 pairs.
* Exact nibble repack: passed for every audited pair.
* Hierarchy endpoint: passed for every test projection/invocation.
* Exact selected atom/bit/page identities: passed in tests and serialized labels.
* Request-level split separation: passed.
* Locked-tree exact-checkpoint confirmation: passed.
* Direct injection/replay identity: not attempted in this branch; no H1–H4 quality claims are made.

## Storage and transfer policy

The network volume was read-only. No network-volume file was edited. No checkpoint, model shard or raw capture is committed. Derived Parquet metrics, compact action labels, manifests, plots, source and tests were copied to the repository before pod shutdown.

The largest committed file is the compressed exploratory action-label Parquet (about 87 MB), below GitHub's 100-MB single-file limit.

## Resource notes

The dense and selective kernels used less than 1 GiB of GPU memory for the sampled-expert evaluator. A monitored deterministic full-model capture rerun peaked at **84,074 MiB GPU memory** and exited successfully. CPU peak memory was not instrumented. The checkpoint download and full-tensor audit were not individually wall-clock instrumented; this is a ledger limitation.

## Output roots

```text
experiments/adaptive_expert_precision_oracle/MXFP4_PROGRESSIVE_HIERARCHY_REPORT.md
experiments/adaptive_expert_precision_oracle/results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/
experiments/adaptive_expert_precision_oracle/EXECUTION_LEDGER_MXFP4_20260818.md
```
