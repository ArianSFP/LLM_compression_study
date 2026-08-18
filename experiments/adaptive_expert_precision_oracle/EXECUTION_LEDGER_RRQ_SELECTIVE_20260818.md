# Selective RRQ Execution Ledger — 2026-08-18

## Scope

Bounded selective atom-major RRQ retention follow-up after the dense-prefix ceiling. Historical result directories were not modified.

## Repository

- Worktree: `/home/arian/LLM_compression_study/rrq_worktree`
- Branch: `agent/rrq-dense-prefix-ceiling`
- Stacked base: `agent/q2-corrected-allocator-low-rank-down`
- Foundation commit: `cb9210edaae306bcbd9856686303c28dbf609290`
- New run ID: `qwen36_q2_rrq_selective_retention_20260818_v1`

## Hardware and environment

- RunPod ID: `gszpnl0eshsw9s`
- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- SSH endpoint: `213.192.2.67:40075`
- Python: `/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python`
- Remote environment: Linux 6.8.0-110-generic/glibc 2.39; PyTorch 2.8.0+cu128; CUDA 12.8; NumPy 2.1.2; pandas 3.0.5; PyArrow 20.0.0.
- Local analysis: Python 3.13.5; pytest 8.3.4.
- Peak CUDA allocation: 159,514,624 bytes (152.1 MiB).
- Main four-layer wall time: 5,673.694 seconds (1 h 34 min 33.7 s), from 16:52:15 to 18:26:49 UTC.
- Missing-layer ranking supplement: completed for layers 20 and 39 before 18:31:26 UTC; its separate wall timer was not instrumented, so only the under-4-min-38-s enclosing bound is claimed.

## Immutable inputs

- Production reference: `/workspace/LLM_prefetch_study/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`
- HF alignment checkpoint: `/workspace/LLM_prefetch_study/artifacts/j_route_0/hf/Qwen3.6-35B-A3B`
- Authoritative captures: `/workspace/LLM_prefetch_study/artifacts/gcrp2/captures/gcrp2r_transformers_bf16_corpus_v1`
- Compact aligned capture: `work/rrq_captures_seed20260817.npz`
- Dense quantizer selection: `results/qwen36_q2_rrq_dense_ceiling_20260818_v1/quantizer_selection.json`
- Input hashes: inherited from the dense run's `run_facts.json` and copied into the result manifest.

The network-volume model, checkpoint, and capture paths were read only.

## Commands

```bash
PYTHONPATH=src pytest -q tests/test_rrq.py tests/test_selective_rrq.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_selective_rrq_retention.py \
  --config configs/q2_rrq_selective_retention.json \
  --captures work/rrq_captures_seed20260817.npz \
  --dense-result results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --output results/qwen36_q2_rrq_selective_retention_20260818_v1

python scripts/run_selective_rrq_ranking_supplement.py \
  --config configs/q2_rrq_selective_retention.json \
  --captures work/rrq_captures_seed20260817.npz \
  --dense-result results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --output results/qwen36_q2_rrq_selective_retention_20260818_v1/ranking_regret_supplement.parquet \
  --layers 20 39

MPLBACKEND=Agg python scripts/analyze_selective_rrq.py \
  --result results/qwen36_q2_rrq_selective_retention_20260818_v1
```

## Checkpointing

The runner atomically rewrites Parquet checkpoints after every expert:

- `selective_metrics.parquet`
- `selected_actions.parquet`
- `stage_support.parquet`
- `ranking_regret.parquet`
- `basis_full_support.parquet`

Exact atom IDs, stages/precision, page IDs, physical/logical increments, and marginal scores are retained. Completion resumes by `(layer, expert_id)`.

## Tests and corrections

- Local focused tests: 12 passed (RRQ plus selective atom-major accounting); final full suite: 28 passed with one unrelated pandas/numexpr warning.
- The selective down path was corrected before paid execution so its selected correction matrix is applied to the same sequential `h̃` produced by the selected gate/up corrections. The authoritative `h_ref` remains only the down support-ranking initialization and projection-level diagnostic.
- The normal update-mode `apply_patch` helper repeatedly failed before reading files because its bubblewrap network namespace could not enter (`RTM_NEWADDR`). Add-file patches still worked; narrowly scoped exact-text replacements were used only inside the new RRQ scripts/reports/ledgers and were compile/diff checked.
- Initial queued smoke failed before model access because the new config pointed at a non-existent copied GGUF Python path. Both selective configs were corrected to the exact `/workspace/LLM_prefetch_study/src/llama.cpp/gguf-py` path proven by the dense run. No scientific row was produced by the failed launch.
- The corrected one-layer smoke completed in 964.760 seconds with 12 metric rows, 2,592 exact action rows, 54 support crossings, and 288 full-support basis rows. Its three test invocations retained a median 79.9%/91.9% of matched dense benefit at 0.5/1.0 physical bpw, which triggered the fixed four-layer run.
- Final integration audit on all 1,356 held-out budget rows and 1,498,464 action rows found zero physical-byte mismatches, zero logical-byte mismatches, zero non-nested atom/stage groups across 1,230,088 groups, and zero malformed page-ID JSON rows.

## Result status

The main run completed with 2,586 metric rows, 1,498,464 exact action-label rows, 8,136 stage-support rows, 660 initial ranking rows, and 6,288 full-support basis rows. The supplement added 660 ranking/beam rows, one held-out-covered audit invocation each for layers 20 and 39. Local analysis generated 1,000-request-cluster bootstrap intervals, standard metrics/summary/per-layer/per-expert tables, exact compute/storage accounting, and eight figures in both PNG and SVG.

Primary difficult-layer recovery at 0.5/1/1.5/2/2.5/3 bpw was 55.81/74.60/82.57/87.43/90.38/92.24% median. On the exact paired PR #2 subset, the 2-bpw gain was +6.48 points with 95% cluster CI [+5.52,+9.41]. This passes the material-improvement criterion but misses the ≥90%-by-2-bpw broad-scaling gate. The result is exploratory because the same test requests informed the hypothesis.

Key transferred-file SHA-256 values (all verified identical before the pod stop):

- `selective_metrics.parquet`: `9632c7ca6b6fa2c4754679bdc57faf843936131284d966c788cbd7345dd9a048`
- `selected_actions.parquet`: `9c71e17b81d11cc3a69777b96503f72a23a676241ef99afab2c2a4428eb3e4e5`
- `stage_support.parquet`: `d934655d9ca055b3c289d053c6e50aecc57376e55d1578e6dcf95cb46f0b8da8`
- `basis_full_support.parquet`: `34316bc546a61b4defe892c528169bfe692880bdf0a85577e4be03d300c4a637`
- `ranking_regret.parquet`: `ffe591a838e465bb65760221dd8e3f3f277e77bdeb7de73c661731fd46d6a606`
- `ranking_regret_supplement.parquet`: `78eeb6f3ac9b375cbc7569172353905ec40895c41d8138839a4adfe21dee6ac5`
- final 43-file `artifact_hashes.json`: `301c31327db39f569150de753af16e7fe04b7ab45bfd110cf07df249c11b9fd2`
- production-reference GGUF: `e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6` (22,182,574,368 bytes)
- capture manifest: `9931e72b07e63d12151fa5f4d11613ce9d44f94f927ea5b437aab1ef97c14a9a` (35,582 bytes)

The compact activation NPZ and model tensors remain external and uncommitted. No network-volume file was modified.

## Pod lifecycle

The user removed the earlier 04:58 UK cutoff. After every primary/supplement artifact and execution log was copied and the SHA-256 values above matched locally, the pod was stopped—not terminated—with:

```bash
runpodctl pod stop gszpnl0eshsw9s
```

RunPod returned `desiredStatus: EXITED` and `lastStatusChange: Exited by user: Tue Aug 18 2026 18:31:26 GMT+0000`. A following `runpodctl pod list` returned no active pods. No terminate command was issued.

## Publication

- Experiment commit: `c3cbabdb6c000fe4495b55917da4b5631d9410af`
- Branch: `agent/rrq-dense-prefix-ceiling`
- Draft review PR: <https://github.com/ArianSFP/LLM_compression_study/pull/3>
- Stacked base branch: `agent/q2-corrected-allocator-low-rank-down`
