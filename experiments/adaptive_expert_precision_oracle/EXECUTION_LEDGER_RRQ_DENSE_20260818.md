# Dense RRQ ceiling execution ledger — 2026-08-18

## Identity and scope

- Repository: `ArianSFP/LLM_compression_study`
- Branch: `agent/rrq-dense-prefix-ceiling`
- Stacked base: `agent/q2-corrected-allocator-low-rank-down`
- Foundation commit: `cb9210edaae306bcbd9856686303c28dbf609290`
- Experiment: dense recurrent-residual Q2-stage ceiling only; no selective
  packet allocator, H4 predictor, top-8 joint allocation, new low-rank run, or
  unvalidated replay claim.
- Test status: paired exploratory. PR #2's held-out outcomes informed this RRQ
  hypothesis, so its 16 test requests are no longer a pristine confirmatory
  set. Quantizer format and calibration are still selected only on validation.

## Authoritative read-only inputs

- Production reference:
  `/workspace/LLM_prefetch_study/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`
- Capture root:
  `/workspace/LLM_prefetch_study/artifacts/gcrp2/captures/gcrp2r_transformers_bf16_corpus_v1`
- Source BF16 checkpoint:
  `/workspace/LLM_prefetch_study/artifacts/j_route_0/hf/Qwen3.6-35B-A3B`
- Exact SHA-256 values and byte counts: `results/.../run_facts.json` (filled by
  the runner after reading the authoritative files).
- Compact capture: six evenly spaced audited segments, token stride 16,
  layers 0/4/20/39, 5,036 records (3,292 train, 760 validation, 984 test), with
  complete-request separation. The compact NPZ is an ephemeral run artifact
  and is not committed.

## Hardware and environment

- RunPod pod ID: `gszpnl0eshsw9s`
- Pod endpoint used: `root@213.192.2.67:40075`
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB
- Driver: 580.159.03
- Remote Python:
  `/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python`
- Local analysis: Python 3.13.5; pytest 8.3.4
- Remote torch/CUDA/package details and peak CUDA allocation are recorded in
  `run_facts.json`.
- This is not a DGX Spark. GPU timings, where noted, are RTX 3090 measurements;
  Spark costs are analytical only.

## Configuration and rate accounting

- Main config: `configs/q2_rrq_dense_ceiling.json`
- Smoke config: `configs/q2_rrq_dense_smoke.json`
- Seed: 20260817
- Sample: 12 hot/median/cold experts per layer; up to four test invocations per
  expert for the 133-invocation exploratory set, retaining the 39-invocation
  exact PR #2 paired subset.
- Resident base: activation-diagonal-MSE Q2, G64, codes and FP16 scales local.
- External capacity: production-reference fallback plus the three-stage RRQ
  package. The resident base is not duplicated externally.
- A symmetric stage is 2.25 effective bpw at G64 and 2.125 at G128 after FP16
  scales. Affine G64 is 2.375 bpw in the implementation because its per-group
  zero points are byte-aligned. Both ideal packed and actual implementation
  fields are retained; primary results use actual bytes.
- Dense stream page accounting: separate contiguous code, scale, and optional
  zero-point streams at 512 B, 1 KiB, 2 KiB, and 4 KiB.

## Exact commands

Local tests:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Result: 26 passed; one unrelated local numexpr-version warning.

Capture extraction on the pod:

```bash
cd /root/adaptive_expert_precision_oracle_rrq
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  scripts/extract_pilot_remote.py \
  --config configs/q2_rrq_dense_ceiling.json \
  --output work/rrq_captures_seed20260817.npz
```

One-layer smoke:

```bash
RRQ_REFERENCE_HASH_CACHE=work/reference_hashes_cache.json \
  /workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  scripts/run_dense_rrq_ceiling.py \
  --config configs/q2_rrq_dense_smoke.json \
  --captures work/rrq_captures_seed20260817.npz \
  --output results/qwen36_q2_rrq_dense_smoke
```

Main four-layer run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RRQ_SEARCH_WORKERS=4 \
RRQ_REFERENCE_HASH_CACHE=work/reference_hashes_cache.json \
  /workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  scripts/run_dense_rrq_ceiling.py \
  --config configs/q2_rrq_dense_ceiling.json \
  --captures work/rrq_captures_seed20260817.npz \
  --output results/qwen36_q2_rrq_dense_ceiling_20260818_v1
```

Local deterministic analysis:

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_dense_rrq.py \
  --result experiments/adaptive_expert_precision_oracle/results/qwen36_q2_rrq_dense_ceiling_20260818_v1 \
  --prior-allocator experiments/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v7/allocator_metrics.parquet \
  --config experiments/adaptive_expert_precision_oracle/configs/q2_rrq_dense_ceiling.json
```

## Failed attempts and corrections

1. The first capture extraction SSH wrapper yielded while its child continued.
   A second extraction was accidentally started; the duplicate was stopped and
   the original was allowed to finish. The network-volume inputs were never
   modified.
2. The first smoke exposed FP16 scale underflow in the historical nested/direct
   transformed-atom controls. That attempt is preserved remotely as
   `qwen36_q2_rrq_dense_smoke_failed_fp16_underflow`. Control encoders were
   corrected to serialize the smallest positive FP16 scale rather than divide
   by zero. RRQ itself retains true zero-scale diagnostics.
3. The first main launch was stopped before scientific output because tuple
   rows lacked exact gate/up/down rate columns. Its partial directory is
   preserved as `qwen36_q2_rrq_dense_ceiling_pre_allocation_columns`; the
   corrected run records all three projection rates.
4. Full-suite collection initially failed without the experiment `PYTHONPATH`.
   With the documented path, all 26 tests passed.

## Timings, memory, and final state

The four-layer runner completed successfully in **19,751.815 seconds
(5 h 29 min 11.8 s)**. It emitted 173,056 complete-expert tuple rows, 32,448
projection rows, and 4,212 stage-diagnostic rows. Peak CUDA allocation was
159,514,624 bytes (152.1 MiB); construction was principally CPU-bound. The
local deterministic analysis produced 8,512 per-invocation frontier rows, 192
summary rows, 1,000-request-cluster bootstrap intervals, and 13 plots in both
PNG and SVG.

The exact 432 serialized stage packages total 133,721,088 bytes. Payload files
were intentionally omitted; their consolidated manifest is 667 KiB with
SHA-256 `48902ce894c174ff8535ae3f361ccc19b17dfa36c9c264415c043283ee82057b`.
The production reference is 22,182,574,368 bytes with SHA-256
`e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6`.
The capture manifest SHA-256 is
`9931e72b07e63d12151fa5f4d11613ce9d44f94f927ea5b437aab1ef97c14a9a`.

Remote environment: Linux 6.8.0-110-generic/glibc 2.39, PyTorch 2.8.0+cu128,
CUDA 12.8, NVIDIA GeForce RTX 3090. The focused/full local test suite ultimately
contains 28 passing tests; the only warning is an unrelated local pandas/
numexpr version advisory.

All dense and selective artifacts were transferred and checksum-verified before
the pod was stopped—not terminated—with `runpodctl pod stop gszpnl0eshsw9s`.
RunPod returned `desiredStatus: EXITED` at 2026-08-18 18:31:26 UTC. Publication
commit and draft-PR details are recorded in the selective follow-up ledger.
