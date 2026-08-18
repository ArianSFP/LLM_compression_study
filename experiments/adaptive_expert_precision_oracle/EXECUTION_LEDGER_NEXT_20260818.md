# Execution ledger — corrected allocator and output-side down oracle — 2026-08-18

## Repository and source identity

- Publication repository: `ArianSFP/LLM_compression_study`.
- Starting branch: `agent/adaptive-expert-precision-oracle-pilot`.
- Starting commit: `ab5070f314189c3e7c9dc30e6af30e0adbfe073b` (`Publish adaptive expert precision oracle pilot`).
- Work branch: `agent/q2-corrected-allocator-low-rank-down`.
- Model: `Qwen3.6-35B-A3B`, HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Model-config SHA-256: `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.
- Production reference: `/workspace/LLM_prefetch_study/models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, 22,182,574,368 bytes, SHA-256 `e1a4925d2ea132576daa9cb980b1102b970d919d896936b7b6e681ef5bc3d3f6`.
- Capture NPZ: `/root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz`, 147,453,931 bytes, SHA-256 `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`.
- Prompt-manifest SHA-256: `916315802286cdead8fc9e4d12a3bdb288fe9feb9d0b3e84c21ecd6ea032c2f7`.
- Tokenizer SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

The model, GGUF, raw capture corpus, and network volume remained read-only. Only `/root/adaptive_expert_precision_oracle` on the pod and the local Git checkout were written.

## Data and split

The four-layer extraction contains 5,036 rows: 1,259 at each of layers 0, 4, 20, and 39. Complete-request splits are 53 train / 12 validation / 16 test requests and 3,292 / 760 / 984 rows. Six evenly spaced audited source segments and token-position stride 16 were used. Seed is 20260817. No request crosses splits.

Run A is deliberately cost-bounded to one held-out invocation for each of up to 12 stratified hot/median/cold experts per layer. Run B retains up to four validation and four test invocations per sampled expert and performs validation-only configuration selection before held-out reporting.

## Hardware and environment

- RunPod ID/name: `oiwjquuc01hepj` / `tall_aquamarine_barnacle`.
- Endpoint during execution: `root@213.192.6.69:40095`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Driver 595.71.05; CUDA runtime 12.8.
- Python 3.12.3; PyTorch 2.8.0+cu128; NumPy 2.1.2; pandas 3.0.5; pyarrow 20.0.0.
- Observed concurrent GPU allocation peaked around 1.7 GiB in diagnostic snapshots. A time-series peak-memory measurement was not instrumented.
- No DGX Spark was used. GPU wall times are not Spark timings; GB10 compute costs are analytical only.

The user required a hard stop by 05:00 BST. A local fail-safe was installed for 04:58 BST as `oracle-runpod-hard-stop-20260818.timer`, executing only `runpodctl pod stop oiwjquuc01hepj`.

## Configurations

- `configs/q2_single_view_baseline_reproduction.json`
- `configs/q2_corrected_page_allocator.json`
- `configs/q2_output_side_down_oracle.json`
- Seed 20260817; group size 64; Q2 resident base; layers 0/4/20/39.
- Run A nested atom precisions 0/2/4/8/16 and page sizes 512/1,024/2,048/4,096 bytes.
- Run B ranks 32/64/128/256/512 and external coefficient Q4/Q8/BF16.
- Final bootstrap uses 1,000 complete-request cluster resamples.

## Exact primary commands

```bash
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  /root/adaptive_expert_precision_oracle/scripts/extract_pilot_remote.py \
  --config /root/adaptive_expert_precision_oracle/configs/q2_corrected_page_allocator.json \
  --output /root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz
```

```bash
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  /root/adaptive_expert_precision_oracle/scripts/run_hybrid_q2_remote.py \
  --config /root/adaptive_expert_precision_oracle/configs/q2_single_view_baseline_reproduction.json \
  --captures /root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz \
  --output /root/adaptive_expert_precision_oracle/results/qwen36_q2_single_view_baseline_20260818_v2
```

```bash
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  /root/adaptive_expert_precision_oracle/scripts/run_corrected_page_allocator.py \
  --config /root/adaptive_expert_precision_oracle/configs/q2_corrected_page_allocator.json \
  --captures /root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz \
  --output /root/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v7
```

```bash
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  /root/adaptive_expert_precision_oracle/scripts/run_output_side_down_oracle.py \
  --config /root/adaptive_expert_precision_oracle/configs/q2_output_side_down_oracle.json \
  --captures /root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz \
  --output /root/adaptive_expert_precision_oracle/results/qwen36_q2_output_side_down_20260818_v3
```

```bash
/workspace/LLM_prefetch_study/artifacts/j_route_0/.venv-jroute0/bin/python \
  /root/adaptive_expert_precision_oracle/scripts/audit_atom_quantizers.py \
  --config /root/adaptive_expert_precision_oracle/configs/q2_corrected_page_allocator.json \
  --captures /root/adaptive_expert_precision_oracle/work/next_oracle_captures_seed20260817.npz \
  --output /root/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v6/independent_quantizer_control.csv
```

```bash
MPLBACKEND=Agg python experiments/adaptive_expert_precision_oracle/scripts/analyze_next_oracles.py \
  --baseline experiments/adaptive_expert_precision_oracle/results/qwen36_q2_single_view_baseline_20260818_v2 \
  --run-a experiments/adaptive_expert_precision_oracle/results/qwen36_q2_corrected_allocator_20260818_v7 \
  --run-b experiments/adaptive_expert_precision_oracle/results/qwen36_q2_output_side_down_20260818_v3 \
  --output experiments/adaptive_expert_precision_oracle/results/qwen36_q2_next_oracles_analysis_20260818_v1 \
  --bootstrap 1000 --seed 20260818
```

## Wall-clock accounting

| Stage | Rows | Wall seconds |
|---|---:|---:|
| Q2 single-view baseline reproduction | 14,364 | 375.44 |
| Corrected Run B validation + test | 358,440 complete-expert + 46,260 down | 484.31 |
| Corrected Run A | 3,969 metrics + 17,868 actions | 1,283.95 resumed-run seconds; 2,105.82 elapsed including the checkpointed pre-resume leg and correction gap |
| Local 1,000-resample analysis and 40 plot files | 1,064 validation-selected Run B test rows | 15.0 |

## Failed attempts and corrections

1. A first extraction used a seed inconsistent with the prior request split. It was discarded; the authoritative extraction uses seed 20260817 and its hash is recorded above.
2. The historical Phase-A physical selector could skip a non-fitting middle atom, return only a count, and then reconstruct the first `count` ranked atoms. New exact-ID/page utilities and a regression test make charged pages, saved IDs, decoded atoms, and arithmetic share one selected set.
3. Corrected-allocator v2 failed before producing science because a repricing call had the wrong argument count. The call and tests were fixed.
4. Corrected-allocator v3 fixed the call but held the down code at authoritative `h_ref`. It was stopped and discarded. The final allocator recomputes down coefficients from current sequential `h_tilde` and stores the exact atom-level bitplane.
5. Corrected-allocator v4 exposed a nested-prefix traversal trap: a damaging 2-bit prefix prevented reaching a beneficial 4/8/16-bit prefix. The final allocator evaluates multi-bit nested paths atomically while charging and recording every intervening bitplane/page.
6. Full 133-invocation Run A runtime projected to roughly 4.4 hours. In accordance with the user’s cost-bounded pilot request it was reduced before completion to one invocation per stratified expert and an 8-coordinate scalable screen. The main four-layer/expert diversity was preserved; sampling uncertainty is reported.
7. A cold layer-4 expert had fewer training masks than the requested four replica regimes. The run stopped after checkpointing 15 invocations. Regime fitting now uses the identifiable number of training-only regimes and deterministically cycles duplicate layouts for the remaining charged replicas; a regression test was added. Resume-by-exact-invocation skipped all completed science and regenerated accounting for every expert.
8. Run B v2 learned the proxy-weighted PCA basis in metric-transformed coordinates but used those vectors directly in raw output space. It was quarantined and not published. v3 maps the subspace back with `G^-1/2`, QR-orthogonalizes it, adds a regression test, and includes validation rows.
9. A local pytest call without `PYTHONPATH` failed during import collection. The documented command passed after setting the source path.
10. Direct replay was audited but could not pass identity because `W_ref` is the production mixed GGUF while the only complete replay graph is the frozen BF16 Transformers checkpoint. No propagation metric was inferred.

## Correctness tests

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src:experiments/adaptive_expert_precision_oracle/scripts \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Result before final publication: 16 passed. Tests cover exact selected IDs/pages, progressive nesting, page increments, sparse-expert regime handling, storage cap, low-rank reconstruction, proxy-basis raw-coordinate mapping, quantizer determinism, basis reconstruction/orthogonality, router weighting, and split leakage.

## Scope limitations recorded at execution time

- No validated direct replay or H1–H4 router/logit/token metric.
- No aligned joint top-8 allocation; the sampled expert weights do not cover all eight experts for every held-out layer-token.
- Replicated layouts are H0 best-replica repricing upper bounds, not a validated H4 layout-choice policy.
- Full-support ideal-byte and independent-quantizer complete-expert controls were not completed; independent atom MSE is saved as a bounded control.
- The full-candidate regret control has only 48 accepted-action depth and is not an equal-depth exact optimum.
- No beam search or local-swap run was completed within the paid pilot bound.
- No Spark timing claim is made.

## Primary output hashes

| File | SHA-256 |
|---|---|
| baseline `hybrid_metrics.parquet` | `7835200af514e31f3d7712ef35c83c3ef9d84efe24d49e725881b3c665d39ed8` |
| Run A `allocator_metrics.parquet` | `7b9d4a375766aa30d15f257a73d6c37b934f2a357cd0dfdb76872b2c8da8b997` |
| Run A `selected_increment_sequence.parquet` | `e101b724ab76e75e8b7391c1029a6bc690a1f57e0df56460d4dba36a4f6715de` |
| Run B `low_rank_complete_expert_metrics.parquet` | `88f9685c40de793e15785446b1b16d452eb09c99404302e4610a1e0d804e8bff` |
| analysis `metrics.parquet` | `cd2e89bc9ced4dbe70d8784ca2b3cb8032cd1c43e70daf7b1b7118f8792317ff` |
| analysis `summary.csv` | `a09ba289e5d5df1aeaa0f853c14a2f5ab621cef8be629863424c6ece765f155c` |
| Run A config | `b421562000ac64ca712c5a015f339f817b8e93fc05eeed466199516df2efdd54` |
| Run B config | `2bed602f59a24dce12bc201b5a9e8cfa5062fa4766247f0448781d969f5e63de` |

The PRO 6000 RunPod `oiwjquuc01hepj` was stopped (desired state `EXITED`, not
terminated) with `runpodctl pod stop` at 2026-08-18 03:17:17 BST after all
artifacts had been copied and verified. The final commit and draft PR URL are
recorded in the publication handoff because a commit cannot contain its own
hash or a PR URL that does not exist until after publication.
