# Execution ledger: codebook granularity study

Date: 2026-08-19  
Repository HEAD tested: 37e7bfcd168545d6a44a3e3df45b6eb27d858a3c  
Branch: agent/mxfp4-sparse-streaming  
New isolated namespace: qwen36_mxfp4_codebook_granularity_20260819_v2

Existing historical result directories were read but never modified. The worktree
already contained unrelated sparse-streaming state; this study uses new paths only.

## Immutable reference

- Repository: pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4
- Revision: 7eceff3a9f7e6f916c824d197266d86676bce695
- Format: compressed-tensors mxfp4-pack-quantized, E2M1 leaf nibbles and one
  uint8 E8M0 exponent per 32 weights.
- config.json SHA256: 52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a
- tensor index SHA256: 842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb
- locked tree SHA256: da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8

The exact checkpoint was downloaded to the RunPod and both checkpoint hashes were
checked before fitting. The old mixed GGUF on the volume was not used.

## Capture identities and split policy

Primary four-layer extension:

- qwen36_mxfp4_md_fresh_capture.npz
- SHA256 75087507e3df27735fc202ab02ea0b891daabcd4461290e58e5e55d4cd0d2698
- layers 0, 4, 20, 39
- 12 train / 6 validation / 6 test requests
- one cold, median and hot expert per layer

Locked baseline reproduction:

- qwen36_exact_confirm_captures.npz was regenerated CPU-only
- observed SHA256 52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931
- 6 train / 3 validation / 3 test requests
- the evaluator locks PR4 expert IDs because CPU/GPU router arithmetic changes a
  few frequency-order selections

The original PR4 GPU capture (84b7e727...) is unavailable. Its published frontier is
retained as frozen artifact evidence; the byte-reproducible current control uses the
52bc capture. The 24-request extension is a separate stronger-sample cohort.

Training requests alone determine functional masses, centroids, family tables, block
IDs and modifiers. Validation selects iterations, Q2 tolerances and promoted designs.
Test rows are used only for final metrics.

## Hardware and environment

- 2 x NVIDIA RTX 3090 (24 GiB each), 1 TiB host RAM.
- Python environment: /workspace/codebook_env.
- PyTorch 2.8.0+cu128, NumPy 2.1.2, pandas 3.0.5, PyArrow 25.0.1,
  SciPy, Matplotlib, safetensors and compressed-tensors 0.18.
- Sufficient for sampled codec/evaluator work and CPU-only capture; insufficient
  for the approximately 84-GiB fully dequantized model on one GPU.

## Principal commands

All remote commands ran from /workspace/codebook_granularity_study with
PYTHONPATH=src:scripts.

~~~bash
python scripts/run_codebook_granularity_study.py fit \\
  --config configs/qwen36_mxfp4_codebook_granularity.json \\
  --captures /workspace/adaptive_expert_precision_oracle_md_20260818/inputs/qwen36_mxfp4_md_fresh_capture.npz \\
  --checkpoint /workspace/qwen36_mxfp4_candidate \\
  --trees locked/selected_trees.json \\
  --output results/qwen36_mxfp4_codebook_granularity_20260819_v2

python scripts/run_codebook_granularity_study.py merge \\
  --config configs/qwen36_mxfp4_codebook_granularity.json \\
  --output results/qwen36_mxfp4_codebook_granularity_20260819_v2

MPLBACKEND=Agg python scripts/analyze_codebook_granularity.py \\
  --result results/qwen36_mxfp4_codebook_granularity_20260819_v2
~~~

Dense evaluation was sharded over layers 0/4 and 20/39. Selective evaluation uses
the validation-retained A/B/C/D/E arms, the direct baseline, ordered-plane and paired
exact actions, all requested rates, a fixed natural 512-byte layout, and exact 9x9x9
complete-SwiGLU rescoring. The analyzer command regenerates every metric summary and
all 16 PNG/SVG plot pairs from the saved Parquet/CSV inputs.

## Reliability event

The first two long foreground SSH selective workers lost their sessions before final
Parquet emission. No completed artifact was overwritten. The runner was changed to
commit every completed candidate as an independent atomic Parquet transaction, and
both shards were restarted detached. This changed execution reliability only; codec,
requests, actions and evaluation math are unchanged.

## Verification

- Exact variants assert leaf-code, packed E2M1 nibble, E8M0 scale and numerical Q4
  equality during dense evaluation.
- Winner packages are reread and repeat code/nibble/scale equality checks.
- Suffix streams are 512-byte aligned to the physical page model.
- Focused codec/serializer tests: 8 passed.
- Complete adaptive-expert-precision test directory: 129 passed, with one existing
  NumExpr version warning.

Final timings, promotion hashes and artifact manifests are retained beside the
machine-readable metrics in this namespace.

## Frozen PR6 baseline reproduction

The copied PR6 interaction runner was executed against the frozen `52bc` capture after
GPU0's detached pilot shard exited cleanly. The exact command, hashes, one-bpw table,
and outage chronology are retained in `metrics/PR6_52BC_BASELINE_REPRODUCTION.md`.

- Launched: 2026-08-19 19:46:17 UTC; script-accounted runtime 2,228.150 seconds
  (19:46:23.353--20:23:31.502 UTC).
- Output: 85 held-out invocations, 8,330-by-28 packet dataframe, all four training-only
  layouts, and seven physical budgets.
- Primary parquet SHA-256:
  `000c340fde29c3827287b903bb7bd83bf41af23f10bd51628ba95017bf85ecf8`, exactly
  matching historical PR6.
- Layout facts SHA-256:
  `82b116c4f66e3c13b16a31117d226994e42b003a0f4989cd829ffabcc40b21b4`, exactly
  matching historical PR6.
- Action/page parquet SHA-256:
  `cd9225d1c4474875f27908d757a422d710e90f3321b3cb30aaa3d94e77f5e4ed`, exactly
  matching historical PR6.
- Frozen-stat validation: 36/36 cohort/rate/method comparisons pass; maximum absolute
  statistic error `4.44e-16`.
- One-bpw all-layer p10/median/p90: current diagonal
  `0.825022/0.891499/0.957836`; exact separate `0.838531/0.897337/0.957036`;
  paired `0.855259/0.920027/0.968948`; hybrid H0 ceiling
  `0.872356/0.923651/0.974856`.
- Reliability: two transient RunPod route/storage outages occurred. The PPID-1 process
  survived the first, recovered from `folio_wait_bit_common` without intervention, and
  finished during the second. Atomic checkpoints and final historical byte equality
  demonstrate intact output.

## Eleven-layer winner promotion

The validation winner was promoted, without refitting eliminated arms, over layers
0, 4, 8, 12, 16, 20, 24, 28, 32, 36 and 39. The isolated local result is
`promotion_11layer/`; its formats are `baseline_pr4_2p25` and
`B16_g32_cluster_functional`.

The broad exact-MXFP4 capture has schema `exact_mxfp4_transformers_confirm_v1`,
1,551 rows (141 per layer), 12 requests (6 train / 3 validation / 3 test), and
SHA256 `da786a99209f03515e68bc21f1849f8613dbe5251d4eebff7e3537e157a0d56f`
over 38,591,171 bytes. Split row counts are 770 / 352 / 429. The CPU-only
capture saved and validated the NPZ before the original process failed while
querying `torch.cuda.get_device_name(0)` for its manifest. No recapture was done:
the manifest was synthesized from the validated NPZ and immutable checkpoint
hashes after fixing CPU manifest reporting (`gpu: null`); fixed script SHA256 is
`b999d0706d04d74544fc84ecc305f9bf3e259f1885a863eb33f92a6199307f18`.

The promotion config SHA256 is
`ab0b78aefbc78d36ac519077270c87c44630b3bd5e2c13b055cc80fb83351443`.
Fit ran on GPU 0; dense and selective layer shards were divided over both GPUs.
B16 fitting took 73.950 seconds inside the runner (about 104 seconds end to end);
dense shards took about 24 and 21 seconds; selective shards took about 1,005 and
1,179 seconds. Capture wall time was 2,842 seconds. Shards were atomically
committed and merged only after key and row validation.

Actual resident accounting over 33 serialized experts and 103,809,024 weights:

- PR4 control: 96 table bytes, 2.250007398 resident bpw.
- B16: 1,624,128 expert selector/header bytes plus 4,608 table bytes = 1,628,736
  metadata bytes, 0.125517874 metadata bpw and 2.375517874 total resident bpw.

Validation locked `B16_g32_cluster_functional` with paired exact Q2-to-Q4 actions
before test inspection, using minimum median absolute damage at one physical bpw.
At that point (135 invocations), B16 median damage was 0.019188862 versus
0.019643623 for PR4. The paired per-invocation damage-reduction distribution has
median 12.227541%, while aggregate damage reduction is 9.659384%
(`1 - 46.302029151 / 51.252726936`). The ratio of marginal medians is 2.31506%;
it is a third aggregation and must not be substituted for either result. B16
validation own-Q2 recovery was p10/median/p90 87.218758% / 93.115850% / 96.996390%.

On held-out test, B16 reduced aggregate Q2 damage by 29.899104% and Q3 damage by
55.793510% relative to PR4; median per-invocation reductions were 21.270596% and
26.866049%. B16 Q3 gap recovery p10/median/p90 was
50.167941% / 67.622870% / 80.208408%, versus
44.314143% / 64.902663% / 78.665259% for PR4. Projection median Q3 gap recovery
(gate/up/down) was 71.178892% / 69.749662% / 67.292097% for B16 and
43.134937% / 73.420721% / 70.882645% for PR4. Exact Q4 equality passed.

Cross-format selective recovery uses each invocation's matched PR4 Q2 base damage.
At one physical bpw with the locked paired packet, B16 common-baseline
p10/median/p90 recovery was 88.802092% / 93.844103% / 97.712284%, versus
87.589464% / 92.970034% / 97.512269% for PR4. Aggregate absolute damage was
59.651895 versus 71.649602; aggregate common recovery was 97.001832% versus
96.398814%, a 16.744974% aggregate damage reduction. B16 crossed median common
recovery 90% / 95% / 97.5% at 0.75 / 1.25 / 1.5 physical bpw; PR4 needed
1.0 / 1.25 / 1.5. Candidate-own-Q2 recovery remains separate.

Promotion integrity hashes:

- `promotion_manifest.json`: `9476e9a3a58dd6b099d22b4a1621fa97fdb8c5959bd3f78085669f4246b474f5`
- `metrics/promotion_summary.json`: `4baa7c3074675cb175a7b29a3e173038db6d1dc748fe013ae74263d3ecbb2f1d`
- `metrics/dense_metrics.parquet`: `13e882660291533b40a39e89e6631a8bc4947bcac833a1cb87579be70dbe2724`
- `metrics/selective_metrics.parquet`: `b7ed8f2a83ff99c90c74efe14879f8830a6e856495d895a93e16c0a3dc648420`
- validation summary: `9ecb1c7cfefbf8547a83656425a134cc693194206ffe55323804110825fa72b0`
- validation metrics: `7f90e55846b8fa18c6b473c6d27261ba32993539cfe5376a6fc84f348ee90a83`
