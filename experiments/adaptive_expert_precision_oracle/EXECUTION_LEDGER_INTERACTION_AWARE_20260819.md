# Execution Ledger: Interaction-Aware MXFP4 Allocator

Date: 2026-08-19 (Europe/London)
Repository: `ArianSFP/LLM_compression_study`
Branch: `agent/mxfp4-interaction-aware-allocator`
Starting commit: `6b31a88ee517359847f940cc74db56898a8b8425`
Seed: `20260818`

## Locked inputs

- Exact checkpoint repository: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`
- Revision: `7eceff3a9f7e6f916c824d197266d86676bce695`
- `config.json` SHA-256: `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`
- tensor index SHA-256: `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`
- Locked PR #4 trees: `results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json`
- Exact capture request split: original 6 train / 3 validation / 3 test prompts.
- Cross-reference capture request split: 53 train / 12 validation / 16 test complete requests.

The codec, leaf/scales, selected trees, checkpoint revision, and request split were not changed. No H4 predictor was trained.

## Hardware

RunPod SSH endpoint supplied by the user. Runtime host `b33e336b8c18`:

```text
GPU: NVIDIA GeForce RTX 3090
VRAM: 24,576 MiB
Driver: 580.159.03
System RAM: 1.0 TiB
```

## Environment

```bash
python3 -m venv --system-site-packages /root/allocator_env
/root/allocator_env/bin/pip install numpy==2.1.2 pandas==3.0.5 pyarrow==25.0.1 \
  safetensors==0.8.0 scipy matplotlib pytest huggingface_hub \
  transformers==5.15.0 accelerate==1.14.0 pydantic loguru
/root/allocator_env/bin/pip install --no-deps compressed-tensors==0.18.0
```

Validated versions:

```text
torch 2.8.0+cu128
transformers 5.15.0
compressed-tensors 0.18.0
pandas 3.0.5
pyarrow 25.0.1
```

## Checkpoint download

```bash
/root/allocator_env/bin/hf download \
  pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695 \
  --local-dir /root/qwen36_mxfp4_candidate
```

39 files downloaded successfully. Every study runner failed closed on the PR #4 config/index hashes before reading weights.

## Broader capture reproduction

```bash
cd /root/interaction_allocator
/root/allocator_env/bin/python scripts/extract_pilot_remote.py \
  --config configs/q2_corrected_page_allocator.json \
  --output work/captures_seed20260817.npz
```

Result: 5,036 rows, layers 0/4/20/39, the same six immutable audited segments, 53/12/16 request split, position stride 16, model revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, and matching model/tokenizer/prompt-manifest hashes.

## Exact capture reproduction on 24-GB GPU

The original PR #4 capture graph peaked near 84 GB on the prior GPU. An initial attempt used Accelerate CPU offload:

```bash
/root/allocator_env/bin/python scripts/capture_exact_mxfp4_confirm.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_exact_confirm_captures.npz \
  --layers 0 4 20 39 --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695 \
  --cpu-offload --max-gpu-memory 20GiB
```

After loading, compressed-tensors dequantization invalidated Accelerate's offload map and raised a `KeyError` for `model.language_model.layers.10.linear_attn.in_proj_qkv.weight`; it produced no capture. The corrected command keeps the graph wholly on CPU and avoids Accelerate hooks:

```bash
/root/allocator_env/bin/python scripts/capture_exact_mxfp4_confirm.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_exact_confirm_captures.npz \
  --layers 0 4 20 39 --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695 \
  --cpu-only
```

Prompts, request IDs, hooks, and 6/3/3 split are unchanged. Device placement is recorded in the new manifest.

Completed result: 564 rows (280 train / 128 validation / 156 test), 47m50.916s wall time. Capture SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`; manifest SHA-256 `428752a60ac09f0f8c5cbcef3ee850ce4ff4ebe466c5d77774e27c7058986fe8`.

CPU/GPU arithmetic changed the frequency-order median/hot IDs in three layers. To keep comparison with PR #4 controlled, the runners explicitly retain the PR #4 sampled experts: layer 0 `{72:cold,160:median,62:hot}`, layer 4 `{17:cold,110:median,154:hot}`, layer 20 `{242:cold,251:median,191:hot}`, and layer 39 `{68:cold,231:median,108:hot}`. These experts retain identical split occurrence counts and sum to exactly 85 held-out invocations.

## Tests

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/src \
  pytest -q experiments/adaptive_expert_precision_oracle/tests
```

Final local result: **48 passed** (one unrelated local numexpr-version warning). The focused interaction allocator suite has **9 tests**, including paid-page closure and incomplete-physical-subset regressions.

## Bounded pilot

```bash
/root/allocator_env/bin/python scripts/run_interaction_projection_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/cross_reference_pilot \
  --source cross_reference --mode pilot
```

The pilot evaluated one previously audited held-out invocation in every layer. All low-rank combinations were screened on one validation invocation per layer/expert/projection; only the global validation winner plus the best member of each factor family was evaluated on the pilot test row.

Pilot gate/up 90% diagonal/exact action ratios were:

| Layer | Gate | Up |
| ---: | ---: | ---: |
| 0 | 2.546x | 2.869x |
| 4 | 2.996x | 3.432x |
| 20 | 2.696x | 3.095x |
| 39 | 2.613x | 3.444x |

This passed the broad interaction-gap expansion condition. No rank <= 64 low-rank candidate retained a positive reproducible fraction of exact-over-diagonal gate/up gain, so low-rank variants were not promoted to full test runs.

## Bounded physical pilot

```bash
/root/allocator_env/bin/python scripts/run_interaction_physical_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/cross_reference_physical_pilot \
  --source cross_reference --mode pilot
```

Completed four audited held-out invocations. The training-mask layout exact-action path repriced to pages has median 1.000x amplification. The sparse exhaustive page-mask oracle has interim/final amplification above 1.2x. The bounded joint gate/up path used 16 global refreshes and 64 gate/up pages; its exact-SwiGLU-rescored rows run at 0.416667 physical bpw and approximately 10--12 seconds per invocation.

## Failures and corrections

1. The normal local patch helper repeatedly failed because its sandbox could not create the loopback interface (`RTM_NEWADDR`). The same `apply_patch` program was run through the approved command executor; no alternate file-writing mechanism was used.
2. The first environment command allowed `compressed-tensors 0.18` to resolve dependencies and began downloading an incompatible prerelease Torch/CUDA 13 stack. It was interrupted before installation. The environment was cleared, system `torch 2.8.0+cu128` was retained, and compressed-tensors was installed with `--no-deps`, matching the working PR #4 environment strategy.
3. The first pilot failed closed before metrics because `audit_checkpoint` does not return a revision field. The guard was corrected to compare the exact PR #4 config and tensor-index SHA-256 hashes, which is stronger local evidence. No scientific rows were produced by the failed invocation.
4. The first all-combination low-rank pilot used CPU BLAS inside the per-action `U @ p` loop. It was stopped before an expert checkpoint was written, moved to CUDA, and bounded to validation-first promotion.
5. Interrupting that SSH command detached its remote Python process. The exact stale PID was identified and terminated; the newer run was left untouched.
6. A second pilot remained broader than intended (30 low-rank combinations on two validation and one test row). It was terminated, and the final pilot used one validation row for all combinations followed by only validation-Pareto test candidates.
7. Accelerate CPU offload loaded the dequantized checkpoint but then failed because compressed-tensors replaced a weight that remained in the offload map. The failed run emitted no capture. The unchanged generator was restarted with `--cpu-only`, eliminating the incompatible offload indirection.
8. The first page-mask implementation removed a paid stage-two page even when other actions on it were not yet nested-eligible. Paid pages are now re-evaluated to closure after every depth change at zero additional page cost, with a dedicated regression test.
9. The first matched-physical low-rank attempt used the inherited page helper, which rebuilt a 2,048-entry inverse permutation for every action. It was stopped before any row was written. An equivalent action-to-page map is now precomputed once per projection and residual energies for all physical budgets are batched on CUDA.
10. The first optimized rerun shadowed the full residual-energy denominator with the zero exact-over-diagonal logical gain at the 2-bpw endpoint. It failed before writing an expert checkpoint. The variables were separated (`total_energy` versus `logical_denominator`) and the empty output directory was verified before rerun.
11. The first completed matched-physical summary divided by exact-minus-diagonal rows even when that denominator was negative. Such rows cannot represent retained exact-over-diagonal gain. The metric now requires a strictly positive denominator and otherwise records NaN; the pilot was rerun from scratch. This removed the spurious validation promotion and produced the final negative rank <= 64 result.
12. The connected RunPod API tool returned HTTP 401 because its configured key was invalid. A systemd shutdown was unavailable inside the container, and terminating container PID 1 only caused RunPod to restart it. The locally authenticated RunPod CLI identified the supplied IP/port as pod `9s3kwxjvs0vdh6`; the pod-level stop below succeeded and was independently verified from the pod list.

## Full exact-checkpoint expansion

```bash
/root/allocator_env/bin/python scripts/run_interaction_projection_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/exact_checkpoint_full \
  --source exact_checkpoint --mode full
```

Completed all 85 held-out invocations in 2,144.454 seconds: 7,140 broad-audit rows, 12,240 selector rows, 180 Gram-spectrum rows, and complete endpoints of 1.0. Primary raw hashes:

- broad audit: `dead5136d3f9c084044e856411ebf85e8161fa7d8b5632b685c00d684bae70d4`
- selector frontier: `5caba62ac9f9a528dc63961600ce7fa8ff0319fb604b954b9913094636b097dc`
- Gram spectrum: `957e39dd46325d9aebdda836bd66b5f7325ba7628ab75806b7c5a0a4f8b03cee`
- run facts: `79596eeba87dbf18c99d1923adb9360d1e07ba202a8f57edf87ee89b193394d6`

## Broader cross-reference expansion

```bash
/root/allocator_env/bin/python scripts/run_interaction_projection_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/cross_reference_full \
  --source cross_reference --mode full
```

Completed all 142 sampled held-out invocations in 4,470.714 seconds: 11,928 broad-audit rows, 20,448 selector rows, 180 Gram-spectrum rows, and complete endpoints of 1.0. Low-rank frontiers were deliberately empty in both full expansions because validation promoted none. Primary raw hashes:

- broad audit: `a7feaa943463b3585da5231c5cc1b2c6ec180ffcce90ab46af52c7877a7255bc`
- selector frontier: `88ece0f4bbf4e6f5a6f29a742e972644c4f6abb09982d5469411212b69d1c746`
- Gram spectrum: `ae0f6f23d059e043bd79b09b2d07f8207430e1f7c6b892920ed042eb5340d16a`
- run facts: `8849362c8bc7b8a638b01e100a6bfed2c13d13c16143ed3a935a344a4525aae1`

## Validation-first low-rank pilot

The corrected matched-physical pilot completed in 951.164 seconds. It contains 3,024 low-rank rows spanning ranks 8/16/32/64/128, all three requested factor families, and FP16/int8. Its low-rank parquet hash is `2d0dfb1085d00ca4602fef635aed53f933e6cf173d0169acc94e73e071ecbccf`. The best repeated held-out rank <= 64 primary-point median was -5.306 for fixed-JL/rank-64/int8, so `configs/interaction_allocator_promotions.json` promotes no low-rank selector.

## Full physical and complete-expert expansion

```bash
/root/allocator_env/bin/python scripts/run_interaction_physical_study.py \
  --config configs/qwen36_mxfp4_interaction_allocator.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/exact_checkpoint_physical_full \
  --source exact_checkpoint --mode full
```

Completed 85 invocations and all four training-only layouts in 1,505.203 seconds. The output has 8,330 complete-expert packet rows and exact endpoints. Raw hashes:

- paired/hybrid frontier: `000c340fde29c3827287b903bb7bd83bf41af23f10bd51628ba95017bf85ecf8`
- storage accounting: `26739edea1cfc1b02c8552f7072ece4c4a927717054dbeaef2ff841d7a43c580`
- layout facts: `82b116c4f66e3c13b16a31117d226994e42b003a0f4989cd829ffabcc40b21b4`

The bounded four-invocation physical pilot supplies 64 exact page-mask rows and four joint first-order/exact-rescore packet rows. Its page and packet hashes are `67f978c65dd698834efae6992c5bcf3e16275f61f18a20ed67fc85a0a0b7045e` and `a2f41fd633c139a9e00ddf419fe588a1a5527f845520bf3d4d636085c3811c6a`.

## Copy-back and final analysis

```bash
scp -o StrictHostKeyChecking=no -P 40025 -i ~/.ssh/id_ed25519 \
  root@213.192.2.120:/root/interaction_allocator/results/cross_reference_physicalmatched_pilot/* \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_physicalmatched_pilot/
scp -o StrictHostKeyChecking=no -P 40025 -i ~/.ssh/id_ed25519 \
  root@213.192.2.120:/root/interaction_allocator/results/cross_reference_physical_pilot/* \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_physical_pilot/
scp -o StrictHostKeyChecking=no -P 40025 -i ~/.ssh/id_ed25519 \
  root@213.192.2.120:/root/interaction_allocator/results/exact_checkpoint_full/* \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/exact_checkpoint_full/
scp -o StrictHostKeyChecking=no -P 40025 -i ~/.ssh/id_ed25519 \
  root@213.192.2.120:/root/interaction_allocator/results/cross_reference_full/* \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_full/
scp -o StrictHostKeyChecking=no -P 40025 -i ~/.ssh/id_ed25519 \
  root@213.192.2.120:/root/interaction_allocator/results/exact_checkpoint_physical_full/* \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/exact_checkpoint_physical_full/

python scripts/analyze_interaction_allocator.py \
  --projection results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/exact_checkpoint_full \
  --projection results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_full \
  --low-rank-pilot results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_physicalmatched_pilot \
  --physical results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/exact_checkpoint_physical_full \
  --physical-pilot results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/cross_reference_physical_pilot \
  --output results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1 \
  --pr4-result results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/exact_checkpoint_confirmation
```

Final row counts are 19,068 broad-audit, 32,688 refresh-frontier, 360 Gram-spectrum, 3,024 low-rank, 64 page, and 8,334 paired/hybrid rows. Six plots were written in both PNG and SVG. `artifact_hashes.sha256` records SHA-256 for every required final artifact and plot, every copied raw artifact, and the fresh-capture manifest.

## Scientific stop decision

- Broad gate/up interaction gap reproduced in all eight layer/projection groups. At 90% recovery, diagonal/exact action ratios were above 2x in every fresh and broader group.
- Rank <= 64 retained median -5.306 of exact-over-diagonal gain at matched physical budgets, far below 0.90. Expansion stopped for low-rank paths as required.
- At one physical correction bpw on fresh difficult layers, PR #4 reported p10/median 0.833365/0.893499. The predeclared exact-training-mask H0 layout produced 0.835098/0.892560: p10 improved, median did not.
- The selected layout's exact action path reprices at 1.000x page amplification, while the sparse exhaustive page-mask oracle is 1.322x and paired page oracle 1.246x.
- Separate/paired/hybrid storage multipliers are 1.471x/1.471x/1.941x, below the 5x cap.

No test layout or metadata tuning followed these outcomes.

## RunPod stop

```bash
runpodctl pod stop 9s3kwxjvs0vdh6
runpodctl get pod | rg '^9s3kwxjvs0vdh6'
```

Verified final state: `EXITED` at 2026-08-19 05:26:36 UTC. The pod disk persists; GPU billing is stopped.

## Git publication

```bash
git commit -m "Add interaction-aware MXFP4 allocator study"
git push -u origin agent/mxfp4-interaction-aware-allocator
```

The staged scope was checked with `git diff --cached --check`; all 48 tests and Python compilation passed immediately before publication.
