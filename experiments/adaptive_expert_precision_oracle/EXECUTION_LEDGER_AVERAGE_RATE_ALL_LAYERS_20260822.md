# PR #13 average-rate allocation: all-layer execution ledger

Date: 2026-08-22

Status: complete

## Scope

This study extends the frozen PR #13 average-rate allocation experiment from
layers `[0, 4, 20, 39]` to every Qwen3-30B-A3B MoE layer in `range(40)`.
It imports PR #13's scientific kernels unchanged and adds only hash-locked
sharded capture loading, per-layer atomic work units, distributed
finalization, and all-layer reporting.

- Frozen PR #13 head: `dfba3748e51d6916dc1f28cb1d3b3250188adbbe`
- Frozen PR #13 runner SHA-256:
  `55dd6f1b28b3dea36913943d5094b007176ddeabf8c12925dfbcf200118acaae`
- Frozen PR #13 config SHA-256:
  `a889dbf9c5ebdf79de0e7e7b7d65165d9f39782d72d669e073c80966e5d83edf`
- All-layer run ID:
  `qwen36_mxfp4_average_rate_all_layers_20260822_v1`

The pre-existing all-layer capture was produced by the PR #12 capture
lineage. That provenance does not make this a PR #12 result: the frozen PR #13
fit and evaluation sentinels below establish exact scientific compatibility,
and this runner hash-locks both the PR #13 runner and config.

## Locked inputs and boundary

- Capture schema: `exact_mxfp4_all_layer_sharded_v1`
- Capture manifest SHA-256:
  `40c06b1495e52f689f4d95217ccec70711a69166d63067c3dcdcb4589078cc4d`
- Capture identity-grid SHA-256:
  `075c851bbda32ad3009d4bfe56815a03458524b304ecb4165bb4c8f119538f00`
- Checkpoint revision:
  `7eceff3a9f7e6f916c824d197266d86676bce695`
- Checkpoint config SHA-256:
  `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`
- Checkpoint index SHA-256:
  `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`
- Selected-tree SHA-256:
  `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`

All four pods independently admitted and hashed the same 80 train/validation
capture shards (102,561,139 bytes). Each layer contains 70 train rows and 32
complete validation top-8 groups. Test shards were not staged, loaded, or
used. The checkpoint audit found 40 layers, 61,440 required tensor entries,
and zero missing entries.

## Pod audit and allocation

Every pod used Python 3.12.3, NumPy 2.1.2, PyTorch 2.8.0+cu128, CUDA 12.8,
one RTX 3090, and a cgroup CPU quota of 27.2 cores. Evaluation used 24 process
workers on every layer.

| Pod | RunPod ID | CPU / RAM | Driver | Assigned scientific work |
| --- | --- | --- | --- | --- |
| A | `cdgrq0vp196lny` | EPYC 7H12 / 117 GiB | 595.71.05 | sentinel 0; layers 1,2,3,5,6,7,8,9 |
| B | `iig7vj7h8szgsn` | EPYC 7H12 / 117 GiB | 595.71.05 | sentinel 4; layers 11-19 |
| C | `29n5b299oe0qb4` | EPYC 7763 / 125 GiB | 580.126.20 | sentinel 20; layers 21-29; stolen tail layer 10 |
| D | `nkyuf55wi0zacu` | EPYC 7763 / 125 GiB | 580.159.03 | sentinel 39; layers 30-38; finalizers and analysis |

Layer 10 was held back from A after layer 9 began evaluation and was assigned
to C as soon as C completed layer 29. The paused A queue controller was then
retired after layer 9's atomic sidecar appeared. No duplicate layer-10 write
was started.

Direct utilization checks showed evaluation workers near 100% CPU each while
the GPU remained at 0% utilization and 556 MiB allocated. The experiment was
CPU-bound. Fit phases were single-process CPU work; evaluations saturated the
available pod CPU quota.

## Compatibility and determinism gates

1. Duplicate layer-0 fits on A and C were bit-for-bit identical across every
   NPZ array despite different CPU generations, kernels, and NVIDIA drivers.
2. All-layer-capture fits for layers 0, 4, 20, and 39 exactly matched the
   frozen PR #13 factor artifacts.
3. The same four sentinel evaluations produced exactly the frozen PR #13
   scientific group and expert fields: 2,112 group rows and 16,896 expert
   rows per layer. Only expected wall-clock/runtime columns differed.
4. The full finalizers rejected failures, missing layers, changed hashes,
   duplicate identities, row-grid changes, and page-budget violations. Both
   finalizers passed.

These gates establish that the PR #12-lineage capture is an admissible input
to the frozen PR #13 scientific calculation.

## Completed evidence

- Fitted layers: 40/40
- Factor NPZ payload: 472,846,521 bytes
- Validation identities: 1,280 complete token/layer top-8 groups
- Routed expert invocations: 10,240
- Aggregate policy rows: 84,480
- Aggregate expert rows: 675,840
- Policies per group: 66
- Test scientific rows admitted or used: false
- Recorded failures and failure history: empty

The full finalized archive is retained on stopped RunPod
`nkyuf55wi0zacu` at:

`/workspace/pr13_average_rate_all_layers_20260822_v1`

Important full-artifact hashes:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Aggregate group frontier | 9,019,668 | `f74e1225789c6502ed7818caa2cad2c50e74633254297b0fb6ebb5c5886f4db6` |
| Aggregate expert allocation | 275,809,057 | `19c2dc0f94cd9a8237dd866b1a1997c4edcdf6a66457f5b924c4a1e9604fb0c4` |
| Validation accounting | 13,348,257 | `ff971cab8701c03194c085e56edf6cee1862e01d3d81af7959b72d0a320bedb4` |
| Evaluation run facts | 13,376,352 | `e0f58de6c5345885488068d5efb24c063f7a910d4b7fc32fb050f9d0f5e79d19` |
| Fit manifest | 41,761 | `60204faca5d9d7974e035ff05088938af942fbd5e0d313c030dddc6748157069` |
| Fit facts | 29,364 | `0a417361b9c7b8f3ce19bade23aaa2b06ea6420aa12d703b356cf6359628fd4d` |

The repository carries the complete analysis, fit manifest/facts, and a
30,251-byte compact evaluation-facts projection. The projection preserves all
40 execution-provenance records and hash-locks both the omitted diagnostics
and the 13.4 MB source facts file.

- Analysis manifest SHA-256:
  `14042103d1d8b5f0d0b660a88de19e679e889dc689d3ae5943815e7deaf83632`
- Compact evaluation facts SHA-256:
  `2bae39a257cdd503275ce32a89819fb4f048d09b87afe7d3d09d60525c1e811d`
- Copied compact files verified against the manifest: 12/12, zero mismatches

## Scientific outcome

Continuation status: `continue_to_predicted_h4_average_rate`.

At the frozen primary point (749 mean correction pages/expert, 1,536-page
burst cap):

- Overall average bpw: 0.9987386068
- qenergy recovery p10: 99.5429%
- qenergy recovery median: 99.7916%
- Paired p10 gain over uniform: +0.0554 percentage points
- Paired median gain over uniform: +0.1867 percentage points
- Selected-frontier repair median gain: +0.0059 percentage points
- Selected page-vector change fraction: 73.125%
- Median-gain gate: pass
- Tail-nonregression gate: pass

The finite-frontier global bound is not certified at the frozen tolerance:
the median/p90/maximum relative gaps are 33.9502% / 36.6751% / 63.1366%.
No configuration is deployable or promotable. This remains an exact-H4
interaction-geometry ceiling, not token accuracy, latency, routing, logit,
model-quality, or end-to-end evidence.

## Analysis finalization note

The first analysis attempt stopped immediately because the pod venv lacked the
declared Matplotlib dependency. Matplotlib 3.10.0, matching frozen PR #13 SVG
metadata, was installed into the isolated analysis venv.

The next attempt completed the approximately 100-minute fail-closed scan and
wrote every report, CSV, plot, and promotion file. Its final manifest step
then exposed two wrapper assumptions:

1. source provenance always pointed at the original four-layer runner; and
2. all manifest paths were assumed to live below the repository root.

The wrapper now selects the runner recorded by the config and computes a
scoped common manifest root for both in-repository and external archive
layouts. Because all ten generated outputs already existed and the analyzer
process had exited, only the missing manifest was recovered. The recovery
rechecked finalized artifact hashes, on-disk source hashes, exact output set,
run IDs, completion/failure flags, and row counts before its atomic write.
Every generated-file hash in the resulting manifest verifies.

## Verification and shutdown

- Focused analyzer/all-layer suite: 13 passed
- Full experiment suite: 344 passed, 1 skipped
- Byte compilation: passed
- `git diff --check`: passed
- Only warning: the pre-existing local pandas/numexpr version warning

Pods A, B, C, and D were stopped with `runpodctl pod stop` after their needed
artifacts were verified locally. An unrelated fifth running pod was not
touched.
