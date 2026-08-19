# Sparse-streaming allocator reproducibility protocol

This document is the reviewer-facing execution contract for the activation-dependent sparse-streaming allocator study. It describes how to reproduce the bounded pilot, freeze validation-only promotion decisions, evaluate the promoted configurations on the fresh exact-checkpoint cohort, and run the broader cross-reference sensitivity check.

It does **not** itself report experimental results. Execution began on 2026-08-19 on RunPod `41rk786odszmk9` (RTX 3090). The fresh capture, bounded pilot, validation-only tile-shape continuation, frozen promotion decision, 85-invocation exact-checkpoint expansion, 142-invocation broader sensitivity run, four-input validation, local final merge, transfer audit, and manifest closure have completed. The execution pod is stopped. Publication is the only remaining handoff.

## Scope and immutable controls

Run commands from the experiment root:

```text
experiments/adaptive_expert_precision_oracle
```

The study branch is `agent/mxfp4-sparse-streaming`. It is anchored at PR #6 commit `1f01edf74ce754fea1615b26a97e4465a829671f`. The allocator may choose which already-encoded suffix pages to read, but it must not alter the embedded Q2→Q3→Q4 codec, selected trees, checkpoint, or request split.

The primary outcome is exact sequential complete-expert qenergy recovery. This is expert-output reconstruction evidence, **not** task accuracy, logit agreement, routing quality, token quality, perplexity, or a downstream benchmark. The experiment does not train an H4 predictor.

The locked identities are:

| Input | Locked identity |
|---|---|
| PR #6 starting commit | `1f01edf74ce754fea1615b26a97e4465a829671f` |
| Sparse-streaming config SHA-256 at protocol freeze | `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326` |
| Checkpoint repository | `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` |
| Checkpoint revision | `7eceff3a9f7e6f916c824d197266d86676bce695` |
| Checkpoint `config.json` SHA-256 | `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` |
| Checkpoint `model.safetensors.index.json` SHA-256 | `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb` |
| Selected trees SHA-256 | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| Fresh exact-checkpoint capture SHA-256 | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| Broader cross-reference capture SHA-256 | `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add` |

The fresh capture contains 6 training, 3 validation, and 3 test requests, with 85 selected held-out test invocations. The broader capture contains 53 training, 12 validation, and 16 test requests, with 142 selected held-out test invocations. The runner requires all three split labels and rejects any request ID shared by train, validation, and test.

The fresh expert IDs are the PR #4 locked hot/median/cold sample. The broader expert IDs are the already-published PR #6 cross-reference cohort, now serialized in the configuration rather than rediscovered from current held-out routing counts. Thus neither pilot nor full execution performs expert selection from validation/test rows.

The runner verifies the capture, tree, checkpoint-config, and checkpoint-index hashes before loading scientific rows, and it records the experiment-config hash for analyzer and promotion binding. It also audits the checkpoint representation for layers 0, 4, 20, and 39. The configured repository revision records the upstream identity; the two local checkpoint metadata hashes and structural checkpoint audit are the executable local guards. Do not silently substitute similarly named model files.

If the committed configuration in the reviewed PR does not hash to the value above, stop. A legitimate protocol change requires an explicit update to this document and the execution ledger before any new run; it must not be reconciled after seeing test data.

## Environment and path setup

Use the environment in which the repository tests pass and CUDA PyTorch can see the 3090. The runner additionally requires NumPy, pandas with a Parquet engine, PyTorch, safetensors, compressed-tensors support used by the existing checkpoint loader, and the repository's existing study dependencies. The analyzer requires pandas, PyArrow, NumPy, and Matplotlib.

Set explicit task-scoped paths. Do not reuse one output directory across phases or capture sources.

```bash
cd /absolute/path/to/LLM_compression_study/experiments/adaptive_expert_precision_oracle

export SPARSE_EXPERIMENT_ROOT="$PWD"
export SPARSE_CONFIG="$SPARSE_EXPERIMENT_ROOT/configs/qwen36_mxfp4_sparse_streaming.json"
export SPARSE_CHECKPOINT="/absolute/path/to/qwen36_mxfp4_checkpoint"
export SPARSE_TREES="$SPARSE_EXPERIMENT_ROOT/results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json"
export SPARSE_EXACT_CAPTURE="/absolute/path/to/qwen36_exact_confirm_captures.npz"
export SPARSE_CROSS_CAPTURE="/absolute/path/to/captures_seed20260817.npz"
export SPARSE_RUN_ROOT="$SPARSE_EXPERIMENT_ROOT/results/qwen36_mxfp4_sparse_streaming_20260819_v1"

export SPARSE_PILOT_EXACT="$SPARSE_RUN_ROOT/pilot_exact_checkpoint"
export SPARSE_INITIAL_VALIDATION="$SPARSE_RUN_ROOT/pilot_validation_only.json"
export SPARSE_TILE_VALIDATION="$SPARSE_RUN_ROOT/validation_tile_shapes_exact_checkpoint"
export SPARSE_AUGMENTED_VALIDATION="$SPARSE_RUN_ROOT/pilot_augmented_validation_only.json"
export SPARSE_PILOT_ANALYSIS="$SPARSE_RUN_ROOT/pilot_analysis"
export SPARSE_FULL_EXACT="$SPARSE_RUN_ROOT/full_exact_checkpoint"
export SPARSE_FULL_CROSS="$SPARSE_RUN_ROOT/full_cross_reference"
export SPARSE_FINAL_ANALYSIS="$SPARSE_RUN_ROOT/final_analysis"
export SPARSE_PROMOTIONS="$SPARSE_PILOT_ANALYSIS/sparse_streaming_promotions.json"
```

The capture basenames above are descriptive placeholders; their contents, not their names, establish identity. `SPARSE_CHECKPOINT` must contain `config.json`, `model.safetensors.index.json`, and every shard referenced by that index.

Before allocating GPU time, verify the repository and locked inputs:

```bash
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 1f01edf74ce754fea1615b26a97e4465a829671f HEAD

sha256sum "$SPARSE_CONFIG"
sha256sum "$SPARSE_CHECKPOINT/config.json"
sha256sum "$SPARSE_CHECKPOINT/model.safetensors.index.json"
sha256sum "$SPARSE_TREES"
sha256sum "$SPARSE_EXACT_CAPTURE"
sha256sum "$SPARSE_CROSS_CAPTURE"

PYTHONPATH=src python -m pytest -q tests/test_sparse_streaming.py tests/test_sparse_streaming_cuda.py tests/test_sparse_streaming_runner.py tests/test_sparse_streaming_analysis.py
```

The hash output must match the table above. Record the exact commands, Git commit, Python/package versions, GPU identity, input hashes, failures, and corrections in `EXECUTION_LEDGER_SPARSE_STREAMING_20260819.md`.

The analyzer's canonical input schema can be inspected without fabricating rows:

```bash
PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --write-schema /tmp/sparse_streaming_input_schema.json
```

## Required execution order

The order is intentionally one-way: fresh exact-checkpoint pilot, validation freeze, fresh held-out expansion, and only then broader-capture sensitivity. Do not run or inspect a Cartesian full sweep first.

### 1. Bounded fresh exact-checkpoint pilot

Pilot mode evaluates the predeclared bounded grid on validation occurrences and exactly one held-out test sanity invocation per pilot expert. The test sanity rows are evidence that the fixed implementation executes; they are not eligible to select configurations.

```bash
PYTHONPATH=src python scripts/run_sparse_streaming_study.py \
  --config "$SPARSE_CONFIG" \
  --captures "$SPARSE_EXACT_CAPTURE" \
  --checkpoint "$SPARSE_CHECKPOINT" \
  --trees "$SPARSE_TREES" \
  --output "$SPARSE_PILOT_EXACT" \
  --source exact_checkpoint \
  --mode pilot \
  --device cuda
```

Pilot mode rejects `--promotions`. It executes the frozen 32×32 tile pilot, exact-refresh-1 shortlist path, and configured unit/budget candidates. Keys named `conditional_followup_*` are deliberately not executed in this first bounded stage; they require a separately recorded validation-only continuation if the first oracle/proxy evidence justifies them. Coverage must reach at least 4 validation invocations from at least 2 validation requests in every audited layer; otherwise the pilot fails rather than weakening the gate.

### 2. Validate the pilot, run any predeclared validation-only continuation, and freeze promotions

First run validation without writing decisions:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --config "$SPARSE_CONFIG" \
  --validate-only >"$SPARSE_INITIAL_VALIDATION"
```

If and only if that validation-only artifact promotes the initial 32×32 tile and therefore activates the predeclared alternate-shape gate, execute the bounded continuation. This continuation reads validation rows only, evaluates no test row, and writes into its own output directory:

```bash
PYTHONPATH=src:scripts python scripts/run_sparse_streaming_tile_followup.py \
  --config "$SPARSE_CONFIG" \
  --captures "$SPARSE_EXACT_CAPTURE" \
  --checkpoint "$SPARSE_CHECKPOINT" \
  --trees "$SPARSE_TREES" \
  --parent-pilot "$SPARSE_PILOT_EXACT" \
  --trigger "$SPARSE_INITIAL_VALIDATION" \
  --output "$SPARSE_TILE_VALIDATION" \
  --device cuda
```

The continuation evaluates only the already-declared 16×64, 32×32, and 64×16 tile shapes with `activation_energy_x_weight`, `cartesian_hidden_input_blocks`, and `exact_dynamic_tile_marginal`. It cannot alter the frozen experiment configuration or the neuron/shortlist evidence. Its facts must say `test_rows_evaluated=0` and `test_rows_consulted_for_selection=false`, and must bind the parent pilot facts/table hashes and the initial validation-decision digest.

When the continuation ran, create the augmented validation-only record and then write the frozen decision and pilot report from the same two inputs:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_TILE_VALIDATION" \
  --config "$SPARSE_CONFIG" \
  --validate-only >"$SPARSE_AUGMENTED_VALIDATION"

MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_TILE_VALIDATION" \
  --config "$SPARSE_CONFIG" \
  --output "$SPARSE_PILOT_ANALYSIS"

sha256sum "$SPARSE_PROMOTIONS"
```

If the initial validation artifact does not activate the continuation gate, skip the continuation and use the original one-input report command instead:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --config "$SPARSE_CONFIG" \
  --output "$SPARSE_PILOT_ANALYSIS"

sha256sum "$SPARSE_PROMOTIONS"
```

`sparse_streaming_promotions.json` is the only legal promotion source for both full runs. It records the config/checkpoint/tree/capture provenance, `selection_split=validation`, `selection_capture_source=exact_checkpoint`, and `test_rows_consulted_for_selection=false`. The analyzer refuses to replace an existing frozen promotion file with different decisions. Its canonical SHA-256 is computed from sorted, indented, finite-only JSON with one terminal newline; this is the digest each full runner records as `validation_promotions_sha256`.

The recorded run exposed one interpretation defect only after that artifact was frozen: the alternative matched-recovery page calculation compared tile curves with an input-coordinate median below the zero-correction recovery of zero. The frozen JSON therefore retains a mechanically computed `0.75` page-reduction field for auditability, but the final report marks that field invalid, forces its interpreted pass flag to false, and makes no page-savings claim from it. The selected exact tile independently passes the separate ≥3-point validation recovery-gain gate. Do not regenerate, edit, or replace the frozen decision after held-out execution; preserving the defect and disclosing it is part of this reproduction contract.

Promotion is based only on fresh exact-checkpoint validation rows. Training requests may fit the already-defined future-proxy metric; validation chooses configurations; test evaluates the frozen choice. If a family misses its predeclared gate, its status is `stop` and full mode leaves that family disabled. A stop is a negative result, not permission to tune a threshold, shape, shortlist size, or selector on test. The broader capture never creates or changes promotions.

### 3. Expand promoted configurations to all 85 fresh test invocations

```bash
PYTHONPATH=src python scripts/run_sparse_streaming_study.py \
  --config "$SPARSE_CONFIG" \
  --captures "$SPARSE_EXACT_CAPTURE" \
  --checkpoint "$SPARSE_CHECKPOINT" \
  --trees "$SPARSE_TREES" \
  --output "$SPARSE_FULL_EXACT" \
  --source exact_checkpoint \
  --mode full \
  --promotions "$SPARSE_PROMOTIONS" \
  --device cuda
```

Full mode rejects a missing, non-analyzer, test-selected, or provenance-mismatched promotion artifact. It evaluates held-out test rows only. A completed fresh full run must report 85 expected and 85 observed unique invocations in `run_facts.json`.

### 4. Run the broader 142-invocation sensitivity check

Use the **same frozen promotion file**. Do not re-analyze the cross-reference validation split to select a different configuration.

```bash
PYTHONPATH=src python scripts/run_sparse_streaming_study.py \
  --config "$SPARSE_CONFIG" \
  --captures "$SPARSE_CROSS_CAPTURE" \
  --checkpoint "$SPARSE_CHECKPOINT" \
  --trees "$SPARSE_TREES" \
  --output "$SPARSE_FULL_CROSS" \
  --source cross_reference \
  --mode full \
  --promotions "$SPARSE_PROMOTIONS" \
  --device cuda
```

A completed broader run must report 142 expected and 142 observed unique invocations. This capture is secondary sensitivity evidence because its activations originate from the aligned BF16 capture source rather than a forward pass through the exact MXFP4 checkpoint. Its stride-16 positions are excluded from adjacent-token claims.

### 5. Merge completed evidence and produce the final report

The final analyzer accepts only completed run directories with exact expected/observed invocation equality and mutually consistent duplicate scientific rows. When pilot and full rows duplicate the same invocation and configuration, the values must agree; the full row is retained and the pilot copy is not double-weighted. It regenerates the promotion payload solely from the pilot's exact-checkpoint validation evidence, computes its canonical digest, and requires every full-run fact to carry that exact digest. Missing, malformed, differing, or merely self-consistent-but-wrong full-run promotion hashes fail analysis.

Because the recorded execution activated the validation-only tile-shape continuation, that directory is an input to both final analyzer calls. It contributes validation evidence only; its facts reject test rows.

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_TILE_VALIDATION" \
  --input "$SPARSE_FULL_EXACT" \
  --input "$SPARSE_FULL_CROSS" \
  --config "$SPARSE_CONFIG" \
  --validate-only

MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_TILE_VALIDATION" \
  --input "$SPARSE_FULL_EXACT" \
  --input "$SPARSE_FULL_CROSS" \
  --config "$SPARSE_CONFIG" \
  --output "$SPARSE_FINAL_ANALYSIS"
```

Primary claims must use the fresh exact-checkpoint held-out cohort. Cross-reference rows must remain visibly labelled as sensitivity evidence. H0 oracle, late H0 proxy, and deployable-H0 proxy regimes must remain separate in tables and prose.

The recorded execution completed all four input directories with expected/observed counts `59/59` pilot, `55/55` validation-only continuation, `85/85` fresh full, and `142/142` broader full. The four-input `--validate-only` command passed and wrote `final_validation_only.json` with SHA-256 `2b219512b7ff56c60933118df623754531694e1d60ef002549c43ceb685874b2`. It independently regenerated the unchanged frozen promotion digest `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f`.

## Resume and failure semantics

The runner checkpoints only after completing an entire `(split, layer, expert)` work unit. Each Parquet file, the internal support cache, and `run_facts.json` are written to a temporary file and atomically replaced, with facts written last. On resume, rows whose expert transaction is absent from the last committed facts journal are discarded and recorded in `resume_discarded_partial_rows`. A process failure therefore cannot mark or retain a partly evaluated expert as complete.

On an exception, the runner writes `completed=false` and appends the exception type and message to `run_facts.json`, then exits nonzero. To resume, correct the environmental failure and rerun the exact same phase command against the same output directory. The runner reloads existing Parquet state and skips `completed_work_units`. A directory whose `run_facts.json` already says `completed=true` is a no-op.

Resume is refused if the experiment-config hash, locked input hashes, or full-run promotion hash differ. Reproducibility requires rerunning with the exact same `SPARSE_PROMOTIONS` file; if inputs, mode, capture source, or promotions intentionally change, use a new empty phase-specific output directory and document why. The final analyzer independently regenerates and canonically hashes the validation decision rather than trusting the runner facts alone. Never point pilot and full mode, or fresh and cross-reference sources, at the same directory.

The analyzer rejects `completed=false`, missing required tables, observed/expected count disagreement, request-separation failure, inconsistent duplicate rows, invalid regime labels, or broken physical/storage arithmetic. Preserve failed directories until their failure and correction are recorded in the ledger.

## Physical and storage accounting conventions

No physical conclusion is inferred from action count.

- One routed expert contains `3 × 2048 × 512 = 3,145,728` weights.
- One physical page is 512 bytes. One physical correction bpw is therefore 393,216 bytes or 768 pages per expert.
- A requested expert budget is converted to pages by `floor(bpw × 3,145,728 / (8 × 512))`. Projection-only matched budgets use `2048 × 512` weights and 256 pages per projection bpw.
- `physical_bytes = physical_pages × 512`. `logical_actions` counts selected semantic actions, while `logical_bytes` records their unpadded payload. `page_amplification = physical_bytes / logical_bytes`.
- A complete neuron-major Q2→Q4 packet is 1,536 bytes: three pages. Progressive Q3 costs two pages and its nested Q4 completion adds one page. At one expert correction bpw, 256 of 512 complete unit packets fit.
- A paired gate/up 32×32 Q2→Q4 tile is exactly one 512-byte page. Complete tile-plus-down suffix coverage ends at 1,536 pages or 2.0 correction bpw.
- A shortlist size `K` means `K` input coordinates, not `K` plane-actions. Both refinement planes are aggregated by `sum_stage_energy`. The frozen identity layout stores both planes for four coordinates per projection page, and fetching a page exposes all co-resident feasible actions.
- Candidate prefetch traffic is charged even when a fetched action is not applied. For shortlist rows, `physical_pages` must equal `fetched_pages`. The primary containment metric compares constrained, exact, and diagonal recovery after repricing all three to the same fetched-page count.
- `selector_runtime_ms` covers one measured complete selector path/frontier construction for an invocation. `selector_bytes_read` is the reproducible unique tensor-footprint lower bound and is explicitly not a hardware traffic counter; repeated scans and Gram-column reads are cache-dependent and excluded. The same values are repeated on each budget snapshot row; never sum them across budgets or treat them as incremental per-budget work.
- Persistent selector metadata is separate from runtime bytes read. `selector_metadata_bpw = 8 × selector_metadata_bytes_per_expert / 3,145,728`.
- Storage uses the MXFP4 reference as denominator: `storage_multiplier = (4.25 reference bpw + suffix_storage_bpw + selector_metadata_bpw) / 4.25`. The resident Q2 parent value of 2.25 bpw is not the denominator. One additional complete 2.0-bpw suffix representation is `6.25 / 4.25 ≈ 1.470588×` before selector metadata.

Exact and dynamic methods labelled `h0_oracle_full_suffix`, `h0_oracle_full_target_residual`, or `h0_oracle_suffix_metadata_heavy` establish headroom only. A deployability claim requires a separately labelled deployable-H0 proxy that passes the same frozen gates while charging its pages, compute, and persistent metadata. Nothing in this protocol implies H4 prefetch performance.

## Artifact inventory

Every runner directory contains these scientific inputs to analysis:

- `neuron_major_frontier.parquet`
- `activation_shortlist_containment.parquet`
- `tile_streaming_frontier.parquet`
- `exact_action_labels.parquet`
- `support_stability.parquet`
- `activation_concentration.parquet`
- `run_facts.json`

`_support_cache.parquet` is an internal atomic-resume input used to rebuild overlap rows; preserve it with the raw run directory, but do not treat it as a reported frontier.

The validation-only continuation additionally contains `alternate_tile_shape_validation_frontier.parquet` and an `executed_code/` snapshot of the exact followup runner, base runner, and selector core used to produce it. The executed continuation's empty companion Parquets are historical null/object-typed zero-row files with an older subset of columns. The analyzer normalizes those empties without treating them as neuron, shortlist, label, stability, or concentration observations, and the current continuation writer derives typed empty schemas from the parent for future reruns. Do not rewrite the raw historical files. Preserve `pilot_validation_only.json` and `pilot_augmented_validation_only.json` beside the run directories as the pre-continuation trigger and post-continuation validation audit respectively; neither is a replacement for the canonical frozen `sparse_streaming_promotions.json`.

The final analysis directory contains:

- `sparse_streaming_promotions.json`
- `neuron_major_summary.csv`
- `neuron_major_by_layer_summary.csv`
- `activation_shortlist_summary.csv`
- `activation_shortlist_by_layer_summary.csv`
- `tile_streaming_summary.csv`
- `tile_streaming_by_layer_summary.csv`
- `support_stability_summary.csv`
- `exact_action_label_summary.csv`
- `activation_concentration_summary.csv`
- `promotion_validation_coverage.csv`
- `sparse_streaming_compute_storage_accounting.json`
- `SPARSE_STREAMING_ALLOCATOR_REPORT.md`
- `analysis_manifest.json`
- PNG and SVG copies of `01_isolated_vs_complete_qenergy`
- PNG and SVG copies of `02_activation_shortlist_containment`
- PNG and SVG copies of `03_tile_complete_expert_frontier`
- PNG and SVG copies of `04_logical_actions_vs_physical_pages`
- PNG and SVG copies of `05_true_adjacent_support_stability`
- PNG and SVG copies of `06_activation_concentration`

The three `*_by_layer_summary.csv` files expose n/p10/median/p90 plus physical/logical accounting for every layer, split, capture source, and selector identity. They are required for the broad-effect audit; pooled promotion statistics cannot by themselves establish a consistent cross-layer effect. The report must continue to distinguish isolated projection recovery, exact sequential complete-expert recovery, logical actions, physical/fetched pages, H0 oracles, late H0 proxies, and deployable approximations.

## Completion and hash verification

Inspect every run fact before accepting analysis:

```bash
python -m json.tool "$SPARSE_PILOT_EXACT/run_facts.json"
python -m json.tool "$SPARSE_TILE_VALIDATION/run_facts.json"
python -m json.tool "$SPARSE_FULL_EXACT/run_facts.json"
python -m json.tool "$SPARSE_FULL_CROSS/run_facts.json"
python -m json.tool "$SPARSE_FINAL_ANALYSIS/analysis_manifest.json"
cmp "$SPARSE_PROMOTIONS" "$SPARSE_FINAL_ANALYSIS/sparse_streaming_promotions.json"
```

Each runner fact must show `completed: true`, `request_separation_verified: true`, the expected locked hashes, and exact equality between expected and observed unique invocations. Both full runs must record the same canonical promotion SHA-256. That digest must equal the SHA-256 of `SPARSE_PROMOTIONS`, the `validation_promotions_sha256` in `analysis_manifest.json`, and the digest independently regenerated by the final analyzer from exact-checkpoint validation evidence.

Verify every final input and output hash recorded by `analysis_manifest.json` from the experiment root:

```bash
python - "$SPARSE_FINAL_ANALYSIS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "analysis_manifest.json").read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

failures = []
for name, expected in manifest["input_sha256"].items():
    path = Path(name)
    actual = sha256(path) if path.is_file() else "MISSING"
    if actual != expected:
        failures.append((str(path), expected, actual))
for name, expected in manifest["outputs"].items():
    path = root / name
    actual = sha256(path) if path.is_file() else "MISSING"
    if actual != expected:
        failures.append((str(path), expected, actual))
if failures:
    for path, expected, actual in failures:
        print(f"FAIL {path}: expected {expected}, got {actual}")
    raise SystemExit(1)
print(f"verified {len(manifest['input_sha256'])} inputs and {len(manifest['outputs'])} outputs")
PY
```

The manifest deliberately hashes the analysis outputs but not itself. The committed final analysis was regenerated locally from repository-relative inputs so the manifest contains portable paths. The final analyzer core SHA-256 is `64307732198d4664fb396e06a95bf6450132a575eb724646025336c4f99be9c6`; the analyzer-test SHA-256 is `d02f50935c006a0ef30973af8e890e34e24ace0568d0a66fccc627161a64e6ac`; and the CLI-wrapper SHA-256 is `1e20c6ba8e577fd74c3bee6f8b25f4721880c1cb0daf2872e5058d53095de353`. The analyzer pins Matplotlib's SVG hash salt to the run ID; an independent second-process run reproduced every final-analysis file byte-for-byte. The local integrated suite passed all 126 tests after the final regeneration.

The recorded manifest closes 29 scientific inputs and 25 outputs. Its verification passed with these identities:

| Artifact | SHA-256 |
|---|---|
| `final_analysis/analysis_manifest.json` | `fc2a4071b032f60d0653fea94f575839e88064f91ccfdd1dc02d610b5264aa2a` |
| `final_analysis/SPARSE_STREAMING_ALLOCATOR_REPORT.md` | `87f24e7680863911268464f3582c8a973aaede0d1ed42cece35fd7d6eea08f46` |
| `final_analysis/sparse_streaming_promotions.json` | `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f` |

The final analysis directory contains 26 files and 2,888,524 bytes with inventory digest `ac74730f8735e37302cc7d28574424268b58d633448258dba47c973b7f63017e`; the whole result package contains 95 files and 59,349,224 bytes with inventory digest `a4d2b58852184ba81a052641bf8f0cf65d50354e1ed5be691c81d01137a45d49`. The copied pre-rerender package contained 95 files and 59,359,634 bytes with inventory digest `0687462b818268e7df23172f67ce794a66e80a88c93fee0aee82a18ba4cfe7c1`. The four copied raw execution directories match their execution-host counterparts byte-for-byte. The only package-level size change after transfer is the intentional relative-path report/plot/manifest rerender; no raw Parquet, run fact, continuation source snapshot, or log changed. Commit the raw run artifacts, final analysis artifacts, report, plots, configuration, tests, code, this protocol, and the completed execution ledger together. Record the final Git commit and pull request in the publication handoff.

## Recorded RunPod shutdown

The study pod `41rk786odszmk9` was stopped after artifact transfer and verification at `2026-08-19 18:03:11 UTC`. Verification returned `desiredStatus=EXITED`, `runtimeStatus=stopped`, and reason `stopped_by_user`; persistent disk remains, while GPU billing has stopped. The first management-connector attempt returned HTTP 401 without changing state, so the already-authenticated local CLI performed the stop. No authentication material is part of this record.

For a future reproduction, stop and verify the exact allocated pod using its own identifier:

```bash
runpodctl pod stop <study-pod-id>
runpodctl pod get <study-pod-id>
runpodctl pod list
```

The final `pod get` evidence must show a stopped runtime state (or an explicitly terminated/deleted state if the owner chose deletion), and the list must show no accidentally running replacement study pod. Record the shutdown command, UTC timestamp, returned state, and any correction in the execution ledger.
