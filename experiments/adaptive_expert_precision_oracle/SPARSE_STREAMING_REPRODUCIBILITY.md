# Sparse-streaming allocator reproducibility protocol

This document is the reviewer-facing execution contract for the activation-dependent sparse-streaming allocator study. It describes how to reproduce the bounded pilot, freeze validation-only promotion decisions, evaluate the promoted configurations on the fresh exact-checkpoint cohort, and run the broader cross-reference sensitivity check.

It does **not** itself report experimental results. Execution began on 2026-08-19 on RunPod `41rk786odszmk9` (RTX 3090). The checkpoint and regenerated broader capture already match their locked hashes; the fresh recapture and allocator stages remain in progress at this protocol update. Results become reviewable only after completed run directories, the final analysis manifest, and the execution ledger have been committed.

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

### 2. Validate the pilot and freeze promotions

First run validation without writing decisions:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --config "$SPARSE_CONFIG" \
  --validate-only
```

Then write the frozen decision and pilot report:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --config "$SPARSE_CONFIG" \
  --output "$SPARSE_PILOT_ANALYSIS"

sha256sum "$SPARSE_PROMOTIONS"
```

`sparse_streaming_promotions.json` is the only legal promotion source for both full runs. It records the config/checkpoint/tree/capture provenance, `selection_split=validation`, `selection_capture_source=exact_checkpoint`, and `test_rows_consulted_for_selection=false`. The analyzer refuses to replace an existing frozen promotion file with different decisions. Its canonical SHA-256 is computed from sorted, indented, finite-only JSON with one terminal newline; this is the digest each full runner records as `validation_promotions_sha256`.

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

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_FULL_EXACT" \
  --input "$SPARSE_FULL_CROSS" \
  --config "$SPARSE_CONFIG" \
  --validate-only

MPLBACKEND=Agg PYTHONPATH=src python scripts/analyze_sparse_streaming.py \
  --input "$SPARSE_PILOT_EXACT" \
  --input "$SPARSE_FULL_EXACT" \
  --input "$SPARSE_FULL_CROSS" \
  --config "$SPARSE_CONFIG" \
  --output "$SPARSE_FINAL_ANALYSIS"
```

Primary claims must use the fresh exact-checkpoint held-out cohort. Cross-reference rows must remain visibly labelled as sensitivity evidence. H0 oracle, late H0 proxy, and deployable-H0 proxy regimes must remain separate in tables and prose.

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

The manifest deliberately hashes the analysis outputs but not itself. Commit the raw run artifacts, final analysis artifacts, report, plots, configuration, tests, code, this protocol, and the completed execution ledger together. Record the final Git commit and manifest SHA-256 in the ledger.

## RunPod shutdown requirement

The active study pod is `41rk786odszmk9`. Keep it running while justified staged experimentation remains. At the end of the study—or immediately after a terminal failure—stop that exact pod and verify its runtime state:

```bash
export SPARSE_POD_ID="the-provided-runpod-id"
runpodctl pod stop "$SPARSE_POD_ID"
runpodctl pod get "$SPARSE_POD_ID"
runpodctl pod list
```

The final `pod get` evidence must show a stopped runtime state (or an explicitly terminated/deleted state if the owner chose deletion), and the list must show no accidentally running replacement study pod. Record the shutdown command, UTC timestamp, returned state, and any correction in the execution ledger. Do not claim the study is complete while its paid pod is still running.
