# Set-utility distillation reproducibility protocol

This is the reviewer-facing execution contract for the bounded
coherent-unit set-utility study built on PR #9. It explains how to reproduce
the train-only fit, exact-checkpoint validation, frozen stop/promote decision,
and—only if validation succeeds—a newly sealed holdout evaluation.

This document does not report cohort scientific results. Single-invocation
benchmark values are implementation preflights and are labelled as such. The
canonical scientific report will be generated as
`validation_analysis/SET_UTILITY_DISTILLATION_REPORT.md`.

## Question and controlled scope

PR #7 established a strong coherent-neuron suffix oracle; PR #9 showed that a
conventional low-rank linear response predictor does not preserve final
complete-expert utility. This study asks three narrower questions:

1. Can direct set-utility supervision predict a useful 256-unit candidate set
   from late-H0 Q2-side features?
2. Can support templates or structured high-rank/low-bit gate/up residual
   metadata outperform global low rank at matched selector storage/compute?
3. Can a bounded coherent/G/D hybrid with direct-GD moves and exact local
   consolidation improve the coherent fixed-greedy path?

The embedded Q2→Q3→Q4 code, selected trees, checkpoint, layer/expert sample,
and request split remain locked. The study does not train H4, modify the
resident Q2 model, optimize kernels, evaluate all layers, or make routing,
logit, perplexity, token-agreement, or downstream-quality claims.

## Immutable controls

Run from:

```text
experiments/adaptive_expert_precision_oracle
```

The branch is `agent/mxfp4-set-utility-distillation`, based exactly on PR #9
commit `56fe7764ec28c92947c83cb3d7dbd16e5630311b`.

| Guard | Expected value |
|---|---|
| Checkpoint revision | `7eceff3a9f7e6f916c824d197266d86676bce695` |
| Checkpoint config SHA-256 | `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` |
| Checkpoint index SHA-256 | `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb` |
| Selected trees SHA-256 | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| Exact-checkpoint capture SHA-256 | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| Cross-reference capture SHA-256 | `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add` |
| Study configuration SHA-256 | `ef5308f7f684dca31a98fb81ecc8c49feded78ce4f201617fa847b4e9b5d09f1` |
| Fit/validation runner SHA-256 | `90b78f665f40ded98d6b17b28bc36ef163e53aee14233cb92acd6aa5cd61eb73` |
| Runner tests SHA-256 | `39b5935bb9415cd7281109c10e5c6b400b6baa081d7cc33f41f31a589c674fe2` |
| Direct predictor core SHA-256 | `82b1b85b01b00658121613776fc8ddcb61352e3b13e7704835c656a6d6ae204c` |
| Direct predictor tests SHA-256 | `10dac8fcfe2faaa2f8b5a424fc875429a2316fc5c826d1ea9d33e76300c5546e` |
| Analyzer core SHA-256 | `66c2a123e078997f1e7d349e161fcce18a3ac6600b0e36294330b74573a79ac4` |
| Analyzer wrapper SHA-256 | `7dff3d084089cf0819703d7d715baa0fa0567734cf2fb1434b7be6b5616410ae` |
| Analyzer tests SHA-256 | `8e277f72ecd8e3acde49b1acef85878a20f9a35a5e6c0730842745c7c10f2247` |
| Analyzer end-to-end tests SHA-256 | `c7c53c6a0ca20e2090c801322e932c25951bb7dd5e110bbc66e353dba4bb4349` |

This table is the authoritative final local pre-launch freeze. Recompute every
digest on the staged execution checkout immediately before launch and require
an exact match.

The runner verifies these identities and the full checkpoint tensor structure
before admitting scientific rows. All recorded SHA-256 values must be 64
lowercase hexadecimal characters. A malformed hash is an immediate contract
failure.

The capture files are monolithic archives and physically contain train,
validation, and test rows. Reproduction claims must use precise semantics:

- fit admits `train` rows only;
- validation admits `train` and `validation` rows only;
- split labels and request-ID metadata for all three splits are inspected only
  for within-capture separation and cross-capture compatibility auditing;
- no test scientific tensor is admitted, evaluated for a metric, used, or
  consulted for tuning;
- request IDs are disjoint across train, validation, and test;
- same-split request-ID sharing across capture sources is generally allowed,
  but any cross-source split mismatch fails closed; this locked pair freezes
  and observes zero shared IDs;
- the runner has no test mode.

Do not claim that the monolithic file was never opened or that its test bytes
were absent.

## Environment and paths

Use a CUDA PyTorch environment with the repository dependencies, NumPy,
pandas plus a Parquet engine, safetensors, compressed-tensors, and Matplotlib.
The recorded execution uses an RTX 3090; one 24 GiB card is sufficient.

Set task-scoped absolute paths. The names below are placeholders; hashes, not
basenames, establish input identity.

```bash
cd /absolute/path/to/LLM_compression_study/experiments/adaptive_expert_precision_oracle

export SETUTIL_ROOT="$PWD"
export SETUTIL_CONFIG="$SETUTIL_ROOT/configs/qwen36_mxfp4_set_utility_distillation.json"
export SETUTIL_CHECKPOINT="/absolute/path/to/qwen36_mxfp4_candidate"
export SETUTIL_TREES="/absolute/path/to/locked/selected_trees.json"
export SETUTIL_EXACT_CAPTURE="/absolute/path/to/qwen36_exact_confirm_captures.npz"
export SETUTIL_CROSS_CAPTURE="/absolute/path/to/captures_seed20260817.npz"
export SETUTIL_PYTHON="/absolute/path/to/study-python"
export SETUTIL_RUN_ROOT="/absolute/path/to/qwen36_mxfp4_set_utility_distillation_20260820_v1"
export SETUTIL_FIT="$SETUTIL_RUN_ROOT/fit"
export SETUTIL_VALIDATION="$SETUTIL_RUN_ROOT/validation_exact_checkpoint"
export SETUTIL_ANALYSIS="$SETUTIL_RUN_ROOT/validation_analysis"
```

Verify provenance before allocating GPU time:

```bash
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 56fe7764ec28c92947c83cb3d7dbd16e5630311b HEAD

sha256sum "$SETUTIL_CONFIG"
sha256sum "$SETUTIL_CHECKPOINT/config.json"
sha256sum "$SETUTIL_CHECKPOINT/model.safetensors.index.json"
sha256sum "$SETUTIL_TREES"
sha256sum "$SETUTIL_EXACT_CAPTURE"
sha256sum "$SETUTIL_CROSS_CAPTURE"

python -m json.tool "$SETUTIL_CONFIG"
python -m py_compile \
  src/oracle_study/unit_set_teacher.py \
  src/oracle_study/set_utility_oracles.py \
  src/oracle_study/set_utility_selector.py \
  src/oracle_study/direct_set_predictor.py \
  src/oracle_study/set_utility_analysis.py \
  scripts/run_set_utility_distillation.py \
  scripts/benchmark_set_utility_hybrid.py \
  scripts/analyze_set_utility_distillation.py

PYTHONPATH=src:scripts "$SETUTIL_PYTHON" -m pytest -q \
  tests/test_unit_set_teacher.py \
  tests/test_set_utility_oracles.py \
  tests/test_set_utility_selector.py \
  tests/test_direct_set_predictor.py \
  tests/test_set_utility_runner.py \
  tests/test_set_utility_analysis.py \
  tests/test_set_utility_analysis_e2e.py

PYTHONPATH=src:scripts "$SETUTIL_PYTHON" -m pytest -q
```

Final local verification against the frozen hashes above recorded 37 runner
tests passing, 41 analyzer tests passing, and 282 tests passing with 1 skipped
in 79.662 s in the full integrated suite. The full suite emitted one NumExpr
warning and no failures. Record execution-host package/GPU facts and staged
hash parity in `EXECUTION_LEDGER_SET_UTILITY_20260820.md` before the fit begins.

## Method reference

### Exact coherent-unit set teacher

For coherent unit corrections `c_i`, let `K_ij = c_i^T G c_j` under the
locked qenergy metric. The teacher uses full-target fixed-coefficient marginal
selection. If `v` indicates remaining target corrections, the gain for
selecting unit `j` is:

```text
2 * (K @ v)[j] - K[j, j]
```

After selecting `j`, set `v_j=0` and update correlations. Coefficients are
fixed; there is no least-squares refit. The correct label is
`exact_marginal_fixed_greedy`, not OMP. The teacher path is not a certificate
of the globally optimal subset.

The implementation avoids storing a deployment Gram. During training and
oracle evaluation it reconstructs the activation-dependent 512×512 coherent
unit Gram from a down-side metric Gram plus `h2/h4`. The down-side Gram and
encoded A/B/C metadata are computed once per expert/proxy and reused across
activations.

### Hybrid action oracle

Each unit can be in `00`, `G`, `D`, or `GD`. The action vocabulary includes
nested one-/two-page transitions and the direct coherent three-page `00→GD`
move. A 2048×2048 primitive transition Gram is built once on GPU and reused
for the 0.5/0.75/1.0-bpw grid. The small fixed-greedy correlation loop is
deterministic on CPU. Each budget takes the best direct-qenergy prefix, so a
negative terminal tail is not forced into the snapshot.

The bounded local search supports exact one-/two-/three-unit add, drop, and
swap moves from four named seeds. It is monotone in exact qenergy and charges
seed construction plus incremental local-search runtime separately.

### Structured predictor families

- Support templates cluster train-only exact masks with utility weighting.
  Per-expert banks are evaluated as charged top-1, top-2 union, and bounded
  repair candidates. Routed exact masks and synthetic all-activation pairings
  retain distinct cohort labels.
- Block-PQ residual synopses split gate/up rows into 32-weight blocks, use
  separate gate/up layer-shared FP16 codebooks, retain the resident E8M0 block
  scales, and store expert/unit/block four-bit codes. One and two additive
  stages are measured. Codebook reads, scale reads, lookup additions, and
  scale multiplications are charged. PQ response features are FP16-rounded and
  then widened for the FP32 reference kernels in both fit and validation, so
  the computation matches their declared two-byte runtime payload.
- High-rank/low-bit controls invert PR #9's low-rank/high-precision trade:
  inherited rank-64 FP8/INT8 baselines are compared with matched-storage
  int2/ternary ranks through 256.
- The direct contextual unit predictor uses actual late-H0 Q2-side features,
  static A/B/C metadata, optional synopsis outputs, and shared per-unit
  context. Active train-only objectives include listwise and boundary ranking;
  hard-set variants add the exact apply-top-192 Gram set loss. The joint
  nested head also receives teacher-top-192 coverage supervision. The
  candidate-256 Gram set loss has zero weight, and the runner supplies no
  exclusion-regret tensor even though the reusable core supports one. Model/normalizer
  parameters are serialized at their charged precision and fixed epochs
  prevent validation-based early stopping.

## Physical accounting and candidate semantics

One expert has `3*2048*512 = 3,145,728` weights. A physical correction bpw is
393,216 bytes or 768 512-byte pages. One coherent unit packet is three pages.

The primary interface fetches 256 units at 1.0 physical bpw and applies 192 at
0.75 logical bpw. Page amplification is 4/3. Candidate overfetch is not free.
For the joint nested predictor, the hard-forward fetched mask is exactly the
apply-head top 192 union the candidate-head top 64 after excluding the apply
support. Its coverage target is the teacher top 192.

The runner emits four policies that must remain separate:

| Policy | Information used after fetch | Role |
|---|---|---|
| `predicted_direct_no_rerank` | Predictor order only | Deployable late-H0 control; promotion eligible |
| `exact_independent_abc_within_fetched_candidates` | FP16-rounded activation/g2/u2/h2, exact Q2→Q4 gate/up residual responses, and decoded-FP16 A/B/C scalar scores | Primary realistic H0 rerank; promotion eligible with total compute charged |
| `contained_fetched_target_exact_h0` | Full fetched-candidate correction geometry | Expensive H0 systems oracle; never promotion primary |
| `full_target_restricted_teacher_oracle` | Complete target residual including unfetched teacher information | Nondeployable teacher-information control |

The contained interaction oracle is not a cheap rerank: a direct 256-way
candidate Gram is approximately 134 million MACs. The full-target restricted
control is also not an upper bound on the globally optimal subset. Neither may
be used to claim deployability.

The independent-ABC correctness reference evaluates all 512 unit responses
for convenience, while its deployable accounting charges response work only
for fetched candidates. Each evaluated Q4 gate/up response charges 1,024 bytes
of resident Q2 parent codes, 128 bytes of E8M0 scales unless the selector has
already read them, and 128 block-scale multiplications. Independent A/B/C
scoring charges six MACs per scored unit; a standalone all-512 score therefore
costs 3,072 MACs. The full-target teacher control charges all 512 responses,
their parent-code and conditional scale payload, the all-512 correction
workspace, the Euclidean Gram, and the rank-4 proxy projection/Gram arithmetic
and metadata.

Contained and full-target interaction accounting is a precise lower bound
only because it excludes resident Q2-down parent-code/scale payload, Q2-down
tree/E8M0 decode and block-scale application, fixed-greedy initial correlation,
and per-selection vector updates. Future-proxy metadata and both Euclidean and
proxy Gram arithmetic are charged, not omitted.

Fetched packet bytes are exact logical physical payload. Selector
`bytes_read` is a logical unique-payload-plus-LUT-read model, not measured DRAM
or PCIe traffic. MACs are analytical logical counts; additions and row-scale
multiplications are separate, including 1,024 high-rank synthesis scale
multiplications per invocation. Already-fetched packets remain charged for
predicted-direct and template rows and are neither erased nor double-counted.
Exact rerank components and lower-bound strings are preserved in
`candidate_rerank_accounting.csv`.

Support-template choice and repair use exact validation path utility and have
no deployable classifier in this study; those rows are nonpromotable. Sampled-
expert high-rank/low-bit controls are transductive diagnostics and are also
nonpromotable.

## Required execution order

### 1. Optional single-invocation feasibility preflight

This step tests runtime, memory, and direct-qenergy parity only. It does not
select a scientific configuration.

```bash
PYTHONPATH=src:scripts "$SETUTIL_PYTHON" \
  scripts/benchmark_set_utility_hybrid.py \
  --config "$SETUTIL_CONFIG" \
  --checkpoint "$SETUTIL_CHECKPOINT" \
  --trees "$SETUTIL_TREES" \
  --exact-captures "$SETUTIL_EXACT_CAPTURE" \
  --layer 4 --expert 17 --device cuda:0

PYTHONPATH=src:scripts "$SETUTIL_PYTHON" \
  scripts/benchmark_set_utility_hybrid.py \
  --config "$SETUTIL_CONFIG" \
  --checkpoint "$SETUTIL_CHECKPOINT" \
  --trees "$SETUTIL_TREES" \
  --exact-captures "$SETUTIL_EXACT_CAPTURE" \
  --layer 4 --expert 17 --device cuda:0 --full-hybrid
```

The recorded real-size preflight used a 2048×2048, 16 MiB primitive Gram and
finished the primitive path in 2.651 s and 2.256 s on repeated runs. The full
five-family/three-budget call took 23.312 s on that one invocation. See the
execution ledger for component timings. These are not cohort results.

### Quarantined attempts before the replacement fit

Never resume any of the four preserved invalid execution bundles:

- `fit_invalid_pre_nested_coverage_fix` is incomplete, with layers 0, 4, and
  20 only, no layer 39, no canonical manifest, `completed=false`, and empty
  failure/history lists. It predates the corrected nested coverage contract.
- `fit_invalid_pre_byte_accounting_fix` stopped pre-layer-0, before any layer
  committed. It has no shard, sidecar, or manifest and records
  `completed_layers=[]`, empty failure/history lists, a clean
  zero-shared/zero-incompatible request audit, and corrected nested semantics.
  It predates the final accounting/runtime and transaction audit.
- `fit_invalid_pre_selector_score_mac_fix` had valid initial provenance,
  request-separation, runtime, and nested-candidate facts, but stopped
  pre-layer-0 when the standalone 512-unit A/B/C score path was found to omit
  its 3,072 analytical MACs. It has no shard, sidecar, or manifest,
  `completed_layers=[]`, and an empty failure list.
- `fit_invalid_pre_candidate_overfetch_fix` completed all four fit layers,
  and the paired `validation_invalid_pre_candidate_overfetch_fix` completed
  exactly 69 exact-checkpoint validation invocations. Both recorded empty
  failure/history lists. The strict analyzer failed closed before any report
  or promotion decision because support-template candidate rows omitted a
  finite `candidate_overfetch` value. Preserve these raw artifacts only as
  correction evidence: no recovery metric from them was summarized, selected,
  or used. The replacement emits finite
  `candidate_overfetch = candidate_units / applied_units`, with a real-path
  regression covering all four rerank semantics.

The replacement must use a fresh absent canonical `fit/` path. Quarantine
artifacts are correction history only and must never enter validation.

### 2. Fit train-only metadata and models

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts \
  "$SETUTIL_PYTHON" scripts/run_set_utility_distillation.py \
  --phase fit \
  --config "$SETUTIL_CONFIG" \
  --checkpoint "$SETUTIL_CHECKPOINT" \
  --trees "$SETUTIL_TREES" \
  --output "$SETUTIL_FIT" \
  --exact-captures "$SETUTIL_EXACT_CAPTURE" \
  --cross-captures "$SETUTIL_CROSS_CAPTURE" \
  --device cuda:0
```

Fit must report `fit_split=train`, no admitted/used test scientific rows, no
validation or test scientific rows used, complete request separation, and a
clean checkpoint audit. It must also report the frozen zero-shared/zero-
incompatible cross-capture audit and the exact nested hard-forward semantics.
It writes one independently hashed NPZ plus JSON sidecar per layer and then a
canonical manifest. A partial layer is never entered into the manifest.
Completion requires exactly layers `[0,4,20,39]`, empty failures, four
verified shard and sidecar hashes, and identical device/GPU/Torch-CUDA/NumPy/
Python/host provenance in facts, every sidecar, and the manifest.

### 3. Evaluate exact-checkpoint validation only

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts \
  "$SETUTIL_PYTHON" scripts/run_set_utility_distillation.py \
  --phase validate \
  --config "$SETUTIL_CONFIG" \
  --checkpoint "$SETUTIL_CHECKPOINT" \
  --trees "$SETUTIL_TREES" \
  --output "$SETUTIL_VALIDATION" \
  --exact-captures "$SETUTIL_EXACT_CAPTURE" \
  --fit-dir "$SETUTIL_FIT" \
  --device cuda:0
```

Do not pass the cross-reference capture to validation; the CLI rejects it.
Validation evaluates every admitted routed occurrence for every locked sampled
expert, bounded by the declared per-expert maximum. Static down Gram and A/B/C
metadata are reused per expert. The same shared hybrid path supplies all three
budgets. A completed run requires exact expected/observed invocation equality
and exact expected/observed hybrid-row equality: 69 unique validation
invocations across the 12 locked layer/expert cells.

### 4. Freeze the validation decision

```bash
MPLBACKEND=Agg PYTHONPATH=src:scripts \
  "$SETUTIL_PYTHON" scripts/analyze_set_utility_distillation.py \
  --config "$SETUTIL_CONFIG" \
  --fit-dir "$SETUTIL_FIT" \
  --validation-dir "$SETUTIL_VALIDATION" \
  --output "$SETUTIL_ANALYSIS"

sha256sum "$SETUTIL_ANALYSIS/set_utility_promotions.json"
sha256sum "$SETUTIL_ANALYSIS/analysis_manifest.json"
```

The analyzer independently recomputes physical/storage/compute arithmetic and
creates a canonical `set_utility_promotions.json`. It rejects incomplete
facts, test consultation, source/config/hash mismatch, malformed selector
regimes, missing coverage, or accounting inconsistency.

Frozen gates are:

- hybrid local path: ≥0.005 absolute one-bpw median gain over the exact-set
  coherent fixed-greedy teacher;
- top-1 template: ≥0.95 median teacher-path gain retention and ≤256 charged
  candidates;
- top-2 template union: ≥0.98 median and ≥0.95 p10 retention, with ≤256
  charged candidates;
- predicted-direct or independent-ABC candidate path: ≥0.94 median recovery,
  ≥0.88 p10 recovery, ≥0.97 p10 exact-set-gain retention, ≤0.35 metadata bpw,
  total selector plus rerank compute strictly below 1,572,864 MACs, and total
  storage below 5×.

If every deployable family fails, stop expansion and publish the negative
result. Do not adjust any threshold or configuration after seeing validation.

### 5. Newly sealed holdout only after promotion

The existing PR #7/#9 test rows are no longer a fresh holdout because they
informed the research direction. This bounded runner has no test phase.

If and only if a deployable family passes, first commit and hash the frozen
promotion, then define a new capture protocol with fresh request IDs and a
sealed hash. Add a separate evaluation entry point that accepts only that
promotion digest. Review and freeze that protocol before capture or test
execution. Do not infer or invent a holdout command from the validation CLI.

If no family passes, this step is intentionally absent.

## Resume and failure semantics

Fit commits one complete layer shard at a time. Its facts file is written
first with `completed=false`, updated after each durable layer, and finalized
only after every shard and sidecar rehashes into the manifest. A rerun binds
all locked hashes and linked code hashes; a provenance mismatch fails rather
than resumes. For this execution, matching provenance explicitly requires the
same device string, GPU names, Torch version/CUDA runtime, NumPy version,
Python version, and host. Cross-hardware or cross-host resume is prohibited.

Validation commits only complete `(validation, layer, expert)` work units.
On resume, rows not named in the last durable `completed_work_units` journal
are discarded before execution. Each Parquet and facts update is atomic, with
facts written last. A completed output is immutable and refuses another run.

On any exception, retain the failed facts and logs. Record the failure,
scientific impact, correction, regression test, and replacement command in the
ledger. Use a new output directory when a scientific input or configuration
changes. Only a restart with the exact runtime provenance above may reuse an
incomplete canonical directory according to the transaction journal. Empty
checkpoint Parquets must still carry the declared columns for all four raw
artifact schemas.

## Required artifacts and closure checks

Fit must contain:

- `fit_facts.json`;
- four `set_utility_fit_layer_<layer>.npz` shards;
- four matching JSON sidecars;
- `set_utility_fit_manifest.json`.

Validation must contain:

- `run_facts.json`;
- `hybrid_oracle_frontier.parquet`;
- `support_template_frontier.parquet`;
- `pq_high_rank_selector_frontier.parquet`;
- `candidate_set_rerank_frontier.parquet`;
- `selector_compute_storage_accounting.json`.

Analysis must contain:

- `SET_UTILITY_DISTILLATION_REPORT.md`;
- `set_utility_promotions.json`;
- `analysis_manifest.json`;
- exactly 14 hybrid, template, synopsis, candidate, coverage, provenance,
  promotion-gate, and accounting CSVs, including
  `candidate_rerank_accounting.csv`;
- four PNG/SVG plot pairs;
- analyzer-recomputed selector accounting.

Before publication:

1. rerun focused and integrated tests from the exact staged source;
2. regenerate analysis twice and verify deterministic file hashes;
3. hash every fit, validation, analysis, failure-log, and documentation file;
4. transfer the complete task-owned result directory locally and compare a
   sorted package digest;
5. ensure no artifact contains an access endpoint, key path, credential, or
   unrelated pod identity;
6. verify the GitHub base/head commits and PR contents;
7. only then stop the task-owned compute pod and verify its stopped state.

Record all final digests, the evidence commit, PR URL, and shutdown evidence in
`EXECUTION_LEDGER_SET_UTILITY_20260820.md`.
