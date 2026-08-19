# Execution Ledger: Activation-Sparse Selective Weight Streaming

Date started: 2026-08-19 (Europe/London)
Repository: `ArianSFP/LLM_compression_study`
Branch: `agent/mxfp4-sparse-streaming`
Starting commit: `1f01edf74ce754fea1615b26a97e4465a829671f` (PR #6 head)
Seed: `20260819`

## Scientific hypothesis

The useful MXFP4 suffix is sparse over `(SwiGLU unit, input block)`, rather than over input coordinates alone. A neuron-major or page-aligned two-dimensional action space may preserve the nonlinear gate/up interaction while exposing a selector that can be localized from resident Q2 activations.

## Locked controls

- Exact checkpoint repository: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`.
- Exact revision: `7eceff3a9f7e6f916c824d197266d86676bce695`.
- Checkpoint config SHA-256: `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`.
- Tensor-index SHA-256: `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
- Selected-tree SHA-256: `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.
- The embedded Q2→Q3→Q4 codec, leaf/scales, selected trees, and exact request splits remain unchanged.
- Exact capture split: 6 train / 3 validation / 3 test requests.
- Broader capture split: 53 train / 12 validation / 16 test requests.
- Training requests may fit metadata or selectors; validation alone promotes configurations; test is evaluated once.
- No H4 predictor is trained in the initial study. No router, logit, token-quality, perplexity, or task-quality claim is made.

## Controlled PR #6 baseline

PR #6 found a median 3.07x gate/up action-count gap between diagonal and exact fixed-coefficient marginal greedy at 90% isolated recovery, but rank≤64 Gram metadata failed. At one physical correction bpw, its predeclared difficult-layer exact-training-mask separate-plane result was p10 0.835098 and median 0.892560 (n=72). Across all 85 exact-checkpoint held-out invocations, the controlled frontiers were: current diagonal layout p10/median 0.8250/0.8915; exact-training separate planes 0.8385/0.8973; paired packets 0.8553/0.9200; H0 hybrid representation 0.8724/0.9237. These are exact sequential complete-expert qenergy recoveries, not end-to-end accuracy measurements.

## Action representations

### Full neuron packet

One Q2→Q4 hidden-unit packet contains one 2,048-weight gate row, one 2,048-weight up row, and one 2,048-weight down column. At two refinement bits per weight this is 1,536 bytes, exactly three 512-byte pages. Since one expert has 3,145,728 weights, one correction bpw supplies 768 pages and exactly 256 full-unit packets.

### Progressive coherent neuron packet

- Stage 1 pays one gate/up Q2→Q3 page plus one down page containing both refinement planes: 1,024 physical bytes for 768 logical bytes.
- Stage 2 is nested after stage 1 and pays the gate/up Q3→Q4 page: 512 incremental physical bytes for the remaining 768 logical bytes already partly prefetched in the down page.
- The full Q4 unit therefore remains 1,536 physical/logical bytes.

The unit output at each coherent level is computed exactly as
`d_i^(level) * SiLU(g_i^(level)) * u_i^(level)`; summing mixed unit levels is an exact mixed-precision expert.

### Gate/up tile page

Every tested tile shape has 1,024 weights per matrix: 16×64, 32×32, or 64×16. Both gate/up matrices and both refinement planes therefore occupy exactly one 512-byte page. Down refinement remains one 512-byte two-plane page per hidden unit. Selecting all 1,024 gate/up tiles plus all 512 down pages reconstructs the Q4 endpoint.

## Primary-source guidance

- Prox (arXiv:2607.27591) supports the exact SwiGLU intermediate product as a channel-salience proxy, but studies channel omission rather than Q2→Q4 refinement.
- WINA (arXiv:2505.19427) supports activation×weight scoring under an orthogonality condition that does not hold for this locked codec; it is treated only as a validation-tested heuristic.
- TEAL (ICLR 2025, arXiv:2408.14690) motivates magnitude baselines and physical sparse layouts, but resident-Q2 sparsification is a separate study because it changes the base computation.
- R-Sparse, DejaVu, and ShadowLLM motivate sparse-plus-residual or learned masks, but do not establish that the PR #6 action Gram is sparse. H4-style prediction remains out of scope until an action representation succeeds at H0.

## Bounded execution order

1. Audit activation-derived containment of PR #6 exact gate/up paths.
2. Evaluate exact full/progressive neuron oracles and resident-Q2 proxy rankings.
3. Promote only validation-Pareto neuron paths.
4. Evaluate page-perfect gate/up tiles with neuron-major down pages, first on the bounded pilot.
5. Evaluate shortlist plus exact residual rescoring and charge every prefetched candidate page.
6. Only if a strong oracle–proxy gap remains, test sparse-neighbour Grams or full-rank low-bit signatures.
7. Expand qualifying paths to the 85 exact-checkpoint and then 142 broader held-out invocations.

Promotion is frozen from validation only. The neuron oracle must beat the PR #6 H0 hybrid at both median 0.9237 and p10 0.8724 at one physical bpw (stretch target 0.95/0.90). An activation shortlist must retain at least 0.95 median and 0.90 p10 of exact-over-diagonal gain with no more than 1.25x paid candidate-page overfetch. The 32x32 tile is tested first; 16x64 or 64x16 is promoted only after at least a 0.03 recovery gain or 25% page reduction at matched validation recovery.

The broader cross-reference capture is secondary sensitivity evidence because its activations originate from the locked BF16-source capture revision rather than the exact MXFP4 checkpoint revision. Adjacent-token stability is evaluated only where true consecutive routed occurrences exist (43 pairs in the fresh capture); stride-16 cross-reference rows are never labelled adjacent.

## RunPod

RunPod `41rk786odszmk9` (`broken_crimson_buzzard`) is the active execution host. The replacement endpoint supplied by the user is `213.192.2.117:40134`. Hardware inspection reports an NVIDIA RTX 3090 with 24,576 MiB VRAM, driver 595.71.05, 1.0 TiB host RAM, and 256 logical CPUs. The user explicitly authorized keeping this pod running throughout the experiment and requested that it be stopped only after experimentation has ended. Shutdown and post-stop verification are therefore terminal ledger events, not pilot cleanup steps.

## Commands, failures, and corrections

Commands and their outputs are appended chronologically below.

### Branch and upstream anchor

```bash
git rev-parse HEAD
git branch --show-current
git ls-remote origin refs/heads/agent/mxfp4-interaction-aware-allocator
```

Both local HEAD and the published PR #6 branch resolved to `1f01edf74ce754fea1615b26a97e4465a829671f`; the new branch is `agent/mxfp4-sparse-streaming`.

### Evidence and raw-input audit

```bash
find /home/arian/LLM_compression_study -type f \( -name '*.npz' -o -name '*.safetensors' \) -printf '%p %s\n'
runpodctl get pod
```

No raw capture NPZ or checkpoint shard is present locally. The configured RunPod account has no running pod; all listed 3090/4090/5090/RTX PRO pods are `EXITED`. The previous paid pod `9s3kwxjvs0vdh6` is also `EXITED`, so no GPU billing was active during local development.

The PR #6 artifact audit established that exact action IDs and raw Grams were not serialized. The new study must restore the immutable captures with SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` (fresh exact checkpoint) and `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add` (broader cross-reference), then recompute labels. The older `selected_bit_actions.parquet` contains PR #4 diagonal paths and is not substituted for PR #6 exact support.

### Locked-tree and baseline hashes

```bash
sha256sum \
  results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/dense/selected_trees.json \
  configs/qwen36_mxfp4_sparse_streaming.json \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/success_gates.json \
  results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/paired_hybrid_frontier.parquet
```

At this point the hashes were respectively `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`, `aec3345e499311f9b69fd9e4a0f21c44782e0ca36978906826a43a131056dacb`, `3908a82df017399ef7810598354b2f05645e452959a300772abbdc09cc0990f9`, and `299849fbcea968752804e8bd0db88e77b0970b3d9f59ca3e860e08eaf778459b`. The `aec3345e...` configuration value is a historical pre-integration hash: it was superseded first by the `680dbd30...` integration checkpoint and then by the final `326edd4c...` pre-pilot freeze recorded below.

### Deterministic selector validation

```bash
PYTHONPATH=src pytest -q tests/test_sparse_streaming.py tests/test_sparse_streaming_cuda.py
PYTHONPATH=src pytest -q
python -m json.tool configs/qwen36_mxfp4_sparse_streaming.json >/dev/null
python -m py_compile \
  src/oracle_study/sparse_streaming.py \
  src/oracle_study/sparse_streaming_cuda.py
```

The focused suite passed `28` tests and the integrated suite passed `76` tests in 10.15 seconds. The only warning was the pre-existing pandas/numexpr version warning. JSON validation and bytecode compilation passed.

### Failure and correction: local sandbox wrapper

The first read-only `git status` and the first direct patch helper both failed before execution with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. Commands were rerun with the managed escalation path; edits continued to use `apply_patch`. This was an environment-wrapper failure, not a code/test failure, and no partial file mutation occurred.

### Correctness correction: progressive greedy tie-breaking

The first Torch progressive selector broke equal gain/page ties only by lower action ID, while the NumPy reference used `(gain/page, raw gain, -incremental pages, -action ID)`. The Torch reduction was corrected to the same lexicographic rule and a constructed cross-stage negative-gain tie fixture was added. The corrected focused CUDA suite passed `20` tests.

### Replacement endpoint and immutable input restoration

The originally supplied endpoint `213.192.2.120:40025` refused the SSH connection. No command was executed and no remote state was changed there. The user supplied the replacement endpoint `213.192.2.117:40134`, which was accepted with the same Ed25519 identity.

The remote study root is `/root/sparse_streaming_study`, the isolated Python environment is `/root/allocator_env`, and the checkpoint is `/root/qwen36_mxfp4_candidate`. The installed execution stack includes PyTorch `2.8.0+cu128`, Transformers `5.15.0`, compressed-tensors `0.18.0`, pandas `3.0.5`, PyArrow `25.0.1`, and safetensors `0.8.0`.

The downloaded checkpoint metadata matched the locked controls exactly:

```text
52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a  config.json
842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb  model.safetensors.index.json
```

The selected-tree file copied to `/root/sparse_streaming_study/locked/selected_trees.json` matched SHA-256 `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.

### Broader capture restoration

The broader capture was regenerated from the immutable aligned raw corpus under `/workspace` and written to `/root/sparse_streaming_study/work/captures_seed20260817.npz`. It contains 5,036 rows with the locked 53 train / 12 validation / 16 test request split. Its SHA-256 is the expected `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`; its regenerated manifest SHA-256 is `94c35a99413e3e3a5d6db32b03c85803c2757d322bb0e6d48a7425567e912520`.

The first extraction attempt prefixed the command with `/usr/bin/time`, which is absent from the pod image, and failed with exit status 127 before creating output. The command was rerun without that optional timing wrapper and completed successfully. This was an environment-only correction; the final NPZ is accepted solely by its locked content hash.

### Fresh exact-checkpoint recapture (in progress)

The following CPU-only forward capture was started so GPU memory remains available for the later allocator run:

```bash
cd /root/sparse_streaming_study
/root/allocator_env/bin/python scripts/capture_exact_mxfp4_confirm.py \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --output work/qwen36_exact_confirm_captures.npz \
  --layers 0 4 20 39 \
  --max-length 64 \
  --revision 7eceff3a9f7e6f916c824d197266d86676bce695 \
  --cpu-only
```

Expected acceptance identity: 564 rows, 280/128/156 train/validation/test rows, and SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`. No allocator result will be accepted until those conditions pass.

A remote full test run was briefly started while this weight-loading capture was active. It competed for host resources and was interrupted; the orphaned pytest PID was terminated and the capture process was verified to be the only remaining study process. Remote tests will be rerun after capture completion. This correction did not change capture inputs or source files.

### Final local integration before execution

```bash
PYTHONPATH=src:scripts pytest -q
python -m py_compile \
  src/oracle_study/sparse_streaming.py \
  src/oracle_study/sparse_streaming_cuda.py \
  src/oracle_study/sparse_streaming_analysis.py \
  scripts/run_sparse_streaming_study.py \
  scripts/analyze_sparse_streaming.py
python -m json.tool configs/qwen36_mxfp4_sparse_streaming.json >/dev/null
```

The integrated suite passed `98` tests in 13.17 seconds; only the pre-existing pandas/numexpr version warning remained. Bytecode and JSON validation passed. The then-current integration config SHA-256 was `680dbd3091a00abd6952e87bf02ee77e668800d47724dc90b19961621c299c25`; the independent audit below occurred before execution and superseded this value with the final protocol freeze.

### Pre-pilot independent audit and corrections

Two independent read-only reviews completed before any allocator result was generated. Neuron-major decomposition, progressive 2+1-page nesting, nonlinear tile state updates, Q4 endpoints, and basic shortlist page closure were cleared. The audit also found four issues that were corrected before pilot launch:

1. The matched-page exact shortlist comparator had post-filtered an unrestricted exact path. Skipped actions therefore remained embedded in later recorded marginals. The comparator now independently reruns `exact_marginal_fixed_greedy` with the physical page budget active, updates the residual only for applied actions, and retains its best cumulative-utility prefix. A deterministic PSD fixture proves the old filtered path can retain the wrong nested action and obtain lower realized recovery.
2. The static diagonal control is now replayed against its actual changing residual and trimmed to its best realized-utility prefix at the same physical charge.
3. The broader sensitivity experts are no longer rediscovered from all request splits. The exact PR #6 cross-reference cohort is serialized in the configuration by layer and stratum: layer 0 `{71,207,238}`, layer 4 `{156,176,134}`, layer 20 `{129,225,72}`, and layer 39 `{180,155,235}` for cold/median/hot respectively.
4. Full-run promotion and resume checks now bind the analyzer generator, frozen pass flags, per-layer validation coverage, run mode, capture source, and exact promotion SHA-256. A completed or partial directory cannot be silently reused under another phase.

Selector byte fields were also standardized: `selector_runtime_ms` remains measured wall time, while `selector_bytes_read` is explicitly a unique tensor-footprint lower bound rather than a cache-dependent hardware traffic counter. Plot K is labelled as input coordinates, and unrestricted gate/up tiles plus down columns are no longer called a neuron-conditioned hybrid.

After these corrections:

```bash
PYTHONPATH=src:scripts pytest -q
python -m py_compile scripts/run_sparse_streaming_study.py src/oracle_study/sparse_streaming_analysis.py
python -m json.tool configs/qwen36_mxfp4_sparse_streaming.json >/dev/null
```

passed `100` tests in 13.81 seconds after adding explicit fixed-cross-cohort and resume-provenance regressions, with the same one pre-existing numexpr warning. This test-count checkpoint is superseded by the verification sequence below. The final pre-pilot configuration SHA-256 is `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326`.

### Pre-pilot selector one-page fast path

The residual-consistent page-budget selector still spent most of its time rebuilding Python page sets for every eligible action. The locked coordinate-packet layout maps each action to exactly one 512-byte page, so `base_gram_candidate_greedy` now uses a specialized vectorized paid-page feasibility mask for that case. Nested Q3→Q4 eligibility, fixed-coefficient residual updates, exact marginal scores, paid-page accounting, and best-prefix semantics are unchanged. The general set-based implementation remains the fallback for actions that span more than one page. A deterministic regression compares the fast path with direct fixed-vector greedy at every page budget.

The implementation and its primary regression fixture have these SHA-256 identities:

| File | SHA-256 |
|---|---|
| `src/oracle_study/sparse_streaming.py` | `eb7b08e4aa2bf490c0edebf5e04a017a6701c56410ab35c61de8cc898f868aeb` |
| `tests/test_sparse_streaming.py` | `2ddea2f2a5970cddbbe6a17bd4be72d470d9c54636bbbc1dce5f83202ccb6cd5` |

A local synthetic real-sized selector benchmark used 2,048 coordinates, hence 4,096 plane-actions, and a rank-32 positive-semidefinite Gram. Representative before/after wall times were:

| Page budget | General path (s) | One-page fast path (s) |
|---:|---:|---:|
| 64 | 0.440 | 0.0405 |
| 128 | 0.705 | 0.116 |
| 192 | 0.962 | 0.101 |
| 256 | 1.210 | 0.133 |
| 384 | 1.498 | 0.192 |
| 512 | 1.515 | 0.241 |

A separate six-budget aggregate timing was 6.29 seconds before and 0.824 seconds after, approximately `7.6×` faster. These are selector implementation diagnostics on a synthetic fixture, not allocator recovery results, 3090 kernel timings, or deployment-performance claims. After this optimization, the then-current local integrated suite passed `103` tests; that chronological checkpoint was followed by the portability correction below. The configuration was not changed and remains pinned at `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326`.

### Verification failures and portability correction

The first focused local test command after the fast-path change omitted the required `PYTHONPATH=src:scripts` prefix and failed during test collection because the local modules were not importable. It produced no scientific artifact. The command was immediately corrected to use the same import path as the documented suite, after which the focused tests and the `103`-test integrated checkpoint passed.

The next remote full-suite attempt collected 102 tests: 101 passed and one report-generation test failed because the pod did not have the optional `tabulate` package used by `pandas.DataFrame.to_markdown`. This was an analyzer portability failure, not a selector, codec, capture, split, or scientific-result failure, and no pilot result was generated or accepted from that attempt.

The correction implemented locally and queued for the remote execution path replaces `DataFrame.to_markdown` with an internal deterministic Markdown table renderer. Its regression deliberately makes `to_markdown` raise `ModuleNotFoundError`, proving that report generation no longer depends on `tabulate`. The corrected analyzer/test identities are:

| File | SHA-256 |
|---|---|
| `src/oracle_study/sparse_streaming_analysis.py` | `15441c7887babe7c26ebe37e8f5dd3dc5fe343022de69e5975be0bd0788a02c7` |
| `tests/test_sparse_streaming_analysis.py` | `465aa35f4754a6f61a53bb2f2a8c01a15ba9c8b2a18a9b808f1c09b7318bae9c` |

The remote targeted analyzer suite then passed all 11 tests in 26.01 seconds. A fresh local integrated rerun passed `104` tests with the single pre-existing pandas/numexpr version warning in 13.31 seconds. A complete remote integrated rerun remains a required pre-pilot verification step. No allocator pilot result is claimed in this ledger entry.

### Final pre-pilot scope and promotion-provenance audit

A final scope review found that pooled summaries alone were insufficient evidence for the predeclared requirement that useful sparsity be broad rather than driven by one layer. The analyzer now emits `neuron_major_by_layer_summary.csv`, `activation_shortlist_by_layer_summary.csv`, and `tile_streaming_by_layer_summary.csv`. Each stratifies n/p10/median/p90 and physical/logical accounting by layer, split, capture source, and selector identity; pooled validation statistics still control promotion, while the layer tables support the separate broad-effect interpretation.

The same review hardened final-analysis provenance. The analyzer regenerates the frozen promotion payload only from pilot exact-checkpoint validation rows, serializes it canonically, and requires every full-run `validation_promotions_sha256` to match that digest. It rejects a missing or malformed digest, differing digests between fresh and cross-reference full runs, and a shared full-run digest that does not equal the regenerated validation decision. The digest is also written to `analysis_manifest.json`.

The final analyzer and analyzer-test SHA-256 identities are:

| File | SHA-256 |
|---|---|
| `src/oracle_study/sparse_streaming_analysis.py` | `120d620041fe0f5ce60532611d879b6f7205ab02327f8f6f52d2c5b6e6eb909d` |
| `tests/test_sparse_streaming_analysis.py` | `5a6a15800c5348121be3f09f483f8f2c5334be4c59d26667c9d3c396263838a9` |

The final local integrated command `PYTHONPATH=src:scripts pytest -q` passed `108` tests with the single pre-existing pandas/numexpr warning in 15.07 seconds. The configuration remains unchanged at SHA-256 `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326`. The preceding `15441c...`/`465aa35...` analyzer checkpoint remains in the chronology as the tabulate-removal state and is superseded by the hashes above. The required complete remote integrated rerun is still outstanding; no pilot result is claimed.
