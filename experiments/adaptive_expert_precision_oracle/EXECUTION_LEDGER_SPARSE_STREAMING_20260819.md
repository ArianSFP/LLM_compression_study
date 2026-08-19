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

RunPod `41rk786odszmk9` (`broken_crimson_buzzard`) was the execution host. Hardware inspection reported an NVIDIA RTX 3090 with 24,576 MiB VRAM, driver 595.71.05, 1.0 TiB host RAM, and 256 logical CPUs. The user explicitly authorized keeping this pod running throughout the experiment and requested that it be stopped only after experimentation ended. The verified shutdown is recorded as the terminal execution event below. Ephemeral SSH endpoints and local private-key paths are intentionally omitted from the publishable ledger.

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

The originally supplied RunPod endpoint refused the SSH connection. No command was executed and no remote state was changed there. The user supplied a replacement endpoint, which was accepted with the same Ed25519 identity.

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

### Fresh exact-checkpoint recapture started

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

The final local integrated command `PYTHONPATH=src:scripts pytest -q` passed `108` tests with the single pre-existing pandas/numexpr warning in 15.07 seconds. The configuration remained unchanged at SHA-256 `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326`. The preceding `15441c...`/`465aa35...` analyzer checkpoint remains in the chronology as the tabulate-removal state and is superseded by the hashes above. At this chronological checkpoint the required complete remote integrated rerun had not yet occurred; its later successful result is recorded below.

### Fresh exact-checkpoint recapture completed

The CPU-only command recorded above completed after 11,963.609 seconds. The accepted capture has 564 rows: 280 train, 128 validation, and 156 test rows from the locked 6/3/3 request split. Its identities are:

| Artifact | SHA-256 |
|---|---|
| `work/qwen36_exact_confirm_captures.npz` | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| `work/qwen36_exact_confirm_captures.manifest.json` | `a2ccad7962e878fa68831e12ace81cd4266c2a7f35fc82587e1efa3abe19c685` |

The NPZ exactly matched the predeclared locked hash, so no recapture correction or request substitution was made. The broader capture remained pinned at `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`.

### Remote pre-pilot verification and source freeze

After capture completion, the complete remote command

```bash
cd /root/sparse_streaming_study
PYTHONPATH=src:scripts /root/allocator_env/bin/python -m pytest -q
```

passed all 108 tests in 25.22 seconds. The pilot was launched from branch commit `37e7bfcd168545d6a44a3e3df45b6eb27d858a3c` with the following executed identities:

| Component | SHA-256 |
|---|---|
| Frozen configuration | `326edd4c5ae771ff1c29f4c3cac8eace0d2eeb9e498575f8ef355be409eef326` |
| Pilot runner | `cba776cf5f3dbdd12c7691bea0080877be249940c435fee2638ad03f1939492a` |
| Analyzer CLI | `1e20c6ba8e577fd74c3bee6f8b25f4721880c1cb0daf2872e5058d53095de353` |
| Selector core | `eb7b08e4aa2bf490c0edebf5e04a017a6701c56410ab35c61de8cc898f868aeb` |
| CUDA selector core | `440a492d0af557f2004a97d33578f041cb0a267eead2ff5631be4d726f82819d` |
| Analyzer core | `120d620041fe0f5ce60532611d879b6f7205ab02327f8f6f52d2c5b6e6eb909d` |

### Bounded exact-checkpoint pilot completed

The executed command was:

```bash
cd /root/sparse_streaming_study
PYTHONPATH=src:scripts /root/allocator_env/bin/python scripts/run_sparse_streaming_study.py \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --source exact_checkpoint \
  --mode pilot \
  --device cuda
```

`run_facts.json` records start `2026-08-19 14:21:05.569949 UTC`, completion `2026-08-19 15:35:38.982792 UTC`, and 4,473.413 seconds elapsed. All 59 expected invocations were observed: 55 validation invocations plus the four predeclared single-invocation test execution checks. The layer/expert work units were validation and sanity-test evaluations for `(0,62)`, `(4,154)`, `(20,191)`, and `(39,108)`. Request separation passed and the failures list is empty. The test sanity rows were not eligible for promotion.

The completed artifact counts and identities are:

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `_support_cache.parquet` | internal | `0e7f4eed8aacc9a8e60486409933ab325f14294c2005b7ec461c655b35ecb34d` |
| `activation_concentration.parquet` | 472 | `f86d68f7efab2c488cfcc339f16537dad4a17d81d65772271174b22fba5459d8` |
| `activation_shortlist_containment.parquet` | 9,912 | `84ded1873cb5d92cc212203cc428f12ed23a60c4e3b32f7123c1986edd15348e` |
| `exact_action_labels.parquet` | 483,328 | `c8871ed9c63cd35ab7ce3a0ab5089a4115c0abbf3ae8daf5984346244e9a31f6` |
| `neuron_major_frontier.parquet` | 3,304 | `cd7626e999fd2618ce7135aa8f21942e778b4c37e165b32ad67f2b13cf80eda2` |
| `support_stability.parquet` | 2,794 | `239eb85241c73bfc64a318c7f635d99cc0506bcfdcd2d1bf4d85895b9906ae39` |
| `tile_streaming_frontier.parquet` | 1,416 | `847dd26bb1f34214ad1db59835250d2028f1ebfc26609eb78849661af0a1daa6` |
| `run_facts.json` | — | `6877fd832a8d2d99c8b7613e3e5f7baba5a52eb10a9411734c779c365b01c998` |

### Initial validation-only decision and conditional trigger

The pilot was first analyzed without writing a report or frozen promotion file:

```bash
cd /root/sparse_streaming_study
MPLBACKEND=Agg PYTHONPATH=src /root/allocator_env/bin/python scripts/analyze_sparse_streaming.py \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --validate-only \
  > results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_validation_only.json
```

The file SHA-256 is `e4f5956b252dd297d12ec906774f2a8ec6fd76b31b140cb1d10623b944519b24`; its canonical validation-decision digest is `183ee7e353b65e9817c1eefae1872f00ce6b8541173220498479acc173dfdbad`. It records `selection_split=validation`, `selection_capture_source=exact_checkpoint`, and `test_rows_consulted_for_selection=false`. Coverage was 7/16/16/16 validation invocations for layers 0/4/20/39, with at least two validation requests in every layer.

These are validation-only gate facts, not held-out result claims:

- The direct three-page `independent_unit_correction_norm` neuron packet passed the predeclared one-bpw gate on 55 validation invocations (median `0.958289`, p10 `0.915015`).
- The activation shortlist family had status `stop`; no shortlist configuration was promoted.
- The exact-refresh-1 32×32 gate/up tile plus neuron-down path activated the predeclared alternate-shape continuation (median `0.964969`, p10 `0.939433`, 768 median pages, and `0.041269` median recovery-point gain over the locked PR #6 H0 hybrid). The frozen analyzer also mechanically recorded `0.75` page reduction against the one-bpw input-coordinate comparator, but that comparator's median recovery was `-14.4413`, below the zero-correction recovery of `0`. The apparent reduction is therefore invalid evidence for page savings and is not counted as a success.

Because the initial 32×32 tile passed its declared trigger, alternate tile shapes were investigated only on validation. No sparse-neighbour Gram, low-bit signature, learned mask, H4 predictor, or resident-Q2 sparsification continuation was activated.

### Validation-only alternate tile-shape continuation

The continuation was implemented as a separate, provenance-bound runner; the locked JSON configuration was not edited. The executed command was:

```bash
cd /root/sparse_streaming_study
PYTHONPATH=src:scripts /root/allocator_env/bin/python scripts/run_sparse_streaming_tile_followup.py \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --parent-pilot results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --trigger results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_validation_only.json \
  --output results/qwen36_mxfp4_sparse_streaming_20260819_v1/validation_tile_shapes_exact_checkpoint \
  --device cuda
```

The run evaluated the predeclared 16×64, 32×32, and 64×16 shapes with `activation_energy_x_weight`, `cartesian_hidden_input_blocks`, and `exact_dynamic_tile_marginal`. It started at `2026-08-19 17:00:37.863607 UTC`, completed at `2026-08-19 17:07:10.060679 UTC`, and took 392.197 seconds. All 55 expected validation invocations were observed; 2,310 tile rows were written; `test_rows_evaluated=0`; `test_rows_consulted_for_selection=false`; request separation passed; and the failure list is empty.

The continuation binds parent pilot facts SHA-256 `6877fd832a8d2d99c8b7613e3e5f7baba5a52eb10a9411734c779c365b01c998`, parent tile SHA-256 `847dd26bb1f34214ad1db59835250d2028f1ebfc26609eb78849661af0a1daa6`, and trigger decision digest `183ee7e353b65e9817c1eefae1872f00ce6b8541173220498479acc173dfdbad`. Its executed-code snapshot records base runner `614260de159cb2715910caedc6d3794ba1d7eec0b2ebe89b2fada916f4efea01`, followup runner `17bdd5ad1377533b4720aeac30c74d30f515c9b89ed7c02650117fcfa2fb551c`, and selector core `8bdfaebdf639d29401c280bfd509c80d84e6c3e440eb72ee21bf3b3fd6fd4b68`.

| Continuation artifact | SHA-256 |
|---|---|
| `alternate_tile_shape_validation_frontier.parquet` | `898f93a93571c321ae47ab4eda42e39c445de58c0ab600920b4f0eaf387b7543` |
| `tile_streaming_frontier.parquet` | `898f93a93571c321ae47ab4eda42e39c445de58c0ab600920b4f0eaf387b7543` |
| `run_facts.json` | `76e65b64ab72d39f70fead94dc8a30f74cce97555a0b3a344e7e77c65d14c73d` |

### Failure and correction: object-typed empty companion Parquets

The first augmented validate-only analyzer call after the continuation failed before writing a decision with:

```text
TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule 'safe'
```

The traceback reached `np.allclose`/`np.isclose` in `sparse_streaming_analysis.py` while checking storage accounting. The continuation's historical zero-row companion Parquets had Arrow null/object schemas and an older subset of columns; after concatenation, those empties contaminated otherwise numeric metadata-byte/bpw, suffix-bpw, and storage-multiplier columns. This was an analysis/schema portability failure after scientific rows were complete, not a selector or recovery failure.

The analyzer now explicitly applies `pd.to_numeric(..., errors="raise").to_numpy(dtype=np.float64)` to those four accounting fields before `np.allclose`. A regression constructs object-typed empty companion Parquets and proves the merged validation succeeds. The continuation writer was also hardened for future reproduction to derive typed zero-row tables from the immutable parent Parquets instead of manufacturing object-typed empties. The executed recovery frontier was not rerun or changed; the exact executed followup source remains preserved under `executed_code/` with SHA-256 `17bdd5...`, while the schema-hardened current followup runner hashes to `a16648a1eb0cf9decc0c458110390023d1192693a0cfe8e8f62302e7a0f3ba63`.

After the correction, the local integrated suite passed 114 tests and the remote focused continuation/analyzer suite passed 21 tests. The corrected analyzer and regression identities are:

| File | SHA-256 |
|---|---|
| `src/oracle_study/sparse_streaming_analysis.py` | `46ee27e9da29eee6abe790741011d9e73cc13fb6d5363a423f9629f5b883e2ac` |
| `tests/test_sparse_streaming_analysis.py` | `a60d857419b7075a8844adf892bc2996f74c0c567d54ee2c2de16ca990ac2e5c` |
| `scripts/run_sparse_streaming_tile_followup.py` | `a16648a1eb0cf9decc0c458110390023d1192693a0cfe8e8f62302e7a0f3ba63` |

Re-analysis then succeeded from the unchanged pilot and continuation Parquets; no held-out recovery was inspected to make this correction.

### Frozen validation promotion artifact

The successful augmented validation and report-generation commands were:

```bash
cd /root/sparse_streaming_study
MPLBACKEND=Agg PYTHONPATH=src /root/allocator_env/bin/python scripts/analyze_sparse_streaming.py \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/validation_tile_shapes_exact_checkpoint \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --validate-only \
  > results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_augmented_validation_only.json

MPLBACKEND=Agg PYTHONPATH=src /root/allocator_env/bin/python scripts/analyze_sparse_streaming.py \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/validation_tile_shapes_exact_checkpoint \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --output results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_analysis
```

`pilot_augmented_validation_only.json` hashes to `2b219512b7ff56c60933118df623754531694e1d60ef002549c43ceb685874b2`. The canonical frozen `pilot_analysis/sparse_streaming_promotions.json` hashes to `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f`. It preserves `selection_split=validation`, `selection_capture_source=exact_checkpoint`, and `test_rows_consulted_for_selection=false`.

The frozen validation-only family decisions are:

- promote the direct neuron `independent_unit_correction_norm` path recorded above;
- stop activation shortlist expansion;
- promote `exact_dynamic_tile_marginal_refresh_1` with the validation-selected 64×16 tile (n=55, median `0.966025`, p10 `0.935183`, 768 median physical pages, and `0.042325` median recovery-point gain over the locked PR #6 H0 hybrid). Its mechanically recorded `0.75` page-reduction diagnostic has the same invalid negative comparator described above and supplies no support for promotion.

These are selection facts on validation, not primary test-cohort conclusions. The frozen file is the only promotion input used by both full runs.

The exact dynamic tile-shape validation rows were close: 16×64 p10/median/p90 was `0.937504/0.965927/0.984044`, 32×32 was `0.939433/0.964969/0.982813`, and 64×16 was `0.935183/0.966025/0.981914`. The selected 64×16 median exceeded 16×64 by only `0.000098` absolute recovery (`0.0098` percentage point). This is a scientific near-tie resolved by the deterministic frozen median ordering, not evidence that 64×16 is materially superior.

The static activation-times-weight medians for 16×64, 32×32, and 64×16 were `0.83521/0.82589/0.82812`; the corresponding Cartesian proxy medians were `0.83290/0.80727/0.83358`. The neuron selection was also an H0 oracle. No deployable-H0 selector passed, and the shortlist family stopped. Thus the frozen promotions establish oracle headroom only; they do not establish a deployable streaming policy or justify H4 training.

The negative comparator exposed a defect in the preregistered alternative page-reduction gate after the `d4572236...` promotion bytes had been frozen and used to start held-out execution. Reopening or rewriting the selection artifact at that point would violate the one-way validation-to-test protocol. The bytes are retained as an audit record, while the final report adds an interpretation guard that forces the page-reduction evidence to invalid whenever the comparator is at or below zero-correction recovery. The selected exact tile independently passes the unchanged ≥3-point recovery-gain gate, so no held-out observation is needed to justify retaining its frozen identity.

### Full 85-invocation exact-checkpoint expansion completed

Only the frozen promoted neuron and tile identities were expanded; the stopped shortlist family remained disabled:

```bash
cd /root/sparse_streaming_study
PYTHONPATH=src:scripts /root/allocator_env/bin/python scripts/run_sparse_streaming_study.py \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --captures work/qwen36_exact_confirm_captures.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_exact_checkpoint \
  --source exact_checkpoint \
  --mode full \
  --promotions results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_analysis/sparse_streaming_promotions.json \
  --device cuda
```

`run_facts.json` records start `2026-08-19 17:13:34.611661 UTC`, completion `2026-08-19 17:26:11.453404 UTC`, and 756.842 seconds elapsed. Expected and observed unique invocations are both 85; request separation passed; the failures list is empty; and `validation_promotions_sha256` is the frozen `d4572236...` digest. Artifact row counts are 680 activation-concentration, 2,210 neuron-frontier, and 1,530 tile-frontier rows; stopped-family shortlist/label/stability tables contain zero rows by construction.

| Full exact artifact | SHA-256 |
|---|---|
| `activation_concentration.parquet` | `a5b1609e1f39ba01fe5abe84437cd802aa93f8d0d82a89aaaa0c92817f851b77` |
| `neuron_major_frontier.parquet` | `6e3d95cb09f14617ec79ba7c77167d36a06071366b8e97b1ce63c6b3c1b0465f` |
| `tile_streaming_frontier.parquet` | `2b13c6b729ab0c774e21cb844631a680627229f354f5688b078f33e8a497f25e` |
| `run_facts.json` | `9bd93219a7f032b6458ca9ddb66ec24e078b920cf293b4f619859a62df0d38ec` |

At this checkpoint completion/provenance was recorded without inspecting held-out recovery. Held-out statistics were read only after the frozen cross run and four-input validation completed, as recorded below.

### Broader 142-invocation sensitivity run completed

The same frozen validation promotion artifact was supplied unchanged to the broader capture:

```bash
cd /root/sparse_streaming_study
PYTHONPATH=src:scripts /root/allocator_env/bin/python scripts/run_sparse_streaming_study.py \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --captures work/captures_seed20260817.npz \
  --checkpoint /root/qwen36_mxfp4_candidate \
  --trees locked/selected_trees.json \
  --output results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_cross_reference \
  --source cross_reference \
  --mode full \
  --promotions results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_analysis/sparse_streaming_promotions.json \
  --device cuda
```

`run_facts.json` records start `2026-08-19 17:26:50.015145 UTC`, completion `2026-08-19 17:47:12.183192 UTC`, and 1,222.168 seconds elapsed. Expected and observed unique invocations are both 142 across the 12 fixed `(layer, expert)` transactions; request separation passed; the failures list is empty; and `validation_promotions_sha256` remains `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f`. Row counts are 1,136 activation-concentration, 3,692 neuron-frontier, and 2,556 tile-frontier rows. The stopped shortlist, exact-label, and stability outputs contain zero scientific rows.

| Full cross-reference artifact | SHA-256 |
|---|---|
| `_support_cache.parquet` | `f120a1583efb9ccfe78db1f59d064550e1dcf9711c00132677acc9a2f078db0e` |
| `activation_concentration.parquet` | `a03e095908c7ee93dab5308c681b92fd35ac0038b9eb44544bea5f832fdbe171` |
| `activation_shortlist_containment.parquet` | `ee386d61793ebedd02c8754e39ba6904775ba53f065c9413da5276cb8dde32e0` |
| `exact_action_labels.parquet` | `455a73d5fa7823ce2278e00379a6570dc9c4c712ed70cccbb23b054eb03bf41b` |
| `neuron_major_frontier.parquet` | `2ad7dbf80f0ed08feaa006b98e2d674dc4669fcc676b85744a55be64bdf23c2e` |
| `support_stability.parquet` | `f0c2dabc468bfce74e4fae0d43248fae86e00f121d9594675678558ae56b13f3` |
| `tile_streaming_frontier.parquet` | `85f92a61857212b666c63760b6c636348bb660d5d979e631779cc0a58e7a5533` |
| `run_facts.json` | `0ed98177f158e17962b14815eaa9a9a3ed36a2239a13ec3cdd1bb42ef01be25a` |

This is secondary sensitivity evidence because the capture comes from the locked aligned BF16-source capture, not a forward pass through the exact MXFP4 checkpoint. It did not select or change any configuration.

### Four-input validation and final analysis

The completed pilot, validation-only tile continuation, fresh full run, and cross-reference full run were jointly validated before report interpretation:

```bash
MPLBACKEND=Agg PYTHONPATH=src /root/allocator_env/bin/python scripts/analyze_sparse_streaming.py \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/pilot_exact_checkpoint \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/validation_tile_shapes_exact_checkpoint \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_exact_checkpoint \
  --input results/qwen36_mxfp4_sparse_streaming_20260819_v1/full_cross_reference \
  --config configs/qwen36_mxfp4_sparse_streaming.json \
  --validate-only \
  > results/qwen36_mxfp4_sparse_streaming_20260819_v1/final_validation_only.json
```

Validation passed. `final_validation_only.json` hashes to `2b219512b7ff56c60933118df623754531694e1d60ef002549c43ceb685874b2`, and independent regeneration still produced frozen promotion SHA-256 `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f`. The remote final analyzer used analyzer core `ecd4a65ea02d95ab3228214d45488cd75823c1df7f6579b5bf793f5a724302de` to create an interim complete report package. The raw package was then copied before the execution host was stopped.

For the committed artifact, final analysis was intentionally regenerated locally from repository-relative input paths. This removes host-specific absolute paths from the manifest and incorporates the final success-gate and invalid-comparator disclosures without changing any raw evidence. The final executed analysis identities are:

| Component | SHA-256 |
|---|---|
| Analyzer core | `64307732198d4664fb396e06a95bf6450132a575eb724646025336c4f99be9c6` |
| Analyzer tests | `d02f50935c006a0ef30973af8e890e34e24ace0568d0a66fccc627161a64e6ac` |
| Analyzer CLI wrapper | `1e20c6ba8e577fd74c3bee6f8b25f4721880c1cb0daf2872e5058d53095de353` |

After regeneration, the local integrated suite passed all 126 tests. The analyzer closed 29 hashed scientific inputs and 25 hashed outputs, and direct manifest verification reported no mismatch. Final identities are:

| Final artifact | SHA-256 |
|---|---|
| `final_analysis/analysis_manifest.json` | `fc2a4071b032f60d0653fea94f575839e88064f91ccfdd1dc02d610b5264aa2a` |
| `final_analysis/SPARSE_STREAMING_ALLOCATOR_REPORT.md` | `87f24e7680863911268464f3582c8a973aaede0d1ed42cece35fd7d6eea08f46` |
| `final_analysis/sparse_streaming_promotions.json` | `d457223630cf3ca595e0e164600f2419996304f8f409128f614be875106b044f` |

The final analysis directory contains 26 files totaling 2,888,524 bytes. Its deterministic inventory digest is `ac74730f8735e37302cc7d28574424268b58d633448258dba47c973b7f63017e`; the complete local result package contains 95 files totaling 59,349,224 bytes with inventory digest `a4d2b58852184ba81a052641bf8f0cf65d50354e1ed5be691c81d01137a45d49`. The cryptographic per-file closure for scientific inputs and report outputs is `analysis_manifest.json`. The final analyzer pins Matplotlib's SVG hash salt to the run ID; an independent second-process rerun reproduced the entire 26-file analysis directory byte-for-byte, including all six SVGs and the manifest.

### Transfer and raw-evidence identity audit

Before local rerendering, the copied remote package contained 95 files totaling 59,359,634 bytes and had inventory digest `0687462b818268e7df23172f67ce794a66e80a88c93fee0aee82a18ba4cfe7c1`. The 10,410-byte aggregate difference in the final local package comes from the deliberate local rerender of generated analysis/report/plot bytes. It is not a raw-result difference.

The four raw execution directories match the execution host byte-for-byte under the transfer audit:

| Raw directory | Remote/local inventory digest |
|---|---|
| `pilot_exact_checkpoint` | `2345c675da99757c09e14edf34d44b9f6b0b4e3368d9f6bb079ec4414608045e` |
| `validation_tile_shapes_exact_checkpoint` | `9259606eb6851029820d9d0f86c9655878b4c8ef0486b84c91e806990b6dd86c` |
| `full_exact_checkpoint` | `7f41d2821d0c8ccf0d842ee1e286e459ebcdb9d262e0f6fb9a124f2ecce726bd` |
| `full_cross_reference` | `6f050fe9dbfbb542b557f8652100b4eca2867333b8d00a34582aca6506e5e1f1` |

Thus the scientific Parquets, run facts, continuation source snapshot, and logs are unchanged. Only the derived final-analysis package was regenerated.

### Held-out result and success-gate disposition

All values below are exact sequential complete-expert qenergy recovery at 1.0 physical correction bpw. They are reconstruction metrics, not end-to-end model quality.

| Cohort | Path | n | p10 | Median |
|---|---|---:|---:|---:|
| Fresh exact-checkpoint, all audited layers | Neuron-major `independent_unit_correction_norm` | 85 | `0.926808` | `0.966073` |
| Fresh exact-checkpoint, all audited layers | Exact 64×16 tile | 85 | `0.937530` | `0.968256` |
| Fresh exact-checkpoint, difficult layers 4/20/39 | Neuron-major `independent_unit_correction_norm` | 72 | `0.922500` | `0.959830` |
| Fresh exact-checkpoint, difficult layers 4/20/39 | Exact 64×16 tile | 72 | `0.936399` | `0.963195` |
| Broader cross-reference sensitivity | Neuron-major `independent_unit_correction_norm` | 142 | `0.898880` | `0.966440` |
| Broader cross-reference sensitivity | Exact 64×16 tile | 142 | `0.930618` | `0.962473` |

The locked PR #6 `success_gates.json` (SHA-256 `3908a82df017399ef7810598354b2f05645e452959a300772abbdc09cc0990f9`) records the controlled n=72 difficult-layer PR #4 p10/median `0.833364894/0.893499343` and PR #6 same-cohort separate-plane p10/median `0.835097846/0.892560229`. On that exact difficult-layer cohort:

| New H0 oracle | Δp10 vs PR #4 | Δmedian vs PR #4 | Δp10 vs PR #6 separate plane | Δmedian vs PR #6 separate plane |
|---|---:|---:|---:|---:|
| Neuron-major | `+0.089135106` | `+0.066330657` | `+0.087402154` | `+0.067269771` |
| Exact 64×16 tile | `+0.103034106` | `+0.069695657` | `+0.101301154` | `+0.070634771` |

These gains are broad H0-oracle headroom across the audited fresh layers, but neither selected method is deployable as measured. The neuron score reads the full suffix, the exact tile allocator refreshes after every selected page—768 global refreshes at one bpw, above the ≤16 target—and no deployable-H0 proxy passed. The activation shortlist stopped. Both selected paths have physical page amplification `1.0` and storage multiplier `1.470588`, so the amplification ≤1.2 and storage <5 limits pass, but the deployable-selector/rank/refresh conjunction fails. The requested overall success conjunction therefore fails.

The validation-selected 64×16 and 16×64 exact tile shapes remain a near-tie: their validation medians differ by only `0.000098`. The frozen apparent 75% page-reduction diagnostic remains invalid because its comparator recovery was below the zero-correction recovery; it is retained only for auditability and is not counted as success. All positive neuron/tile results are labelled `h0_oracle`; no H4 predictor was trained and no router, logit, token-quality, perplexity, task-quality, or deployment claim is made.

### RunPod shutdown completed

After all raw artifacts had been copied and audited, RunPod `41rk786odszmk9` was stopped with `runpodctl pod stop` at `2026-08-19 18:03:11 UTC`. Post-stop inspection reported `desiredStatus=EXITED`, `runtimeStatus=stopped`, and reason `stopped_by_user`. Persistent disk remains available for recovery, while GPU billing has stopped.

The first management-connector attempt returned HTTP 401 because that connector was not authenticated. It made no state change. The already-authenticated local CLI was then used successfully; no credential material is recorded here.

### Publication handoff

- Final Git commit: `TBD at publication`.
- Push result: `TBD at publication`.
- Pull request: `TBD at publication`.
- Intended base/head: `agent/mxfp4-interaction-aware-allocator` ← `agent/mxfp4-sparse-streaming`.
