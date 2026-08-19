# Execution ledger: neuron-selector distillation

All times are Europe/London on 2026-08-19 unless noted. Ephemeral connector
addresses and private-key paths are intentionally omitted from this committed
record.

## Scope and branch

- Base: PR #7 commit `9ef21d519a4f2cd844e9fa75d04cd601e37d5a11`.
- Branch: `agent/mxfp4-neuron-selector-distillation`.
- Worktree: a clean dedicated worktree; unrelated codec experiments in the
  original shared worktree were excluded.
- Locked checkpoint revision, codec, tree, and capture hashes are declared in
  `configs/qwen36_mxfp4_neuron_selector_distillation.json`.

## Implementation and corrections

1. Implemented exact four-state `00/G/D/GD` unit decomposition, exact
   marginal-per-page fixed-greedy with nonidentity proxy metric, complete
   endpoint and nested eligibility tests.
2. Implemented the exact A/B/C sufficient statistic, response-optimal
   layer-shared and sampled per-expert factors, FP16/FP8/INT8 synthesis
   encodings, unit-ranking metrics, candidate fetch plus exact H0 reranking,
   and physical/storage accounting.
3. Added validation-only selection, immutable comparator hashes, split
   predicate reads, minimum layer/request coverage, and held-out gate
   recomputation.
4. Corrected A/B/C FP16 overflow risk by normalizing each vector and storing
   one FP32 scale. The final compact cost is `3*(512*2+4)=3,084` bytes per
   expert for FP16 A/B/C metadata.
5. Added expert-boundary resume. On restart, only transactions named in the
   last durable facts file survive; partially written rows are discarded.
   Resume also binds runner, selector core, base runner, capture, checkpoint,
   tree, fit bundle, and frozen-promotion hashes.
6. Added minimum named schemas for legitimately stopped/empty artifact
   families and an end-to-end invocation/accounting regression.
7. Replaced the projected 250–300 MB monolithic fit bundle with four independently hashed layer shards plus a canonical manifest. Evaluation verifies hash, byte count, array count, and layer ownership, then keeps only one decoded layer in memory.
8. Hardened FP8 synthesis metadata with one FP16 scale per row so large fitted factors cannot silently overflow E4M3FN; non-finite FP16/FP8 inputs or decodes now fail closed.
9. Replaced the unbounded CPU fitting path before execution: response sample Grams now use the GPU eigensolver, all max-rank B transforms share one dual/primal ridge factorization per cohort, and response matrices sharing a latent basis reuse one GPU pseudoinverse. Numerical projection, ridge, and synthesis parity tests were added.
10. The repository patch helper repeatedly failed before reading files because
   its sandbox could not create a loopback interface. Narrow patches were then
   applied with `git apply --recount`; a few exact mechanical substitutions
   used Perl after both the patch helper and an initial patch transport failed.
   One zero-byte temporary file created by that failed transport was inspected
   and removed before staging.

## Verification

```text
python -m py_compile \
  src/oracle_study/neuron_selector.py \
  src/oracle_study/neuron_selector_analysis.py \
  scripts/run_neuron_selector_distillation.py \
  scripts/analyze_neuron_selector_distillation.py
python -m json.tool configs/qwen36_mxfp4_neuron_selector_distillation.json
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_neuron_selector.py \
  tests/test_neuron_selector_runner.py \
  tests/test_neuron_selector_analysis.py
PYTHONPATH=src:scripts python -m pytest -q
```

Latest pre-run result: 26 focused tests passed; 147 integrated tests passed in
26.84 seconds. The only warning is the inherited pandas/numexpr version
warning.

## Remote execution

The supplied 3090 endpoint was attempted eight times with a 15-second
connection timeout. Every attempt failed before authentication with `No route
to host`. No remote command ran, no pod state changed, and no validation or
held-out scientific row was read. The original endpoint remained unavailable. A replacement endpoint exposed a verified RTX 3090. Its checkpoint metadata, exact capture, selected tree, synced runner/core/config, and six PR #7 comparator hashes matched the frozen protocol. The locked cross capture was absent, but its immutable raw corpus and audited extractor were present, so deterministic regeneration was started. Two full-history clone attempts were stopped and only their incomplete task-owned directories removed; a 786 KiB checksummed source archive and a 151 KiB checksummed comparator archive were used instead. The first archive extraction failed only while restoring local UID/GID ownership; rerunning with `--no-same-owner` succeeded and all execution hashes matched. The regeneration remained I/O-bound at this checkpoint; no validation or held-out row had been evaluated.

### Replacement-pod chronology

- The deterministic cross-reference extraction completed with 5,036 rows
  (3,292 train, 760 validation, 984 test; 53/12/16 complete requests). Its
  NPZ SHA-256 was exactly the locked
  `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add`.
- The corrected remote focused suite passed 26/26 before fitting. The first
  training-only fit failed before emitting a model artifact because an FP16
  row scale underflowed to zero during FP8 metadata encoding. Commit
  `29b154e` added the missing tiny-row regression for FP8 and INT8.
- A second training-only attempt exposed downward FP16 scale rounding: the
  normalized row could land fractionally above finite E4M3. Commit `d7baf1c`
  rounds stored scales upward and tests 4,096 magnitudes across fourteen
  decades. The remote focused suite then passed 29/29.
- The third fit completed on the 3090. It used training rows only and wrote
  four independently hashed layer shards of 39.9/49.5/49.5/49.1 MB, all
  below GitHub's per-file limit. Bundle-manifest SHA-256:
  `73b0285be3e3611fa44ac0995bd69c3b4390cd02b0fa9ac03a9626397c878d60`.
  Both failed fit logs were retained beside the successful output.
- Validation attempt 1 completed 69/69 invocations with no failures. Its
  first promotion (`d7733971...`) stopped both families, but a bounded
  validation-only audit found that factorized rows blindly applied the final
  path state even when an earlier cumulative prefix had higher exact
  qenergy. This was an evaluation-policy defect, not a tuned hyperparameter.
  Commit `a26a78a` selects the best cumulative prefix under each physical
  budget while still allowing a negative prerequisite to unlock a later
  profitable transition. The complete first attempt and promotion were
  renamed and preserved; no test row had been opened. Corrected validation
  was then restarted from a new output directory after 30/30 remote tests.

- Corrected validation completed 69/69 invocations over all twelve sampled
  experts with no failure. Row counts were 414 factorized/coherent frontier,
  13,041 direct predictor, 17,388 candidate-rerank, and 4,347 response-
  diagnostic rows. The final promotion SHA-256 was
  `7430a885d6dfa303ac1a54357fa7a86f54bd9000df46d23f2727fd45fe961bc4`.
- On the 55-invocation matched PR #7 cohort, factorized G/D one-bpw
  p10/median was `0.869842/0.932810`, versus `0.935183/0.966025` for the exact
  64×16 tile. Factorization improved the coherent packet at 0.5 bpw but not
  at 0.75 or 1.0 bpw. The family stopped.
- The best deployable response candidate was combined-train, separate
  gate/up, rank 64, row-FP8. At 256 fetched/192 applied units it retained
  `0.916435/0.956179` p10/median independent-score utility but reached only
  `0.755966/0.888333` recovery at `0.184926` metadata bpw. It stopped. The
  sampled per-expert upper bound also failed, so the predeclared activation-
  top-k and cluster-basis continuations were not run.
- Since both families stopped, held-out exact and cross-reference evaluation
  were not launched. This is a deliberate terminal condition, not missing
  evidence; test rows remain unconsulted.
- The complete remote result directory was transferred locally. All four fit
  shards rehashed exactly, corrected facts were complete with empty failures,
  and the final analysis was regenerated from relative repository paths. Its
  manifest closes 17 inputs and 15 outputs, including five CSV summaries,
  accounting JSON, report, and four PNG/SVG plot pairs.
- A second complete local analysis rerun was byte-identical after pinning Matplotlib SVG IDs and date metadata, then normalizing
  generated SVG trailing whitespace. No new file exceeds 49.6 MB; the full
  result package is 184 MB. Focused tests passed 32/32 and the integrated
  suite passed 153/153 in 21.67 seconds with only the inherited numexpr
  warning.

### Executed commands

The following ran inside the scoped experiment checkout. Connection commands
and private-key locations are intentionally omitted.

```bash
python scripts/extract_pilot_remote.py \
  --config configs/q2_corrected_page_allocator.json \
  --output /workspace/neuron_selector_study/work/captures_seed20260817.npz

python scripts/run_neuron_selector_distillation.py --phase fit \
  --config configs/qwen36_mxfp4_neuron_selector_distillation.json \
  --checkpoint /workspace/qwen36_mxfp4_candidate \
  --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
  --output "$RESULT_ROOT/fit" \
  --exact-captures /workspace/codebook_granularity_study/inputs/qwen36_exact_confirm_captures.npz \
  --cross-captures /workspace/neuron_selector_study/work/captures_seed20260817.npz \
  --device cuda

python scripts/run_neuron_selector_distillation.py --phase evaluate \
  --config configs/qwen36_mxfp4_neuron_selector_distillation.json \
  --checkpoint /workspace/qwen36_mxfp4_candidate \
  --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
  --output "$RESULT_ROOT/validation_exact_checkpoint" \
  --captures /workspace/codebook_granularity_study/inputs/qwen36_exact_confirm_captures.npz \
  --source exact_checkpoint --split validation --fit-dir "$RESULT_ROOT/fit" \
  --device cuda

python scripts/analyze_neuron_selector_distillation.py --stage promote \
  --config configs/qwen36_mxfp4_neuron_selector_distillation.json \
  --fit-dir "$RESULT_ROOT/fit" \
  --validation-dir "$RESULT_ROOT/validation_exact_checkpoint" \
  --output "$RESULT_ROOT/validation_analysis"
```

`RESULT_ROOT` denotes the task-owned result directory named in the committed
paths. The final analyzer was rerun locally with repository-relative paths.
The deterministic 51-file result-package digest (`find . -type f -print0 |
sort -z | xargs -0 sha256sum | sha256sum`, run from the package root) is
`76f5ef0fdc10171b36d9f82e0a39d9a5efe42a345dc6dc9961192b3ea4d90998`.

## Publication and shutdown

- Evidence/results commit: `454ac8afa667723ece5a19c9a461ea9259f1534b`
  (`Add neuron selector distillation results`). The branch push succeeded.
- Draft PR: https://github.com/ArianSFP/LLM_compression_study/pull/9
- Verified GitHub base/head:
  `agent/mxfp4-sparse-streaming at commit 9ef21d519a4f2cd844e9fa75d04cd601e37d5a11`
  → `agent/mxfp4-neuron-selector-distillation at commit 454ac8afa667723ece5a19c9a461ea9259f1534b`.
  GitHub reported draft/open, 62 changed files, and a clean merge state.
- After the evidence push and PR verification, study pod `osfs2vbk08exds` was
  stopped with `runpodctl pod stop` at `2026-08-19 22:07:37 UTC`. Post-stop
  inspection reported `desiredStatus=EXITED`, `runtimeStatus=stopped`, and
  reason `stopped_by_user`; GPU billing has stopped. The account list no
  longer showed this study pod as running. One pre-existing two-GPU pod was
  left untouched because it was not the supplied study endpoint.
