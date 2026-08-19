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

## Pending chronological entries

- training-only fit command, duration, hashes, and failures/corrections;
- validation command, cohort counts, artifact hashes, and frozen decision;
- any validation-only conditional continuation;
- held-out exact confirmation and cross sensitivity, if promoted;
- final analysis/report/plots and manifest verification;
- commit, push, draft PR, and pod-stop verification.
