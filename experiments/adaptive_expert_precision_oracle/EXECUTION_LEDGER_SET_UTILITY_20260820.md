# Execution ledger: set-utility distillation

All times are Europe/London on 2026-08-20 unless explicitly stated otherwise.
Ephemeral access addresses, private-key paths, service credentials, and host
secrets are intentionally excluded from this committed record.

This ledger records implementation and execution provenance. The validation
fit, exact-checkpoint validation, analysis, and STOP decision are recorded
below. Repository publication, transfer verification, and shutdown remain
deliberately incomplete until those events happen.

## Scope and branch

- Base: PR #9 branch `agent/mxfp4-neuron-selector-distillation`, commit
  `56fe7764ec28c92947c83cb3d7dbd16e5630311b`.
- Study branch: `agent/mxfp4-set-utility-distillation`.
- Run ID: `qwen36_mxfp4_set_utility_distillation_20260820_v1`.
- Worktree: a dedicated worktree rooted at the exact PR #9 head. Unrelated
  changes in the shared checkout were not imported.
- Scientific scope: investigate whether coherent-unit sparse suffix selection
  can be predicted by direct set-utility objectives, structured high-rank
  metadata, or recurring support templates; separately test whether a bounded
  coherent/G/D hybrid oracle improves the PR #9 action path.
- Non-goals: no codec or tree search, no H4 predictor training, no all-layer
  expansion, no router/logit/token-quality claims, and no model-quality claim.

The locked embedded Q2→Q3→Q4 codec, selected trees, exact checkpoint revision,
and train/validation/test request separation are inherited unchanged. This is
an allocator/selector study, not a codec study.

## Locked identities

| Input | Locked identity |
|---|---|
| PR #9 base commit | `56fe7764ec28c92947c83cb3d7dbd16e5630311b` |
| Checkpoint repository | `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` |
| Checkpoint revision | `7eceff3a9f7e6f916c824d197266d86676bce695` |
| Checkpoint `config.json` SHA-256 | `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a` |
| Checkpoint `model.safetensors.index.json` SHA-256 | `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb` |
| Selected trees SHA-256 | `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8` |
| Exact-checkpoint capture SHA-256 | `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| Cross-reference capture SHA-256 | `3307216e92a7c9fae24ad0f213f064a325f403685f0e97952862b70b86b91add` |
| Study configuration SHA-256 | `ef5308f7f684dca31a98fb81ecc8c49feded78ce4f201617fa847b4e9b5d09f1` |
| Fit/validation runner SHA-256 | `90b78f665f40ded98d6b17b28bc36ef163e53aee14233cb92acd6aa5cd61eb73` |
| Runner tests SHA-256 | `39b5935bb9415cd7281109c10e5c6b400b6baa081d7cc33f41f31a589c674fe2` |
| Direct predictor core SHA-256 | `82b1b85b01b00658121613776fc8ddcb61352e3b13e7704835c656a6d6ae204c` |
| Direct predictor tests SHA-256 | `10dac8fcfe2faaa2f8b5a424fc875429a2316fc5c826d1ea9d33e76300c5546e` |
| Analyzer core SHA-256 | `66c2a123e078997f1e7d349e161fcce18a3ac6600b0e36294330b74573a79ac4` |
| Analyzer wrapper SHA-256 | `277b43b9cbecafd427b4bef32fcec1d6f628d228c75f95f95946779f18c76ac9` |
| Analyzer tests SHA-256 | `8e277f72ecd8e3acde49b1acef85878a20f9a35a5e6c0730842745c7c10f2247` |
| Analyzer end-to-end tests SHA-256 | `eeb0152e2d4b4c729f1a6be3d39d18f81505355ee194105755ecb0b2aef6b456` |

The supplied compute host exposed two NVIDIA GeForce RTX 3090 GPUs. The study
fits and evaluates layers 0, 4, 20, and 39 and uses the locked hot/median/cold
expert samples serialized in the configuration. One 24 GiB RTX 3090 is
sufficient; no PRO 6000-specific capacity is required for this bounded study.

The table above is the authoritative final local pre-launch freeze. Every
digest must be recomputed on the staged execution checkout immediately before
the replacement launch and must match exactly. The publication manifest must
bind these configuration and source hashes and fail if they differ from the
fit or validation facts.

## Terminology and evidence boundaries

- `exact_marginal_fixed_greedy` means residual-aware, fixed-coefficient
  marginal greedy selection. There is no least-squares coefficient refit, so
  this method is not standard OMP. Old serialized labels remain readable only
  for compatibility.
- `coherent_exact_set_fixed_greedy_teacher` is a full-target fixed-greedy
  teacher, not a proof of the globally optimal cardinality-constrained subset.
- `full_target_restricted_teacher_oracle` restricts that teacher to a fetched
  candidate set while still using the complete unfetched target residual. It
  is a teacher-information control, not a deployable selector and not a global
  optimum. Its reported retention may exceed a fixed-greedy denominator.
- `contained_fetched_target_exact_h0` uses only the fetched candidate
  corrections but constructs their interaction geometry. It is an H0 systems
  oracle, not the low-cost promotion path.
- The direct selector receives the actual activation and Q2-side values
  `(x, g2, u2, h2)`. It is therefore a late-H0 selector. It is not an H4
  predictor or evidence that future activations can be predicted.
- Every recovery value is exact sequential complete-expert qenergy recovery
  under the locked future-proxy metric. It is expert-output reconstruction
  evidence only.

## Implementation chronology

1. Added a compact exact coherent-unit teacher. A static down-side metric Gram
   is built once per `(layer, expert, proxy)` and reused across activations.
   The activation-dependent 512×512 correction Gram is reconstructed from
   `h2`, `h4`, and the static down Gram. Training teacher paths are 512-action
   permutations and use fixed-coefficient exact marginal updates.
2. Added exact Gram/direct-output qenergy parity assertions, nested support
   tests, deterministic tie-breaking, nonidentity-metric tests, and real-size
   fixtures.
3. Added a four-state `00/G/D/GD` hybrid action vocabulary with a direct
   three-page `00→GD` action, exact one-/two-/three-unit monotone local moves,
   and four seed basins. The final grid reports five controlled families:
   PR #9 independent coherent, exact-set coherent fixed-greedy, PR #9
   factorized best-prefix, direct-GD hybrid best-prefix, and forward plus
   bounded local search.
4. Replaced the infeasible repeated static-down-Gram construction with one
   expert-level precompute. This avoids repeating an approximately
   512×2048-by-2048×512 contraction for every synthetic activation.
5. Added a GPU primitive-Gram implementation for the hybrid oracle. It builds
   the 2048×2048 Gram for the four primitive transition vectors per unit once
   on GPU, transfers it once, and runs a deterministic CPU correlation-update
   loop. Direct-GD actions are evaluated as sums of primitive transitions.
   One shared path supplies all three physical budgets.
6. Added per-expert support-template banks trained only from training masks,
   with utility-weighted clustering, charged top-1/top-2 unions, and 0/16/32/64
   repair-unit controls. Actual routed exact-training masks remain separate
   from synthetic all-activation pairings.
7. Added activation-weighted, scale-aware block product-quantized gate/up
   residual synopses. Gate and up use separate layer-shared FP16 codebooks;
   expert rows store packed four-bit codes and use the locked resident E8M0
   block scales. One- and two-additive-codebook controls are predeclared. PQ
   response features are now FP16-rounded and then widened for the FP32
   reference kernels in both fitting and validation, matching their declared
   two-byte runtime representation.
8. Added matched-storage high-rank/low-bit residual controls, including the
   inherited rank-64 FP8/INT8 baselines and rank-128/rank-256 int2 or ternary
   diagnostics.
9. Added a compact shared contextual per-unit predictor with listwise and
   boundary-ranking supervision. Predeclared hard-set variants additionally
   use the exact apply-top-192 Gram set loss. The joint nested head uses a
   teacher-top-192 coverage loss; its hard-forward candidate mask is the
   apply-head top 192 union the best 64 candidate-head units excluding those
   192. Candidate-256 Gram set loss is disabled. The reusable core supports
   optional exclusion-regret weights, but this runner supplies none. Training
   uses Q2-side features and exact correction vectors/Grams only as teacher
   information. No full Gram or correction tensor is deployable metadata.
10. Encoded A/B/C score metadata as three FP16-normalized vectors with one
    FP32 scale each: `3*(512*2+4)=3,084` bytes per expert. Train-only feature
    normalization uses stable `log(A)`, `B/sqrt(A*C)`, and `log(C)` transforms.
11. Added request-round-robin sampling, explicit routed-versus-synthetic
    cohort labels, fixed training epochs, deterministic seeds, layer-sharded
    fit artifacts, atomic writes, and strict hash-bound resume.
12. Added four distinct candidate-application records and separate rerank
    compute/byte/runtime accounting. Promotion is limited to predicted-direct
    or exact-independent-ABC-within-fetched policies. Standalone unit-score
    selectors charge six A/B/C score MACs per unit, or 3,072 MACs for all 512
    units. Q4 response accounting separately charges resident Q2 gate/up
    parent codes, conditionally deduplicated E8M0 scale payload, and block-scale
    multiplications; exact components and bound strings are preserved in
    `candidate_rerank_accounting.csv`.
13. Added strict execution provenance and transaction guards. Fit resume,
    every layer sidecar, the manifest, fit-to-validation transition, and
    validation resume bind the device string, GPU names, Torch/CUDA runtime,
    NumPy, Python, and host. Cross-hardware resume is prohibited for this run.
    Empty checkpoint Parquets retain the declared schema for all four raw
    artifact families.
14. Hardened analyzer accounting serialization after two fail-closed partial
    analysis attempts. Support-template selector additions and scale
    multiplications are family-inapplicable and are encoded only as explicit
    JSON `null`; accounting schema v5 rejects every unexpected missing or
    infinite value. Both partial output directories remain quarantined, and
    neither emitted nor informed a scientific decision. Canonical analysis
    was regenerated from the unchanged immutable validation evidence.

## Physical candidate interface

One coherent Q2→Q4 unit packet is three 512-byte pages: two gate/up pages and
one down page. The primary interface fetches 256 packets and applies 192:

- fetched traffic: 768 pages = 393,216 bytes = 1.0 physical correction bpw;
- applied correction: 576 pages = 294,912 bytes = 0.75 logical correction bpw;
- page amplification: `768/576 = 4/3`.

All candidate traffic is charged. The four application policies are:

1. `predicted_direct_no_rerank`: apply the apply-head top 192. For the joint
   nested model the fetched set is exactly those 192 units union the best 64
   candidate-head units excluding the apply support. The candidate head is
   supervised for teacher-top-192 coverage. This is the most direct late-H0
   deployment control.
2. `exact_independent_abc_within_fetched_candidates`: FP16-round the
   activation and resident `g2/u2/h2`, compute exact Q2→Q4 gate/up residual
   responses, evaluate decoded-FP16 A/B/C scalar scores, and apply the best
   192 fetched candidates. The correctness reference evaluates all 512 unit
   responses for convenience, but accounting charges only fetched candidates:
   resident Q2 gate/up parent codes, E8M0 scale bytes unless the selector
   already read them, 128 response block-scale multiplications per unit, and
   six A/B/C score MACs per fetched unit. This is the primary realistic H0
   rerank and is eligible for promotion only after total selector plus rerank
   compute is charged.
3. `contained_fetched_target_exact_h0`: use fetched corrections to construct
   and greedily solve the contained interaction problem. This is an expensive
   systems oracle. A direct 256-candidate construction is on the order of
   `256^2*2048 = 134,217,728` multiply-accumulates before the small greedy
   loop, far above the declared selector gate.
4. `full_target_restricted_teacher_oracle`: greedily select from the fetched
   candidates against the full target residual. This uses unfetched teacher
   information and is nondeployable. It charges all-512 Q4 response work, the
   all-512 gate/up parent-code payload, conditional E8M0 scales and block-scale
   applications, the all-512 correction workspace, the Euclidean Gram, and
   the rank-4 proxy projection/Gram arithmetic and metadata. Its accounting is
   a precise lower bound only because resident Q2-down parent-code/scale
   payload, Q2-down tree/E8M0 decode and block-scale application, fixed-greedy
   initial correlation, and per-selection vector updates remain excluded. The
   contained interaction path has the same explicit lower-bound omissions.

The first two policies are the only promotion-primary interfaces. Policies
three and four diagnose candidate quality and interaction regret; they cannot
rescue a failed deployable configuration.

The 393,216 fetched packet bytes are exact logical physical payload. Selector
metadata `bytes_read` is a logical unique-payload-plus-LUT-read model, not a
measurement of DRAM or PCIe transactions. Reported MACs are analytical logical
MAC counts; additions and scale multiplications are separate fields. A Q4
gate/up response requires 1,024 bytes of resident Q2 parent codes, 128 bytes of
E8M0 scales, and 128 block-scale multiplications per unit; scale bytes are
deduplicated only when the selector already read them. The two high-rank
synthesis matrices charge 1,024 row-scale multiplications per invocation.
Already-fetched suffix-packet bytes remain charged for predicted-direct and
template rows and are not subtracted to zero or double-counted.

## Verification checkpoints

Commands use the experiment root as the current directory.

```text
python -m py_compile \
  src/oracle_study/unit_set_teacher.py \
  src/oracle_study/set_utility_oracles.py \
  src/oracle_study/set_utility_selector.py \
  src/oracle_study/direct_set_predictor.py \
  src/oracle_study/set_utility_analysis.py \
  scripts/run_set_utility_distillation.py \
  scripts/benchmark_set_utility_hybrid.py \
  scripts/analyze_set_utility_distillation.py

python -m json.tool configs/qwen36_mxfp4_set_utility_distillation.json

PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_unit_set_teacher.py \
  tests/test_set_utility_oracles.py \
  tests/test_set_utility_selector.py \
  tests/test_direct_set_predictor.py

PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_set_utility_runner.py

PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_set_utility_analysis.py \
  tests/test_set_utility_analysis_e2e.py

PYTHONPATH=src:scripts python -m pytest -q
```

The authoritative final local verification against the hashes above is:

- runner-focused suite: 37 passed;
- analyzer-focused suite: 42 passed;
- full integrated suite: 282 passed and 1 skipped in 79.662 s, with one NumExpr
  warning and no failures.
- staged remote runner-plus-analyzer gate: 78 passed in 60.518 s, with no
  failures.

These results supersede intermediate test counts produced while the source
freeze was still changing. The staged execution checkout must reproduce the
exact digests above before the replacement fit is admitted.

## Failed benchmark and correction

The first real benchmark invoked `scripts/benchmark_set_utility_hybrid.py`
against the locked exact capture, checkpoint, and selected-tree file. It
failed closed before scientific compute because the configuration contained a
56-character tree SHA rather than a valid 64-character lowercase hexadecimal
SHA-256. This was a provenance-contract failure, not a model or CUDA failure.

Correction:

1. replace the malformed value with the exact locked tree digest
   `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`;
2. add a regression that rejects any locked digest that is not exactly 64
   lowercase hexadecimal characters;
3. rerun the identical locked benchmark.

No recovery or timing row was emitted by the failed invocation. The failed
attempt is retained here so a reviewer can see that the input contract, not a
post-hoc scientific decision, caused the correction.

## Real-size benchmark preflight

The corrected benchmark used one locked layer-4/expert-17 validation
invocation on a real RTX 3090. It is a feasibility and parity preflight, not
broad validation evidence.

| Quantity | First run | Repeat |
|---|---:|---:|
| Primitive Gram shape | 2048×2048 | 2048×2048 |
| Primitive Gram device bytes | 16 MiB | 16 MiB |
| Gram build | 0.096 s | 0.106 s |
| Device-to-host transfer | 0.259 s | 0.126 s |
| Deterministic greedy update | 1.512 s | 1.741 s |
| Budget snapshots/direct checks | 0.572 s | 0.264 s |
| Total | 2.651 s | 2.256 s |
| Candidate evaluations | 548,959 | 548,959 |

Direct-qenergy versus Gram parity passed. The single-invocation recovery
snapshots at 0.5/0.75/1.0 physical correction bpw were
0.9484/0.9595/0.9644. These numbers are explicitly pilot-only and must not be
quoted as a cohort result.

The full `_hybrid_rows` preflight on the same one invocation evaluated all
five families at all three budgets. Wall time was 23.312 s. Bounded local
search incremental times at 0.5/0.75/1.0 bpw were
5.713/7.600/6.654 s; total local rows including all seed construction were
8.854/10.577/9.636 s. The corresponding pilot-only local-search recoveries
were 0.96019/0.98033/0.99056. These values establish bounded execution and
correct accounting scope only; they are not evidence of broad gain.

## Quarantined fit attempts

Four superseded execution bundles were preserved outside the canonical
`fit/` and validation paths. None may be resumed or cited as scientific
evidence.

1. `fit_invalid_pre_nested_coverage_fix` was stopped before validation after
   the nested candidate-head coverage contract was corrected. It contains
   incomplete layer shards and sidecars for layers 0, 4, and 20 only; layer 39
   and the canonical manifest are absent. Facts record `completed=false` and
   empty failure/history lists.
2. `fit_invalid_pre_byte_accounting_fix` was stopped pre-layer-0, before any
   layer committed, while final candidate-byte, runtime-rounded
   independent-ABC, execution-provenance, and empty-Parquet transaction
   audits were being completed. It contains no shard, sidecar, or manifest.
   Facts record
   `completed_layers=[]`, empty failure/history lists, a clean cross-capture
   audit with zero shared and zero incompatible request IDs, and the corrected
   nested candidate semantics.
3. `fit_invalid_pre_selector_score_mac_fix` had valid initial provenance,
   request-separation, runtime, and nested-candidate facts, but was stopped
   pre-layer-0 when the standalone 512-unit A/B/C score path was found to omit
   its 3,072 analytical MACs. It contains no shard, sidecar, or manifest;
   `completed_layers=[]` and the failure list was empty. No output from this
   attempt is scientific evidence.
4. `fit_invalid_pre_candidate_overfetch_fix` completed all four fit layers,
   and its paired `validation_invalid_pre_candidate_overfetch_fix` completed
   exactly 69 exact-checkpoint validation invocations. Both recorded empty
   failure/history lists. The strict analyzer then failed closed before any
   report or promotion decision because support-template candidate rows did
   not carry a finite `candidate_overfetch` value. The raw fit and validation
   artifacts are preserved only as correction evidence; no recovery metric
   from them was summarized, selected, or used. The replacement runner emits
   finite `candidate_overfetch = candidate_units / applied_units` and has a
   real-path regression covering all four rerank semantics.

The replacement launch must start from a newly absent canonical `fit/` path
and a new task-owned log. Quarantine contents remain immutable evidence of
the corrections, not resume inputs.

## Train-only fit launch

The successful replacement fit launched only after the staged checkout
matched the frozen configuration/source hashes and focused tests passed. The
literal access command remains omitted. Within the scoped experiment checkout,
the executed command semantics were:

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts \
  /absolute/path/to/study-python \
  scripts/run_set_utility_distillation.py \
  --phase fit \
  --config configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /absolute/path/to/qwen36_mxfp4_candidate \
  --trees /absolute/path/to/locked/selected_trees.json \
  --output /absolute/path/to/run-root/fit \
  --exact-captures /absolute/path/to/qwen36_exact_confirm_captures.npz \
  --cross-captures /absolute/path/to/captures_seed20260817.npz \
  --device cuda:0
```

The process was detached by the job launcher and its stdout/stderr sent to a
task-owned log. The runner admitted only `train` scientific rows from both
monolithic capture files. Those files physically contain test rows, so this
ledger does not claim the bytes were absent or that the archive was never
opened. Split labels and request-ID metadata for train, validation, and test
are inspected solely to verify within-capture separation and cross-capture
split compatibility. Same-split sharing is generally compatible, while a
cross-source split mismatch fails closed; this locked capture pair freezes and
observes zero shared IDs. No test scientific tensor is admitted into a fit or
validation cohort, evaluated for a metric, used for tuning, or reported.
The successful replacement `fit_facts.json` records:

- `fit_split=train`;
- `validation_or_test_rows_used=false`;
- `test_rows_admitted=false` and `test_rows_used=false`;
- request separation verified;
- cross-capture request audit: zero shared IDs, zero incompatible overlaps,
  and zero train-to-nontrain overlaps;
- exact hard-forward nested candidate semantics;
- locked device/GPU/Torch-CUDA/NumPy/Python/host runtime provenance;
- checkpoint tensor audit passed for all 6,144 required entries;
- both capture hashes, checkpoint hashes, tree hash, and all linked source
  hashes bound before layer fitting.

Fit completed at `2026-08-20T07:27:30Z` with exactly layers `[0,4,20,39]`,
`completed=true`, and empty failure/history lists. `fit_facts.json` SHA-256 is
`9aa5f692bbebbf94871cb110e3c40a49ed7d9be93220b655241a67ee87904759`;
the manifest SHA-256 is `afb87ff360e38d191c77abd543f41a8c686afe0e4b0f0bbda4209880db968387`.

| Layer | JSON sidecar SHA-256 | NPZ shard SHA-256 |
|---:|---|---|
| 0 | `87a65751b82c051edfc315f497f2a96ee04e8b53a8adb37747e8ecd512563f38` | `4c9a7ae2554bf366f74454451c7aac2ec2597417318b0cd04e8732bd36511524` |
| 4 | `c866dbe2026e53147321d80111aac2a9311999b59a019d71e72c5521d9587395` | `2572b4ef1195ae923ed6f7ffd3a8d0b090f16beab1ae9a5f0ef8311c53ec3ace` |
| 20 | `068466e61380efc34d9ea435793c9f8c6801ed22312381b0471e34b6d7fc4342` | `145bf96562d86965dafec9d6f6ea9ae5d81b424fcdc771a2be25b6a01055dba3` |
| 39 | `7cae024be153a084150e0f80bf73c5db0f2d2ddf459dfc27ee12e6a1b6db4d34` | `06a6ddc578858b376c81a9e05083c417ef8eb92dccc962fe8d4078303c3b7815` |

Every shard and sidecar rehashed into the manifest with identical runtime
provenance. Routed and synthetic train-only cohort counts remain distinct in
the manifest; no validation or test scientific row was used for fitting.

## Validation-only execution and stop/promote contract

Validation may begin only after the fit facts and manifest say complete and
all layer-shard hashes verify. Validation loads the exact-checkpoint capture
with an allow-list of `train` and `validation`; it rejects any admitted test
row. The cross-reference capture is not a legal validation argument.

The frozen command is:

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:scripts \
  /absolute/path/to/study-python \
  scripts/run_set_utility_distillation.py \
  --phase validate \
  --config configs/qwen36_mxfp4_set_utility_distillation.json \
  --checkpoint /absolute/path/to/qwen36_mxfp4_candidate \
  --trees /absolute/path/to/locked/selected_trees.json \
  --output /absolute/path/to/run-root/validation_exact_checkpoint \
  --exact-captures /absolute/path/to/qwen36_exact_confirm_captures.npz \
  --fit-dir /absolute/path/to/run-root/fit \
  --device cuda:0
```

The analyzer command is:

```bash
MPLBACKEND=Agg PYTHONPATH=src:scripts \
  /absolute/path/to/study-python \
  scripts/analyze_set_utility_distillation.py \
  --config configs/qwen36_mxfp4_set_utility_distillation.json \
  --fit-dir /absolute/path/to/run-root/fit \
  --validation-dir /absolute/path/to/run-root/validation_exact_checkpoint \
  --output /absolute/path/to/run-root/validation_analysis
```

The analyzer applies the following one-way validation gates:

- hybrid continuation: at least +0.005 absolute median recovery over the
  exact-set coherent fixed-greedy teacher at 1.0 physical bpw;
- top-1 template: median teacher-path gain retention at least 0.95 and no
  more than 256 charged candidates;
- top-2 union template: median retention at least 0.98, p10 at least 0.95,
  and no more than 256 charged candidates;
- deployable candidate interface: median recovery at least 0.94, p10 at
  least 0.88, p10 exact-set-gain retention at least 0.97, metadata no more
  than 0.35 bpw, total selector plus rerank compute strictly below 1,572,864
  MACs, and total storage below 5×;
- only `predicted_direct_no_rerank` and
  `exact_independent_abc_within_fetched_candidates` are eligible for the
  deployable gate.

No threshold, candidate count, template count, loss weight, PQ stage, rank,
or local-search setting may change after observing validation. If no
deployable policy passes, stop and publish the validation-only negative result.
If a policy passes, freeze exactly one Pareto-relevant configuration in
`set_utility_promotions.json` before any new holdout is captured or evaluated.

The PR #7/#9 test split scientific values have already informed this research
direction, so they are not a fresh holdout and must remain unconsulted in this
bounded runner. A
promoted method requires a newly captured, sealed holdout under a separately
frozen protocol. This runner deliberately exposes no test phase.

Validation completed at `2026-08-20T08:01:22Z`: exactly 69/69 unique
exact-checkpoint validation invocations and all 12 locked layer/expert cells,
with empty failure/history lists. Raw row counts are 1,035 hybrid, 8,832
template, 1,173 selector, and 40,020 candidate-interface rows. The completed
`run_facts.json` SHA-256 is
`5699f45b3983b0e189aebaaf83351e62ce991c1f4af9b74db496388dc68c2e7e`.

The frozen decision is **STOP**. `set_utility_promotions.json` SHA-256 is
`5e9928ee2ba0c72f4b34cd5e270d2f47e169cf1078bf6fcf0520347fc4fc0ebd`.
The exact one-bpw hybrid local-search oracle reaches complete-expert recovery
p10/median `96.05%/97.89%` and passes the oracle continuation gate.
It improves paired recovery by `+2.261/+5.439` percentage points in p10/median
over coherent exact, and by `+0.031/+2.123` points over the PR #9 independent
control. The `K=128` template plus 64 oracle repair units retains
p10/median utility `96.89%/98.77%` at 256 candidates, but is a support-regime
existence oracle only; no template classifier was trained or evaluated.
The best promotable PQ-stage-2 plus independent-ABC candidate/rerank path
reaches recovery p10/median `78.72%/88.63%`, p10 exact-set-gain retention
`90.45%`, `0.1771` metadata bpw, `1.184M` total selector MACs, and `1.512x`
storage. It fails the frozen recovery and set-retention gates. All 34
predicted/independent rows fail the scientific gates, and direct set models
perform worse. The authoritative report SHA-256 is
`3fef9cfe4b8fb8dcc50421bc5847312d686d876275cad12c358991c5643f6964`;
analysis-manifest SHA-256 is
`e6a9894a630c8a7d81d4da064f295d56cd64cd7016b7a1d4e672c213ff4cf869`;
analyzer accounting JSON SHA-256 is
`35831588074bca7a89b1634c1f530fd33d0d56d05bb48f4b71be42917af51960`.

No sealed holdout was captured or evaluated because no deployable selector
passed every validation gate. The bounded study stopped exactly as frozen.

## Artifact inventory

Train-only fit artifacts:

- `fit/fit_facts.json` — complete; SHA-256 `9aa5f692bbebbf94871cb110e3c40a49ed7d9be93220b655241a67ee87904759`;
- `fit/set_utility_fit_layer_{0,4,20,39}.npz` — four verified hashes in the completion table above;
- `fit/set_utility_fit_layer_{0,4,20,39}.json` — four verified hashes in the completion table above;
- `fit/set_utility_fit_manifest.json` — SHA-256 `afb87ff360e38d191c77abd543f41a8c686afe0e4b0f0bbda4209880db968387`;
- successful-run failures/history are empty; preserved correction bundles are enumerated under quarantined attempts.

Validation raw artifacts:

- `validation_exact_checkpoint/run_facts.json` — complete; SHA-256 `5699f45b3983b0e189aebaaf83351e62ce991c1f4af9b74db496388dc68c2e7e`;
- `validation_exact_checkpoint/hybrid_oracle_frontier.parquet` — 1,035 rows; SHA-256 `5fab3125c4bad9329ebbe9a566a7a4ef41cfba14b823a588f3fac9904c083a34`;
- `validation_exact_checkpoint/support_template_frontier.parquet` — 8,832 rows; SHA-256 `c62c1bbfcdbe4bf91d625ce7e34bf8f6b4eab9cc413a6bf463946ebaa45f02bd`;
- `validation_exact_checkpoint/pq_high_rank_selector_frontier.parquet` — 1,173 rows; SHA-256 `108fd14313fb0a16a42171d46558d5ab8dffdab1e943cf9b7f4083f6a743d7bc`;
- `validation_exact_checkpoint/candidate_set_rerank_frontier.parquet` — 40,020 rows; SHA-256 `d8b8699cae7c7ffc7bf04d798504e4f6944fe92ff509a797a7c9321e66f2df3f`;
- `validation_exact_checkpoint/selector_compute_storage_accounting.json` — SHA-256 `cceadc876a6b347cc5a73479a1142093631064b5fa04d32f3007c01e9555b8f7`.

Validation-analysis artifacts:

- `validation_analysis/SET_UTILITY_DISTILLATION_REPORT.md` — SHA-256 `3fef9cfe4b8fb8dcc50421bc5847312d686d876275cad12c358991c5643f6964`;
- `validation_analysis/set_utility_promotions.json` — STOP; SHA-256 `5e9928ee2ba0c72f4b34cd5e270d2f47e169cf1078bf6fcf0520347fc4fc0ebd`;
- `validation_analysis/analysis_manifest.json` — SHA-256 `e6a9894a630c8a7d81d4da064f295d56cd64cd7016b7a1d4e672c213ff4cf869`;
- exactly 14 CSVs, including `candidate_rerank_accounting.csv`, are hash-bound in the analysis manifest;
- four deterministic PNG/SVG plot pairs are hash-bound in the analysis manifest;
- analyzer-recomputed `selector_compute_storage_accounting.json` — SHA-256 `35831588074bca7a89b1634c1f530fd33d0d56d05bb48f4b71be42917af51960`;
- package-level `artifact_hashes.sha256` binds 42 files; SHA-256 `68cc3f54ba346fbb3632ec50b5813da3817e46cd8867375540b5342fec95f3c5`.

Publication must preserve train-fit, exact validation, any newly sealed
holdout, and analysis directories as distinct evidence planes. It must not
copy pilot-only benchmark values into cohort summary tables.

## Publication and shutdown

- **Result interpretation:** hybrid action/local-search structure is the only
  continuation-positive result; the deployable coherent-unit selectors stop
  because all predicted/independent rows fail the frozen scientific gates.
  Templates establish a secondary support regime and PQ is secondary synopsis
  evidence, not a passing selector. No sealed holdout was run.
- **EVIDENCE COMMIT PLACEHOLDER:** `[commit SHA and subject]`.
- **PUSH PLACEHOLDER:** `[remote branch and verified head SHA]`.
- **PR PLACEHOLDER:** `[URL, base/head, draft/open state, changed-file count,
  mergeability/check status]`.
- **TRANSFER AUDIT PLACEHOLDER:** `[remote/local package digest and file count]`.
- **POD SHUTDOWN PLACEHOLDER:** `[stop time, task-owned pod identity, verified
  exited/stopped state]`.

The compute pod must remain running while experiments or artifact transfer are
still active. It may be stopped only after every required artifact has been
transferred, rehashed locally, committed, pushed, and the PR has been
verified. No unrelated pod may be stopped.
