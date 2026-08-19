# Frozen historical selective baselines

This fragment freezes the historical exact-MXFP4 H0 comparison rows used by the codebook-granularity study. The machine-readable companion is `frozen_historical_selective_baselines.csv`. It contains both the all-four-layer cohort (`n=85`) and the primary difficult-layer cohort (`layers 4/20/39`, `n=72`) at streamed budgets `0, 0.5, 0.75, 1, 1.25, 1.5, 2` bpw.

## Locked reference and codec

- Repository audit HEAD: `37e7bfcd168545d6a44a3e3df45b6eb27d858a3c` on `agent/mxfp4-sparse-streaming`.
- Codec publication commit: `6b31a88ee517359847f940cc74db56898a8b8425`.
- Interaction-aware allocator publication commit: `1f01edf74ce754fea1615b26a97e4465a829671f`.
- Checkpoint: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`, revision `7eceff3a9f7e6f916c824d197266d86676bce695`.
- Checkpoint `config.json` SHA-256: `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`.
- Checkpoint index SHA-256: `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`.
- Format: compressed-tensors `mxfp4-pack-quantized`, E2M1 leaves, one E8M0 scale byte per 32 weights, exact 4.25-bpw reference.
- Frozen selected-tree SHA-256: `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`.
- Historically reported resident base: 2-bit parent plus shared E8M0 scales, 2.25 bpw and 884,736 bytes per expert.

Recovery is exact sequential complete-expert proxy-weighted damage recovery relative to the embedded resident Q2 parent:

```text
recovery = 1 - D(candidate) / D(embedded Q2)
```

The evaluator uses the actual sequential SwiGLU expert and the training-fitted rank-4 future-router proxy. These rows are H0 expert-output reconstruction evidence, not task accuracy, logit agreement, routing quality, or an H4 prediction result.

## Source artifacts

| Source ID | Artifact | SHA-256 | Capture |
|---|---|---|---|
| `pr4_headline` | `results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/exact_checkpoint_confirmation/headline_rate_summary.csv` | `32c3b34f750684923210d5acb1d269a4bb7c7b8c6fc73bab43bf3dd555f61d60` | Original exact-checkpoint capture SHA-256 `84b7e727d774f6393d7e4e4cb5304c7df283b817002b48813a30ff55e6262e61` |
| `pr6_paired_hybrid` | `results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/raw/exact_checkpoint_physical_full/paired_hybrid_frontier.parquet` | `000c340fde29c3827287b903bb7bd83bf41af23f10bd51628ba95017bf85ecf8` | CPU-regenerated exact capture SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931` |
| `pr6_paired_hybrid` aggregate | `results/qwen36_mxfp4_interaction_aware_allocator_20260819_v1/paired_hybrid_frontier.parquet` | `299849fbcea968752804e8bd0db88e77b0970b3d9f59ca3e860e08eaf778459b` | Same CPU-regenerated exact capture |
| storage | `results/qwen36_full_mxfp4_embedded_hierarchy_20260818_v1/storage_accounting.json` | `88502e3b0ba2008e3ec5ecc7f3ed7c7494100ffbc91e2dc52f2830c473d0730c` | N/A |

## Method boundaries

- `pr4_deployable_ordered_diagonal_coselection` is the published deployment-oriented H0 baseline: ordered Q2→Q3→Q4 planes, activation-dependent diagonal ordering, one training-only co-selection layout, and 512-byte pages.
- `pr6_exact_training_separate` keeps separate ordered planes but uses exact fixed-coefficient full-Gram marginal selection for gate/up, diagonal down, an exact-training-mask layout, and exact sequential 9×9×9 complete-expert allocation. The representation is fixed; the selector is an H0 full-Gram oracle and is not deployable metadata.
- `pr6_exact_training_paired` uses exact Q2→Q4 coordinate packets with the same H0 full-Gram selector boundary.
- `pr6_exact_training_hybrid` chooses separate or paired packets independently per projection and rate from H0 damage. It is a representation-choice oracle and requires both suffix layouts: 4 external suffix bpw rather than 2.

At one streamed bpw on all 85 invocations, the frozen p10/median/p90 values are 0.834754/0.896134/0.960089 for PR4 ordered, 0.838531/0.897337/0.957036 for PR6 separate, 0.855259/0.920027/0.968948 for PR6 paired, and 0.872356/0.923651/0.974856 for the PR6 H0 hybrid ceiling.

## CSV accounting conventions and caveats

- Recovery percentiles are recomputed from the raw PR6 invocation rows. PR4 percentiles come directly from its published headline CSV.
- PR6 byte and amplification columns are per-invocation min/median/max or p10/median/p90 across the named cohort. Some invocations finish below the nominal budget; therefore `physical_bytes_min` can be below the median even when median physical bpw equals the requested budget.
- The PR4 headline artifact does not retain per-invocation byte columns. Its bytes are reconstructed from the published exact 1.000x packed-plane accounting: `budget_bpw * 3,145,728 / 8`. Thus PR4 byte min/median/max are identical.
- Budget-zero rows are explicit synthetic resident-only endpoints. Their recovery, byte, bpw, and amplification fields are zero.
- `resident_bpw_reported=2.25` preserves historical accounting. It excludes the learned per-expert co-selection permutation, which was neither serialized nor charged. A straightforward uint16 gate/up/down inverse map would add 9,216 bytes/expert, or 0.0234375 bpw. New codebook results must serialize and count any retained layout metadata rather than silently inheriting this omission.
- The original PR4 raw capture is not committed. Its published aggregate remains frozen evidence, while the PR6 CPU-regenerated capture and locked expert IDs are the reproducible current control.
