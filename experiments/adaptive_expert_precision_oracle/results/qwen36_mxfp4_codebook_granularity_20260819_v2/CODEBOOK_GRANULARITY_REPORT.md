# Codebook granularity study: progressive MXFP4 at approximately 2.375 resident bpw

Date: 2026-08-19  
Repository commit tested: `37e7bfcd168545d6a44a3e3df45b6eb27d858a3c`  
Branch: `agent/mxfp4-sparse-streaming`  
Result namespace: `qwen36_mxfp4_codebook_granularity_20260819_v2`  
Status: **complete: bounded design study, validation packet lock, 11-layer
promotion, exact baseline reproduction and winner serialization**

## Executive summary

This bounded H0 study asks whether 0.125 additional resident bpw can improve the
resident Q2 and first-refinement Q3 levels of an exact progressive MXFP4
hierarchy. It does. The validation-locked dense recommendation is a 16-family
codebook selected once per native 32-weight block, with tables shared by
expert-frequency cluster and projection and fitted from the complete-expert
functional proxy.

The promoted format occupies **2.375517874 logical resident bpw** across 11
layers and 33 experts; the standalone framed 12-expert pilot package occupies
**2.380167643 bpw**. On 334 broader held-out invocations, B reduces aggregate
resident Q2 damage by **29.899%** relative to the PR4 parent. Per-invocation Q2
common recovery p10/median/p90 is **-16.081/21.271/50.211%**. After one dense
refinement plane, common recovery is **55.697/75.720/87.378%**, and aggregate
same-level PR4-Q3 damage falls by **55.794%**. The gate projection remains the
largest beneficiary.

The validation-only rule locks **paired** packets by minimum median absolute
damage at one physical bpw. On the broader test cohort, paired B reaches
**88.802/93.844/97.712%** p10/median/p90 common recovery versus
**87.589/92.970/97.512%** for the matched PR4 control. Aggregate absolute damage
is **59.651895 versus 71.649602**, a **16.745%** reduction. On the sampled rate
grid, B first exceeds median 90/95/97.5% recovery at
**0.75/1.25/1.5 physical bpw**, versus **1.0/1.25/1.5** for PR4.

This is a strong resident and dense-Q3 success and a moderate selective
success. The one-bpw median recovery gain is **0.874 points**, below the stated
2--3-point target, but the 90% median crossing moves one 0.25-bpw grid step
earlier and aggregate one-bpw damage falls materially. Exact Q4 code equality
passes. The recommended H4 representation is therefore exact-MXFP4
`B16_g32_cluster_functional` with paired packets, retaining separate packets as
the ordered-plane control.

## 1. Scope and research boundary

This is an H0 codec/oracle experiment. It does not train or evaluate an H4
predictor. The deployment shape under study is:

```text
resident Q2-like weights + activation-dependent streamed refinement pages
```

The exact variants retain the two one-bit-per-weight refinement levels:

```text
Q2 --one streamed bit/weight--> Q3 --one streamed bit/weight--> exact MXFP4 Q4
```

Training requests alone determine objective weights, trees, centroids, family
assignments and Design C modifiers. Validation chooses optimizer snapshots,
epsilon tradeoffs and promoted formats. Test data is held out for reported
results.

## 2. Immutable reference and baseline

### 2.1 Authoritative checkpoint

- Repository: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`
- Revision: `7eceff3a9f7e6f916c824d197266d86676bce695`
- `config.json` SHA-256:
  `52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a`
- tensor index SHA-256:
  `842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb`
- representation: compressed-tensors `mxfp4-pack-quantized`
- leaf type: E2M1, four bits per weight
- scale type: one uint8 E8M0 exponent per native 32-weight block
- exact reference rate: 4.25 bpw
- locked PR4 tree SHA-256:
  `da4675181fd81c87bd2d3db7ff2d3dea1f075db4ea98f23ab90f14943ee2bfe8`

The checkpoint and index hashes were verified before fitting. An older mixed
GGUF on the remote volume was not used.

### 2.2 Expert geometry and progressive baseline

Each routed expert has gate and up matrices of shape 512 by 2,048 and a down
matrix of shape 2,048 by 512: 3,145,728 weights in total. The existing resident
representation contains a two-bit parent code per weight plus the original
E8M0 block scales:

| Resident component | Bytes/expert | bpw |
|---|---:|---:|
| Two-bit parent codes | 786,432 | 2.000000 |
| E8M0 scales, one byte/32 weights | 98,304 | 0.250000 |
| **Historical resident total** | **884,736** | **2.250000** |

Each dense suffix plane is 393,216 bytes/expert, or one bpw. Both suffixes
recover the original leaf through a balanced `(parent, r1, r2)` bijection.

### 2.3 Frozen latest historical selective frontier

The historical baseline is retained unchanged and is not mixed statistically
with the fresh 24-request codebook cohort. The published deployment-oriented
PR4 allocator uses ordered Q2-to-Q3-to-Q4 planes, activation-dependent diagonal
ordering, one training-only co-selection layout and 512-byte pages. On its
all-four-layer cohort (`n=85`):

| Physical streamed bpw | Physical bytes/expert | p10 recovery | Median recovery | p90 recovery |
|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 196,608 | 0.6114 | 0.7345 | 0.8633 |
| 0.75 | 294,912 | 0.7282 | 0.8353 | 0.9229 |
| 1.00 | 393,216 | 0.8348 | 0.8961 | 0.9601 |
| 1.25 | 491,520 | 0.9046 | 0.9452 | 0.9805 |
| 1.50 | 589,824 | 0.9534 | 0.9732 | 0.9945 |
| 2.00 | 786,432 | 1.0000 | 1.0000 | 1.0000 |

The original PR4 raw byte rows are unavailable. The displayed physical bytes
are reconstructed from its published exactly packed-plane accounting; the
recovery percentiles are the published aggregate evidence.

The strongest frozen H0 representation-choice ceiling at one bpw is the PR6
separate/paired hybrid: 0.8724/0.9237/0.9749 p10/median/p90. It requires both
suffix layouts, four physical suffix bpw, and oracle per-projection
representation choice; it is not the deployable control.

The full PR6 physical-page artifact was then rerun from exact-checkpoint capture
SHA-256 `52bc9eb2d726e014efef77f24e2f6eaf4401b5f7abd7dddb2d84f1a55483a931`.
The reproduced dataframe is exactly equal at shape 8,330 by 28; all three frozen
artifact hashes match and all **36/36** cohort/method/rate comparisons pass over
the 85-invocation all-four cohort and 72-invocation difficult cohort. Maximum
frozen-stat error is `4.44e-16`. At one physical bpw on all four layers:

| Reproduced PR6 arm | p10 | Median | p90 | Median damage | Physical/logical bpw |
|---|---:|---:|---:|---:|---:|
| Current diagonal separate | 82.502% | 89.150% | 95.784% | 0.090688 | 1.0 / 1.0 |
| Exact-training separate | 83.853% | 89.734% | 95.704% | 0.088770 | 1.0 / 1.0 |
| Exact-training paired | 85.526% | 92.003% | 96.895% | 0.066103 | 1.0 / 1.0 |
| Hybrid representation-choice ceiling | 87.236% | 92.365% | 97.486% | 0.053095 | 1.0 / 1.0 |

The validation record is
`baseline_reproduction/pr6_exact_checkpoint_physical_52bc/frozen_pr6_validation.json`,
SHA-256
`fe87346211f04382ab7d4044a075252460509b6bf7ee42e120dedaaae9439f97`.
The reproduced one-bpw summary is
`metrics/pr6_52bc_one_bpw_reproduction_summary.csv`, SHA-256
`8e80a3e1de73cb52fa744b66c3519136f47c1074d39920f654712f35795c6a8c`.

Historical PR4 accounting reports 2.25 resident bpw but did not serialize or
charge its co-selection inverse map. A straightforward uint16 gate/up/down map
would add 9,216 bytes/expert, or 0.0234375 bpw. All new codebook metadata is
charged from actual binary representations.

## 3. Pilot/promotion cohorts, hardware and evaluator

The primary capture is
`qwen36_mxfp4_md_fresh_capture.npz`, SHA-256
`75087507e3df27735fc202ab02ea0b891daabcd4461290e58e5e55d4cd0d2698`.
It covers layers 0, 4, 20 and 39 and selects one cold, median and hot expert per
layer subject to complete split coverage. The request split is 12 train, six
validation and six test requests. After routed positions and per-expert caps,
the complete-expert dense evaluator contains 258/135/152
train/validation/test invocations.

The promoted capture is
`qwen36_exact_broad_11layer_captures.npz`, SHA-256
`da786a99209f03515e68bc21f1849f8613dbe5251d4eebff7e3537e157a0d56f`.
It adds layers 8, 12, 16, 24, 28, 32 and 36 for 11 total layers, retains one
cold/median/hot expert per layer (33 experts), and evaluates 534/315/334
train/validation/test complete-expert invocations. Only validation locks the
packet; the 334 test invocations report the promoted result.

The remote host has two RTX 3090 GPUs with 24 GiB each and 1 TiB host RAM. This
is sufficient for the sampled compressed-tensor codec, fitting and evaluator.
It is not sufficient to hold the roughly 84-GiB fully dequantized model on one
GPU. No result here is a full-model task-accuracy measurement.

For an invocation, the exact expert reference is

```text
g = Gx
u = Ux
h = SiLU(g) * u
y = Dh
```

and complete damage is

```text
D(e) = ||e||_2^2 + beta ||e P||_2^2,
```

where `P` and `beta` are the training-fitted rank-4 future-router proxy. Dense
evaluation replaces gate, up and down at the same hierarchy level and executes
the actual sequential SwiGLU expert. Projection errors are also recorded, but
projection damage is not added to complete-expert damage.

## 4. Codec designs

### 4.1 Exact progressive book

One exact book is a 32-byte binary object:

- eight bytes packing the 16 leaf-to-`(parent,r1,r2)` mappings;
- four FP16 Q2 centroids;
- eight FP16 Q3 centroids.

Every parent owns four exact leaves, and the 16 triples form a bijection. Q2
and Q3 use the serialized FP16 centroids. Q4 ignores those centroids and decodes
the original E2M1 leaf. The scored representation is the rounded serialized
representation, not an unrounded fitting table.

### 4.2 Compared formats

| Design | Resident specialization | Selector stream | Additional resident data |
|---|---|---:|---|
| A | Independent hierarchy for every 2,048-weight gate row, up row and down column | none | 1,536 books/expert, 49,152 bytes |
| B | Shared family selected per native 32-weight block | 4 bits/block | 16-book family plus selector |
| C | Eight-book family selected per 32-weight block | 3 bits/block | four FP16 affine values per functional vector: Q2 scale/offset and Q3 scale/offset |
| D | Four-book family selected per 16 weights | 2 bits/group | selector plus shared family |
| E | 16, 32 or 64 trained books selected per 64 weights | 8 bits/group | selector plus shared family |

Design B tests global/projection, layer/projection,
expert-frequency-cluster/projection and expert/projection sharing. It also
compares raw weight MSE, activation/projection weighting and the complete-expert
functional proxy. Design A includes shrinkage toward the layer/projection leaf
mass prior, with `alpha = 8 / (training_invocations + 8)`. Design C clips its
fitted normalized affine scales to `[0.5,1.5]` and offsets to `[-1,1]`, then
stores the four values as FP16.

### 4.3 Learned-leaf diagnostic

`learned_16_leaf_scalar_global` fits an unbalanced 4-by-2-by-2 scalar tree and
learns all 16 Q4 terminal values. Its book is 64 bytes. It is a global,
one-family diagnostic, not a capacity-matched learned-leaf B arm, and it is not
exact MXFP4.

## 5. Fitting and validation protocol

Three training-only diagonal objectives are implemented:

1. raw weight MSE;
2. activation/projection weighting from mean squared projection inputs;
3. a complete-expert functional proxy using the local SiLU derivatives, up/gate
   interactions, down weights and rank-4 output metric.

All leaf masses include the squared E8M0 scale. Family fitting alternates:

1. refit each balanced hierarchy from the training leaf masses assigned to it;
2. reassign every group by lexicographic Q2-first/Q3-second distortion;
3. score the aligned validation masses;
4. stop on unchanged assignments or two stale iterations, up to six configured
   iterations.

The balanced tree search uses deterministic fixed and seeded leaf orders. The
Q2 tolerance sweep is 0, 0.1%, 0.25%, 0.5%, 1%, 2% and 5%. Within an epsilon,
Q3 is minimized only among trees within the Q2 band. Each epsilon arm is a fresh
non-convex fit with a candidate-specific seed, so tiny non-monotonic differences
are not interpreted as a causal epsilon effect.

Final design promotion is also lexicographic: minimize summed validation Q2,
retain candidates within 0.5%, then minimize summed validation Q3. Test metrics
do not select the primary format.

## 6. Metric definitions

Two recovery denominators are retained deliberately:

- **common recovery:**
  `1 - D(candidate level) / D(PR4 resident Q2)` for the same invocation;
- **own-Q2 recovery (`rho3`):**
  `1 - D(candidate Q3) / D(candidate Q2)`.

Common recovery is required for cross-format resident comparisons. Own-Q2
recovery measures refinement efficiency, but it can be large simply because a
candidate starts from a poor Q2. “Aggregate” values below are ratios of summed
damage. p10/median/p90 are percentiles of per-invocation ratios and therefore
need not rank candidates in the same order as summed damage.

## 7. Serialized resident-rate accounting

The per-expert metadata files use a fixed 64-byte `CGBMXF4` header and 16-byte
projection alignment. The payloads already align, so measured alignment padding
is zero. Research NPZ files and JSON manifests are excluded; tables, selectors,
modifiers and metadata headers are included. The first table below is the
model-amortized design comparison. It keeps the historical two-bit parent and
E8M0 streams at their logical raw sizes; the standalone pilot bundle accounting
that follows additionally charges each projection's `CGWMXF4` header and
512-byte stream alignment.

| Candidate | Selector bytes | Modifier bytes | Header/padding bytes | Amortized tables bytes/expert | Metadata bytes/expert | Metadata bpw | Total resident bpw |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR4 parent | 0 | 0 | 0 | 0.009 | 0.009 | 0.000000 | 2.250000 |
| A vector | 0 | 0 | 0 | 49,152.000 | 49,152.000 | 0.125000 | 2.375000 |
| **B16/g32 cluster** | **49,152** | 0 | 64 | 0.450 | **49,216.450** | **0.125164** | **2.375164** |
| B16/g32 expert diagnostic | 49,152 | 0 | 64 | 1,536.000 | 50,752.000 | 0.129069 | 2.379069 |
| B16/g32 layer | 49,152 | 0 | 64 | 6.000 | 49,222.000 | 0.125178 | 2.375178 |
| C8/g32 layer | 36,864 | 12,288 | 64 | 3.000 | 49,219.000 | 0.125170 | 2.375170 |
| D4/g16 layer | 49,152 | 0 | 64 | 1.500 | 49,217.500 | 0.125167 | 2.375167 |
| E64/g64 layer | 49,152 | 0 | 64 | 24.000 | 49,240.000 | 0.125224 | 2.375224 |

The selected dense B format has 933,952.45 resident bytes/expert. Its complete
three-cluster-by-three-projection-by-16-book table set is 4,608 bytes model-wide;
the per-expert selector/header file dominates. The expert-specific B diagnostic
still fits the requested `2.375 +/- 0.01` band, but it missed the locked 0.5%
validation Q2 tolerance.

For selected B, one complete logical dense refinement plane is 393,216
bytes/expert and one bpw. Its dense effective rates are therefore 2.375164 bpw
at Q2, 3.375164 bpw at Q3 and 4.375164 bpw at exact Q4. Under selective
streaming, total effective bpw is 2.375164 plus the actually charged physical
streamed bpw; logical suffix bpw and page amplification are reported
separately.

The retained 12-expert deployment bundle provides the stricter actual-byte
accounting:

| Candidate | Weights | Framed parent + E8M0 resident bytes | Resident metadata bytes | Total resident bytes | Resident bpw | Q3 bytes/bpw | Q4 bytes/bpw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Framed PR4 control | 37,748,736 | 10,635,264 | 384 | 10,635,648 | 2.253987630 | 4,718,592 / 1.0 | 4,718,592 / 1.0 |
| **B16/g32 cluster** | **37,748,736** | **10,635,264** | **595,776** | **11,231,040** | **2.380167643** | **4,718,592 / 1.0** | **4,718,592 / 1.0** |

B's actual metadata consists of twelve 49,216-byte expert selector/header files
(590,592 bytes) and nine 576-byte codebook-table files (5,184 bytes, including
their 64-byte headers). The 36 projection files add 18,432 bytes of weight-stream
header/alignment overhead relative to raw 2.25-bpw parent-plus-scale packing.
Thus the bounded package is 0.005168 bpw above the 2.375 target, still inside the
requested `2.375 +/- 0.01` comparison band. Its exact-Q4 total effective rate is
4.380167643 bpw; at one physical streamed bpw it is 3.380167643 bpw.

The promoted 11-layer logical accounting spans 103,809,024 weights in 33
experts. Its 33 expert metadata files occupy 1,624,128 bytes and its nine shared
tables occupy 4,608 bytes, for 1,628,736 total metadata bytes or
**0.125517874 bpw**. Including logical Q2 parents and E8M0 scales gives
**2.375517874 resident bpw**. The mode-matched PR4 control is 2.250007398 bpw
after charging its 96 table bytes. This broad logical rate remains within the
requested band; the separately retained 12-expert bundle above is the strict
standalone framed-file accounting. Each broad suffix plane is 12,976,128 bytes,
exactly one bpw, so the promoted effective rate is 3.375517874 bpw at one dense
plane and 4.375517874 bpw at exact Q4.

## 8. Dense results

### 8.1 Validation-locked winners

| Design winner | Validation D2 | Validation D3 | Test D2 | Test D3 | Test aggregate common Q2 | Test aggregate common Q3 | Test own rho3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A vector | 1294.179 | 248.573 | 1192.595 | 280.920 | 3.14% | 77.19% | 76.44% |
| **B16/g32 cluster** | **849.935** | 212.382 | 771.303 | **221.998** | 37.36% | **81.97%** | 71.22% |
| C8/g32 hybrid | 904.930 | **208.490** | 857.916 | 238.087 | 30.32% | 80.66% | 72.25% |
| D4/g16 | 969.423 | 232.804 | 781.676 | 252.573 | 36.52% | 79.49% | 67.69% |
| E64/g64 | 915.378 | 219.515 | **749.186** | 240.666 | **39.15%** | 80.45% | 67.88% |
| PR4 parent | 1359.763 | 420.803 | 1231.294 | 435.516 | 0.00% | 64.63% | 64.63% |
| Learned-16-leaf diagnostic | 1448.636 | 308.060 | 1398.787 | 322.496 | -13.60% | 73.81% | 76.94% |

B is the global lexicographic winner. C has the smallest validation Q3 among
the five per-design winners but is 6.47% worse than B on validation Q2. E64 has
the lowest test Q2 sum, but promoting it for that would be test selection; it
was 7.70% worse than B on validation Q2.

### 8.2 Held-out complete-expert distribution

| Candidate | Q2 common recovery p10/median/p90 | Q3 common recovery p10/median/p90 | Q2 damage p10/median/p90 | Q3 damage p10/median/p90 |
|---|---|---|---|---|
| A vector | -13.99 / 11.31 / 51.38% | 55.12 / 76.98 / 87.11% | 0.0519 / 0.2740 / 19.7030 | 0.0128 / 0.0776 / 5.4465 |
| **B16/g32 cluster** | **-7.46 / 30.88 / 64.51%** | **62.07 / 77.93 / 87.90%** | **0.0378 / 0.2350 / 16.4823** | **0.0119 / 0.0899 / 4.8819** |
| C8/g32 hybrid | -15.74 / 23.69 / 56.99% | 61.56 / 77.47 / 88.93% | 0.0415 / 0.2464 / 18.1564 | 0.0136 / 0.0802 / 5.4030 |
| D4/g16 | -13.53 / 30.55 / 58.84% | 56.15 / 77.06 / 86.40% | 0.0354 / 0.2654 / 14.5783 | 0.0123 / 0.0909 / 5.8465 |
| E64/g64 | -12.27 / 24.58 / 57.38% | 57.01 / 78.61 / 88.34% | 0.0400 / 0.2407 / 17.7492 | 0.0133 / 0.0771 / 5.4811 |
| PR4 parent | 0 / 0 / 0% | 37.84 / 64.35 / 77.12% | 0.0517 / 0.3977 / 21.7292 | 0.0187 / 0.1348 / 8.0980 |

Selected B lowers median complete-expert relative output error from
0.5498/0.3338 at PR4 Q2/Q3 to 0.4853/0.2729. Its negative Q2 p10 means the
resident hierarchy is materially better in aggregate and at the median but is
not uniformly better per invocation.

### 8.3 Projection and gate effect

| Projection | PR4 D2 sum | B D2 sum | B common Q2 | PR4 D3 sum | B D3 sum | B common Q3 | B own rho3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate | 15295.106 | 5629.693 | **63.19%** | 7109.426 | 908.730 | **94.06%** | **83.86%** |
| Up | 2872.192 | 2102.840 | 26.79% | 701.952 | 610.256 | 78.75% | 70.98% |
| Down | 603.136 | 323.497 | 46.36% | 181.733 | 108.277 | 82.05% | 66.53% |

Gate is the central positive result. The old PR4 first plane removes 53.52% of
summed gate Q2 damage; B removes 83.86% of its own gate Q2 damage and leaves
5.94% of the old PR4 gate-Q2 damage at Q3. B also improves resident up and down,
unlike A, whose down Q2 sum is 3.38% worse than PR4.

### 8.4 Objective and sharing ablations

Global B16/g32 objective comparison:

| Objective | Validation D2 | Validation D3 | Test D2 | Test D3 | Test median common Q2 / Q3 |
|---|---:|---:|---:|---:|---:|
| Weight MSE | 1212.747 | 224.996 | 1023.009 | 237.006 | 16.09 / 77.58% |
| Activation/projection | 939.505 | **211.535** | 776.954 | 252.209 | **33.32 / 76.84%** |
| Complete-expert functional | **870.018** | 213.589 | **760.284** | 240.962 | 29.93 / 76.86% |

Raw MSE is not competitive at Q2. Functional fitting is the validation resident
winner and gives better test D2 and D3 sums than activation weighting.

Functional B16/g32 sharing comparison:

| Sharing | Resident bpw | Validation D2 | Validation D3 | Test D2 | Test D3 | Test median common Q2 / Q3 |
|---|---:|---:|---:|---:|---:|---:|
| Global/projection | 2.375163 | 870.018 | 213.589 | 760.284 | 240.962 | 29.93 / 76.86% |
| Layer/projection | 2.375178 | 956.811 | 212.877 | 868.486 | 226.358 | 25.76 / 78.62% |
| **Cluster/projection** | **2.375164** | **849.935** | 212.382 | 771.303 | 221.998 | 30.88 / 77.93% |
| Expert/projection diagnostic | 2.379069 | 857.968 | **195.949** | **733.499** | **207.422** | 29.71 / **80.94%** |

The expert diagnostic misses the 0.5% Q2 band by a small amount: validation Q2
is 0.95% above cluster B, while Q3 is 7.74% lower. It is a useful secondary arm
if a future experiment explicitly permits about a 1% resident sacrifice.

### 8.5 A/C/D/E, granularity and family count

| Layer-shared format | Validation D2 | Validation D3 | Test D2 | Test D3 | Test own rho3 |
|---|---:|---:|---:|---:|---:|
| B16/g32 | 956.811 | 212.877 | 868.486 | **226.358** | **73.94%** |
| **C8/g32 + vector affine** | **904.930** | 208.490 | 857.916 | 238.087 | 72.25% |
| D4/g16 | 969.423 | 232.804 | 781.676 | 252.573 | 67.69% |
| E16/g64 | 955.303 | **207.375** | 827.729 | 246.564 | 70.21% |
| E32/g64 | 989.277 | 221.968 | 846.114 | 241.396 | 71.47% |
| E64/g64 | 915.378 | 219.515 | **749.186** | 240.666 | 67.88% |

C is the best balanced layer-shared validation arm, but changing the family
count and adding affine modifiers in the same arm prevents an isolated causal
estimate of the modifier. D's 16-weight locality helps test Q2 but is worse than
native-32 B on validation Q2 and both validation/test Q3. Native 32 is therefore
preferred for the first streamed bit and for MXFP4 block alignment.

At G64, E64 improves resident Q2, but E16 has the best validation Q3 and nearly
the same test median common Q3. E32 is non-monotonically worse. More than 16
families can buy resident specialization, but Q3 appears saturated near 16. A
256-family fit was not run because the bounded sample did not support it.

Design A uses its 49,152 bytes poorly: its test Q2 sum is only 3.14% below PR4
and its Q3 sum is 26.54% above B. Shrinkage alpha spans 0.143--0.727 and does not
improve held-out totals. This is not clean train-set memorization—A is already
0.08% worse than PR4 on training Q2—but it is unsupported or objective-misaligned
capacity and should not be scaled.

### 8.6 Q2 tolerance

| Epsilon | Validation D2 | Validation D3 | Validation own rho3 | Test D2 | Test D3 | Test own rho3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 914.694 | 216.801 | 76.30% | 826.827 | 232.594 | 71.87% |
| 0.1% | 930.713 | 221.238 | 76.23% | 758.741 | 228.000 | 69.95% |
| 0.25% | 861.298 | 210.968 | 75.51% | **699.110** | 238.319 | 65.91% |
| 0.5% | 870.018 | 213.589 | 75.45% | 760.284 | 240.962 | 68.31% |
| 1% | **857.738** | 207.220 | 75.84% | 767.510 | 235.471 | 69.32% |
| 2% | 917.737 | 214.139 | 76.67% | 774.309 | 242.280 | 68.71% |
| 5% | 988.032 | **202.378** | **79.52%** | 786.529 | **207.000** | **73.68%** |

The 5% arm establishes that a large allowed Q2 sacrifice can strengthen Q3. The
sub-percent fits are noisy and non-monotonic; they do not establish that a
0.1--0.5% sacrifice reliably buys Q3. The locked 0.5% rule remains appropriate.

### 8.7 Exact endpoint and learned-Q4 diagnostic

Every exact candidate has code equality and zero Q4 damage on every one of its
545 dense rows across all splits. The learned-leaf diagnostic is worse at Q2
than PR4 but improves the global Q3 median from 64.35% to 69.96% common recovery.
It remains behind exact B at Q3: test summed damage is 322.496 versus 221.998.
Its inexact test Q4 has 116.853 summed damage, 83.83/91.49/95.57% common-recovery
p10/median/p90 and median relative output error 0.1734.

This unmatched diagnostic does not show that exact leaf preservation materially
limits the best hierarchy. A capacity-matched learned-leaf B16/g32 ablation
would be needed to isolate that constraint.

### 8.8 Generalization and family diagnostics

Selected B aggregate common recovery is 33.51/81.93% on train Q2/Q3,
37.49/84.38% on validation and 37.36/81.97% on test. Its Q2 improvement is
stable across splits. Layer 39 dominates held-out absolute damage: 743.114 of
771.303 Q2 damage and 213.315 of 221.998 Q3 damage. Layer-39 common-recovery
p10/median/p90 is -15.04/26.61/50.82% at Q2 and
61.45/76.62/89.75% at Q3. The first dense plane makes the tail much more
uniform, but resident Q2 remains vulnerable.

The saved family-utilization/predictability diagnostic is the global B16/g32
functional epsilon-1% arm, not cluster B. All 16 IDs are used in every one of
36 expert/projection groups. Effective family count has median 13.16, minimum
5.09 and maximum 15.51. A nearest-centroid classifier from simple leaf/block
statistics predicts held-out IDs at 19.63% versus 15.75% majority accuracy.
Family IDs have some statistical structure but are not well replaced by a
simple deterministic feature rule.

### 8.9 Promoted 11-layer dense result

The pilot-selected B format was refit and evaluated over 11 layers without
reopening the A--E design choice. On 334 test invocations:

| Level | B median damage | PR4 median damage | B/PR4 aggregate damage reduction | Median paired damage reduction | Common recovery p10/median/p90 | B/PR4 median relative output error |
|---|---:|---:|---:|---:|---|---|
| Q2 | 0.778186 | 0.966594 | **29.899%** | **21.271%** | **-16.081 / 21.271 / 50.211%** | 0.514677 / 0.565954 |
| Q3 | 0.246748 | 0.307562 | **55.794%** | **26.866%** | **55.697 / 75.720 / 87.378%** | 0.296981 / 0.335480 |
| Q4 | 0 | 0 | exact endpoint | exact endpoint | 100 / 100 / 100% | 0 / 0 |

“Aggregate damage reduction” is the ratio of summed B and same-level PR4
damage. “Median paired damage reduction” is the median of per-invocation
same-level reductions. “Common recovery” always uses the matched PR4 Q2 damage
as denominator; these quantities are intentionally not interchanged.

The median per-invocation same-level B reductions on train/validation/test are
27.382/17.653/21.271% at Q2 and 32.726/27.163/26.866% at Q3. The benefit
generalizes beyond the fit split, with a weaker but still positive resident
validation effect.

Projection damage confirms that gate remains the principal win:

| Projection | Q2 mean damage B / PR4 | Q2 reduction | Q3 mean damage B / PR4 | Q3 reduction | B own-Q2 Q3 median recovery |
|---|---:|---:|---:|---:|---:|
| Gate | 20.663 / 39.863 | **48.16%** | 5.035 / 20.849 | **75.85%** | **71.18%** |
| Up | 14.473 / 18.874 | 23.32% | 4.374 / 4.915 | 11.00% | 69.75% |
| Down | 1.740 / 2.544 | 31.58% | 0.520 / 0.576 | 9.66% | 67.29% |

The promoted endpoint is exact MXFP4: every saved Q4 dense row has zero damage
and the codec retains the exact leaf mapping.

## 9. Selective streaming method

The current H0 selective evaluator supports both representations requested by
the study:

- **separate:** ordered Q2-to-Q3 and Q3-to-Q4 actions in distinct planes;
- **paired:** exact Q2-to-Q4 coordinate packets carrying both suffix bits.

For gate and up, action order uses exact fixed-coefficient full-Gram marginal
selection. Down uses the proxy-weighted diagonal order. For each projection and
packet type, candidate rates are 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75 and
2.0 bpw. Suffixes are mapped to natural 512-byte pages. Pages are closed to all
contained actions and charged physically. The evaluator then exhaustively
rescales the 9-by-9-by-9 gate/up/down rate grid, executes the complete nonlinear
expert, and selects the lowest-damage combination satisfying each physical
budget in 0.25, 0.5, 0.75, 1.0, 1.25, 1.5 and 2.0 bpw.

This is an activation-dependent H0 oracle. It establishes representation and
allocator ceilings; it is not a deployable H4 selector.

<!-- BEGIN SELECTIVE_RESULTS -->

## 10. Selective streaming results

The bounded-pilot merge contains 19,152 complete selective rows and 57,456
saved projection action rows. Every pilot B row below has 152 held-out expert
invocations. Candidate selection remains locked to
`B16_g32_cluster_functional` from validation dense Q2; the two packet modes are
reported independently before the broader promotion result.

### 10.1 Pilot validation-locked B frontier

| Packet | Physical bpw | Common recovery p10/median/p90 | Own-Q2 recovery p10/median/p90 | Absolute damage p10/median/p90 | Logical bpw | Page amp. |
|---|---:|---|---|---|---:|---:|
| Paired | 0.25 | 61.716 / 75.604 / 88.086% | 48.616 / 65.209 / 78.243% | 0.011109 / 0.086249 / 5.842689 | 0.25 | 1.000 |
| Paired | 0.50 | 77.697 / 88.105 / 94.668% | 70.790 / 82.241 / 90.416% | 0.004803 / 0.048978 / 3.493343 | 0.50 | 1.000 |
| Paired | 0.75 | 87.266 / 92.754 / 96.622% | 80.027 / 88.787 / 94.910% | 0.002892 / 0.028955 / 2.010220 | 0.75 | 1.000 |
| **Paired** | **1.00** | **90.910 / 95.606 / 97.875%** | **86.849 / 93.224 / 96.532%** | **0.002014 / 0.017178 / 1.118354** | **1.00** | **1.000** |
| Paired | 1.25 | 94.234 / 97.287 / 98.750% | 92.061 / 96.071 / 98.055% | 0.001133 / 0.011188 / 0.669657 | 1.25 | 1.000 |
| Paired | 1.50 | 96.659 / 98.560 / 99.775% | 95.285 / 98.040 / 99.686% | 0.000136 / 0.006887 / 0.345253 | 1.50 | 1.000 |
| Paired | 2.00 | 100.000 / 100.000 / 100.000% | 100.000 / 100.000 / 100.000% | 2.52e-14 / 2.43e-13 / 1.55e-11 | 2.00 | 1.000 |
| Separate | 0.25 | 55.340 / 71.700 / 84.885% | 44.174 / 57.967 / 71.994% | 0.013738 / 0.100707 / 6.934038 | 0.25 | 1.000 |
| Separate | 0.50 | 72.825 / 84.530 / 92.706% | 63.202 / 76.544 / 86.720% | 0.006935 / 0.061333 / 3.901236 | 0.50 | 1.000 |
| Separate | 0.75 | 82.069 / 90.280 / 95.864% | 75.321 / 85.667 / 92.331% | 0.004182 / 0.037921 / 2.688107 | 0.75 | 1.000 |
| **Separate** | **1.00** | **88.019 / 93.396 / 96.935%** | **83.126 / 90.788 / 95.178%** | **0.002555 / 0.025382 / 1.495198** | **1.00** | **1.000** |
| Separate | 1.25 | 92.737 / 96.150 / 98.186% | 88.281 / 94.657 / 97.060% | 0.001647 / 0.014562 / 0.926246 | 1.25 | 1.000 |
| Separate | 1.50 | 95.745 / 97.819 / 99.056% | 93.530 / 96.788 / 98.514% | 0.000800 / 0.008991 / 0.505521 | 1.50 | 1.000 |
| Separate | 2.00 | 100.000 / 100.000 / 100.000% | 100.000 / 100.000 / 100.000% | 2.56e-14 / 2.43e-13 / 1.55e-11 | 2.00 | 1.000 |

Physical and logical medians coincide at every retained budget and page
amplification is 1.0. This is not an assumption: the allocator's natural
coordinate chunks close exactly onto the charged 512-byte pages in these
summaries. At two bpw the tiny nonzero output damage is floating-point evaluator
residual; the serialized code and packed-leaf equality test is exact.

### 10.2 Pilot projection bandwidth and upgrades

| Packet | Bpw | Gate/up/down bytes | Gate/up/down Q2-to-Q3 upgrades | Gate/up/down Q3-to-Q4 upgrades |
|---|---:|---|---|---|
| Paired | 0.25 | 32,768 / 32,768 / 32,768 | 256 / 256 / 64 | 256 / 256 / 64 |
| Paired | 0.50 | 65,536 / 65,536 / 65,536 | 512 / 512 / 128 | 512 / 512 / 128 |
| Paired | 0.75 | 98,304 / 98,304 / 98,304 | 768 / 768 / 192 | 768 / 768 / 192 |
| Paired | 1.00 | 131,072 / 131,072 / 98,304 | 1,024 / 1,024 / 192 | 1,024 / 1,024 / 192 |
| Paired | 1.25 | 163,840 / 196,608 / 131,072 | 1,280 / 1,536 / 256 | 1,280 / 1,536 / 256 |
| Paired | 1.50 | 229,376 / 262,144 / 131,072 | 1,792 / 2,048 / 256 | 1,792 / 2,048 / 256 |
| Paired | 2.00 | 262,144 / 262,144 / 262,144 | 2,048 / 2,048 / 512 | 2,048 / 2,048 / 512 |
| Separate | 0.25 | 32,768 / 32,768 / 32,768 | 436 / 464 / 92 | 48 / 40 / 44 |
| Separate | 0.50 | 65,536 / 65,536 / 65,536 | 840 / 848 / 180 | 128 / 124 / 108 |
| Separate | 0.75 | 98,304 / 98,304 / 98,304 | 1,172 / 1,244 / 243 | 264 / 288 / 149 |
| Separate | 1.00 | 131,072 / 131,072 / 131,072 | 1,492 / 1,552 / 283 | 488 / 536 / 196 |
| Separate | 1.25 | 163,840 / 196,608 / 131,072 | 1,736 / 1,900 / 309 | 772 / 1,108 / 225 |
| Separate | 1.50 | 229,376 / 229,376 / 131,072 | 2,008 / 2,016 / 316 | 1,568 / 1,576 / 246 |
| Separate | 2.00 | 262,144 / 262,144 / 262,144 | 2,048 / 2,048 / 512 | 2,048 / 2,048 / 512 |

These are independently computed component medians; their sum need not equal
the median total bytes. Paired packets upgrade the same coordinate count at both
levels by construction. The one-bpw paired median allocates less to down than
gate/up, consistent with gate/up dominating functional damage.

### 10.3 Pilot one-bpw design comparison

| Candidate | Packet | p10 recovery | Median recovery | p90 recovery | Median absolute damage |
|---|---|---:|---:|---:|---:|
| A vector | Paired | 90.372% | 94.652% | 97.558% | 0.020450 |
| **B16/g32 cluster** | **Paired** | **90.910%** | **95.606%** | **97.875%** | **0.017178** |
| C8/g32 hybrid | Paired | **91.384%** | 95.602% | 97.864% | 0.017492 |
| D4/g16 | Paired | 91.274% | **95.611%** | 97.751% | 0.018928 |
| E64/g64 | Paired | 91.034% | 95.182% | **97.921%** | 0.020296 |
| Fresh PR4 control | Paired | 89.538% | 94.624% | 97.332% | 0.022092 |
| A vector | Separate | 87.512% | 93.155% | 96.585% | 0.025853 |
| **B16/g32 cluster** | **Separate** | 88.019% | 93.396% | 96.935% | 0.025382 |
| C8/g32 hybrid | Separate | **89.093%** | 93.319% | 97.121% | 0.025829 |
| D4/g16 | Separate | 87.425% | **93.951%** | **97.236%** | **0.023097** |
| E64/g64 | Separate | 87.943% | 93.349% | 97.038% | 0.024525 |
| Fresh PR4 control | Separate | 83.969% | 91.584% | 95.886% | 0.033728 |

The held-out table does not override the validation lock. D's paired median edge
over B is only 0.000052 recovery (0.0052 percentage points), and D has worse
paired median absolute damage. C has the best p10 in both packet modes. Thus no
single test-selected alternative dominates B, and the dense-Q3 ordering does
not strictly determine the selective ordering.

Against the mode-matched fresh PR4 control at one bpw, paired B improves
p10/median/p90 by **1.372/0.982/0.543 points** and lowers absolute-damage
p10/median/p90 by **12.49/22.24/25.99%**. Separate B improves those recoveries
by **4.050/1.812/1.048 points** and lowers damage by
**25.43/24.75/35.62%**. The frozen historical ordered-plane PR4 row is a
different cohort; descriptively, separate B is 4.54/3.79/0.93 recovery points
higher at one bpw, but that difference is not a paired treatment estimate.

### 10.4 Pilot interpolated recovery-rate crossings

All crossings use linear interpolation of the saved physical-bpw frontier.
Values in parentheses are the relative physical-rate reduction from the
mode-matched fresh PR4 control.

| Paired statistic | B 90% / control | B 95% / control | B 97.5% / control |
|---|---:|---:|---:|
| p10 | 0.938 / 1.030 (**9.01%**) | 1.329 / 1.395 (**4.75%**) | 1.626 / 1.671 (**2.70%**) |
| **Median** | **0.602 / 0.692 (13.06%)** | **0.947 / 1.047 (9.58%)** | **1.292 / 1.371 (5.76%)** |
| p90 | 0.323 / 0.431 (**25.13%**) | 0.542 / 0.684 (**20.68%**) | 0.925 / 1.035 (**10.63%**) |

| Separate statistic | B 90% / control | B 95% / control | B 97.5% / control |
|---|---:|---:|---:|
| p10 | 1.105 / 1.232 (**10.30%**) | 1.438 / 1.545 (**6.94%**) | 1.706 / 1.773 (**3.75%**) |
| **Median** | **0.738 / 0.907 (18.63%)** | **1.146 / 1.237 (7.36%)** | **1.452 / 1.497 (3.02%)** |
| p90 | 0.413 / 0.522 (**20.79%**) | 0.682 / 0.899 (**24.19%**) | 1.113 / 1.230 (**9.51%**) |

The requested 15--25% median rate reduction is met only for separate packets at
the 90% crossing (18.63%), not broadly at 95% or 97.5%. Paired is the stronger
absolute frontier and reaches 95% median below one physical bpw, but its matched
rate reduction is 9.58%.

### 10.5 Pilot validation-only packet lock

Packet selection was repeated on the validation split only after B had already
been locked by the dense validation rule. The selection rule was minimum median
complete-expert absolute damage at one physical bpw; test recovery was not
available to the rule.

| Validation candidate | Packet | n | Recovery p10/median/p90 | Median absolute damage | Median physical/logical bpw |
|---|---|---:|---|---:|---:|
| **B16/g32 cluster** | **Paired** | **135** | **87.219 / 93.116 / 96.996%** | **0.0191889** | **1.0 / 1.0** |
| B16/g32 cluster | Separate | 135 | 84.835 / 90.110 / 95.435% | 0.0239746 | 1.0 / 1.0 |
| Fresh PR4 control | Paired | 135 | 88.867 / 94.390 / 97.994% | 0.0196436 | 1.0 / 1.0 |

Paired lowers B's median validation damage by 19.96% relative to separate and
is therefore the locked packet. B also has 2.32% lower median absolute damage
than the paired PR4 control even though its median recovery ratio is lower;
this is possible because recovery is a per-invocation normalized ratio and the
median of ratios need not rank like median absolute damage. The lock deliberately
uses the latter objective.

### 10.6 Promoted 11-layer selective frontier

The broader result uses only common-baseline recovery: every candidate is
normalized by the matched PR4 Q2 damage for that invocation. It does not use
the candidate-own-Q2 recovery stored in the raw promotion output.

| Physical bpw | B paired common recovery p10/median/p90 | PR4 paired p10/median/p90 | B aggregate damage | PR4 aggregate damage |
|---:|---|---|---:|---:|
| 0.25 | 51.781 / 70.128 / 85.763% | 47.938 / 62.896 / 81.732% | 372.178804 | 486.068305 |
| 0.50 | 72.087 / 84.107 / 93.343% | 71.018 / 80.752 / 92.414% | 178.399667 | 209.434081 |
| 0.75 | 82.930 / 90.345 / 96.561% | 81.394 / 88.810 / 95.950% | 96.390791 | 118.879649 |
| **1.00** | **88.802 / 93.844 / 97.712%** | **87.589 / 92.970 / 97.512%** | **59.651895** | **71.649602** |
| 1.25 | 92.880 / 96.082 / 98.613% | 92.491 / 95.626 / 98.507% | 35.440429 | 42.459368 |
| 1.50 | 96.122 / 97.926 / 99.448% | 95.605 / 97.586 / 99.382% | 18.586895 | 21.548145 |
| 2.00 | 100.000 / 100.000 / 100.000% | 100.000 / 100.000 / 100.000% | 1.02e-9 | 1.04e-9 |

At one bpw the promoted B p10/median/p90 gains are
**1.213/0.874/0.200 recovery points** and aggregate damage falls by
**16.745%**. Median physical bpw is exactly the requested grid point at every
row; the small difference in mean physical bpw comes from requests whose
available action sets saturate early. The exact two-bpw residual is numerical,
not a code mismatch.

For the ordered-plane control at one bpw, B separate reaches
84.838/91.536/96.604% versus PR4 separate at 82.120/89.208/95.637%; aggregate
damage is 81.403778 versus 115.870213, a 29.746% reduction. Paired remains the
locked representation because its absolute B damage is lower: 59.651895 versus
81.403778 aggregate and 0.069038 versus 0.089762 median.

The promoted sampled-grid median crossings are:

| Candidate/packet | 90% | 95% | 97.5% |
|---|---:|---:|---:|
| **B paired** | **0.75** | **1.25** | **1.50** |
| PR4 paired | 1.00 | 1.25 | 1.50 |

These are first sampled budgets, not interpolated crossings. Thus B saves one
0.25-bpw step (25%) at the 90% median target and ties PR4 at 95% and 97.5%.
Section 10.4's smoother 0.602/0.947/1.292 values are pilot interpolations on a
different cohort and must not be substituted for the promoted sampled-grid
answer.

<!-- END SELECTIVE_RESULTS -->

## 11. Decode and compute accounting

All exact designs can decode with one reconstruction-table lookup per weight if
the loader expands the inverse Q4 map once. The table-base cadence and extra
work differ:

| Design | Maximum table-base selections/weight | Selector handling | Main extra work |
|---|---:|---|---|
| PR4 | one per 1,048,576 | none | projection-global base |
| A | one per 2,048 | none | advance per functional vector |
| B | one per 32 | packed 4-bit ID | nibble extraction and family stride |
| C | one per 32 | packed 3-bit ID | straddling extraction plus affine FMA |
| D | one per 16 | packed 2-bit ID | twice B's table-base cadence |
| E | one per 64 | direct uint8 ID | family stride; two native scales/group |

B is the cleanest local-family fusion target because the selector, E8M0 scale
and 32 codes share a native decode unit. C is feasible but adds one scale/offset
application at Q2/Q3. D introduces a half-block control boundary. E has cheaper
selector unpack but spans two independently scaled native blocks.

Under the existing two-useful-FLOP convention and assuming all added metadata is
read once, 0.125 resident bpw reduces estimated useful arithmetic intensity at
one streamed bpw from 9.8462 to approximately 9.481 FLOP/byte, a 3.7% decrease.
The points remain analytically memory-bound under the prior GB10 roofline
assumptions. Cache reuse, indirect addressing, register pressure and fused
instruction issue are unmeasured. **No GB10 latency or throughput is claimed.**

The promoted selective result supports B's native-block decode path: paired B
lowers aggregate one-bpw damage by 16.745%, improves p10 recovery by 1.213
points and median recovery by 0.874 points. The added complexity is justified
for a tail/damage-sensitive H4 study, but a fused-kernel benchmark is still
required before a deployment latency claim.

<!-- BEGIN FINAL_PROMOTION_SERIALIZATION -->

## 12. Final promotion and winning serialization

The validation-only policy locks `B16_g32_cluster_functional` with **paired**
packets by minimum median absolute damage at one physical bpw. The completed
11-layer promotion retains that representation and packet without consulting
test metrics.

The validation lock artifacts are:

- `metrics/selective_validation_metrics.parquet`: 3,780 rows, 115,170 bytes,
  SHA-256
  `7f90e55846b8fa18c6b473c6d27261ba32993539cfe5376a6fc84f348ee90a83`;
- `metrics/selected_validation_actions.parquet`: 11,340 rows, 3,869,002 bytes,
  SHA-256
  `0898943d8a48f7a65f24dbc7349007b751a81d10231e2280fe5d1299fd6cc634`;
- `metrics/validation_allocator_lock_summary.json`: 14,068 bytes, SHA-256
  `9ecb1c7cfefbf8547a83656425a134cc693194206ffe55323804110825fa72b0`.

The summary records only the `validation` split, four pilot layers, 135
invocations per one-bpw row and zero duplicate keys.

The broader promotion artifacts are rooted at `promotion_11layer/`:

- `promotion_manifest.json`: 94,819 bytes, SHA-256
  `9476e9a3a58dd6b099d22b4a1621fa97fdb8c5959bd3f78085669f4246b474f5`;
- `metrics/promotion_summary.json`: 71,212 bytes, SHA-256
  `4baa7c3074675cb175a7b29a3e173038db6d1dc748fe013ae74263d3ecbb2f1d`;
- `metrics/dense_metrics.parquet`: 7,098 rows, SHA-256
  `13e882660291533b40a39e89e6631a8bc4947bcac833a1cb87579be70dbe2724`;
- `metrics/selective_metrics.parquet`: 9,352 rows, SHA-256
  `b7ed8f2a83ff99c90c74efe14879f8830a6e856495d895a93e16c0a3dc648420`;
- `metrics/selected_actions.parquet`: 28,056 rows, SHA-256
  `5ab7b6386ba2109ca1a0bf49fb4ea3d4780ba1514c213a7a7c2acd00b30f32e1`.

The promotion also retains 21,294 projection rows, all 11 target layers, the
33 selected experts, the exact capture/config/code hashes, serialized B tables
and metadata, and zero dense/selective duplicate keys. Its B codec manifest is
`promotion_11layer/serialized_manifests/codecs/B16_g32_cluster_functional/manifest.json`,
SHA-256
`1fcf46239f25d4787fcbaf7020727769d010f92b34c9e5984d43a751829a405f`.

The validation-locked B codec and the mode-matched PR4 control are retained at
`serialized_manifests/winner_packages_20260819/`. This standalone framed bundle
provides the strict stream-level reader/equality proof; the promotion directory
provides the broader logical-rate and quality evidence.

- Package index: `index.json`, 3,533 bytes, SHA-256
  `8a93a3ad36fb508dac9dbd3617cfc974a98b3994c6cba25b0e8eb401b125f28b`.
- B manifest: 92,809 bytes, SHA-256
  `0620c0614da8f191b08904725d263a688d58a3b06669b6d5c914734c7792f45c`.
- Nine cluster/projection tables are each 576 bytes (64-byte header plus 512
  payload bytes), 5,184 bytes total. For example,
  `cluster_cold__gate.cgt` has SHA-256
  `1ba7f42878962d16ab8890faedaf6db5105816776b29facaa1ec687f283c76f0`.
- Twelve per-expert metadata files are each 49,216 bytes, 590,592 bytes total.
  For example, `layer_00_expert_052.cgb` has SHA-256
  `d825fcc49c4bfe556c893c70831ecf6692955a15944a9361b275e4a614ef8278`.
- All 36 projection files are 557,568 bytes. A representative
  `layer_00/expert_052/gate.cgw` has SHA-256
  `c08e6d09fb515ec2a840a3e627ebc8746a90bc7971fbbbb96c5082519ed1b738`.
  It contains a 262,144-byte logical Q2 parent stream stored in 262,592 bytes
  (448 bytes alignment padding), 32,768 E8M0 bytes, and 131,072 bytes for each
  Q3 and Q4 suffix. The individual stream SHA-256 values are respectively
  `1638f73eb79ecdbf2488bc81ef42fcf768f968b64e65dc5edc1fff752795800d`,
  `731bbe662a0d5c0d15354f99851698ef28ec0de7675ca08dbf0d3a5017316db3`,
  `d32f1dbf14fd4aa1981db0c266d11d871860740ab18061c5d364bbd112bfff6a`
  and `07e6a539f90825b51f21eb6edd5a136439adf5b201b1e14588b859ae07351ed6`.
- Reader verification passes for all projections: parent/r1/r2 recombination
  reproduces the source packed E2M1 leaf codes exactly, and the E8M0 stream hash
  matches the source. The package-level `exact_q4_code_equality` flag is true.
- Across 37,748,736 weights, B has 10,635,264 framed resident weight bytes,
  595,776 metadata bytes and 11,231,040 total resident bytes
  (**2.380167643 bpw**). Q3 and Q4 are each 4,718,592 bytes (**1.0 bpw**).
  The framed control is **2.253987630 resident bpw**.
- `selective_metrics.parquet` has 19,152 rows, 488,403 bytes and SHA-256
  `77038a7e6407ef1823050e43860047cb771f716e15675fc0ff539440d0668d7a`.
  `selected_actions.parquet` has 57,456 rows, 20,801,802 bytes and SHA-256
  `064d36b30cded26adebd0abb8d48ceb5458f966ef14c6dd29164e4e4edd72585`.

The B manifest contains the size, role and SHA-256 of every table, expert
metadata file, projection file and internal stream; the representative entries
above are enough to verify the format without duplicating all 57 file records
in prose.

<!-- END FINAL_PROMOTION_SERIALIZATION -->

## 13. Answers to the primary questions

1. **Does 0.125 resident bpw materially improve Q2?** Yes. Promoted B reduces
   aggregate test Q2 damage by 29.899%; per-invocation common recovery is
   -16.081/21.271/50.211% p10/median/p90. The negative p10 is the remaining
   resident failure tail.
2. **Does it materially improve Q3?** Yes. Promoted B reduces aggregate
   same-level PR4-Q3 damage by 55.794% and reaches
   55.697/75.720/87.378% common recovery. Gate Q3 mean damage falls 75.85%
   against PR4 Q3.
3. **A, B or C?** B is the validation-locked resident winner. C is the best
   balanced layer-shared secondary arm. A should be eliminated.
4. **G32 or G16?** G32 is preferred: G16 has mixed Q2 gains but materially worse
   validation and test Q3 and twice the table-switch cadence.
5. **More than 16 families?** It can improve resident Q2 at G64, but Q3 saturates
   near 16 and the 32/64-family results are non-monotonic.
6. **Can simple statistics predict family IDs?** Only weakly: 19.63% nearest-
   centroid accuracy versus 15.75% majority. The explicit selector remains
   useful.
7. **Do exact leaves materially hurt?** Not in the available experiment. The
   unmatched learned-leaf diagnostic improves a global Q3 but remains well
   behind exact B and has a materially inexact endpoint.
8. **Does dense-Q3 win selective?** Not as a strict pilot test ordering: C has
   the best one-bpw p10 and D's paired median is only 0.000052 above B. Those
   test-descriptive differences do not override the validation B-paired lock;
   promoted B then improves all broader paired common-recovery percentiles and
   aggregate damage versus PR4.
9. **One-bpw p10/median/p90 gain?** On the promoted cohort, paired B gains
   **1.213/0.874/0.200 points** over matched PR4 and lowers aggregate damage by
   **16.745%**. The bounded pilot gains were 1.372/0.982/0.543 points and are
   reported separately in Section 10.3.
10. **Bpw for 90/95/97.5% recovery?** On the promoted sampled grid, paired B
    first crosses at **0.75/1.25/1.50 bpw**, versus **1.00/1.25/1.50** for PR4.
    The 0.602/0.947/1.292 values in Section 10.4 are pilot interpolations, not
    promoted grid observations.
11. **Is decode complexity justified?** For a tail/damage-sensitive next study,
    yes: B stays on the native 32-weight boundary and materially reduces damage.
    The promoted paired median recovery gain is only 0.874 points, so a fused-kernel
    benchmark is required before deployment. No GB10 latency was measured.
12. **Format for the next H4 study?** Use exact-MXFP4 B16/g32
    cluster/projection functional with validation-locked paired packets; retain
    separate packets as the ordered-plane control. The 11-layer promotion
    confirms this recommendation.

## 14. Success-criteria assessment

| Criterion | Dense status | Final status |
|---|---|---|
| Meaningful resident improvement | **Met:** promoted aggregate Q2 damage falls 29.899%; median common recovery is +21.271 points | Confirmed over 11 layers |
| Stronger first streamed bit | **Met densely:** promoted PR4-Q3 aggregate damage falls 55.794%; gate Q3 mean damage falls 75.85% | **Moderate selectively:** paired one-bpw aggregate damage falls 16.745% |
| +2--3 recovery points at one physical bpw | Not a dense metric | **Not met:** promoted paired p10/median/p90 gains are +1.213/+0.874/+0.200 points |
| 15--25% fewer physical bytes at matched quality | Not a dense metric | **Partially met:** promoted median 90% moves from 1.00 to 0.75 bpw (25%); 95% and 97.5% crossings tie |
| High-90s recovery near one physical bpw | Not a dense metric | **Not met at median:** promoted paired median is 93.844%; p90 is 97.712% |

The codec is a strong resident/dense-Q3 success and a moderate selective
success. It advances the 90% median crossing and lowers aggregate damage, but
does not deliver the headline +2--3 median points or high-90s median at one
physical bpw.

## 15. Limitations

- This is an 11-layer, 33-expert H0 promotion, not a full-40-layer model-quality
  or downstream task-accuracy run.
- The frozen PR6 reproduction, four-layer pilot and broader promotion are
  different cohorts. Only the PR4 rows evaluated on the same broader capture
  are matched controls for promoted codec deltas.
- The PR6 `52bc` artifact is byte-identically reproduced. The bounded fresh
  pilot still uses a CPU-regenerated exact-checkpoint capture with locked expert
  IDs, so its frequency ordering should not be mixed with the frozen cohort.
- Complete-expert quality is a rank-4 proxy-weighted output reconstruction
  metric. It is not downstream token accuracy, routing accuracy or H4 quality.
- Only one cold/median/hot expert per promoted layer is retained. Expert and
  layer strata therefore still have limited sample support.
- The family optimizer is non-convex and epsilon arms use different deterministic
  seeds. Small non-monotonic differences should not be overinterpreted.
- C changes family count and adds vector modifiers simultaneously, so the
  modifier contribution is not isolated.
- The learned-leaf diagnostic is not capacity matched to B.
- Block-feature and scale/rare-leaf diagnostics are available for the saved
  pilot family-specialization arm, but were not repeated over all promoted
  blocks; rare-pattern conclusions remain pilot-scale.
- B and paired packets are validation locked. D/C one-bpw differences are
  descriptive pilot test observations only; the promotion does not re-open the
  eliminated design search.
- Promoted crossings are first observed points on a 0.25-bpw grid. Pilot
  crossings are linear interpolations; neither should be represented as the
  other.
- Analyzer plots 04 and 05 choose the displayed candidate/packet by held-out
  test recovery. They are descriptive visual summaries only and do not override
  the saved validation packet lock.
- Decode instruction counts are analytical estimates. No fused kernel or GB10
  latency was measured.

## 16. Reproduction and artifacts

Remote commands, run environment, capture identities and reliability events are
recorded in `EXECUTION_LEDGER_CODEBOOK_GRANULARITY_20260819.md`. The principal
analysis regeneration command is:

```bash
MPLBACKEND=Agg python scripts/analyze_codebook_granularity.py \
  --result results/qwen36_mxfp4_codebook_granularity_20260819_v2
```

It regenerated the metric summaries and PNG/SVG versions of all 16 requested
plot stems from the complete dense, diagnostic and selective inputs. The saved
`analysis_manifest.json` records 34,335 dense rows, 19,152 selective rows and
all 32 plot files.

The required plot stems are:

1. `01_resident_error_vs_metadata`;
2. `02_dense_q3_recovery_vs_metadata`;
3. `03_selective_recovery_vs_physical`;
4. `04_selective_quantile_frontier`;
5. `05_design_comparison_one_bpw`;
6. `06_q2_tolerance_vs_q3`;
7. `07_family_size_vs_quality`;
8. `08_group_granularity_vs_quality`;
9. `09_train_validation_test`;
10. `10_family_utilization_histogram`;
11. `11_family_specialization_heatmap`;
12. `12_metadata_bytes_breakdown`;
13. `13_projection_bandwidth_allocation`;
14. `14_absolute_damage_vs_mxfp4`;
15. `15_decode_cost_vs_quality`;
16. `16_pareto_frontier`.

Each stem exists as both PNG and SVG. Plots 03/04/05/13/14/16 were generated
from the complete selective Parquet; plot 11 and the scale/rare-leaf strata use
`family_block_features.parquet`. Codec manifests are retained locally so byte
accounting can be recomputed. These plots visualize the bounded design stage;
the promoted 11-layer rows are retained separately under `promotion_11layer/`
and summarized in Sections 8.9 and 10.6.

Available now:

- `DENSE_SCIENTIFIC_MEMO.md`
- `DECODE_ACCOUNTING_MEMO.md`
- `decode_accounting.csv`
- `EXECUTION_LEDGER_CODEBOOK_GRANULARITY_20260819.md`
- `metrics/dense_metrics.parquet`
- `metrics/projection_metrics.parquet`
- `metrics/dense_summary.csv`
- `metrics/projection_summary.csv`
- `metrics/validation_selection.csv`
- `metrics/heldout_stratified_summary.csv`
- `metrics/fit_history.parquet`
- `metrics/metadata_accounting.csv`
- `metrics/family_utilization.csv`
- `metrics/family_predictability.json`
- `metrics/frozen_historical_selective_baselines.csv`
- `metrics/FROZEN_BASELINE_PROVENANCE.md`
- `metrics/selective_metrics.parquet`
- `metrics/selected_actions.parquet`
- `metrics/selective_summary.csv`
- `metrics/recovery_crossings.csv`
- `metrics/selective_validation_metrics.parquet`
- `metrics/selected_validation_actions.parquet`
- `metrics/validation_allocator_lock_summary.json`
- `metrics/merge_manifest.json` with complete shard inventory;
- `metrics/family_block_features.parquet`,
  `metrics/family_specialization_summary.csv` and
  `metrics/projection_heldout_stratified_summary.csv`;
- `serialized_manifests/candidate_index.json`, `fit_run_facts.json` and all
  candidate codec manifests/research states;
- `serialized_manifests/winner_packages_20260819/index.json` plus actual packed
  B/control parent, scale, Q3 and Q4 streams and equality proofs;
- `plots/` with all 16 PNG/SVG pairs;
- `analysis_manifest.json`;
- `configs/qwen36_mxfp4_codebook_granularity.json` and
  `configs/qwen36_mxfp4_codebook_promotion_11layer.json`.
- `promotion_11layer/promotion_manifest.json`;
- `promotion_11layer/metrics/promotion_summary.json`, dense/projection/selective
  Parquets, action rows, shard manifests and serialized B state;
- `baseline_reproduction/pr6_exact_checkpoint_physical_52bc/` with the
  byte-identical frontier, layout facts, action-to-page frontier, storage
  accounting and validation JSON;
- `metrics/pr6_52bc_one_bpw_reproduction_summary.csv`.

The result-local configs mirror the canonical files under
`experiments/adaptive_expert_precision_oracle/configs/`; the bounded pilot and
broader 11-layer promotion configurations are both retained.

Dense raw rows already retain request/position/layer/expert identity, split,
damage, reference energy, relative error and hierarchy level. Candidate research
states retain family selectors and Design C modifiers. The final winner bundle
materializes the actual packed Q2 parent, E8M0, Q3 and Q4 streams. The merged
`selected_actions.parquet` retains per-invocation depth arrays, physical page
IDs, logical/physical bytes and projection allocation. These raw artifacts, not
only aggregate CSVs, are now locally retained.

## 17. Final-response handoff checklist

| # | Required final item | Current source/status |
|---:|---|---|
| 1 | Exact commit tested | Complete: report/ledger header |
| 2 | Exact reference format | Complete: Section 2.1 |
| 3 | Current baseline reproduction | Complete: PR6 52bc dataframe byte-identical, 85 all-four invocations and 36/36 comparisons; broader same-cohort PR4 control retained |
| 4 | Best resident design | Promoted: B16/g32 cluster functional, 29.899% aggregate Q2 damage reduction |
| 5 | Best Q3 design | Promoted B under the Q2 lock reduces PR4-Q3 aggregate damage 55.794%; C is only the pilot pure-D3 winner and violates the B Q2 lock |
| 6 | Best selective one-bpw result | Promoted validation-locked B paired: 93.844% median and 16.745% lower aggregate damage than PR4 |
| 7 | Selective p10/median/p90 | Promoted B paired at one bpw: 88.802/93.844/97.712%; PR4: 87.589/92.970/97.512% |
| 8 | Bpw for 90/95/97.5% recovery | Promoted sampled-grid B paired: 0.75/1.25/1.50; PR4 paired: 1.00/1.25/1.50 |
| 9 | Best family size | Dense answer: 16 for Q3; 64 can improve resident Q2 at G64 |
| 10 | Best group granularity | Dense answer: native 32 weights overall |
| 11 | Whether per-vector tables overfit | Complete: no clean memorization signature, but unsupported/misaligned and inferior to shared B |
| 12 | Learned-Q4 versus exact leaves | Complete with unmatched-diagnostic caveat |
| 13 | Serialized resident metadata cost | Promoted logical: 1,628,736 metadata bytes, 2.375517874 resident bpw over 103,809,024 weights; framed pilot bundle: 595,776 metadata bytes and 2.380167643 resident bpw |
| 14 | Estimated decode overhead | Complete analytical estimate; no GB10 latency claim |
| 15 | Recommended next format | Exact-MXFP4 B16/g32 cluster functional with paired packets; separate retained as ordered-plane control |
| 16 | Paths to reports, metrics, plots, configs and ledger | Complete: bounded study, promotion, reproduction, serialized packages and ledger all retained |
