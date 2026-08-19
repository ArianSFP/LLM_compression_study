# Dense scientific memo: codebook granularity pilot

Date: 2026-08-19  
Repository HEAD tested: `37e7bfcd168545d6a44a3e3df45b6eb27d858a3c`  
Result namespace: `qwen36_mxfp4_codebook_granularity_20260819_v2`

## Scope and metric discipline

This memo is an independent reading of the completed **dense** metrics only. It
does not use the frozen historical selective frontier or any partial selective
output. Consequently, it makes no claim about the one-physical-bpw selective
winner.

The reference is the exact checkpoint
`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` at revision
`7eceff3a9f7e6f916c824d197266d86676bce695`: compressed-tensors
`mxfp4-pack-quantized`, E2M1 leaf nibbles and one uint8 E8M0 scale per 32
weights. The pilot uses layers 0, 4, 20 and 39, one cold/median/hot expert per
layer, and a request-level 12/6/6 train/validation/test split. The resulting
complete-expert invocation counts are 258/135/152.

Two recovery definitions must not be conflated:

- **Common-baseline recovery** is
  `1 - D(candidate level) / D(PR4 resident Q2)` on the same invocation. This is
  the only recovery used to compare resident Q2 values across formats.
- **Own-Q2 recovery**, or dense `rho3`, is
  `1 - D(candidate Q3) / D(candidate Q2)`. It measures how effective that
  candidate's first refinement plane is, but a large value can coexist with a
  poor resident Q2.

Unless explicitly called “aggregate,” p10/median/p90 are percentiles of
per-invocation ratios. Aggregate recovery is the ratio of summed damage and is
more sensitive to the high-damage layer-39/high-activation rows. Absolute
damage is always versus exact MXFP4.

Validation selection is lexicographic within each design: minimize summed Q2
damage, admit candidates within 0.5% of that minimum, then minimize summed Q3
damage. Test is used only after this lock.

## Result in one paragraph

The validation-locked recommendation is **Design B, 16 families selected per
native 32-weight block, shared by expert cluster and projection, fitted with the
complete-expert functional proxy**. It occupies 2.375164 resident bpw. On test,
its resident Q2 reduces summed complete-expert damage by 37.36% relative to the
2.25-bpw PR4 parent; per-invocation common recovery is
**-7.46% / 30.88% / 64.51%** at p10/median/p90. Dense Q3 reaches
**62.07% / 77.93% / 87.90%** common recovery, removes 71.22% of its own summed
Q2 damage, and reduces PR4-Q3 summed damage by 49.03%. The p10 resident result remains
negative, so the extra metadata materially improves the center and aggregate
but does not eliminate the resident failure tail. All exact candidates recover
the MXFP4 leaves exactly at Q4.

## Validation-locked design winners

The table below uses summed complete-expert damage. “Common Q2/Q3” is the ratio
of sums against the PR4 resident Q2 sum; `rho3` is the ratio of each candidate's
own Q3 and Q2 sums.

| Design winner | Resident bpw | Validation D2 | Validation D3 | Validation common Q2 | Validation common Q3 | Validation own rho3 | Test D2 | Test D3 | Test common Q2 | Test common Q3 | Test own rho3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A `A_vector_functional` | 2.375000 | 1294.179 | 248.573 | 4.82% | 81.72% | 80.79% | 1192.595 | 280.920 | 3.14% | 77.19% | 76.44% |
| **B `B16_g32_cluster_functional`** | **2.375164** | **849.935** | 212.382 | **37.49%** | 84.38% | 75.01% | 771.303 | **221.998** | 37.36% | **81.97%** | 71.22% |
| C `C8_g32_layer_functional` | 2.375170 | 904.930 | **208.490** | 33.45% | **84.67%** | 76.96% | 857.916 | 238.087 | 30.32% | 80.66% | 72.25% |
| D `D4_g16_layer_functional` | 2.375167 | 969.423 | 232.804 | 28.71% | 82.88% | 75.99% | 781.676 | 252.573 | 36.52% | 79.49% | 67.69% |
| E `E64_g64_layer_functional` | 2.375224 | 915.378 | 219.515 | 32.68% | 83.86% | 76.02% | **749.186** | 240.666 | **39.15%** | 80.45% | 67.88% |
| PR4 2.25-bpw parent | 2.250000 | 1359.763 | 420.803 | 0.00% | 69.05% | 69.05% | 1231.294 | 435.516 | 0.00% | 64.63% | 64.63% |
| Learned-16-leaf global diagnostic | 2.250000 | 1448.636 | 308.060 | -6.54% | 77.34% | 78.73% | 1398.787 | 322.496 | -13.60% | 73.81% | 76.94% |

Design B is the global lexicographic winner because it has the smallest
validation Q2 damage. Design C has the smallest validation Q3 damage among the
five per-design winners, but its Q2 is 6.47% worse than B's and it does not
retain its Q3 lead on test. E64 has the lowest test Q2 sum, but promoting it on
that fact would be test selection; it was 7.70% worse than B on validation Q2.

## Held-out percentiles and absolute damage

Common-baseline complete-expert recovery on the test split:

| Candidate | Q2 p10 | Q2 median | Q2 p90 | Q3 p10 | Q3 median | Q3 p90 |
|---|---:|---:|---:|---:|---:|---:|
| A vector | -13.99% | 11.31% | 51.38% | 55.12% | 76.98% | 87.11% |
| **B16/g32 cluster** | **-7.46%** | **30.88%** | **64.51%** | **62.07%** | 77.93% | 87.90% |
| C8/g32 hybrid | -15.74% | 23.69% | 56.99% | 61.56% | 77.47% | **88.93%** |
| D4/g16 | -13.53% | 30.55% | 58.84% | 56.15% | 77.06% | 86.40% |
| E64/g64 | -12.27% | 24.58% | 57.38% | 57.01% | **78.61%** | 88.34% |
| PR4 parent | 0.00% | 0.00% | 0.00% | 37.84% | 64.35% | 77.12% |
| Learned-16-leaf | -65.55% | -9.82% | 24.92% | 48.35% | 69.96% | 83.95% |

The corresponding absolute complete-expert damage versus exact MXFP4 is:

| Candidate | Q2 damage p10 | Q2 median | Q2 p90 | Q3 damage p10 | Q3 median | Q3 p90 | Median relative output error Q2 / Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A vector | 0.051947 | 0.274034 | 19.703002 | 0.012753 | 0.077558 | 5.446529 | 0.5282 / 0.2810 |
| **B16/g32 cluster** | **0.037756** | **0.234991** | 16.482341 | **0.011927** | 0.089947 | **4.881925** | **0.4853 / 0.2729** |
| C8/g32 hybrid | 0.041519 | 0.246444 | 18.156411 | 0.013557 | 0.080165 | 5.402986 | 0.4948 / 0.2761 |
| D4/g16 | 0.035369 | 0.265358 | **14.578293** | 0.012300 | 0.090921 | 5.846532 | 0.4809 / 0.2783 |
| E64/g64 | 0.039995 | 0.240684 | 17.749169 | 0.013257 | **0.077082** | 5.481105 | 0.4911 / 0.2746 |
| PR4 parent | 0.051664 | 0.397743 | 21.729198 | 0.018683 | 0.134813 | 8.097995 | 0.5498 / 0.3338 |
| Learned-16-leaf | 0.061852 | 0.448032 | 23.744296 | 0.018121 | 0.124793 | 6.319715 | 0.5770 / 0.3058 |

The different orderings of ratio percentiles, absolute-damage percentiles and
summed damage are expected: they weight invocations differently. B is the most
stable validation-locked choice, not the winner of every test statistic.

## Projection and gate effects

Projection damage is evaluated before the complete nonlinear SwiGLU
composition, so its magnitude must not be added to or compared numerically with
complete-expert damage. Within a projection, however, the comparison is direct.

For the selected B hierarchy on test:

| Projection | PR4 D2 sum | B D2 sum | B common Q2 | PR4 D3 sum | B D3 sum | B common Q3 | B own rho3 | B median relative error Q2 / Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gate | 15295.106 | 5629.693 | **63.19%** | 7109.426 | 908.730 | **94.06%** | **83.86%** | 0.2248 / 0.1240 |
| Up | 2872.192 | 2102.840 | 26.79% | 701.952 | 610.256 | 78.75% | 70.98% | 0.2907 / 0.1585 |
| Down | 603.136 | 323.497 | 46.36% | 181.733 | 108.277 | 82.05% | 66.53% | 0.3322 / 0.1852 |

Gate is the decisive win. The old PR4 gate first plane removed only 53.52% of
summed gate Q2 damage; B removes 83.86% of its own gate damage and leaves only
5.94% of the old PR4 gate-Q2 damage. The B gate per-invocation common-recovery
p10/median/p90 moves from 11.64/23.76/72.57% at Q2 to
74.19/78.48/97.24% at Q3.

Gate aggregate comparisons across the validation-locked exact designs are:

| Design | Gate common Q2 | Gate common Q3 | Gate own rho3 |
|---|---:|---:|---:|
| A | 57.50% | 92.90% | 83.30% |
| **B** | 63.19% | 94.06% | **83.86%** |
| C | 68.51% | 92.75% | 76.97% |
| D | 61.96% | 90.48% | 74.98% |
| E64 | **68.84%** | **94.25%** | 81.55% |

A's apparently strong own-Q2 gate rho3 does not make it the best hierarchy: its
up Q2 improves only 6.35% in aggregate and its down Q2 is 3.38% worse than PR4.
B improves all three projections at the resident level.

## Design A: independent vectors and shrinkage

Design A spends exactly 49,152 bytes/expert on 1,536 functional-vector tables
(512 gate rows, 512 up rows and 512 down columns),
but it converts that capacity poorly into resident functional quality. Its test
Q2 sum is only 3.14% better than PR4, versus 37.36% for cluster-shared B. A does
remove 76.44% of its own summed Q2 damage at Q3, but its absolute Q3 sum remains
26.54% higher than B's (280.920 versus 221.998).

The shrinkage strength of eight yields per-expert alpha values from 0.143 to
0.727 (median 0.348) for 3--48 training invocations. It does not rescue A:

| A variant | Validation D2 | Validation D3 | Test D2 | Test D3 | Test median common Q2 | Test median common Q3 |
|---|---:|---:|---:|---:|---:|---:|
| Independent | **1294.179** | 248.573 | **1192.595** | **280.920** | **11.31%** | **76.98%** |
| Shrunk to layer/projection prior | 1312.015 | **246.940** | 1204.566 | 283.529 | 10.05% | 76.60% |

For cold experts, shrinkage worsens test Q2 summed damage from 442.236 to
454.297, while improving Q3 only from 66.215 to 65.819. For hot experts it
worsens both Q2 (454.404 to 462.599) and Q3 (126.724 to 130.540). Median-expert
Q2 improves, but not enough to change the global decision.

This is **not a clean classic memorization-overfit signature**: unshrunk A is
already 0.08% worse than PR4 on aggregate training Q2, then 4.82% better on
validation and 3.14% better on test. The stronger conclusion is that
independent-vector capacity is poorly supported or poorly aligned with the
complete-expert objective in this bounded sample. Relative to shared B it is
inferior on every split, and shrinkage gives no held-out rescue. It should be
eliminated from scaling.

## Design B: objective and sharing

Functional weighting is important at Q2. For globally shared 16-family tables:

| Objective | Validation D2 | Validation D3 | Test D2 | Test D3 | Test median common Q2 | Test median common Q3 |
|---|---:|---:|---:|---:|---:|---:|
| Raw weight MSE | 1212.747 | 224.996 | 1023.009 | **237.006** | 16.09% | **77.58%** |
| Activation/projection | 939.505 | **211.535** | 776.954 | 252.209 | **33.32%** | 76.84% |
| Complete-expert functional proxy | **870.018** | 213.589 | **760.284** | 240.962 | 29.93% | 76.86% |

Raw weight MSE gives a relatively large own-Q2 Q3 recovery because it starts
from a much worse Q2. It is not competitive in absolute resident damage.
Functional fitting is the validation Q2 winner and has the best test D2 and a
better test D3 sum than activation weighting.

The functional sharing sweep is:

| Sharing | Resident bpw | Validation D2 | Validation D3 | Test D2 | Test D3 | Test median common Q2 | Test median common Q3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global/projection | 2.375163 | 870.018 | 213.589 | 760.284 | 240.962 | 29.93% | 76.86% |
| Layer/projection | 2.375178 | 956.811 | 212.877 | 868.486 | 226.358 | 25.76% | 78.62% |
| **Expert-cluster/projection** | **2.375164** | **849.935** | 212.382 | 771.303 | 221.998 | 30.88% | 77.93% |
| Expert/projection diagnostic | 2.379069 | 857.968 | **195.949** | **733.499** | **207.422** | 29.71% | **80.94%** |

Cluster sharing is selected because it has the best validation Q2. The
per-expert diagnostic has 0.95% more validation Q2 damage, outside the locked
0.5% tolerance, but 7.74% less validation Q3 damage and the best test sums. It
is a useful ceiling: if the next study deliberately permits about a 1% Q2
sacrifice, expert-shared families should be retained as a secondary arm. Layer
sharing is not supported by this pilot.

## Designs C, D and E: modifier, granularity and family count

The layer-shared formats provide the cleanest available, though not perfectly
factorial, comparison:

| Candidate | Selector/modifier | Resident bpw | Validation D2 | Validation D3 | Test D2 | Test D3 | Test own rho3 |
|---|---|---:|---:|---:|---:|---:|---:|
| B16/g32 layer | 4-bit block ID | 2.375178 | 956.811 | 212.877 | 868.486 | **226.358** | **73.94%** |
| **C8/g32 layer** | 3-bit block ID + 8-byte/vector FP16 Q2/Q3 affine modifiers | 2.375170 | **904.930** | **208.490** | 857.916 | 238.087 | 72.25% |
| D4/g16 layer | 2-bit ID per 16 | 2.375167 | 969.423 | 232.804 | 781.676 | 252.573 | 67.69% |
| E16/g64 layer | 8-bit ID per 64 | 2.375178 | 955.303 | 207.375 | 827.729 | 246.564 | 70.21% |
| E32/g64 layer | 8-bit ID per 64 | 2.375193 | 989.277 | 221.968 | 846.114 | 241.396 | 71.47% |
| E64/g64 layer | 8-bit ID per 64 | 2.375224 | 915.378 | 219.515 | **749.186** | 240.666 | 67.88% |

Design C is the best balanced layer-shared validation arm: compared with
B16/g32 layer, it lowers validation D2 by 5.42% and D3 by 2.06%. On test its D2
is 1.22% lower but D3 is 5.18% higher. Because C changes both the family count
and adds affine modifiers, this experiment does not identify the modifier's
isolated causal contribution. It supports keeping C as a secondary dense arm,
not replacing B cluster.

The 16-weight D arm gives a surprisingly good test Q2, but validation Q2 is
1.32% worse than native-32 B and validation Q3 is 9.36% worse. On test it also
leaves 11.58% more Q3 damage than native-32 B. Thus finer spatial locality helps
some resident cases, while 32-weight grouping is the safer choice for the first
streamed bit and aligns with the native MXFP4 block.

At 64-weight granularity, increasing the family count beyond 16 is
non-monotonic. E64 wins resident Q2 on validation and test, but E16 has the best
validation Q3 (207.375 versus 219.515) and essentially the same test median
common Q3 recovery (78.67% versus 78.61%). E32 is worse than both on validation.
The evidence is therefore: **more than 16 families can help resident Q2, but Q3
specialization is already saturated around 16**. A 256-family arm was not run;
the 16/32/64 behavior and sample support do not justify it in this bounded
pilot.

## Q2-tolerance sweep

This sweep uses global/projection B16/g32 functional tables. `rho3` is the ratio
of summed Q3 and own-Q2 damage.

| Epsilon | Validation D2 | Validation D3 | Validation rho3 | Test D2 | Test D3 | Test rho3 | Test median common Q2 | Test median common Q3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 914.694 | 216.801 | 76.30% | 826.827 | 232.594 | 71.87% | 26.45% | 75.91% |
| 0.1% | 930.713 | 221.238 | 76.23% | 758.741 | 228.000 | 69.95% | 32.76% | 77.63% |
| 0.25% | 861.298 | 210.968 | 75.51% | **699.110** | 238.319 | 65.91% | 31.74% | 75.58% |
| 0.5% | 870.018 | 213.589 | 75.45% | 760.284 | 240.962 | 68.31% | 29.93% | 76.86% |
| 1% | **857.738** | 207.220 | 75.84% | 767.510 | 235.471 | 69.32% | 31.37% | 76.71% |
| 2% | 917.737 | 214.139 | 76.67% | 774.309 | 242.280 | 68.71% | 29.63% | 77.55% |
| 5% | 988.032 | **202.378** | **79.52%** | 786.529 | **207.000** | **73.68%** | 30.76% | **78.93%** |

The 5% arm demonstrates a real Q3-oriented endpoint: it has the smallest
validation and test D3, but validation Q2 is 15.19% worse than the 1% arm. The
sub-percent arms do not form a smooth Pareto curve. Each epsilon run performs a
fresh non-convex alternating fit with a candidate-specific seed, so small
non-monotonic differences cannot be interpreted as the causal effect of epsilon
alone. The bounded evidence does **not** establish that a 0.1--0.5% Q2 sacrifice
reliably buys a stronger Q3. It does establish that a much larger allowed
sacrifice can do so. The 0.5% validation lock remains the defensible default.

## Learned-Q4 diagnostic

The learned-16-leaf scalar hierarchy is a global/projection, one-family
diagnostic rather than a capacity-matched version of B. It worsens test resident
Q2: summed damage is 13.60% above PR4 and median common recovery is -9.82%. Its
Q3 is better than PR4 Q3 (322.496 versus 435.516 summed damage; median common
recovery 69.96% versus 64.35%), showing that movable leaves can reshape a
low-capacity first plane.

It does not outperform the strong exact hierarchy: selected exact B has 31.16%
less Q3 damage than the learned-leaf diagnostic (221.998 versus 322.496) and a
higher common-recovery median (77.93% versus 69.96%). The learned endpoint is
also materially inexact on test:

- Q4 summed damage: 116.853;
- Q4 common-recovery p10/median/p90: 83.83/91.49/95.57%;
- Q4 relative-output-error p10/median/p90: 0.1437/0.1734/0.1929.

Across all splits, every one of the 545 level-4 rows for every exact candidate
has exact leaf/code equality, zero maximum damage and zero summed damage. None
of the 545 learned-leaf rows is exact.

Thus the available diagnostic gives **no evidence that exact MXFP4 leaf
preservation materially limits the best Q2/Q3 hierarchy**. It suggests a modest
global-Q3 opportunity from learned leaves, but a matched B16/g32 learned-leaf
ablation would be required to isolate the leaf constraint itself.

## Train/validation/test generalization and held-out strata

Aggregate common-baseline recovery by split:

| Candidate | Train Q2 | Train Q3 | Validation Q2 | Validation Q3 | Test Q2 | Test Q3 |
|---|---:|---:|---:|---:|---:|---:|
| A vector | -0.08% | 78.64% | 4.82% | 81.72% | 3.14% | 77.19% |
| A shrunk | 0.28% | 78.71% | 3.51% | 81.84% | 2.17% | 76.97% |
| **B16/g32 cluster** | **33.51%** | **81.93%** | **37.49%** | 84.38% | 37.36% | **81.97%** |
| C8/g32 hybrid | 26.32% | 79.56% | 33.45% | **84.67%** | 30.32% | 80.66% |
| D4/g16 | 30.52% | 79.51% | 28.71% | 82.88% | 36.52% | 79.49% |
| E64/g64 | 29.26% | 80.45% | 32.68% | 83.86% | **39.15%** | 80.45% |
| Learned-16-leaf | -9.00% | 75.86% | -6.54% | 77.34% | -13.60% | 73.81% |

B's aggregate Q2 improvement is stable across train/validation/test and its Q3
test recovery is close to train. A 2.4--4.0-point validation-to-test Q3 drop
appears across B/C/D/E, so it is more consistent with split difficulty
than design-specific overfit.

For selected B on the test split:

| Stratum | Count | Q2 p10 / median / p90 common recovery | Q3 p10 / median / p90 common recovery | Q2 / Q3 summed damage |
|---|---:|---:|---:|---:|
| Layer 0 | 33 | -6.73 / 30.90 / 54.11% | 62.14 / 77.07 / 86.87% | 1.643 / 0.472 |
| Layer 4 | 41 | -9.37 / 21.33 / 43.59% | 58.87 / 78.06 / 87.08% | 19.074 / 5.472 |
| Layer 20 | 39 | 4.40 / 52.72 / 76.05% | 73.96 / 81.67 / 88.83% | 7.471 / 2.739 |
| Layer 39 | 39 | -15.04 / 26.61 / 50.82% | 61.45 / 76.62 / 89.75% | 743.114 / 213.315 |
| Cold experts | 14 | -3.94 / 19.66 / 61.54% | 69.84 / 77.45 / 91.99% | 155.616 / 29.065 |
| Median experts | 26 | -10.00 / 22.23 / 75.48% | 62.94 / 75.05 / 86.63% | 263.615 / 89.530 |
| Hot experts | 112 | -5.32 / 31.57 / 59.30% | 61.43 / 79.76 / 87.90% | 352.072 / 103.403 |
| Low activation norm | 51 | -7.64 / 27.57 / 53.03% | 62.01 / 77.84 / 86.87% | 6.638 / 1.785 |
| Median activation norm | 50 | -4.46 / 31.50 / 65.84% | 63.06 / 81.23 / 88.57% | 19.410 / 5.763 |
| High activation norm | 51 | -10.63 / 33.53 / 74.28% | 62.56 / 76.33 / 88.83% | 745.255 / 214.450 |

Layer 39 and high-activation invocations dominate absolute damage. The weakest
resident tail remains visible in every expert-frequency group and in three of
four layers. Dense Q3 is much more uniform: every listed stratum has a positive
p10 above 58% common recovery. This is strong evidence for a better first plane,
but also a reason not to describe resident Q2 as solved.

The copied dense summary contains layer, expert-frequency and activation-norm
strata. It does not contain independent test aggregates by block-scale exponent
or rare leaf pattern; those questions require the raw block-feature artifact
and should not be inferred from the invocation summaries.

## Family use and predictability

The saved family diagnostic is for global B16/g32 functional at epsilon 1%, not
for the cluster-shared validation winner. Across its 36 expert/projection
groups, every group uses all 16 IDs. Effective-family count has median 13.16,
minimum 5.09 and maximum 15.51; medians are 13.76 for gate, 13.16 for up and
12.90 for down. The four selector bits are therefore not generally wasted.

A nearest-centroid classifier using zero/half fractions, sign imbalance,
absolute mean, variance, max/RMS, scale exponent, activation importance and
distinct-leaf count predicts held-out IDs with 19.63% accuracy, versus 15.75%
for the majority class. Gate and up reach about 22%, while down reaches 14.74%.
The IDs have some interpretable statistical structure but behave mainly as
nontrivial clusters; simple block statistics are not an adequate replacement
for the selector stream.

## Serialized resident cost

Actual accounting for the principal candidates is:

| Candidate | Selector bytes/expert | Modifier bytes/expert | Header/padding bytes/expert | Amortized table bytes/expert | Total metadata bytes/expert | Metadata bpw | Resident bpw |
|---|---:|---:|---:|---:|---:|---:|---:|
| A vector | 0 | 0 | 0 | 49152.00 | 49152.00 | 0.125000 | 2.375000 |
| **B16/g32 cluster** | **49152** | 0 | 64 | 0.45 | **49216.45** | **0.125164** | **2.375164** |
| B16/g32 expert diagnostic | 49152 | 0 | 64 | 1536.00 | 50752.00 | 0.129069 | 2.379069 |
| C8/g32 hybrid | 36864 | 12288 | 64 | 3.00 | 49219.00 | 0.125170 | 2.375170 |
| D4/g16 | 49152 | 0 | 64 | 1.50 | 49217.50 | 0.125167 | 2.375167 |
| E64/g64 | 49152 | 0 | 64 | 24.00 | 49240.00 | 0.125224 | 2.375224 |

Selected B has 933,952.45 resident bytes per expert including the historical
2.25-bpw parent and all charged metadata. The pilot's serialized B-cluster table
payload is 4,608 bytes. These figures are from the saved serialized-accounting
table, not nominal selector-bit arithmetic.

## Dense-only scientific recommendation

Use **exact-MXFP4 B16/g32 cluster/projection functional** as the primary
representation for the next selective/H4 study. It is validation locked, sits
within 0.000164 bpw of the 2.375 target, makes the largest reliable resident
improvement, transforms the gate first plane, preserves the native 32-weight
block, and reaches exact Q4.

Retain two secondary diagnostics:

1. B16/g32 expert/projection if a deliberate 1% resident-Q2 tolerance is
   acceptable; it is the strongest observed absolute Q3 arm but was outside the
   locked 0.5% band.
2. C8/g32 plus affine vector modifier, because it is the best balanced
   layer-shared validation arm and directly tests vector specialization without
   independent-vector tables.

Do not scale A, D, E32 or the learned-leaf diagnostic. E64 is worth retaining
only if resident Q2 is prioritized above Q3; selecting it as the primary arm
would rely on its test-only resident win and accept a larger 64-weight table
domain. The decisive unresolved question is whether B's dense gate/Q3 advantage
survives the physical-page selective allocator. That answer must come from the
completed selective metrics, not from this memo.

## Machine-readable sources

- `metrics/validation_selection.csv`
- `metrics/dense_metrics.parquet`
- `metrics/dense_summary.csv`
- `metrics/projection_metrics.parquet`
- `metrics/projection_summary.csv`
- `metrics/heldout_stratified_summary.csv`
- `metrics/fit_history.parquet`
- `metrics/metadata_accounting.csv`
- `metrics/family_utilization.csv`
- `metrics/family_predictability.json`
