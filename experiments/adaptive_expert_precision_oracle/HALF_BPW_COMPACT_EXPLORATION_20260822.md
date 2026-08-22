# Compact 0.5-bpw interaction-field exploration

This note records rapid, nonpromotable screens performed on 2026-08-22 after
the frozen PR #13 all-layer result. The immutable source branch is
`agent/mxfp4-average-rate-allocation-all-layers` at
`089deb4bd41eb71864779ced0cbb3db41e9dcb63`. No PR #13 source or artifact was
modified.

## Objective and cost boundary

The target is approximately 99% p10 recovery at 0.5 streamed correction bpw
without the exact joint field's 128-MiB group geometry. A compact candidate must
remain in the same bandwidth and compute order as PR #13. Up to 10x PR #13
resident interaction metadata is permitted.

The cross-layer experiment below does not add interaction metadata:

- rank-8 Hadamard INT4 factor plus encoded A/B/C: 9,232 bytes/expert;
- top-8 group geometry: 73,856 bytes;
- permitted correction stream: 384 pages/expert on average, or 3,072
  512-byte pages per token/layer top-8 group;
- nominal correction rate: 0.5 bpw;
- nominal all-in rate including metadata: 0.523478 bpw.

The factor geometry is loaded once per expert and reused by its coordinate and
local-search passes. Cross-layer pooling changes where correction pages are
spent, not the total correction bytes or factor payload.

## Compact execution method

The probe reuses the frozen all-layer train factors and exact validation
captures. It regenerates the existing per-expert price frontier, injects the
published PR #13 selected states as dominance anchors, and evaluates a dense
mean-page grid around the 0.5-bpw point. Forty layers are split into five
independent eight-layer jobs. Each job evaluates the same two aligned
validation token positions with two group workers. The five jobs run in
parallel, so 80 group frontiers complete in under four minutes on the supplied
dual-socket EPYC host.

The host reports two 64-core AMD EPYC 7H12 sockets, two threads/core, 128
physical cores, 256 logical CPUs, full 0-255 affinity, and 1 TiB RAM. Ten
frontier workers were used across the five jobs; this did not oversubscribe the
machine. The RTX 3090 was not the bottleneck for this replay.

The pooled allocator keeps the total budget fixed at 245,760 pages:

`2 tokens * 40 layers * 8 experts * 384 pages`.

Five objectives were screened from the same saved options: normalized residual
damage, squared damage, fourth-power damage, and hinge loss above 1.0% or 0.5%
remaining damage. Fourth-power damage produced the best p10.

## Result

| Selection field | Rate policy | Actual correction bpw | Actual all-in bpw | p10 recovery | Median recovery | Mean recovery | Minimum recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact rank-8 field | Uniform 0.5 per layer | <=0.500000 | <=0.523478 | 97.060% | 99.063% | 98.614% | 94.513% |
| Compact rank-8 field | 40-layer pooled, fourth-power residual | 0.499449 | 0.522927 | **97.855%** | 99.016% | 98.858% | 97.230% |
| Exact H0 group damage | Uniform 0.5 per layer | <=0.500000 | <=0.523478 | 97.107% | 99.080% | 98.642% | 94.711% |
| Exact H0 group damage | 40-layer pooled, fourth-power residual | 0.499963 | 0.523442 | **98.353%** | 99.037% | 98.918% | 98.105% |

At identical allowed streamed pages, compressed pooling improves p10 by 0.7950
percentage points. The exact tail-aware ceiling improves p10 by 1.2463 points.
The latter also raises the minimum by 3.3937 points. Median recovery moves
slightly down because the objective deliberately transfers pages from already
saturated groups into the tail.

These are strong positive allocation results, but neither reaches the 99% p10
target. They show that layer heterogeneity is worth exploiting and that a
tail-aware objective is materially better than mean normalized damage at this
rate.

## Scientific and systems boundary

This is a compact validation screen over two token positions, not a promotion
result. In particular:

- the exact result uses exact H0 combined group damage and is an oracle;
- even the compressed selection pools frontiers for all 40 future layers, so it
  is future-aware and not yet a causal streaming policy;
- the cross-layer DP and its buffering are not included in the frozen PR #13
  selector MAC/latency accounting;
- a leave-one-token-out static layer schedule did not generalize: its best p10
  was 96.136% for the compressed field and 95.878% for the exact field;
- no H1-H4 causal replay, downstream routing, logit, or token-quality claim is
  made.

The bandwidth statement is exact for this probe: actual page use does not
exceed the frozen average budget, and the resident sidecar remains exactly PR
#13-sized. A deployable continuation needs either a small causal rolling
allocator or a train-fitted rate predictor, with allocator state, compute,
latency, and burst buffering charged explicitly.

## Full-burst and direct p10 follow-up

The initial compact frontier retained operating points only through 576
pages/expert (0.75 correction bpw) because that was enough for the smooth
fourth-power allocation. To test whether the tail was artificially capped, the
nine sub-99% layers on the harder token were regenerated through the full
1,536-page/expert endpoint. The pooled budget stayed fixed at 122,880 pages per
40-layer token, or 0.5 correction bpw on average.

Adding those burst columns does not change the useful fourth-power result:
compressed p10/median remains 97.855%/99.016%, and exact H0 p10/median remains
98.353%/99.037%. The allocations use 245,494 and 245,742 pages respectively
across both tokens, within the 245,760-page limit. Thus the earlier ceiling was
not being held down by the 0.75-bpw per-group frontier truncation.

A deliberately harsh quantile control then maximized a conservative empirical
p10 floor by requiring 37 of each token's 40 layer groups to exceed a common
threshold. It raises the aggregate exact p10 to 98.768% and the compressed p10
to 98.073%, but it does so by assigning zero pages to three groups per token;
both controls have zero minimum recovery and are not usable streaming policies.
They are retained only as upper-bound diagnostics.

The harder token is decisive. Under exact H0 selection its highest feasible
37-of-40 floor is 98.732%. Reaching 99% for 37 groups requires at least 136,034
pages, 13,154 pages or 10.70% above the allowed 122,880. The compressed
selector requires 140,945 pages, 18,065 pages or 14.70% above budget. The other
token can reach the 99% floor. Therefore 99% p10 is outside the observed
Q2-to-Q4 rate frontier at 0.5 correction bpw for the hard token, even with
exact tail-aware allocation and a full two-bpw burst endpoint.

This is a stronger boundary than simply saying that the current allocator
missed the target: allocation loss can still be reduced, but the current state
representation needs a better low-rate frontier to close the final gap.

## Wide low-bit geometry screen

A two-group layers-4/20 smoke screen used the permitted metadata headroom to
test a rank-256 Hadamard ternary field and rank-512 Hadamard sign field. Both
store FP16 row scales and per-unit FP16 2-by-2 Q4/Q2 calibration transforms.
Including the existing PR #13 sidecar, each reads 647,328 bytes/top-8 group,
8.765x PR #13 and below the 10x limit. Charged solver work is 1.715x and 2.427x
PR #13 respectively.

Both are clear negatives on this smoke: rank-256 ternary reaches 97.343% p10 /
98.105% median and rank-512 sign reaches 97.326% / 98.130%, versus 98.488% /
98.955% for the frozen PR #13 witness on the same two groups. The smaller
rank-128 INT4 shared field is better than either. Increasing nominal rank with
one- or two-bit scalar codes therefore does not preserve the relevant
interaction basis, and these formats were not scaled to all layers.

## Reproduction and immutable probe hashes

Exploratory source commit before the uncommitted probes:
`f0d9599f001776767f71c1cc781e6597df17774f`.

- probe: `2ec2dc3ffa3a05c29d81f6ac8687b957f72dfd2817d06d89a18c75dc60956231`
- original analyzer: `10eee2cff75df8375058bcfd36be1964cec5073a6df036f7e3c7316e62d38807`
- supplemental-frontier analyzer: `377be6b6eb3aed74b0f4357a015a093ca0e3759c698d8d7b7d5579ec835ec3c0`
- merged JSON: `9dd9e7fd394f823ca2c612c9207bf5c7548a1f4334d4ae61d7071c2092357f47`
- chunks 0-7 / 8-15 / 16-23 / 24-31 / 32-39:
  `2cf6259f8f5b2edca4ebfa9d87eff9751c39828fcaf80054f25b837d8a233307`,
  `f66af50b07b6dbf6374321605910bbfa894d1ffbdbf4d13331b0a82323508231`,
  `7369e21192dcf6bf648ae6b7b47f855f14edb11e76e63452dfe69f62d4dc9bc1`,
  `f3a97439cfd6c6d61af9bfac3863161c3b5d9996086135c19f8337f9b7273a78`,
  and `936e2bf3cab79f74a20188a181b66040bc116bec5b261368750471241b38d4fc`.

- nine-layer full-burst frontier: `db25b550dbc321786786a1aa2e6bfd0cef49548a2dce5efea49f232097db4cfa`
- corrected merged full-burst/tail analysis: `3fb19b75c8550d120a52e0af2795ee70bd6ca1444bd99c5f07a9c14719227469`
- wide low-bit smoke summary/frontier: `dac81b45af2aee9978c13c71a002f5622d29dec43054a63a8719d513e66c5079` / `843c7aa3fd4a3e380fa71f61216967f7b0367557fed619a9f0151d133b06bcb4`
- wide low-bit probe/core/test: `9647a0065b422c2ce0247bd7407fc9be496b6e20d2b0029737d6cafa4c2e4105` / `9daede580f0a6c19afdd93e2851e79639bef023fb77e41c9cefa077eb0ea8312` / `aa9ec9485b450ab0fa01acd20757730718e0a0ad9e7a951dc27f9e08ee16a37b`

The 852-KiB canonical evidence subset is packaged under
[`results/half_bpw_compact_exploration_20260822/`](results/half_bpw_compact_exploration_20260822/):
five base frontier chunks, the original analysis, the nine-layer full-burst
supplement, the corrected merged analysis, and the wide-low-bit summary and
Parquet. Regenerate the merged analysis with:

```bash
PYTHONPATH=experiments/adaptive_expert_precision_oracle/scripts \
python experiments/adaptive_expert_precision_oracle/scripts/analyze_half_bpw_cross_layer_rate.py \
  experiments/adaptive_expert_precision_oracle/results/half_bpw_compact_exploration_20260822/cross_layer_frontier_layers_*.json \
  --supplements experiments/adaptive_expert_precision_oracle/results/half_bpw_compact_exploration_20260822/cross_layer_fullburst_hard9.json \
  --output /tmp/half_bpw_fullburst_analysis.json
```

The regenerated JSON must hash to the recorded corrected-analysis digest.
