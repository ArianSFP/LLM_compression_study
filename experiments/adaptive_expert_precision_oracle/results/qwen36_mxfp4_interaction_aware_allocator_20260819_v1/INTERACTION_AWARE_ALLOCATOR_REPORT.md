# Interaction-Aware MXFP4 Allocator Study

Run date: 2026-08-19
Branch: `agent/mxfp4-interaction-aware-allocator`
Starting commit: `6b31a88ee517359847f940cc74db56898a8b8425`

## Executive conclusion

The broad H0 interaction hypothesis is supported, but the compact deployable-Gram hypothesis is not. Across the audited gate/up rows, diagonal ranking requires a median **3.07x** as many actions as exact fixed-coefficient marginal greedy at 90% recovery (minimum invocation **2.05x**). Refresh selectors retain much of that isolated oracle gain.

Rank <= 64 metadata did not meet the 90% feasibility gate. The best held-out median fraction of exact-over-diagonal gain retained was **-5.306**. These variants were selected on validation, evaluated once on held-out pilot rows, and were not promoted to the broad full runs. This is a negative deployability result, not a prompt to tune on test.

## Locked scientific controls

- Checkpoint revision remained `7eceff3a9f7e6f916c824d197266d86676bce695`; config and tensor-index hashes were checked before every run.
- The PR #4 gate/up/down trees and embedded Q2->Q3->Q4 deltas were read unchanged.
- Exact greedy fixes every selected coefficient; it performs no least-squares refit. The canonical label is `exact_marginal_fixed_greedy`; old `exact_marginal_omp` artifacts remain readable.
- Training requests alone fitted activation-weighted metadata and physical layouts; validation selected hyperparameters; test supplied only held-out activations and explicitly labeled H0 oracle choices.
- No H4 predictor was trained. No router, logit, token-quality, or model-quality claim is made.

## Isolated projection recovery

The file `exact_broad_ranking_audit.parquet` contains 80/90/95/99% crossings by source, request, layer, projection, expert stratum, router rank, and refinement plane. Both Q2->Q3 and Q3->Q4 are reported separately from the joint nested path. Gate/up gains reproduce across layers rather than depending on one invocation. Down interactions are smaller and inconsistent, so down remains diagonal in any deployable interpretation.

## Refresh selector frontier

`selector_refresh_frontier.parquet` records logical actions, selector time, and analytical bytes read; `action_to_page_frontier.parquet` separately records physical pages. `block_refresh_4/8/16` are H0 Pareto-relevant; initial-full-marginal and backward elimination are not reliable gate/up replacements. Exact and backward paths reach the complete endpoint, but backward is poor at low rate.

## Low-rank feasibility and storage

`gram_spectrum.parquet` and `low_rank_selector_frontier.parquet` cover ranks 8/16/32/64/128, truncated SVD, fixed JL, activation-weighted training SVD, and FP16/int8 metadata. Exact diagonals are always retained. Metadata bytes are reported per projection and extrapolated over all 40x256 routed experts.

Across all three projections, the largest evaluated metadata configuration occupies **2,396,160 bytes per expert**, or **24,536,678,400 bytes** extrapolated over 40x256 routed experts. Per-rank and per-encoding values remain in the frontier rather than being collapsed to this maximum.

The primary matched-budget metric is `(recovery_approx-recovery_diagonal)/(recovery_exact-recovery_diagonal)`. Rank <= 64 fails the 0.90 target on held-out gate/up, so no low-rank configuration was promoted to the 85/142-invocation expansion.

## Physical pages and layouts

`action_to_page_frontier.parquet` reports the bounded 512-byte page-mask oracle and exact action paths repriced to pages. Layouts are: current diagonal-trained co-selection, exact-training-mask layout, balanced hypergraph, and activation-feature support clusters. Test masks never build layouts. Page mask enumeration is exact up to 2^8 for separate gate/up pages, 2^4 for paired gate/up pages, and 2^2 for down pages.

## Paired/hybrid packets and complete experts

`paired_hybrid_frontier.parquet` distinguishes separate planes, exact Q2->Q4 coordinate packets, and H0 hybrid representation choice. Every row carries exact storage multipliers, physical bytes/pages, logical bytes/actions, and exact sequential SwiGLU 9x9x9 recovery.

The physical pilot also includes `joint_gate_up_first_order_exact_page_rescore_down_diagonal`: 16 global refreshes shortlist pages using the requested first-order gate/up SwiGLU effects propagated through the current diagonal-down reconstruction and future-proxy metric. Within each shortlist, every chosen page is rescored with actual SwiGLU and qenergy using exhaustive 2^8-or-smaller masks. This remains a bounded H0 point, not a deployable selector.

PR #4 fresh difficult-layer one-bpw result: p10 **83.34%**, median **89.35%**. Training-mask-selected exact H0 physical layout: p10 **83.51%**, median **89.26%**.

At the same selected layout and one physical bpw, median complete-expert recovery is **89.26%** for separate planes, **91.00%** for paired packets, and **92.07%** for the H0 hybrid representation choice.

These are complete-expert H0 oracle results, not deployable low-rank results. Exact page/path oracles require Gram information that exceeds the compact metadata target.

## Success gates

1. Fresh difficult-layer p10 and median improvement versus PR #4: **failed** for the predeclared exact-training-mask layout. This remains an H0 oracle result.
2. Rank <= 64 with compact metadata: **failed**.
3. Training-mask layout page amplification: exact action path **1.000x**; sparse exhaustive page-mask oracle **1.322x**. Each is judged independently against 1.2x in `success_gates.json`.
4. External storage: separate/paired/hybrid multipliers are 1.471x/1.471x/1.941x, all below 5x.
5. Broad gate/up gain: **passed at the H0 exact-oracle level**.

## Interpretation

Static diagonal paths are a real bottleneck, and short refresh blocks recover much of the exact isolated gain. However, the interaction Gram is not compact enough at rank 64 for the requested deployable selector. The correct next step is not H4 training or test-set tuning; it is a different structured metadata model or a serving-time shortlist source that does not attempt to compress the full Gram into one low-rank PSD factor.
