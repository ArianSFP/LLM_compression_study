# Direct replay identity audit — 2026-08-18

## Result

Replay identity did **not** pass, so no H1–H4 hidden-state, router, logit, or token metric is reported.

## What is available

The compact four-layer extraction has exact request/sequence IDs, token positions, prefix hashes, pre-MoE expert input `x`, captured routed BF16 output, post-MoE state `xplus`, top-8 expert IDs and weights, current router logits, and H1–H4 post-MoE states/router logits/expert IDs. The source capture corpus also records deterministic greedy generation events and was produced by a frozen BF16 Transformers checkpoint at model revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.

## Identity blocker

The correction study's `W_ref` is dequantized from the production mixed-format GGUF `Qwen3.6-35B-A3B-MXFP4_MOE.gguf`, while the only complete replayable execution graph and KV state correspond to the BF16 Transformers checkpoint. The two are closely aligned but not identical: sampled expert tensor relative RMSE ranges from roughly 1.8% to 11.6%, depending on layer and projection. A zero-delta or unmodified-output injection in the BF16 graph therefore cannot establish identity for the production-reference GGUF computation used by Run A and Run B.

The compact extraction also omits the complete teacher token stream, attention KV cache or a deterministic reconstruction checkpoint, the pre-MoE residual/shared-expert contribution needed to replace exactly one routed aggregate, and final current/future logits. Existing repository "replay" utilities are trace/cache-policy replays or MTP capture hooks, not an arbitrary routed-MoE-output injection path for the production GGUF runtime.

## Minimum next capture/runtime change

1. Add a hook in the exact production GGUF/llama.cpp build at the routed-expert aggregation boundary, before residual addition.
2. Save and validate the authoritative routed aggregate, shared-expert contribution, pre-add residual, token IDs, positions, attention/KV state or deterministic prefix, router logits/weights, final logits, and build/quantization hashes.
3. Support replacement by either the unchanged aggregate or an arbitrary delta.
4. Require the unchanged replacement to reproduce H0 state, H1–H4 teacher-forced states/routes, and logits within a documented numeric tolerance.
5. Only after that identity test passes, inject Run A/Run B outputs and report causal propagation metrics.

Running a BF16 replay and labelling it as production-reference replay would mix model/quantization provenance and was intentionally rejected.
