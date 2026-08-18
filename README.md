# LLM Compression Study

This repository contains the external-review package for the **Adaptive Expert Precision H0 Oracle Pilot** on Qwen3.6-35B-A3B.

Start here:

- [Full oracle study report](experiments/adaptive_expert_precision_oracle/ORACLE_STUDY_REPORT.md)
- [Experiment README and reproduction commands](experiments/adaptive_expert_precision_oracle/README.md)
- [Execution ledger, environment, commands, timings, and hashes](experiments/adaptive_expert_precision_oracle/EXECUTION_LEDGER_20260818.md)
- [Principal machine-readable results](experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817/)
- [Plots in PNG and SVG](experiments/adaptive_expert_precision_oracle/results/qwen36_mxfp4_varied_pilot_extended_20260817/plots/)

## Headline finding

The hypothesis is **partially supported and strongly layer-dependent**. A deployable generalized-gate/generalized-up/native-down hybrid recovered 78.31% median complete-expert future-proxy-weighted damage at 1.9896 physical streamed bpw over layers 0/20/39. Layer 0 was exceptional, reaching 96.69% at 2 bpw, but the six-layer aggregate reached only 76.65% at 2 bpw and 89.65% at 4 bpw. Q2 is the more promising resident base: at 2 correction bpw it reached 85.53%, a paired mean gain of 3.50 percentage points over W1.

The report distinguishes deployable single-view results from non-deployable multi-view H0 ceilings and explicitly marks unmeasured direct H1-H4 replay, joint top-8 allocation, and token/logit metrics. No synthetic propagation results are substituted.

## Review scope

The repository includes source, tests, exact configs, CSV/Parquet statistics, JSON accounting/manifests, and all generated PNG/SVG figures. Raw activation captures, checkpoints, and model tensors are intentionally omitted because of size and licensing; their external paths and cryptographic hashes are recorded in the execution ledger and capture manifest.
