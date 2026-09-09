# Tail-value feasibility and effective-prefix audit

Parent result: `8cd2230a21bbe4aca2910ec7fb6b5426bd43eda7`.
Methods: [TAIL_VALUE_3090_METHODS.md](../../TAIL_VALUE_3090_METHODS.md).

## Completed cohort findings

The old authenticated manifest has 32 calibration request IDs but 26 distinct
consumed inputs; 96 evaluation IDs but 74 distinct consumed inputs. All 12
evaluation summarisation rows share one input and all 12 long-context rows share
another. Both inputs occur in calibration: 24 evaluation rows repeat an input
used for calibration. This is effective-input overlap, not evidence that old
calibration consumed terminal labels.

The repeated summarisation input contributes 99.1773% of the net layer-6
360-vs-384 product regression. Equal weighting of distinct inputs changes both
product mean signs, but also changes domain weights and the estimand. It is a
descriptive sensitivity calculation, not replacement confirmatory inference.
The old Experiment A promotion decision remains **failed**.

## Corrected cohort

All 128 old complete prompts are excluded from the 256-request source. The
remaining 128 prompts are split deterministically, within each domain, into 32
development, 32 validation, and 64 reserved requests. The revised position rule
is `min(63,n-2)`. All 128 revised consumed inputs are distinct and none equals
an old consumed input. The source lacks document/template grouping metadata;
uniqueness alone does not establish independence of related templates.

No reserved request may receive a model outcome in this experiment. This
reservation is scoped to this experiment, not an assertion about every earlier
project's use of the source corpus.

## Reproduction

`scripts/audit_tail_value_cohort.py` generates the cohort audit and manifests from
the pinned original inputs; `scripts/analyze_effective_prefix_sensitivity.py`
reproduces the descriptive reweighting from the published parquets and saved
effective-input membership hashes. The candidate generator reads no terminal
labels and the new model runner seals its banks before label execution.

GPU execution status and final handoff will be recorded after preflight and the
feasible 3090 work finish. The pod is identified by its exact SSH endpoint;
shutdown will use `runpodctl` after artifacts are preserved.
