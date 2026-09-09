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

## Completed development experiment

All 32 development requests, six layers and two budgets completed: 192
request-layer banks, 384 budget cells and 5,806 candidate rows. Every request
passed uncached-to-cached capture replay and repeated reference gates. All seven
available continuation probes passed their first-token consistency checks.

Peak allocated GPU memory was 12.821 GiB on the RTX 3090. The pod has an EPYC
7C13, a 27.2-core CPU quota and a 116.4-GiB RAM limit; these are not Zen4 timing
measurements. Sampled bank-preparation RAM use was about 99.3 GiB. The cached
runner took 4,849 seconds, excluding earlier captures and implementation smoke
runs. Full-bank performance variants matched original masks, residuals and
candidate records exactly.

Mean request-average terminal KL differences below use six layers per request.
Negative values improve on the indicated PR13 reference.

| Payload | Policy | Versus PR13 same payload | Versus external PR13 +24 pages/expert |
|---:|---|---:|---:|
| 360 | Within-bank hindsight best | -0.00430597 | -0.00443782 |
| 360 | Fitted metric | -0.00050096 | -0.00063282 |
| 360 | Fitted metric + D1 nonworsening | -0.00123284 | -0.00136469 |
| 725 | Within-bank hindsight best | -0.00316562 | -0.00347042 |
| 725 | Fitted metric | +0.00152622 | +0.00122142 |
| 725 | Fitted metric + D1 nonworsening | +0.00138711 | +0.00108231 |

The oracle demonstrates headroom in this bank; its choices use terminal labels.
The small fixed-feature PSD metric recovers only 11.6% of same-payload headroom
at 360 and is harmful at 725 even on development. D1 nonworsening here means
no increase in missing next-router members relative to PR13, not mandatory zero
crossings or a replication of the previous bounded D1 repair.

Current-token oracle choices improve mean future KL by 0.002275 at 360 but
worsen it by 0.005069 at 725 over seven short continuation probes. Almost all
of the latter mean regression comes from one dialogue case whose immediate
improvement is only about 3.12e-7. These small probes do not establish general
cache safety. Multilingual continuation is unavailable at the fixed position.

A descriptive coordinate-dominance check excludes about 42%/39% of exact
winners from the positive-local/nonnegative-proxy metric family, but its
optimistic nondominated-plus-incumbent envelope retains about 95% of headroom.
That envelope is not evidence that shared fitted coefficients can recover it.

## Frozen validation and artifact status

The fitted model is frozen at SHA256
`70538ee7d349ed1b97d43cbfbe23005725af35149b524ed316675ab2f63f149e`.
Validation is running with 32 held-out requests and unchanged coefficients.
No validation outcomes informed the table or model above. The 64-request
reserve remains unused. Experiment B remains paused.

Raw banks, captures and pinned inputs are mirrored locally at
`/home/arian/LLM_compression_study/tail_value_3090_artifacts` and on persistent
pod storage at `/workspace/tail_value_3090_20260909`. The committed bank seal
authenticates all 192 development banks. Shutdown using runpodctl is pending
completion and preservation of the validation work.
