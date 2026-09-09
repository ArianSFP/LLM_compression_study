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

## Completed frozen validation

The model remained frozen at SHA256
`70538ee7d349ed1b97d43cbfbe23005725af35149b524ed316675ab2f63f149e`.
All 32 held-out requests, 192 banks, 384 budget cells and 5,826 candidate rows
passed verification. Five continuation probes passed repeated-reference and
first-token consistency gates. Three first-domain requests have no continuation
at the fixed position (dialogue/instruction, multilingual and reasoning).
The run took 5,763 seconds and peaked at 12.821 GiB allocated GPU memory.

Negative KL/NLL differences improve on PR13 at the same payload. These are
means of per-request averages across six layers, with 32 requests per row.

| Payload | Policy | Mean delta KL | Mean delta NLL | Fraction of requests with worse KL |
|---:|---|---:|---:|---:|
| 360 | metric | -0.00143915 | +0.00291358 | 46.9% |
| 360 | metric_d1_nonworsening | -0.00143099 | +0.00286427 | 46.9% |
| 360 | oracle | -0.00349014 | -0.02188374 | 0.0% |
| 360 | oracle_d1_nonworsening | -0.00348910 | -0.02232689 | 0.0% |
| 725 | metric | +0.00081754 | -0.00428953 | 53.1% |
| 725 | metric_d1_nonworsening | +0.00079901 | -0.00140974 | 50.0% |
| 725 | oracle | -0.00342503 | -0.01632303 | 0.0% |
| 725 | oracle_d1_nonworsening | -0.00342021 | -0.01558010 | 0.0% |

The fitted metric recovers 41.2% of same-payload hindsight headroom at 360,
but worsens mean KL at 725. Its 360 KL improvement accompanies slightly worse
mean NLL, so the two objectives should not be conflated. The D1 restriction
barely changes the oracle headroom and does not repair the metric's 725 result.
The full [analysis](validation/analysis.json) includes p95, maxima, external
higher-payload comparisons and other fixed selectors; request and stratum
parquets preserve the underlying contrasts.

Short continuation probes compare future exact decoding after a single
layer-6 compressed token. They average five selected requests, not all 32.

| Payload | Policy | Mean future delta KL | Fraction of probed requests harmed |
|---:|---|---:|---:|
| 360 | current_token_oracle | +0.00243762 | 60% |
| 360 | metric | +0.01115649 | 100% |
| 725 | current_token_oracle | -0.00345552 | 20% |
| 725 | metric | +0.00732257 | 60% |

Metric+D1 selects the same choices as the metric in these probes. Future KL
regressions at both budgets prevent a cache-safety claim. The hindsight oracle's
future effects also change sign between budgets and cohorts. Current-token KL
alone is insufficient evidence of safe continuation. The validation dominance
envelope retains about 97% of headroom; this remains an optimistic capacity
diagnostic, not a realizable shared selector.

## Targeted live all-layer pilot — in progress

The plan was sealed before validation labels. Two selected development cases
(first code and the dialogue continuation outlier), two budgets and two
policies produce eight independent trajectories. Each runs two compressed
tokens across all 40 routed-expert layers, followed by four exact future tokens.
Every candidate bank is rebuilt from the trajectory's live activations and
routes. The two-choice oracle compares PR13 with exact-local allocation using
current-token KL. This is a targeted diagnostic, not a population estimate,
global oracle or runtime controller. No validation results alter its settings.

## Artifact and promotion status

The 64-request reserve remains unused. Experiment A remains failed and
Experiment B remains paused. No deployment or runtime speed claim follows from
these oracle emulations. A larger GPU is unnecessary for these numerical
experiments, but longer/broader live studies will benefit from more resident
expert weights. These measured timings use EPYC 7C13, not Zen4; the model's
CPU-resident BF16 expert weights also require substantial system RAM.

Raw banks, captures and pinned inputs are mirrored locally at
`/home/arian/LLM_compression_study/tail_value_3090_artifacts` and on persistent
pod storage at `/workspace/tail_value_3090_20260909`. All development and
validation banks and all 40 original factor files have verified local hashes.
Shutdown using runpodctl is pending completion and preservation of the live
pilot. Changes are committed locally; public publication needs explicit
approval for the included tokenized manifests and request-level artifacts.
