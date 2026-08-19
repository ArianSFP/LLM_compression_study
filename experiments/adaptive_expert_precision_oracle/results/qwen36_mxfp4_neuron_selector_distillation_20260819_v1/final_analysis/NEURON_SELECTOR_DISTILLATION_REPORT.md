# Neuron-selector distillation report

## Outcome

Both predeclared workstreams stop on exact-checkpoint validation. Factorized
gate/up-versus-down actions help at 0.5 bpw but do not close the nonlinear tile
gap at 1.0 bpw. Compact residual-response models recover much independent
unit-score utility, yet fail complete-expert recovery. No test or cross-
reference evaluation was launched.

This is exact sequential complete-expert qenergy recovery, not task accuracy,
token quality, logits, or routing quality. Every studied predictor is H0-late
and requires a second pass; no H4 predictor was trained.

## Workstream A: factorized G/D neuron actions

### Matched frontier

| representation_label | physical_budget_bpw | recovery_n | recovery_p10 | recovery_median | recovery_p90 | physical_pages_median |
|---|---|---|---|---|---|---|
| PR7 independent-unit score | 0.500000 | 55 | 0.717999 | 0.865659 | 0.987667 | 384.000000 |
| PR7 independent-unit score | 0.750000 | 55 | 0.838805 | 0.925812 | 0.999145 | 576.000000 |
| PR7 independent-unit score | 1.000000 | 55 | 0.915015 | 0.958289 | 1.000000 | 768.000000 |
| coherent complete-unit packet | 0.500000 | 55 | 0.717999 | 0.865659 | 0.987667 | 384.000000 |
| coherent complete-unit packet | 0.750000 | 55 | 0.838805 | 0.925812 | 0.999145 | 576.000000 |
| coherent complete-unit packet | 1.000000 | 55 | 0.915015 | 0.958289 | 1.000000 | 768.000000 |
| exact 64x16 tile | 0.500000 | 55 | 0.893324 | 0.938685 | 0.979288 | 384.000000 |
| exact 64x16 tile | 0.750000 | 55 | 0.926209 | 0.960589 | 0.981732 | 576.000000 |
| exact 64x16 tile | 1.000000 | 55 | 0.935183 | 0.966025 | 0.981914 | 768.000000 |
| factorized G/D fixed-greedy | 0.500000 | 55 | 0.827137 | 0.901541 | 0.958717 | 384.000000 |
| factorized G/D fixed-greedy | 0.750000 | 55 | 0.858382 | 0.923228 | 0.962574 | 576.000000 |
| factorized G/D fixed-greedy | 1.000000 | 55 | 0.869842 | 0.932810 | 0.965431 | 768.000000 |

At 0.5 bpw the factorized path materially improves on coherent packets, but
the advantage disappears by 0.75 bpw and reverses at 1.0 bpw.

### One-bpw accounting

| representation_label | recovery_n | recovery_p10 | recovery_median | recovery_p90 | physical_pages_median | logical_actions_median | selector_runtime_ms_median |
|---|---|---|---|---|---|---|---|
| PR7 independent-unit score | 55 | 0.915015 | 0.958289 | 1.000000 | 768.000000 | 256.000000 | 5.669367 |
| coherent complete-unit packet | 55 | 0.915015 | 0.958289 | 1.000000 | 768.000000 | 256.000000 | 4.014462 |
| exact 64x16 tile | 55 | 0.935183 | 0.966025 | 0.981914 | 768.000000 | 768.000000 | 1395.575506 |
| factorized G/D fixed-greedy | 55 | 0.869842 | 0.932810 | 0.965431 | 768.000000 | 536.000000 | 320.884643 |

Decision: **STOP**. Factorized p10/median was
0.869842/0.932810,
trailing the frozen 64x16 tile by
0.065342/0.033215.
The fixed-greedy path is not a global subset optimum; it retains the best
cumulative prefix under each paid-page cap.

### One-bpw layer breakdown (matched validation cohort)

| representation_label | layer | recovery_n | recovery_p10 | recovery_median | physical_pages_median |
|---|---|---|---|---|---|
| PR7 independent-unit score | 0 | 7 | 1.000000 | 1.000000 | 768.000000 |
| PR7 independent-unit score | 4 | 16 | 0.902255 | 0.947783 | 768.000000 |
| PR7 independent-unit score | 20 | 16 | 0.910499 | 0.933510 | 768.000000 |
| PR7 independent-unit score | 39 | 16 | 0.956278 | 0.983987 | 768.000000 |
| coherent complete-unit packet | 0 | 7 | 1.000000 | 1.000000 | 768.000000 |
| coherent complete-unit packet | 4 | 16 | 0.902255 | 0.947783 | 768.000000 |
| coherent complete-unit packet | 20 | 16 | 0.910499 | 0.933510 | 768.000000 |
| coherent complete-unit packet | 39 | 16 | 0.956278 | 0.983987 | 768.000000 |
| exact 64x16 tile | 0 | 7 | 0.945143 | 0.967635 | 768.000000 |
| exact 64x16 tile | 4 | 16 | 0.937467 | 0.961106 | 768.000000 |
| exact 64x16 tile | 20 | 16 | 0.938186 | 0.965790 | 768.000000 |
| exact 64x16 tile | 39 | 16 | 0.938839 | 0.974929 | 768.000000 |
| factorized G/D fixed-greedy | 0 | 7 | 0.871796 | 0.948150 | 559.000000 |
| factorized G/D fixed-greedy | 4 | 16 | 0.887194 | 0.909235 | 768.000000 |
| factorized G/D fixed-greedy | 20 | 16 | 0.882703 | 0.926787 | 768.000000 |
| factorized G/D fixed-greedy | 39 | 16 | 0.850400 | 0.943990 | 768.000000 |

## Workstream B: predict h4 and score with A/B/C

The exact scalar statistic is compact: three normalized FP16 vectors plus
FP32 scales require 3,084 bytes/expert
(0.007843 bpw). Ranks 8/16/32/64/128, joint
and separate layer bases, two training cohorts, FP16/row-FP8/row-INT8, and a
sampled per-expert upper bound were evaluated.

Best deployable candidate by recovery (metadata <= 0.35 bpw):

| training_cohort | basis_variant | rank | synthesis_encoding | candidate_recovery_p10 | candidate_recovery_median | candidate_utility_p10 | candidate_utility_median | selector_metadata_bpw | selector_compute_macs |
|---|---|---|---|---|---|---|---|---|---|
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | fp8_e4m3fn_per_row | 0.755966 | 0.888333 | 0.916435 | 0.956179 | 0.184926 | 327680 |

Decision: **STOP**. Even the sampled per-expert upper bound failed, so neither
the activation-top-k nor cluster-basis continuation was run.

### Ten strongest candidate-rerank configurations

| training_cohort | basis_variant | rank | synthesis_encoding | candidate_recovery_p10 | candidate_recovery_median | candidate_utility_p10 | candidate_utility_median | selector_metadata_bpw | sampled_per_expert_upper_bound |
|---|---|---|---|---|---|---|---|---|---|
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | fp8_e4m3fn_per_row | 0.755966 | 0.888333 | 0.916435 | 0.956179 | 0.184926 | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | fp16 | 0.758823 | 0.886906 | 0.919229 | 0.954232 | 0.346385 | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 128 | fp8_e4m3fn_per_row | 0.779949 | 0.886576 | 0.918793 | 0.963084 | 0.356801 | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 64 | fp16 | 0.767803 | 0.885793 | 0.916576 | 0.956998 | 1.007843 | True |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 128 | fp16 | 0.778739 | 0.885306 | 0.919919 | 0.962996 | 0.684926 | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 128 | fp16 | 0.752259 | 0.885230 | 0.910979 | 0.966659 | 2.007843 | True |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 32 | fp8_e4m3fn_per_row | 0.754477 | 0.884473 | 0.911682 | 0.953416 | 0.098989 | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | int8_per_row | 0.758823 | 0.884350 | 0.916005 | 0.956062 | 0.184926 | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 128 | int8_per_row | 0.764245 | 0.884125 | 0.918386 | 0.962029 | 0.351593 | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 128 | fp16 | 0.750122 | 0.884125 | 0.918667 | 0.962029 | 0.679718 | False |

## Interpretation

PR #7's neuron sparsity remains real, but independently scoring units from
predicted h4 is insufficient. Factorization is useful only at low rates; its
partially refined states make the one-bpw greedy endgame worse than coherent
packets. The next justified target is marginal complete-expert utility or
support templates on a much larger exact-checkpoint corpus—not another rank
or activation-top-k sweep on these validation requests.

## Controls and provenance

- fit used complete training requests only;
- selection used 69 exact-checkpoint validation invocations;
- frozen matched PR #7 comparison used 55 validation invocations;
- test rows consulted: **false**; held-out evaluation launched: **false**;
- codec, selected trees, checkpoint revision and request separation unchanged;
- config: `41eb0e21cdb016017d8f3b4a48bcdd4425323c463d96afedd0bf585cfbedf3d4`;
- fit bundle manifest: `73b0285be3e3611fa44ac0995bd69c3b4390cd02b0fa9ac03a9626397c878d60`;
- frozen validation promotion: `7430a885d6dfa303ac1a54357fa7a86f54bd9000df46d23f2727fd45fe961bc4`.
