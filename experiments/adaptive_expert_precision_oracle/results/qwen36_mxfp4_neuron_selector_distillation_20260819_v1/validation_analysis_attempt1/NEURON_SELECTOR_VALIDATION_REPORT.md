# Neuron selector distillation: validation decision

This document contains validation-only evidence. No held-out test row was
loaded by the promotion path. Recovery is exact sequential complete-expert
qenergy recovery, not accuracy, logits, routing quality or token quality.

## Factorized G/D neuron oracle

- validation overlap with the frozen PR #7 exact 64x16 tile: 55 invocations;
- factorized p10 / median at one physical bpw: 0.869842 / 0.932810;
- tile p10 / median: 0.935183 / 0.966025;
- gap to tile, p10 / median: 0.065342 / 0.033215;
- decision: **STOP**.

## Residual-response predictor candidates

| training_cohort | basis_variant | rank | synthesis_encoding | candidate_recovery_p10 | candidate_recovery_median | candidate_utility_p10 | candidate_utility_median | direct_recovery_p10 | direct_recovery_median | selector_metadata_bpw | passes_candidate | passes_direct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 8 | fp16 | 0.752086 | 0.874731 | 0.899230 | 0.945447 | 0.800815 | 0.897328 | 0.049835 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 8 | fp8_e4m3fn_per_row | 0.753170 | 0.874731 | 0.897473 | 0.945447 | 0.800120 | 0.897933 | 0.034210 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 8 | int8_per_row | 0.752937 | 0.874731 | 0.899230 | 0.945447 | 0.801270 | 0.897328 | 0.034210 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 16 | fp16 | 0.738666 | 0.876580 | 0.906161 | 0.950673 | 0.805169 | 0.890897 | 0.091827 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 16 | fp8_e4m3fn_per_row | 0.747737 | 0.868751 | 0.906161 | 0.950513 | 0.803307 | 0.890437 | 0.055369 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 16 | int8_per_row | 0.738666 | 0.876580 | 0.906161 | 0.950673 | 0.802110 | 0.892092 | 0.055369 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 32 | fp16 | 0.740103 | 0.877852 | 0.906702 | 0.950736 | 0.806151 | 0.892383 | 0.175812 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 32 | fp8_e4m3fn_per_row | 0.740103 | 0.874340 | 0.906702 | 0.948028 | 0.801184 | 0.896432 | 0.097687 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 32 | int8_per_row | 0.742024 | 0.874340 | 0.906702 | 0.952095 | 0.805104 | 0.894635 | 0.097687 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 64 | fp16 | 0.785019 | 0.872731 | 0.918463 | 0.957011 | 0.808849 | 0.886970 | 0.343781 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 64 | fp8_e4m3fn_per_row | 0.765384 | 0.883882 | 0.917397 | 0.954745 | 0.805343 | 0.886120 | 0.182322 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 64 | int8_per_row | 0.782407 | 0.872731 | 0.918177 | 0.957011 | 0.805183 | 0.883967 | 0.182322 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 128 | fp16 | 0.750122 | 0.884125 | 0.918667 | 0.962029 | 0.821878 | 0.902296 | 0.679718 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 128 | fp8_e4m3fn_per_row | 0.755193 | 0.881033 | 0.918239 | 0.962056 | 0.811040 | 0.902491 | 0.351593 | False | False |
| combined_exact_and_cross_train | layer_shared_joint_gate_up | 128 | int8_per_row | 0.764245 | 0.884125 | 0.918386 | 0.962029 | 0.808055 | 0.902491 | 0.351593 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 8 | fp16 | 0.739971 | 0.864743 | 0.897846 | 0.942951 | 0.787888 | 0.879360 | 0.050161 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 8 | fp8_e4m3fn_per_row | 0.739971 | 0.870771 | 0.897846 | 0.941312 | 0.793551 | 0.880634 | 0.034536 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 8 | int8_per_row | 0.745324 | 0.864743 | 0.897846 | 0.942951 | 0.792569 | 0.883328 | 0.034536 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 16 | fp16 | 0.744483 | 0.880174 | 0.903517 | 0.951212 | 0.796644 | 0.895003 | 0.092478 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 16 | fp8_e4m3fn_per_row | 0.744633 | 0.873976 | 0.904284 | 0.951212 | 0.798760 | 0.893151 | 0.056020 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 16 | int8_per_row | 0.744483 | 0.880174 | 0.903517 | 0.950751 | 0.794724 | 0.896249 | 0.056020 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 32 | fp16 | 0.748666 | 0.880767 | 0.912522 | 0.953310 | 0.786480 | 0.898970 | 0.177114 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 32 | fp8_e4m3fn_per_row | 0.754477 | 0.884473 | 0.911682 | 0.953416 | 0.792652 | 0.899846 | 0.098989 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 32 | int8_per_row | 0.750120 | 0.880767 | 0.912470 | 0.954195 | 0.786721 | 0.898970 | 0.098989 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | fp16 | 0.758823 | 0.886906 | 0.919229 | 0.954232 | 0.804542 | 0.894950 | 0.346385 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | fp8_e4m3fn_per_row | 0.755966 | 0.888333 | 0.916435 | 0.956179 | 0.801593 | 0.897129 | 0.184926 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 64 | int8_per_row | 0.758823 | 0.884350 | 0.916005 | 0.956062 | 0.804922 | 0.894950 | 0.184926 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 128 | fp16 | 0.778739 | 0.885306 | 0.919919 | 0.962996 | 0.806039 | 0.904108 | 0.684926 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 128 | fp8_e4m3fn_per_row | 0.779949 | 0.886576 | 0.918793 | 0.963084 | 0.811488 | 0.905534 | 0.356801 | False | False |
| combined_exact_and_cross_train | layer_shared_separate_gate_up | 128 | int8_per_row | 0.780287 | 0.883714 | 0.918793 | 0.962425 | 0.804551 | 0.901277 | 0.356801 | False | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 8 | fp16 | 0.754277 | 0.860886 | 0.896561 | 0.947225 | 0.797695 | 0.886688 | 0.132843 | False | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 16 | fp16 | 0.757743 | 0.882760 | 0.907111 | 0.950391 | 0.803657 | 0.894648 | 0.257843 | False | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 32 | fp16 | 0.750229 | 0.880770 | 0.911357 | 0.947771 | 0.797167 | 0.897270 | 0.507843 | False | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 64 | fp16 | 0.767803 | 0.885793 | 0.916576 | 0.956998 | 0.804949 | 0.903170 | 1.007843 | False | False |
| combined_exact_and_cross_train | per_expert_joint_gate_up_upper_bound | 128 | fp16 | 0.752259 | 0.885230 | 0.910979 | 0.966659 | 0.803704 | 0.910413 | 2.007843 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 8 | fp16 | 0.756169 | 0.874915 | 0.903746 | 0.949318 | 0.785090 | 0.885775 | 0.049835 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 8 | fp8_e4m3fn_per_row | 0.754893 | 0.874915 | 0.903746 | 0.949318 | 0.779211 | 0.884628 | 0.034210 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 8 | int8_per_row | 0.742491 | 0.874915 | 0.903033 | 0.949033 | 0.786016 | 0.885775 | 0.034210 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 16 | fp16 | 0.742403 | 0.867897 | 0.912274 | 0.951874 | 0.781842 | 0.887647 | 0.091827 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 16 | fp8_e4m3fn_per_row | 0.742403 | 0.869090 | 0.913410 | 0.952895 | 0.775981 | 0.885798 | 0.055369 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 16 | int8_per_row | 0.742403 | 0.867897 | 0.908706 | 0.952665 | 0.781842 | 0.887647 | 0.055369 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 32 | fp16 | 0.758201 | 0.876495 | 0.913671 | 0.962408 | 0.799594 | 0.896837 | 0.175812 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 32 | fp8_e4m3fn_per_row | 0.753561 | 0.878158 | 0.915085 | 0.958926 | 0.781778 | 0.898740 | 0.097687 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 32 | int8_per_row | 0.759627 | 0.876495 | 0.913647 | 0.962408 | 0.804551 | 0.896837 | 0.097687 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 64 | fp16 | 0.725754 | 0.882939 | 0.918087 | 0.960953 | 0.786482 | 0.901135 | 0.343781 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 64 | fp8_e4m3fn_per_row | 0.716501 | 0.878560 | 0.916031 | 0.960400 | 0.782132 | 0.894229 | 0.182322 | False | False |
| exact_checkpoint_train_only | layer_shared_joint_gate_up | 64 | int8_per_row | 0.732753 | 0.883036 | 0.915904 | 0.957468 | 0.781915 | 0.896113 | 0.182322 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 8 | fp16 | 0.758066 | 0.866793 | 0.905854 | 0.949318 | 0.788872 | 0.888987 | 0.050161 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 8 | fp8_e4m3fn_per_row | 0.757953 | 0.870991 | 0.905854 | 0.948290 | 0.793126 | 0.887171 | 0.034536 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 8 | int8_per_row | 0.758066 | 0.866793 | 0.905854 | 0.949318 | 0.788872 | 0.887864 | 0.034536 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 16 | fp16 | 0.745329 | 0.867266 | 0.912838 | 0.952395 | 0.789801 | 0.888617 | 0.092478 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 16 | fp8_e4m3fn_per_row | 0.745072 | 0.867559 | 0.912838 | 0.952395 | 0.781888 | 0.889820 | 0.056020 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 16 | int8_per_row | 0.746603 | 0.867559 | 0.912818 | 0.952612 | 0.785916 | 0.888617 | 0.056020 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 32 | fp16 | 0.757753 | 0.877104 | 0.911297 | 0.962360 | 0.802971 | 0.901525 | 0.177114 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 32 | fp8_e4m3fn_per_row | 0.753164 | 0.874932 | 0.914510 | 0.963144 | 0.791317 | 0.899554 | 0.098989 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 32 | int8_per_row | 0.758851 | 0.876219 | 0.911297 | 0.960334 | 0.802971 | 0.896935 | 0.098989 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 64 | fp16 | 0.748628 | 0.875969 | 0.914372 | 0.960953 | 0.782870 | 0.903961 | 0.346385 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 64 | fp8_e4m3fn_per_row | 0.735294 | 0.873223 | 0.917800 | 0.961248 | 0.778042 | 0.900777 | 0.184926 | False | False |
| exact_checkpoint_train_only | layer_shared_separate_gate_up | 64 | int8_per_row | 0.746897 | 0.872603 | 0.915811 | 0.960523 | 0.783130 | 0.900777 | 0.184926 | False | False |
| exact_checkpoint_train_only | per_expert_joint_gate_up_upper_bound | 8 | fp16 | 0.760906 | 0.872534 | 0.911041 | 0.949647 | 0.781724 | 0.884497 | 0.132843 | False | False |
| exact_checkpoint_train_only | per_expert_joint_gate_up_upper_bound | 16 | fp16 | 0.742608 | 0.869478 | 0.909781 | 0.952102 | 0.787719 | 0.893343 | 0.257843 | False | False |
| exact_checkpoint_train_only | per_expert_joint_gate_up_upper_bound | 32 | fp16 | 0.732479 | 0.880736 | 0.914431 | 0.960825 | 0.789212 | 0.898128 | 0.507843 | False | False |
| exact_checkpoint_train_only | per_expert_joint_gate_up_upper_bound | 64 | fp16 | 0.744761 | 0.873223 | 0.911195 | 0.963867 | 0.780015 | 0.906058 | 1.007843 | False | False |

Decision: **STOP**. Conditional
follow-up: response predictor stopped; no conditional expansion.

The sampled per-expert basis is an upper bound and is never promoted as a
bank-wide deployable selector from this small expert sample.

## Provenance

- config SHA-256: `41eb0e21cdb016017d8f3b4a48bcdd4425323c463d96afedd0bf585cfbedf3d4`
- fit bundle-manifest SHA-256: `73b0285be3e3611fa44ac0995bd69c3b4390cd02b0fa9ac03a9626397c878d60`
- promotions SHA-256: `d7733971c71f8e4ce9fa9674562346c314624d406b95b47d38daa08b58c79c5d`
- test rows consulted for selection: `false`
