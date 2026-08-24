#!/usr/bin/env bash
set -euo pipefail

d1_decode_root=/workspace/pr13_d1_decode_slice_pilot_20260824_v2
d1_decode_exp=${d1_decode_root}/code/experiments/adaptive_expert_precision_oracle

cd "${d1_decode_exp}"

for d1_decode_cell in "$@"; do
  d1_decode_rate=${d1_decode_cell%%:*}
  d1_decode_layer=${d1_decode_cell##*:}
  d1_decode_layer_padded=$(printf '%02d' "${d1_decode_layer}")
  d1_decode_log=${d1_decode_root}/evidence/rate_${d1_decode_rate}_layer${d1_decode_layer_padded}.log
  bash scripts/run_d1_decode_slice_lane.sh "${d1_decode_cell}" 2>&1 \
    | tee "${d1_decode_log}"
done
