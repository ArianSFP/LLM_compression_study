#!/usr/bin/env bash
set -euo pipefail

source /workspace/codebook_env/bin/activate

d1_matched_root=/workspace/pr13_d1_decode_matched_rates_20260824_v2
d1_source_root=/workspace/pr13_d1_decode_slice_pilot_20260824_v2
d1_exp=${d1_source_root}/code/experiments/adaptive_expert_precision_oracle
d1_checkpoint=/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint

cd "${d1_exp}"
mkdir -p "${d1_matched_root}/results" "${d1_matched_root}/evidence"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=src

for d1_cell in "$@"; do
  d1_rate=${d1_cell%%:*}
  d1_layer=${d1_cell##*:}
  d1_layer_padded=$(printf '%02d' "${d1_layer}")
  d1_output=${d1_matched_root}/results/rate_${d1_rate}_layer${d1_layer_padded}
  d1_log=${d1_matched_root}/evidence/rate_${d1_rate}_layer${d1_layer_padded}.log

  python scripts/run_d1_decode_slice_oracle.py \
    --checkpoint "${d1_checkpoint}" \
    --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
    --fit-dir /workspace/pr13_average_rate_all_layers_20260822_v1/results/fit \
    --capture-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/baseline \
    --same-host-layers-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/layers \
    --pr13-config configs/qwen36_mxfp4_average_rate_all_layers.json \
    --same-host-config configs/qwen36_mxfp4_same_host_causal_controls.json \
    --output-dir "${d1_output}" \
    --layers "${d1_layer}" \
    --rates "${d1_rate}" \
    --eta 0 0.00005 0.0001 0.00025 0.0005 0.001 \
    --temperature 0.0625 \
    --minimum-prefix-tokens 1 \
    --coarse-geometry-cache-dir "${d1_source_root}/cache" \
    --workers 16 \
    --column-generated-frontiers \
    --historical-pr13-mode omit 2>&1 | tee "${d1_log}"
done
