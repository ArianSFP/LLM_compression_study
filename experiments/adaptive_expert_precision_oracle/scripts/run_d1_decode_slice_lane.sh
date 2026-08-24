#!/usr/bin/env bash
set -euo pipefail

source /workspace/codebook_env/bin/activate

d1_decode_root=/workspace/pr13_d1_decode_slice_pilot_20260824_v2
d1_decode_exp=${d1_decode_root}/code/experiments/adaptive_expert_precision_oracle
d1_decode_checkpoint=/workspace/pr13_average_rate_all_layers_20260822_v1/inputs/checkpoint

cd "${d1_decode_exp}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=src

for d1_decode_cell in "$@"; do
  d1_decode_rate=${d1_decode_cell%%:*}
  d1_decode_layer=${d1_decode_cell##*:}
  d1_decode_layer_padded=$(printf '%02d' "${d1_decode_layer}")
  d1_decode_output=${d1_decode_root}/results/rate_${d1_decode_rate}_layer${d1_decode_layer_padded}

  python scripts/run_d1_decode_slice_oracle.py \
    --checkpoint "${d1_decode_checkpoint}" \
    --trees /workspace/codebook_granularity_study/locked/selected_trees.json \
    --fit-dir /workspace/pr13_average_rate_all_layers_20260822_v1/results/fit \
    --capture-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/baseline \
    --same-host-layers-dir /workspace/pr13_same_host_causal_controls_20260823_v1/results/same_host/layers \
    --pr13-config configs/qwen36_mxfp4_average_rate_all_layers.json \
    --same-host-config configs/qwen36_mxfp4_same_host_causal_controls.json \
    --output-dir "${d1_decode_output}" \
    --layers "${d1_decode_layer}" \
    --rates "${d1_decode_rate}" \
    --eta 0 0.00005 0.0001 0.00025 0.0005 0.001 \
    --temperature 0.0625 \
    --minimum-prefix-tokens 1 \
    --coarse-geometry-cache-dir "${d1_decode_root}/cache" \
    --workers 16 \
    --column-generated-frontiers \
    --historical-pr13-mode required
done
