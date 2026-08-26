#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path("/workspace/LLM_compression_study_d1_nested")
EXPERIMENT = REPO / "experiments" / "adaptive_expert_precision_oracle"
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

import run_d1_nested_safe_oracle as nested  # noqa: E402


request_id = "9118185608909195810"
layer = 6
capture_dir = Path(
    "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/captures_authenticated"
)
capture = nested.load_capture(
    nested.capture_path(capture_dir, request_id), layer,
)
position = 33
model_slice = nested.load_d1_layer_slice(
    Path("/workspace/qwen36_mxfp4_candidate"), layer, device="cuda",
)
stored_hidden = torch.as_tensor(
    np.asarray(capture["hidden"][layer], np.float32),
    device=model_slice.device,
    dtype=torch.bfloat16,
).unsqueeze(0)
stored_current = stored_hidden[:, position : position + 1]
with torch.inference_mode():
    prefix = model_slice.decode_prefix_cache(stored_hidden[:, :position])
    raw_logits, _, raw_ids, _ = model_slice.next_router_outputs_decode(
        stored_current, nested.clone_decode_cache(prefix), position,
    )

raw_cuda = raw_logits[0, 0].float()
raw = raw_cuda.detach().cpu().numpy()
raw_ids_np = raw_ids[0, 0].detach().cpu().numpy()
target = np.asarray(capture["router_logits"][layer + 1, position], np.float32)
target_ids = np.asarray(capture["router_ids"][layer + 1, position], np.int64)
anchor = np.asarray(target - raw, np.float32)
anchored_cuda = raw_cuda + torch.as_tensor(
    anchor, device=raw_cuda.device, dtype=torch.float32,
)
_, anchored_ids_cuda = nested._cuda_softmax_top8(anchored_cuda)
anchored = anchored_cuda.detach().cpu().numpy()
anchored_ids = anchored_ids_cuda.detach().cpu().numpy()

raw_order = torch.topk(
    torch.softmax(raw_cuda, dim=-1), 16, largest=True, sorted=True,
).indices.detach().cpu().numpy()
target_tensor = torch.as_tensor(target, device="cuda", dtype=torch.float32)
target_prob, target_top16 = torch.topk(
    torch.softmax(target_tensor, dim=-1), 16, largest=True, sorted=True,
)
target_order = target_top16.detach().cpu().numpy()

def rows(order: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "expert": int(expert),
            "raw": float(raw[int(expert)]),
            "target": float(target[int(expert)]),
            "anchor": float(anchor[int(expert)]),
        }
        for expert in order
    ]

payload = {
    "request_id": request_id,
    "layer": layer,
    "position": position,
    "raw_max_abs": float(np.max(np.abs(anchor), initial=0.0)),
    "anchored_logits_bit_identical": bool(np.array_equal(anchored, target)),
    "raw_ids": raw_ids_np.tolist(),
    "target_ids": target_ids.tolist(),
    "anchored_ids": anchored_ids.tolist(),
    "raw_vs_target_ordered_equal": bool(np.array_equal(raw_ids_np, target_ids)),
    "raw_vs_target_set_equal": bool(set(raw_ids_np.tolist()) == set(target_ids.tolist())),
    "anchored_vs_target_ordered_equal": bool(np.array_equal(anchored_ids, target_ids)),
    "raw_top16": rows(raw_order),
    "target_top16": rows(target_order),
    "target_rank8_rank9_probability_gap": float(target_prob[7] - target_prob[8]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
