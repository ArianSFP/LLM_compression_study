#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path("/workspace/LLM_compression_study_d1_nested")
EXPERIMENT = REPO / "experiments" / "adaptive_expert_precision_oracle"
sys.path[:0] = [str(EXPERIMENT / "src"), str(EXPERIMENT / "scripts")]

import run_d1_nested_safe_oracle as nested  # noqa: E402


manifest = Path(
    "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/inputs/"
    "d1_nested_requests.jsonl"
)
capture_dir = Path(
    "/workspace/pr13_d1_nested_safe_oracle_20260825_v1/captures_authenticated"
)
checkpoint = Path("/workspace/qwen36_mxfp4_candidate")
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
layers = (0, 1, 4, 6, 12, 23)
details: list[dict[str, object]] = []
contexts = 0
raw_order_matches = 0
raw_set_matches = 0
anchored_order_matches = 0
anchored_bit_matches = 0
maximum_anchor = 0.0
by_split = {
    split: {"contexts": 0, "raw_order_mismatches": 0, "raw_set_mismatches": 0}
    for split in ("calibration", "evaluation")
}

for layer in layers:
    model_slice = nested.load_d1_layer_slice(checkpoint, layer, device="cuda")
    print(f"[raw-route audit] layer={layer} requests={len(rows)}", file=sys.stderr, flush=True)
    for offset, row in enumerate(rows, start=1):
        request_id = str(row["request_id"])
        split = str(row["split"])
        position = int(row["decode_position"])
        capture = nested.load_capture(
            nested.capture_path(capture_dir, request_id), layer,
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
        target = np.asarray(
            capture["router_logits"][layer + 1, position], np.float32,
        )
        target_ids = np.asarray(
            capture["router_ids"][layer + 1, position], np.int64,
        )
        anchor = np.asarray(target - raw, np.float32)
        anchor_max = float(np.max(np.abs(anchor), initial=0.0))
        anchored_cuda = raw_cuda + torch.as_tensor(
            anchor, device=raw_cuda.device, dtype=torch.float32,
        )
        _, anchored_ids_cuda = nested._cuda_softmax_top8(anchored_cuda)
        anchored = anchored_cuda.detach().cpu().numpy()
        anchored_ids = anchored_ids_cuda.detach().cpu().numpy()
        raw_order_equal = bool(np.array_equal(raw_ids_np, target_ids))
        raw_set_equal = bool(
            set(raw_ids_np.tolist()) == set(target_ids.tolist())
        )
        anchored_bit_equal = bool(np.array_equal(anchored, target))
        anchored_order_equal = bool(np.array_equal(anchored_ids, target_ids))
        contexts += 1
        by_split[split]["contexts"] += 1
        raw_order_matches += int(raw_order_equal)
        raw_set_matches += int(raw_set_equal)
        anchored_bit_matches += int(anchored_bit_equal)
        anchored_order_matches += int(anchored_order_equal)
        maximum_anchor = max(maximum_anchor, anchor_max)
        by_split[split]["raw_order_mismatches"] += int(not raw_order_equal)
        by_split[split]["raw_set_mismatches"] += int(not raw_set_equal)
        if not raw_order_equal or not raw_set_equal:
            details.append({
                "request_id": request_id,
                "split": split,
                "domain": str(row["domain"]),
                "layer": int(layer),
                "position": position,
                "raw_max_abs": anchor_max,
                "raw_ids": raw_ids_np.tolist(),
                "target_ids": target_ids.tolist(),
                "anchored_ids": anchored_ids.tolist(),
                "raw_order_equal": raw_order_equal,
                "raw_set_equal": raw_set_equal,
                "anchored_logits_bit_identical": anchored_bit_equal,
                "anchored_order_equal": anchored_order_equal,
            })
        if not anchored_bit_equal or not anchored_order_equal or anchor_max > 0.0625:
            raise RuntimeError(
                f"anchored hard gate failed request={request_id} layer={layer}"
            )
        if offset % 32 == 0:
            print(
                f"[raw-route audit] layer={layer} {offset}/{len(rows)}",
                file=sys.stderr,
                flush=True,
            )
        del capture, prefix, raw_logits, raw_ids
    del model_slice
    gc.collect()
    torch.cuda.empty_cache()

print(json.dumps({
    "schema": "pr13_d1_raw_slice_route_grid_audit_v1",
    "contexts": contexts,
    "requests": len(rows),
    "layers": list(layers),
    "raw_order_matches": raw_order_matches,
    "raw_order_mismatches": contexts - raw_order_matches,
    "raw_set_matches": raw_set_matches,
    "raw_set_mismatches": contexts - raw_set_matches,
    "anchored_logits_bit_identical": anchored_bit_matches,
    "anchored_order_matches": anchored_order_matches,
    "maximum_raw_anchor_abs": maximum_anchor,
    "by_split": by_split,
    "mismatches": details,
}, indent=2, sort_keys=True))
