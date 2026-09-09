#!/usr/bin/env python3
"""Same-host correctness and memory gate for the offline streaming reference."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oracle_study.d1_decode import cache_state_metrics
from tail_value_streaming import load_streamed, forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    model = load_streamed(a.checkpoint)
    load_seconds = time.perf_counter() - started
    row = next(json.loads(l) for l in a.manifest.read_text().splitlines() if json.loads(l)["domain"] == "summarisation")
    tokens, position = row["prompt_token_ids"], row["decode_position"]
    t = time.perf_counter()
    pref = forward(model, tokens[:position])
    prefix_seconds = time.perf_counter() - t
    cache = pref.past_key_values
    del pref
    captured = {}
    experts = model.model.language_model.layers[6].mlp.experts
    handle = experts.register_forward_hook(lambda m, args, out: captured.update(args=tuple(x.detach().clone() for x in args), output=out.detach().clone()))
    t = time.perf_counter()
    first = forward(model, [tokens[position]], copy.deepcopy(cache), position)
    decode_seconds = time.perf_counter() - t
    handle.remove()
    second = forward(model, [tokens[position]], copy.deepcopy(cache), position)
    repeat = bool(torch.equal(first.logits, second.logits))
    cm = cache_state_metrics(first.past_key_values, second.past_key_values).to_dict()
    # Compare full-bank native eager against selected-expert transfers using identical operands.
    cpu = {name: parameter.data for name, parameter in experts.named_parameters()}
    experts.to("cuda")
    with torch.inference_mode():
        resident = experts._resident_forward(*captured["args"])
    native_equal = bool(torch.equal(resident, captured["output"]))
    native_max = float((resident.float() - captured["output"].float()).abs().max())
    for name, parameter in experts.named_parameters():
        parameter.data = cpu[name]
    facts = dict(schema="tail_value_3090_benchmark_v1", load_seconds=load_seconds,
                 prefix_seconds=prefix_seconds, decode_seconds=decode_seconds,
                 position=position, request_id=row["request_id"],
                 gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                 peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                 repeat_logits_bit_exact=repeat, native_expert_bit_exact=native_equal,
                 native_expert_max_abs=native_max, **cm)
    facts["passed"] = repeat and native_equal and cm["cache_bit_identical"]
    (a.output / "benchmark.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2), flush=True)
    if not facts["passed"]:
        raise RuntimeError("reference execution gate failed")


if __name__ == "__main__":
    main()
