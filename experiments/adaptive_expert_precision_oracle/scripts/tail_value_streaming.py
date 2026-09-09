"""Offline BF16 reference execution with CPU-resident expert banks.

Uses the native eager arithmetic, fetching only experts hit by the live routes.
No predictor, production latency, or cross-device bit-equivalence claim.
"""
import time
import types
import hashlib
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F

_EXPERT_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LIMIT = 8 * 2**30
_CACHE_HITS = 0
_CACHE_MISSES = 0


def expert_cache_stats():
    return dict(bytes=_CACHE_BYTES,limit_bytes=_CACHE_LIMIT,hits=_CACHE_HITS,misses=_CACHE_MISSES)


def expert_weights(module,expert,device):
    global _CACHE_BYTES,_CACHE_HITS,_CACHE_MISSES
    key=(id(module),int(expert),str(device))
    if key in _EXPERT_CACHE:
        _CACHE_HITS+=1
        _EXPERT_CACHE.move_to_end(key)
        return _EXPERT_CACHE[key]
    _CACHE_MISSES+=1
    weights=(module.gate_up_proj[expert].to(device),module.down_proj[expert].to(device))
    size=sum(t.numel()*t.element_size() for t in weights)
    if size<=_CACHE_LIMIT:
        while _CACHE_BYTES+size>_CACHE_LIMIT:
            _,old=_EXPERT_CACHE.popitem(last=False)
            _CACHE_BYTES-=sum(t.numel()*t.element_size() for t in old)
        _EXPERT_CACHE[key]=weights
        _CACHE_BYTES+=size
    return weights


def streamed_experts(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
    for expert in hits:
        pos, tokens = torch.where(mask[expert])
        gate_up,down = expert_weights(self,expert,hidden_states.device)
        gate, up = F.linear(hidden_states[tokens], gate_up).chunk(2, dim=-1)
        value = F.linear(self.act_fn(gate) * up, down)
        value = value * top_k_weights[tokens, pos, None]
        final.index_add_(0, tokens, value.to(final.dtype))
    return final


def load_streamed(checkpoint):
    from transformers import AutoConfig, CompressedTensorsConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    started = time.perf_counter()
    pins = {"config.json": "52411f11bf654a1a5b3bfb15c84be894ee57d97e95dbdd37656f6c75846edc1a",
            "model.safetensors.index.json": "842c9ba65c2bebe47cc834eb8e8a744b7ba8f610b9fc8cd751cbc7a4564d39fb"}
    for name, expected in pins.items():
        if hashlib.sha256((Path(checkpoint)/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"checkpoint identity differs: {name}")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    quant = CompressedTensorsConfig.from_dict(config.quantization_config)
    quant.dequantize = True
    config.text_config._attn_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    print("loading CPU BF16 reference", flush=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, device_map={"": "cpu"}, quantization_config=quant)
    model.eval().requires_grad_(False)
    layers = model.model.language_model.layers
    excluded = {id(layer.mlp.experts) for layer in layers}
    if hasattr(model.model, "visual"):
        excluded.add(id(model.model.visual))
    if hasattr(model, "mtp"):
        excluded.add(id(model.mtp))

    def move(module):
        if id(module) in excluded:
            return
        for name, parameter in module.named_parameters(recurse=False):
            parameter.data = parameter.data.to("cuda")
        for name, value in module.named_buffers(recurse=False):
            setattr(module, name, value.to("cuda"))
        for child in module.children():
            move(child)
    move(model)
    for layer in layers:
        experts = layer.mlp.experts
        experts._resident_forward = experts.forward
        experts.forward = types.MethodType(streamed_experts, experts)
    torch.cuda.synchronize()
    print(f"loaded in {time.perf_counter()-started:.1f}s; GPU allocated {torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)
    return model


def forward(model, tokens, cache=None, position=0):
    ids = torch.tensor([tokens], device="cuda")
    with torch.inference_mode():
        output = model(input_ids=ids,
                       attention_mask=torch.ones((1, position+len(tokens)), device="cuda", dtype=torch.long),
                       past_key_values=cache,
                       cache_position=torch.arange(position, position+len(tokens), device="cuda"),
                       use_cache=True, return_dict=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return output
