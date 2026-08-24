"""Exact two-layer slices for the D1 route-boundary oracle.

The slice reconstructs the output of an injected MoE layer from the captured
pre-MoE residual, post-attention-normalized input, and routed-expert output.
It then executes only the smooth pre-MoE portion and router of the next layer.
No routed expert from either layer is loaded.

Checkpoint loading is intentionally tensor-local.  Loading a complete Qwen
checkpoint just to test one D1 boundary is both slower and much more memory
intensive than reading the handful of tensors used by the slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


__all__ = [
    "D1SliceParity",
    "SliceTensorLoader",
    "load_d1_layer_slice",
    "load_linear_d1_layer_slice",
    "slice_parity",
    "topk_membership",
]


def topk_membership(logits: np.ndarray, k: int = 8) -> np.ndarray:
    """Return stable descending top-k membership for every token."""

    values = np.asarray(logits)
    width = int(k)
    if values.ndim != 2 or width < 1 or values.shape[1] < width:
        raise ValueError("router logits must be [tokens,experts] with k <= experts")
    return np.argsort(-values, axis=1, kind="stable")[:, :width]


@dataclass(frozen=True)
class D1SliceParity:
    """Hard parity measurements for one captured request and layer slice."""

    current_hidden_bit_identical: bool
    current_hidden_max_abs: float
    next_router_bit_identical: bool
    next_router_max_abs: float
    next_top8_exact_fraction: float
    next_top8_set_exact_fraction: float

    def __post_init__(self) -> None:
        for name in ("current_hidden_max_abs", "next_router_max_abs"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite nonnegative")
        for name in ("next_top8_exact_fraction", "next_top8_set_exact_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


class SliceTensorLoader:
    """Read and dequantize individual tensors from an indexed checkpoint."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint)
        index_path = self.checkpoint / "model.safetensors.index.json"
        payload = json.loads(index_path.read_text())
        self.weight_map = dict(payload["weight_map"])
        self._quantization_scheme: Any | None = None
        self._compressor: Any | None = None

    def has(self, name: str) -> bool:
        return str(name) in self.weight_map

    def tensor(self, name: str) -> Any:
        """Read one tensor without materializing the rest of its shard."""

        from safetensors import safe_open

        key = str(name)
        if key not in self.weight_map:
            raise KeyError(f"checkpoint tensor is absent: {key}")
        path = self.checkpoint / self.weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def _mxfp4(self) -> tuple[Any, Any]:
        if self._compressor is None or self._quantization_scheme is None:
            from compressed_tensors.compressors.mxfp4.base import (
                MXFP4PackedCompressor,
            )
            from compressed_tensors.quantization import (
                QuantizationArgs,
                QuantizationScheme,
            )

            arguments = QuantizationArgs(
                num_bits=4,
                type="float",
                strategy="group",
                group_size=32,
                symmetric=True,
            )
            self._quantization_scheme = QuantizationScheme(
                targets=["Linear"], weights=arguments,
            )
            self._compressor = MXFP4PackedCompressor()
        return self._compressor, self._quantization_scheme

    def linear_weight(self, prefix: str) -> Any:
        """Return a dense weight from either BF16 or packed MXFP4 storage."""

        dense = f"{prefix}.weight"
        if self.has(dense):
            return self.tensor(dense)
        packed = f"{prefix}.weight_packed"
        scale = f"{prefix}.weight_scale"
        if not self.has(packed) or not self.has(scale):
            raise KeyError(f"linear checkpoint fields are absent: {prefix}")
        compressor, scheme = self._mxfp4()
        result = compressor.decompress(
            {
                "weight_packed": self.tensor(packed),
                "weight_scale": self.tensor(scale),
            },
            scheme,
        )
        return result["weight"]


class _D1LayerSlice:
    """Standalone current composition and next-layer pre-MoE/router path."""

    def __init__(
        self,
        *,
        layer: int,
        current_shared: Any,
        current_shared_gate: Any,
        next_input_norm: Any,
        next_mixer: Any,
        next_mixer_type: str,
        next_config: Any,
        next_rotary: Any | None,
        next_post_norm: Any,
        next_router: Any,
        device: Any,
    ) -> None:
        self.layer = int(layer)
        self.current_shared = current_shared
        self.current_shared_gate = current_shared_gate
        self.next_input_norm = next_input_norm
        self.next_mixer = next_mixer
        self.next_mixer_type = str(next_mixer_type)
        self.next_config = next_config
        self.next_rotary = next_rotary
        self.next_post_norm = next_post_norm
        self.next_router = next_router
        self.device = device

    @property
    def modules(self) -> tuple[Any, ...]:
        modules = (
            self.current_shared,
            self.current_shared_gate,
            self.next_input_norm,
            self.next_mixer,
            self.next_post_norm,
            self.next_router,
        )
        return modules + (() if self.next_rotary is None else (self.next_rotary,))

    def compose_current_output(
        self,
        residual: Any,
        normalized_input: Any,
        routed_output: Any,
        routed_delta: Any | None = None,
    ) -> Any:
        """Reproduce true pre-residual routed-output replacement.

        The delta convention is ``approximate - exact Q4``.  Addition is
        performed in FP32 and rounded once to BF16, exactly as in the
        same-host causal-control scientific injection.
        """

        import torch

        residual_value = torch.as_tensor(
            residual, device=self.device, dtype=torch.bfloat16,
        )
        x = torch.as_tensor(
            normalized_input, device=self.device, dtype=torch.bfloat16,
        )
        routed = torch.as_tensor(
            routed_output, device=self.device, dtype=torch.bfloat16,
        )
        if residual_value.ndim == 2:
            residual_value = residual_value.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if routed.ndim == 3 and routed.shape[0] == 1:
            routed = routed[0]
        if routed_delta is not None:
            delta = torch.as_tensor(
                routed_delta, device=self.device, dtype=torch.float32,
            )
            if delta.ndim == 3 and delta.shape[0] == 1:
                delta = delta[0]
            routed = (routed.float() + delta).to(torch.bfloat16)
        flat = x.reshape(-1, x.shape[-1])
        shared = self.current_shared(flat)
        shared = torch.sigmoid(self.current_shared_gate(flat)) * shared
        moe = routed.reshape_as(shared) + shared
        return residual_value + moe.reshape_as(residual_value)

    def next_router_outputs(
        self,
        layer_output: Any,
        attention_mask: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Execute D1 and return exact GPU logits, weights, and top-k IDs."""

        import torch

        hidden = layer_output
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(
                hidden, device=self.device, dtype=torch.bfloat16,
            )
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        mask = attention_mask
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=self.device)
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        residual = hidden
        normalized_input = self.next_input_norm(hidden)
        if self.next_mixer_type == "linear_attention":
            mixed = self.next_mixer(
                hidden_states=normalized_input,
                cache_params=None,
                attention_mask=mask,
            )
        elif self.next_mixer_type == "full_attention":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                create_causal_mask,
            )

            batch, sequence = hidden.shape[:2]
            position_ids = torch.arange(
                sequence, device=hidden.device, dtype=torch.long,
            ).view(1, 1, -1).expand(4, batch, -1)
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
            causal_mask = create_causal_mask(
                config=self.next_config,
                inputs_embeds=hidden,
                attention_mask=mask,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            if self.next_rotary is None:
                raise RuntimeError("full-attention D1 slice omitted rotary embeddings")
            position_embeddings = self.next_rotary(hidden, rope_position_ids)
            mixed, _ = self.next_mixer(
                hidden_states=normalized_input,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
            )
        else:
            raise RuntimeError(f"unsupported D1 next mixer: {self.next_mixer_type}")
        pre_moe = residual + mixed
        normalized = self.next_post_norm(pre_moe)
        logits, scores, expert_ids = self.next_router(normalized)
        shape = (hidden.shape[0], hidden.shape[1])
        return (
            logits.reshape(*shape, -1),
            scores.reshape(*shape, -1),
            expert_ids.reshape(*shape, -1),
        )

    def next_router_logits(
        self,
        layer_output: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Execute the complete smooth D1 path and return router logits."""

        return self.next_router_outputs(layer_output, attention_mask)[0]

    def decode_prefix_cache(self, prefix_layer_output: Any) -> Any:
        """Build the exact next-mixer cache from an uncompressed prefix.

        Only the next layer is executed.  The returned hybrid cache is a
        pre-step snapshot: callers must clone it before every candidate
        because both Qwen attention implementations mutate cache state.
        """

        import torch
        from transformers import DynamicCache

        hidden = prefix_layer_output
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(
                hidden, device=self.device, dtype=torch.bfloat16,
            )
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("decode prefix must be [1,prefix,hidden]")
        cache = DynamicCache(config=self.next_config)
        sequence = int(hidden.shape[1])
        if sequence == 0:
            return cache
        normalized = self.next_input_norm(hidden)
        if self.next_mixer_type == "linear_attention":
            self.next_mixer(
                hidden_states=normalized,
                cache_params=cache,
                attention_mask=None,
            )
        elif self.next_mixer_type == "full_attention":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                create_causal_mask,
            )

            positions = torch.arange(
                sequence, device=hidden.device, dtype=torch.long,
            ).view(1, 1, -1).expand(4, 1, -1)
            text_positions = positions[0]
            rope_positions = positions[1:]
            mask = create_causal_mask(
                config=self.next_config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=None,
                position_ids=text_positions,
            )
            if self.next_rotary is None:
                raise RuntimeError("full-attention decode slice omitted rotary embeddings")
            self.next_mixer(
                hidden_states=normalized,
                position_embeddings=self.next_rotary(hidden, rope_positions),
                attention_mask=mask,
                position_ids=text_positions,
                past_key_values=cache,
            )
        else:
            raise RuntimeError(f"unsupported D1 next mixer: {self.next_mixer_type}")
        return cache

    def next_router_outputs_decode(
        self,
        current_layer_output: Any,
        prefix_cache: Any,
        position: int,
    ) -> tuple[Any, Any, Any, Any]:
        """Execute one cached decode token and return router outputs and cache.

        ``prefix_cache`` is mutated exactly once.  It must be a private clone
        of an exact-prefix snapshot rather than shared among candidates.
        """

        import torch

        hidden = current_layer_output
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(
                hidden, device=self.device, dtype=torch.bfloat16,
            )
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, 1):
            raise ValueError("cached D1 decode requires one [1,1,hidden] token")
        token_position = int(position)
        if token_position < 0:
            raise ValueError("decode position must be nonnegative")
        residual = hidden
        normalized = self.next_input_norm(hidden)
        if self.next_mixer_type == "linear_attention":
            mixed = self.next_mixer(
                hidden_states=normalized,
                cache_params=prefix_cache,
                attention_mask=None,
            )
        elif self.next_mixer_type == "full_attention":
            positions = torch.full(
                (4, 1, 1), token_position,
                device=hidden.device,
                dtype=torch.long,
            )
            if self.next_rotary is None:
                raise RuntimeError("full-attention decode slice omitted rotary embeddings")
            mixed, _ = self.next_mixer(
                hidden_states=normalized,
                position_embeddings=self.next_rotary(hidden, positions[1:]),
                attention_mask=None,
                position_ids=positions[0],
                past_key_values=prefix_cache,
            )
        else:
            raise RuntimeError(f"unsupported D1 next mixer: {self.next_mixer_type}")
        normalized_moe = self.next_post_norm(residual + mixed)
        logits, scores, expert_ids = self.next_router(normalized_moe)
        return (
            logits.reshape(1, 1, -1),
            scores.reshape(1, 1, -1),
            expert_ids.reshape(1, 1, -1),
            prefix_cache,
        )

    def next_router_logits_decode(
        self,
        current_layer_output: Any,
        prefix_cache: Any,
        position: int,
    ) -> tuple[Any, Any]:
        """Return one-token cached router logits and the committed cache."""

        logits, _, _, cache = self.next_router_outputs_decode(
            current_layer_output, prefix_cache, position,
        )
        return logits, cache


def _copy_state(module: Any, state: Mapping[str, Any]) -> None:
    missing, unexpected = module.load_state_dict(dict(state), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"slice module state mismatch: missing={missing}, unexpected={unexpected}",
        )


def load_d1_layer_slice(
    checkpoint: str | Path,
    layer: int,
    *,
    device: str = "cuda",
) -> _D1LayerSlice:
    """Load a current layer and its next linear/full-attention D1 slice."""

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeAttention,
        Qwen3_5MoeMLP,
        Qwen3_5MoeRMSNorm,
        Qwen3_5MoeTextRotaryEmbedding,
        Qwen3_5MoeTopKRouter,
    )

    current = int(layer)
    checkpoint_path = Path(checkpoint)
    full_config = AutoConfig.from_pretrained(
        checkpoint_path, local_files_only=True,
    )
    config = full_config.text_config
    following = current + 1
    if current < 0 or following >= int(config.num_hidden_layers):
        raise ValueError("D1 slice requires layer in [0,num_hidden_layers-2]")
    next_mixer_type = str(config.layer_types[following])
    if next_mixer_type not in ("linear_attention", "full_attention"):
        raise ValueError(f"unsupported D1 next mixer: {next_mixer_type}")
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    loader = SliceTensorLoader(checkpoint_path)
    target = torch.device(device)
    dtype = torch.bfloat16

    shared = Qwen3_5MoeMLP(
        config, intermediate_size=int(config.shared_expert_intermediate_size),
    ).to(device=target, dtype=dtype)
    shared_gate = torch.nn.Linear(
        int(config.hidden_size), 1, bias=False, device=target, dtype=dtype,
    )
    current_prefix = f"model.language_model.layers.{current}.mlp"
    _copy_state(shared, {
        name: loader.linear_weight(f"{current_prefix}.shared_expert.{name[:-7]}")
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    })
    _copy_state(shared_gate, {
        "weight": loader.linear_weight(f"{current_prefix}.shared_expert_gate"),
    })

    next_prefix = f"model.language_model.layers.{following}"
    input_norm = Qwen3_5MoeRMSNorm(
        int(config.hidden_size), eps=float(config.rms_norm_eps),
    ).to(device=target, dtype=dtype)
    post_norm = Qwen3_5MoeRMSNorm(
        int(config.hidden_size), eps=float(config.rms_norm_eps),
    ).to(device=target, dtype=dtype)
    router = Qwen3_5MoeTopKRouter(config).to(device=target, dtype=dtype)
    _copy_state(input_norm, {
        "weight": loader.tensor(f"{next_prefix}.input_layernorm.weight"),
    })
    _copy_state(post_norm, {
        "weight": loader.tensor(f"{next_prefix}.post_attention_layernorm.weight"),
    })
    _copy_state(router, {
        "weight": loader.linear_weight(f"{next_prefix}.mlp.gate"),
    })
    rotary = None
    mixer_state: dict[str, Any] = {}
    if next_mixer_type == "linear_attention":
        mixer = Qwen3_5MoeGatedDeltaNet(config, following).to(
            device=target, dtype=dtype,
        )
        mixer_prefix = f"{next_prefix}.linear_attn"
        linear_weights = {
            "out_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight",
            "in_proj_b.weight", "in_proj_a.weight",
        }
        for name in mixer.state_dict():
            prefix = (
                f"{mixer_prefix}.{name[:-7]}" if name.endswith(".weight") else ""
            )
            if name in linear_weights:
                mixer_state[name] = loader.linear_weight(prefix)
            else:
                mixer_state[name] = loader.tensor(f"{mixer_prefix}.{name}")
    else:
        mixer = Qwen3_5MoeAttention(config, following).to(
            device=target, dtype=dtype,
        )
        rotary = Qwen3_5MoeTextRotaryEmbedding(config).to(
            device=target, dtype=dtype,
        )
        mixer_prefix = f"{next_prefix}.self_attn"
        projection_weights = {
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        }
        for name in mixer.state_dict():
            if name in projection_weights:
                mixer_state[name] = loader.linear_weight(
                    f"{mixer_prefix}.{name[:-7]}",
                )
            else:
                mixer_state[name] = loader.tensor(f"{mixer_prefix}.{name}")
    _copy_state(mixer, mixer_state)

    modules = (
        shared, shared_gate, input_norm, mixer, post_norm, router,
        *((rotary,) if rotary is not None else ()),
    )
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    return _D1LayerSlice(
        layer=current,
        current_shared=shared,
        current_shared_gate=shared_gate,
        next_input_norm=input_norm,
        next_mixer=mixer,
        next_mixer_type=next_mixer_type,
        next_config=config,
        next_rotary=rotary,
        next_post_norm=post_norm,
        next_router=router,
        device=target,
    )


def load_linear_d1_layer_slice(
    checkpoint: str | Path,
    layer: int,
    *,
    device: str = "cuda",
) -> _D1LayerSlice:
    """Compatibility loader that retains the historical linear-only guard."""

    result = load_d1_layer_slice(checkpoint, layer, device=device)
    if result.next_mixer_type != "linear_attention":
        raise ValueError("requested D1 layer is followed by full attention")
    return result


def slice_parity(
    reconstructed_hidden: Any,
    reference_hidden: np.ndarray,
    reconstructed_router_logits: Any,
    reference_router_logits: np.ndarray,
    *,
    k: int = 8,
) -> D1SliceParity:
    """Compare a zero-delta slice against immutable captured tensors."""

    def numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, np.float32)

    hidden = numpy(reconstructed_hidden)
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    router = numpy(reconstructed_router_logits)
    if router.ndim == 3 and router.shape[0] == 1:
        router = router[0]
    hidden_reference = np.asarray(reference_hidden, np.float32)
    router_reference = np.asarray(reference_router_logits, np.float32)
    if hidden.shape != hidden_reference.shape or router.shape != router_reference.shape:
        raise ValueError("slice and reference shapes changed")
    observed_ids = topk_membership(router, k)
    reference_ids = topk_membership(router_reference, k)
    exact = np.mean(np.all(observed_ids == reference_ids, axis=1))
    set_exact = np.mean([
        set(left.tolist()) == set(right.tolist())
        for left, right in zip(observed_ids, reference_ids)
    ])
    return D1SliceParity(
        current_hidden_bit_identical=bool(np.array_equal(hidden, hidden_reference)),
        current_hidden_max_abs=float(np.max(np.abs(hidden - hidden_reference), initial=0.0)),
        next_router_bit_identical=bool(np.array_equal(router, router_reference)),
        next_router_max_abs=float(np.max(np.abs(router - router_reference), initial=0.0)),
        next_top8_exact_fraction=float(exact),
        next_top8_set_exact_fraction=float(set_exact),
    )
