#!/usr/bin/env python3
"""Microbenchmark the exhaustive and device-resident D1 page scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_runtime_controller import (  # noqa: E402
    TokenLatentExpert,
    score_single_page_transitions,
    score_single_page_transitions_torch,
)


def _summary(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, np.float64) * 1e3
    return {
        "samples": int(values.size),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "minimum_ms": float(np.min(values)),
    }


def benchmark(
    *,
    device: str,
    warmup: int,
    samples: int,
    numpy_samples: int,
    shortlist: int,
    seed: int,
) -> dict[str, object]:
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(int(seed))
    down2 = torch.randn(8, 512, 8, device=target, generator=generator)
    down4 = torch.randn(8, 512, 8, device=target, generator=generator)
    hidden = torch.randn(8, 512, 4, device=target, generator=generator)
    states = torch.zeros(8, 512, dtype=torch.int64, device=target)
    weights = torch.full((8,), 0.125, device=target)
    adjoint = torch.randn(8, device=target, generator=generator)
    uncertainty = 0.01 * torch.rand(
        8, 512, 3, device=target, generator=generator,
    )

    def tensor_call() -> None:
        score_single_page_transitions_torch(
            down2,
            down4,
            hidden,
            states,
            weights,
            adjoint,
            uncertainty_reduction=uncertainty,
            shortlist=int(shortlist),
        )

    for _ in range(int(warmup)):
        tensor_call()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    tensor_times = []
    for _ in range(int(samples)):
        started = time.perf_counter()
        tensor_call()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        tensor_times.append(time.perf_counter() - started)

    result: dict[str, object] = {
        "device": str(target),
        "device_name": (
            torch.cuda.get_device_name(target) if target.type == "cuda" else platform.processor()
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "experts": 8,
        "units_per_expert": 512,
        "latent_rank": 8,
        "legal_q2_transitions": 12_288,
        "shortlist": int(shortlist),
        "tensor_scorer": _summary(tensor_times),
    }
    if int(numpy_samples) > 0:
        down2_numpy = down2.detach().cpu().numpy().astype(np.float64)
        down4_numpy = down4.detach().cpu().numpy().astype(np.float64)
        hidden_numpy = hidden.detach().cpu().numpy().astype(np.float64)
        experts = tuple(
            TokenLatentExpert(down2_numpy[index], down4_numpy[index], hidden_numpy[index])
            for index in range(8)
        )
        states_numpy = tuple(np.zeros(512, np.int64) for _ in range(8))
        weights_numpy = weights.detach().cpu().numpy()
        adjoint_numpy = adjoint.detach().cpu().numpy()
        uncertainty_numpy = uncertainty.detach().cpu().numpy()

        def numpy_call() -> None:
            score_single_page_transitions(
                experts,
                states_numpy,
                weights_numpy,
                adjoint_numpy,
                uncertainty_reduction=uncertainty_numpy,
                shortlist=int(shortlist),
                unique_units=True,
            )

        numpy_call()
        numpy_times = []
        for _ in range(int(numpy_samples)):
            started = time.perf_counter()
            numpy_call()
            numpy_times.append(time.perf_counter() - started)
        result["numpy_scorer"] = _summary(numpy_times)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--numpy-samples", type=int, default=50)
    parser.add_argument("--shortlist", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.numpy_samples) < 0 or args.samples < 1:
        raise ValueError("benchmark sample counts are invalid")
    result = benchmark(
        device=args.device,
        warmup=args.warmup,
        samples=args.samples,
        numpy_samples=args.numpy_samples,
        shortlist=args.shortlist,
        seed=args.seed,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
