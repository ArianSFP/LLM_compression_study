#!/usr/bin/env python3
"""Analyze token-dependent D1 predictor labels before sequential execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))

from oracle_study.d1_accounting import (  # noqa: E402
    controller_accounting,
    provisional_decode_accounting,
)
from oracle_study.d1_runtime_controller import (  # noqa: E402
    certify_topk,
    corrected_logit_metrics,
    signed_effect_metrics,
)


REQUIRED = (
    "predicted_signed_effects",
    "exact_signed_effects",
    "signed_effect_error_radius",
    "predicted_corrected_logits",
    "exact_corrected_logits",
    "corrected_logit_error_radius",
    "candidate_pool_contains_exact_topk",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _certificate_metrics(
    arrays: dict[str, np.ndarray], top_k: int, radius_multiplier: float,
) -> dict[str, Any]:
    predicted = np.asarray(arrays["predicted_corrected_logits"], np.float64)
    exact = np.asarray(arrays["exact_corrected_logits"], np.float64)
    multiplier = float(radius_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("confidence radius multiplier must be positive finite")
    radius = multiplier * np.asarray(
        arrays["corrected_logit_error_radius"], np.float64,
    )
    if predicted.ndim != 2 or exact.shape != predicted.shape or radius.shape != predicted.shape:
        raise ValueError("corrected logits/radii must be equal [groups,candidates] arrays")
    if predicted.shape[1] <= int(top_k):
        raise ValueError("candidate logits do not contain a top-k outsider")
    if "candidate_ids" in arrays:
        ids = np.asarray(arrays["candidate_ids"], np.int64)
        if ids.ndim == 1:
            ids = np.broadcast_to(ids, predicted.shape)
        if ids.shape != predicted.shape:
            raise ValueError("candidate_ids must be [candidates] or [groups,candidates]")
    else:
        ids = np.broadcast_to(np.arange(predicted.shape[1]), predicted.shape)
    pool_contains = np.asarray(
        arrays["candidate_pool_contains_exact_topk"], bool,
    ).reshape(-1)
    if pool_contains.shape != (predicted.shape[0],):
        raise ValueError("candidate-pool labels must contain one boolean per group")
    excluded = arrays.get("excluded_outsider_upper_bound")
    if excluded is not None:
        excluded = np.asarray(excluded, np.float64).reshape(-1)
        if excluded.shape != (predicted.shape[0],):
            raise ValueError("excluded outsider bounds must contain one value per group")
    certified = []
    false = []
    gaps = []
    for row in range(predicted.shape[0]):
        certificate = certify_topk(
            predicted[row],
            radius[row],
            top_k=int(top_k),
            candidate_ids=ids[row],
            excluded_upper_bound=(None if excluded is None else float(excluded[row])),
        )
        predicted_set = set(certificate.top_ids.tolist())
        exact_order = np.argsort(-exact[row], kind="stable")[: int(top_k)]
        exact_set = set(ids[row, exact_order].tolist())
        certified.append(certificate.certified)
        false.append(
            certificate.certified
            and (predicted_set != exact_set or not bool(pool_contains[row]))
        )
        gaps.append(certificate.certificate_gap)
    admitted = int(np.count_nonzero(certified))
    false_count = int(np.count_nonzero(false))
    return {
        "radius_multiplier": multiplier,
        "groups": int(predicted.shape[0]),
        "candidate_pool_misses": int(np.count_nonzero(~pool_contains)),
        "candidate_pool_miss_rate": float(np.mean(~pool_contains)),
        "certificate_coverage": admitted / float(predicted.shape[0]),
        "false_certificates": false_count,
        "false_certificate_rate_among_certified": (
            false_count / float(admitted) if admitted else 0.0
        ),
        "certificate_gap_median": float(np.median(gaps)),
        "certificate_gap_p10": float(np.quantile(gaps, 0.1)),
    }


def analyze(config: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    missing = sorted(set(REQUIRED) - set(arrays))
    if missing:
        raise ValueError(f"runtime predictor bundle is missing: {missing}")
    signed = signed_effect_metrics(
        arrays["predicted_signed_effects"],
        arrays["exact_signed_effects"],
        error_radius=arrays["signed_effect_error_radius"],
        top_pages=int(config.get("top_page_metric_count", 128)),
    )
    logits = corrected_logit_metrics(
        arrays["predicted_corrected_logits"],
        arrays["exact_corrected_logits"],
        error_radius=arrays["corrected_logit_error_radius"],
    )
    top_k = int(config.get("top_k", 8))
    certificate = _certificate_metrics(arrays, top_k, 1.0)
    confidence_curve = [
        _certificate_metrics(arrays, top_k, float(multiplier))
        for multiplier in config.get(
            "confidence_radius_multipliers", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        )
    ]
    rates = [float(value) for value in config["charged_bpw"]]
    accounting = controller_accounting(
        rates,
        rank=int(config.get("latent_rank", 8)),
        candidate_logits=int(config.get("initial_candidate_logits", 16)),
        refreshes=int(config.get("accounting_refreshes", 6)),
        response_synthesis_bytes_per_expert=int(
            config.get("response_synthesis_bytes_per_expert", 10_240)
        ),
        shared_bytes_per_layer=int(config.get("shared_bytes_per_layer", 98_304)),
        uncertainty_bytes_per_expert=int(
            config.get("uncertainty_bytes_per_expert", 1_600)
        ),
        response_synthesis_macs_per_expert=int(
            config.get("response_synthesis_macs_per_expert", 10_240)
        ),
        pr13_selector_macs=int(
            config.get("pr13_primary_selector_macs_per_group", 51_343_360)
        ),
    ).to_dict()
    contexts = [
        provisional_decode_accounting(int(value)).to_dict()
        for value in config.get(
            "provisional_decode_contexts", [1024, 4096, 8192, 32768],
        )
    ]
    return {
        "run_id": str(config["run_id"]),
        "experiment": "B_predictor_prequalification",
        "token_dependent_effects_required": True,
        "exact_labels_used_only_for_evaluation": True,
        "signed_effect_metrics": signed,
        "corrected_logit_metrics": logits,
        "certificate_metrics": certificate,
        "confidence_coverage_curve": confidence_curve,
        "controller_accounting": accounting,
        "provisional_decode_accounting": contexts,
        "sequential_runtime_claim": False,
        "d2_d4_in_scope": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    with np.load(args.input, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    atomic_json(args.output, analyze(config, arrays))


if __name__ == "__main__":
    main()
