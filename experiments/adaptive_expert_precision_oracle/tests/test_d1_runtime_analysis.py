from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_d1_runtime_predictions import analyze


def test_runtime_analysis_reports_predictor_certification_and_costs() -> None:
    config = {
        "run_id": "test",
        "charged_bpw": [0.5234781901041666, 0.9987386067708334],
        "latent_rank": 8,
        "initial_candidate_logits": 10,
        "accounting_refreshes": 4,
        "top_k": 8,
        "top_page_metric_count": 2,
        "provisional_decode_contexts": [1024],
    }
    exact_logits = np.asarray([
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    arrays = {
        "predicted_signed_effects": np.asarray([2.0, -1.0, 0.5]),
        "exact_signed_effects": np.asarray([1.9, -1.1, 0.4]),
        "signed_effect_error_radius": np.full(3, 0.2),
        "predicted_corrected_logits": exact_logits.copy(),
        "exact_corrected_logits": exact_logits,
        "corrected_logit_error_radius": np.full_like(exact_logits, 0.1),
        "candidate_pool_contains_exact_topk": np.asarray([True, True]),
    }
    result = analyze(config, arrays)
    assert result["signed_effect_metrics"]["sign_accuracy"] == 1.0
    assert result["certificate_metrics"]["certificate_coverage"] == 1.0
    assert result["certificate_metrics"]["false_certificates"] == 0
    assert result["certificate_metrics"]["candidate_pool_miss_rate"] == 0.0
    assert len(result["confidence_coverage_curve"]) == 6
    assert result["controller_accounting"]["direct_candidate_macs"] > 0
    assert result["sequential_runtime_claim"] is False
