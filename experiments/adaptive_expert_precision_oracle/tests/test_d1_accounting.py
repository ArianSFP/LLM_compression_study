from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_study.d1_accounting import (
    controller_accounting,
    matched_integral_pages,
    metadata_bpw,
    repair_objective_accounting,
    provisional_decode_accounting,
)


RATES = (
    0.5234781901041666,
    0.7734781901041666,
    0.9987386067708334,
    1.0234781901041666,
)


def test_pr13_metadata_and_integral_page_accounting() -> None:
    assert metadata_bpw(6148 + 3084) == pytest.approx(0.023478190104166668)
    assert [matched_integral_pages(rate, 6148 + 3084) for rate in RATES] == [
        384, 576, 749, 768,
    ]


def test_joint_controller_cost_and_matched_rates() -> None:
    result = controller_accounting(RATES)
    assert result.direct_candidate_macs == 9_699_328
    assert result.one_adjoint_macs == 851_968
    assert result.response_synthesis_macs == 81_920
    assert result.direct_total_macs == 9_781_248
    assert result.one_adjoint_total_macs == 933_888
    assert result.pr13_primary_selector_macs == 51_343_360
    assert result.direct_fraction_of_pr13 == pytest.approx(0.1905, rel=1e-3)
    assert result.one_adjoint_fraction_of_pr13 == pytest.approx(0.01819, rel=1e-3)
    assert result.current_model_metadata_mib == pytest.approx(90.15625)
    assert result.joint_model_metadata_mib == pytest.approx(193.90625)
    assert result.joint_uncertainty_model_metadata_mib == pytest.approx(209.53125)
    assert result.current_group_metadata_kib == pytest.approx(72.125)
    assert result.joint_group_metadata_kib == pytest.approx(155.125)
    assert result.joint_uncertainty_group_metadata_kib == pytest.approx(167.625)
    assert list(result.matched_pages.values()) == [363, 555, 728, 747]
    assert list(result.matched_pages_with_uncertainty.values()) == [360, 552, 725, 744]


def test_hybrid_provisional_pass_scales_with_context() -> None:
    short = provisional_decode_accounting(1024)
    long = provisional_decode_accounting(32768)
    assert short.average_macs == pytest.approx(35.91e6, rel=2e-3)
    assert long.average_macs == pytest.approx(102.59e6, rel=2e-3)
    assert 0.55 < short.fraction_of_normal_layer < 0.57
    assert 0.77 < long.fraction_of_normal_layer < 0.79
    assert short.average_packed_megabytes < short.average_bf16_megabytes
    assert long.average_packed_megabytes > short.average_packed_megabytes


def test_repair_overlay_and_severity_sidecar_accounting() -> None:
    result = repair_objective_accounting(RATES)
    assert result.repair_overlay_total_macs == 52_277_248
    assert result.repair_overlay_fraction_of_pr13 == pytest.approx(1.01819, rel=1e-4)
    assert result.scalar_adjoint_total_macs == 688_128
    assert result.scalar_adjoint_fraction_of_pr13 == pytest.approx(0.01340, rel=1e-3)
    assert result.routing_mass_increment_bytes_per_expert == 0
    assert result.functional_embedding_bytes_per_expert == 16
    assert result.functional_embedding_bpw_increment == pytest.approx(0.00004069, rel=1e-4)
    assert result.functional_embedding_model_mib == pytest.approx(0.15625)
    assert result.full_pair_table_bytes_per_expert == 512
    assert result.full_pair_table_bpw_increment == pytest.approx(0.00130208, rel=1e-4)
    assert result.full_pair_table_model_mib == pytest.approx(5.0)
