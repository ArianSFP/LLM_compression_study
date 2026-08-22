from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from oracle_study.refinement_rotation import (
    activation_pca_basis,
    block_joint_diagonalize,
    correction_eigenbasis,
    diagonal_order,
    exact_fixed_coefficient_order,
    fixed_contributions,
    materialize_block_basis,
    offdiagonal_fraction,
    orthogonal_joint_diagonalize,
    quantize_columns_symmetric,
    recovery_at_counts,
    transform_refinement,
)
from run_refinement_rotation_audit import (
    apply_native_order,
    native_exact_greedy,
    native_states,
    output_jacobian_basis_torch,
    transformed_occurrence,
)
from analyze_refinement_rotation_audit import build_summaries, markdown_table


def test_correction_eigenbasis_reconstructs_and_orthogonalizes() -> None:
    rng = np.random.default_rng(3)
    residual = rng.normal(size=(4, 7))
    metric_root = rng.normal(size=(4, 4))
    metric = metric_root.T @ metric_root
    basis, eigenvalues = correction_eigenbasis(residual, metric)
    assert basis.shape == (7, 4)
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-10)
    transformed = residual @ basis
    gram = transformed.T @ metric @ transformed
    assert np.allclose(gram, np.diag(eigenvalues), atol=1e-9)
    assert np.allclose(residual, transformed @ basis.T, atol=1e-10)


def test_joint_basis_makes_diagonal_and_exact_orders_equivalent() -> None:
    rng = np.random.default_rng(7)
    gate = rng.normal(size=(3, 6))
    up = rng.normal(size=(3, 6))
    activation = rng.normal(size=6)
    basis, _ = correction_eigenbasis((gate, up))
    contributions = np.concatenate((
        fixed_contributions(gate, basis, activation),
        fixed_contributions(up, basis, activation),
    ), axis=1)
    diagonal = diagonal_order(contributions)
    exact, _ = exact_fixed_coefficient_order(contributions)
    assert np.array_equal(diagonal, exact)
    diagonal_curve = recovery_at_counts(contributions, diagonal, range(7))
    exact_curve = recovery_at_counts(contributions, exact, range(7))
    assert diagonal_curve == exact_curve
    assert np.isclose(diagonal_curve[6], 1.0)


def test_activation_pca_is_uncentered_and_orthogonal() -> None:
    activations = np.asarray([[4.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    basis, energy = activation_pca_basis(activations)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-10)
    assert energy[0] >= energy[1]
    assert abs(basis[0, 0]) > abs(basis[1, 0])


def test_transform_refinement_exact_with_square_basis() -> None:
    rng = np.random.default_rng(11)
    residual = rng.normal(size=(5, 5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    activation = rng.normal(size=5)
    atoms, coefficients = transform_refinement(residual, basis, activation)
    assert np.allclose(atoms @ coefficients, residual @ activation, atol=1e-10)


def test_column_quantization_accounting_and_endpoint() -> None:
    matrix = np.asarray([[1.0, -2.0], [0.5, 0.0], [-1.0, 1.0]])
    encoded = quantize_columns_symmetric(matrix, 4)
    assert encoded.decoded.shape == matrix.shape
    assert encoded.code_bytes_per_column == 2
    assert encoded.code_bytes == 4
    assert encoded.metadata_bytes == 4
    assert np.max(np.abs(encoded.decoded - matrix)) <= np.max(encoded.scales) / 2 + 1e-6


def test_int2_quantization_and_fp16_scale_underflow_are_finite() -> None:
    matrix = np.asarray([
        [1.0, 1.0e-12], [-0.5, -1.0e-12], [0.25, 0.0], [0.0, 0.0],
    ], dtype=np.float32)
    encoded = quantize_columns_symmetric(matrix, 2)
    assert encoded.code_bytes_per_column == 1
    assert encoded.code_bytes == 2
    assert np.all(encoded.scales > 0.0)
    assert np.all(np.isfinite(encoded.decoded))


def test_joint_diagonalization_reduces_offdiagonal_energy() -> None:
    angle = 0.37
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrices = np.stack([
        rotation @ np.diag([4.0, 1.0]) @ rotation.T,
        rotation @ np.diag([2.0, 0.5]) @ rotation.T,
    ])
    before = offdiagonal_fraction(matrices)
    basis, diagonalized, facts = orthogonal_joint_diagonalize(matrices, sweeps=4)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-10)
    assert offdiagonal_fraction(diagonalized) < before * 1e-10
    assert facts["final_offdiagonal_fraction"] < facts["initial_offdiagonal_fraction"]


def test_block_joint_diagonalization_materializes_orthogonal_basis() -> None:
    rng = np.random.default_rng(13)
    residuals = [rng.normal(size=(5, 8)), rng.normal(size=(4, 8))]
    blocks, facts = block_joint_diagonalize(residuals, block_size=4, sweeps=2)
    basis = materialize_block_basis(blocks)
    assert blocks.shape == (2, 4, 4)
    assert np.allclose(basis.T @ basis, np.eye(8), atol=1e-10)
    assert facts["blocks"] == 2


def test_transformed_occurrence_scores_against_exact_not_truncated_endpoint() -> None:
    basis = np.eye(2)
    gate_residual = np.asarray([[1.0, 0.0], [0.0, 0.0]])
    up_residual = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    activation = np.asarray([2.0, 3.0])
    gate_base = np.asarray([0.2, -0.1])
    up_base = np.asarray([0.5, 0.7])
    gate_target = gate_residual @ activation
    up_target = up_residual @ activation
    down = np.eye(2)
    target_output = np.asarray(
        np.array([gate_base[0] + gate_target[0], gate_base[1] + gate_target[1]])
    )
    target_output = (
        1.0 / (1.0 + np.exp(-target_output)) * target_output
        * (up_base + up_target)
    )
    base_output = (
        1.0 / (1.0 + np.exp(-gate_base)) * gate_base * up_base
    )
    rows, _, _, _ = transformed_occurrence(
        method="fixture", gate_atoms=gate_residual, up_atoms=up_residual,
        gate_basis=basis, up_basis=basis, activation=activation,
        gate2=gate_base, up2=up_base,
        target_gate_correction=gate_target, target_up_correction=up_target,
        target_output=target_output, base_output=base_output, down4=down,
        proxy=np.empty((2, 0)), beta=0.0, budgets=[1, 2], targets=[0.9, 0.95, 0.99],
        accounting={},
    )
    assert rows[-1]["isolated_gate_up_recovery"] == 1.0
    assert np.isclose(rows[-1]["expert_output_recovery"], 1.0)


def test_transformed_occurrence_treats_two_coordinate_page_as_indivisible() -> None:
    basis = np.eye(4)
    gate_atoms = np.eye(4)
    up_atoms = np.eye(4)
    activation = np.asarray([4.0, 3.0, 2.0, 1.0])
    target = activation.copy()
    rows, _, contributions, order = transformed_occurrence(
        method="fixture_int2", gate_atoms=gate_atoms, up_atoms=up_atoms,
        gate_basis=basis, up_basis=basis, activation=activation,
        gate2=np.zeros(4), up2=np.ones(4),
        target_gate_correction=target, target_up_correction=target,
        target_output=np.ones(4), base_output=np.zeros(4), down4=np.eye(4),
        proxy=np.empty((4, 0)), beta=0.0, budgets=[1, 2],
        targets=[0.9, 0.95, 0.99],
        accounting={"coordinates_per_action": 2},
    )
    assert contributions.shape == (2, 8)
    assert order.tolist() == [0, 1]
    assert rows[0]["logical_actions"] == 1
    assert rows[-1]["isolated_gate_up_recovery"] == 1.0


def test_native_exact_greedy_reaches_exact_joint_state() -> None:
    gate2 = np.asarray([-0.2, 0.3])
    gate4 = np.asarray([0.8, -0.7])
    up2 = np.asarray([0.4, 0.6])
    up4 = np.asarray([1.2, -0.5])
    hidden = native_states(gate2, gate4, up2, up4)
    down_columns = np.asarray([[1.0, 0.2], [0.1, 0.9]])
    order = native_exact_greedy(hidden, down_columns, np.empty((2, 0)), 0.0)
    assert sorted(order.tolist()) == list(range(4))
    assert np.allclose(apply_native_order(hidden, order, 4), hidden[:, 3])


def test_output_jacobian_basis_is_train_only_orthogonal_eigensystem() -> None:
    rng = np.random.default_rng(17)
    gate2 = rng.normal(size=(2, 4)).astype(np.float32)
    up2 = rng.normal(size=(2, 4)).astype(np.float32)
    down2 = rng.normal(size=(3, 2)).astype(np.float32)
    matrices = {
        "gate": [gate2, gate2, gate2 + 0.1 * rng.normal(size=(2, 4))],
        "up": [up2, up2, up2 + 0.1 * rng.normal(size=(2, 4))],
        "down": [down2, down2, down2 + 0.1 * rng.normal(size=(3, 2))],
    }
    basis, values = output_jacobian_basis_torch(
        matrices, rng.normal(size=(5, 4)), np.empty((3, 0)), 0.0,
        torch.device("cpu"), rank=4,
    )
    assert basis.shape == (4, 4)
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-5)
    assert np.all(values[:-1] >= values[1:])


def test_rotation_analyzer_builds_frontier_target_and_fit_summaries() -> None:
    rows = []
    for invocation, recovery in enumerate((0.8, 1.0)):
        rows.append({
            "request_id": f"r{invocation}", "position": invocation,
            "layer": 0, "expert_id": 1, "method": "fixture",
            "physical_pages": 64, "isolated_gate_up_recovery": recovery,
            "expert_output_recovery": recovery - 0.1,
            "actions_at_90pct": 10 + invocation, "actions_at_95pct": 12 + invocation,
            "actions_at_99pct": 16 + invocation, "physically_encoded": True,
            "budget_semantics": "physical_512_byte_pages", "action_payload_bytes": 512,
            "coordinates_per_action": 1,
            "representation_code_bytes": 512, "representation_scale_bytes": 2,
            "representation_bytes": 514, "basis_scope": "layer_shared",
            "basis_storage_bytes": 128, "transform_macs": 64,
        })
    omp = pd.DataFrame([{
        "method": "fixture", "target_recovery": 0.99,
        "diagonal_to_exact_ratio": 1.25,
    }])
    fit = {"layers": {"0": {
        "activation_top_rank_energy_fraction": 0.7,
        "uniform_top_rank_energy_fraction": 0.8,
        "router_top_rank_energy_fraction": 0.9,
        "wall_seconds": 1.0,
        "ajd": {
            "mean_initial_offdiagonal_fraction": 0.4,
            "mean_final_offdiagonal_fraction": 0.2,
        },
    }}}
    summaries = build_summaries(pd.DataFrame(rows), omp, fit)
    frontier = summaries["refinement_rotation_frontier_summary.csv"]
    assert np.isclose(frontier.iloc[0]["isolated_recovery_median"], 0.9)
    assert summaries["refinement_rotation_actions_to_target.csv"].shape[0] == 3
    assert "| method" in markdown_table(frontier[["method", "physical_pages"]])
