import numpy as np

from oracle_study.rrq import QuantizerSpec, atom_packet_bytes, atom_stage_matrix, build_atom_rrq_series


def test_atom_rrq_is_atom_major_and_exactly_accounted():
    rng = np.random.default_rng(7)
    atoms = rng.standard_normal((128, 17), dtype=np.float32)
    series = build_atom_rrq_series(atoms, (QuantizerSpec("affine_mse", 64),) * 3)
    assert series.stages[0].shape == (17, 128)
    assert atom_stage_matrix(series, 1).shape == atoms.shape
    logical, physical = atom_packet_bytes(series.stages[0], 512)
    assert logical == 38  # 32 code + 4 FP16 scale + 2 one-byte zero point.
    assert physical == 512
    assert logical * 17 == series.stages[0].serialized_bytes() - 64


def test_rrq_stage_support_is_nested():
    depths = np.zeros(8, dtype=np.int8)
    for atom, requested_stage in ((3, 1), (3, 2), (1, 1), (3, 3)):
        assert requested_stage == depths[atom] + 1
        depths[atom] = requested_stage
    assert depths.tolist() == [0, 1, 0, 3, 0, 0, 0, 0]
