import numpy as np

from oracle_study.page_allocator import nested_levels


def test_each_atom_can_have_an_independent_nested_precision():
    atoms = np.asarray([[1.0, -0.25], [-0.5, 0.75]], dtype=np.float32)
    levels = [nested_levels(atoms[:, index]) for index in range(2)]
    codes = np.asarray([2.0, 3.0], dtype=np.float32)
    precision = [4, 2]
    mixed = levels[0][precision[0]] * codes[0] + levels[1][precision[1]] * codes[1]
    uniform4 = levels[0][4] * codes[0] + levels[1][4] * codes[1]
    assert mixed.shape == (2,)
    assert not np.array_equal(mixed, uniform4)
    for atom in range(2):
        np.testing.assert_allclose(levels[atom][16], atoms[:, atom], rtol=2e-4, atol=2e-4)
