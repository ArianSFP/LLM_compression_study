import numpy as np

from oracle_study.page_allocator import exact_id_page_selection, reconstruct_selected


def test_skipped_middle_atom_never_becomes_prefix_reconstruction():
    pages = {0: {"p0"}, 1: {"p1", "p2"}, 2: {"p0"}}
    result = exact_id_page_selection(
        [0, 1, 2], lambda atom: pages[atom], lambda atom: 10,
        byte_budget=4096, page_size=4096, skip_nonfitting=True,
    )
    assert result.atom_ids == [0, 2]
    assert result.pages == {"p0"}
    atoms = np.eye(3, dtype=np.float32)
    codes = np.asarray([1.0, 10.0, 100.0], dtype=np.float32)
    exact = reconstruct_selected(atoms, codes, result.atom_ids)
    prefix_bug = reconstruct_selected(atoms, codes, [0, 1])
    np.testing.assert_array_equal(exact, [1.0, 0.0, 100.0])
    assert not np.array_equal(exact, prefix_bug)
