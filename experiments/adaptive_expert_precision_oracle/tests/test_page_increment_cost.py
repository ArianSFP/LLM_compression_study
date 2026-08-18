from oracle_study.page_allocator import incremental_page_cost


def test_resident_page_has_zero_incremental_external_cost():
    selected = {("layout", 0), ("layout", 1)}
    assert incremental_page_cost({("layout", 1)}, selected, 4096) == 0
    assert incremental_page_cost({("layout", 1), ("layout", 2)}, selected, 4096) == 4096
