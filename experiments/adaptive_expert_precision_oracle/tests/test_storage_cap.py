import numpy as np

from oracle_study.page_allocator import PacketLayout, regime_layouts, storage_multiplier


def test_replica_storage_is_counted_and_cap_can_be_enforced():
    orders = {"gate": np.arange(8), "up": np.arange(8), "down": np.arange(4)}
    lengths = {"gate": 4, "up": 4, "down": 8}
    layouts = [PacketLayout(orders, lengths, 512, f"r{index}") for index in range(4)]
    one = storage_multiplier(4096, layouts[:1], [4])
    four = storage_multiplier(4096, layouts, [4, 4, 4, 4])
    assert four["external_bytes"] > one["external_bytes"]
    assert four["external_multiplier"] > one["external_multiplier"]


def test_cold_expert_regime_layouts_cycle_when_masks_are_fewer_than_replicas():
    incidence = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    layouts, assignment = regime_layouts(incidence, replicas=4, seed=9)
    assert len(layouts) == 4
    assert assignment.shape == (2,)
    assert all(sorted(layout.tolist()) == [0, 1, 2] for layout in layouts)
