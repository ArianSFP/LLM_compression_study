from __future__ import annotations

import analyze_half_bpw_cross_layer_rate as analysis


def _option(pages: int, recovery: float) -> dict[str, float | int | str]:
    damage = 1.0 - float(recovery)
    return {
        "source": f"synthetic_{pages}",
        "pages": int(pages),
        "recovery": float(recovery),
        "exact_normalized_damage": damage,
        "predicted_normalized_damage": damage,
    }


def test_supplemental_options_are_pareto_merged_by_selection_objective():
    first = [_option(0, 0.0), _option(2, 0.8), _option(4, 0.9)]
    second = [_option(2, 0.85), _option(3, 0.82), _option(5, 0.95)]
    merged = analysis._merge_options(first, second, "exact")
    assert [(row["pages"], row["recovery"]) for row in merged] == [
        (0, 0.0), (2, 0.85), (4, 0.9), (5, 0.95),
    ]


def test_tail_floor_protects_37_of_40_groups_at_fixed_budget():
    groups = []
    for layer in range(40):
        options = [_option(0, 0.9), _option(1, 0.995)]
        groups.append({
            "request_id": "request", "position": 0, "layer": layer,
            "compressed": [dict(row) for row in options],
            "exact": [dict(row) for row in options],
        })
    selected, used, floor, target_pages, unprotected, filtered = (
        analysis._tail_floor_allocate(groups, "exact", 37)
    )
    recoveries = [
        float(group["exact"][choice]["recovery"])
        for group, choice in zip(filtered, selected)
    ]
    assert used == 37
    assert target_pages == 37
    assert len(unprotected) == 3
    assert floor >= 0.995 - 1e-12
    assert analysis._summarize(recoveries)["p10"] >= 0.995 - 1e-12
