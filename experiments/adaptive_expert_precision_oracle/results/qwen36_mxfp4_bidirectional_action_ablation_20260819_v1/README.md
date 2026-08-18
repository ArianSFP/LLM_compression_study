# Bidirectional action ablation

This run uses one identical A/B page layout and separates two changes that the
primary bidirectional policy introduced together:

* `ordered_a_with_direct_ab`: A is the only legal first one-bit description,
  but a direct two-bit AB action is allowed;
* `bidirectional_ab_no_direct`: A or B may arrive first, but direct AB is
  disabled.

Together with the primary ordered/bidirectional rows, these policies attribute
the small measured gain to direct AB rather than B-first eligibility. Exact
actions and charged pages are retained in `selected_description_actions.parquet`.
