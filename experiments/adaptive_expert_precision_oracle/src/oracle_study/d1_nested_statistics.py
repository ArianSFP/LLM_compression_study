"""Request-level inference utilities for the D1 nested-safe oracle.

The functions in this module deliberately keep allocation and inference
separate.  They never select a policy.  Repeated layers or observations are
first reduced within a request; bootstrap and sign-flip inference then treat
the request as the only independent sampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    requests: int
    seed: int


@dataclass(frozen=True)
class SignFlipResult:
    statistic: float
    p_value: float
    alternative: str
    method: str
    permutations: int
    requests: int
    nonzero_requests: int
    seed: int


@dataclass(frozen=True)
class PairedDistributionSummary:
    requests: int
    mean: float
    median: float
    win_fraction: float
    p90: float
    p95: float
    maximum: float


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _stable_tuple(values: Sequence[object]) -> tuple[object, ...]:
    return tuple(sorted(set(values), key=lambda value: (type(value).__name__, repr(value))))


def _validate_identity_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    if frame.empty:
        raise ValueError(f"{name} cannot be empty")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name} identity columns cannot contain missing values")
    for column in columns:
        if frame[column].dtype == object or isinstance(frame[column].dtype, pd.StringDtype):
            if frame[column].astype(str).str.len().eq(0).any():
                raise ValueError(f"{name} identity columns cannot contain empty strings")


def _numeric_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    for column in columns:
        try:
            values = frame[column].to_numpy(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} column {column!r} must be numeric") from error
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} column {column!r} must be finite")


def _expected_arm_rate_pairs(
    frame: pd.DataFrame,
    *,
    arm_col: str,
    rate_col: str,
    expected_arms: Sequence[object] | None,
    expected_rates: Sequence[object] | None,
    expected_arm_rates: Mapping[object, Sequence[object]] | None,
) -> tuple[tuple[object, object], ...]:
    actual_arms = _stable_tuple(frame[arm_col].tolist())
    if expected_arms is not None:
        declared_arms = _stable_tuple(tuple(expected_arms))
        if actual_arms != declared_arms:
            raise ValueError(
                f"arm set differs from declaration: actual={actual_arms}, "
                f"expected={declared_arms}"
            )
    else:
        declared_arms = actual_arms

    if expected_arm_rates is not None and expected_rates is not None:
        raise ValueError("declare expected_rates or expected_arm_rates, not both")
    if expected_arm_rates is not None:
        declared_keys = _stable_tuple(tuple(expected_arm_rates))
        if declared_keys != declared_arms:
            raise ValueError(
                "expected_arm_rates keys must exactly match the declared arm set"
            )
        expected_pairs = {
            (arm, rate)
            for arm in declared_arms
            for rate in tuple(expected_arm_rates[arm])
        }
    elif expected_rates is not None:
        rates = _stable_tuple(tuple(expected_rates))
        expected_pairs = {(arm, rate) for arm in declared_arms for rate in rates}
    else:
        expected_pairs = set(zip(frame[arm_col], frame[rate_col], strict=True))

    actual_pairs = set(zip(frame[arm_col], frame[rate_col], strict=True))
    if actual_pairs != expected_pairs:
        actual = _stable_tuple(tuple(actual_pairs))
        expected = _stable_tuple(tuple(expected_pairs))
        raise ValueError(
            f"arm/rate grid differs from declaration: actual={actual}, expected={expected}"
        )
    return tuple(
        sorted(expected_pairs, key=lambda pair: (repr(pair[0]), repr(pair[1])))
    )


def validate_paired_alignment(
    frame: pd.DataFrame,
    *,
    request_col: str = "request_id",
    split_col: str = "split",
    arm_col: str = "arm",
    rate_col: str = "rate_pages_per_expert",
    layer_col: str = "injection_layer",
    observation_cols: Sequence[str] = (),
    expected_arms: Sequence[object] | None = None,
    expected_rates: Sequence[object] | None = None,
    expected_arm_rates: Mapping[object, Sequence[object]] | None = None,
    expected_layers: Sequence[object] | None = None,
) -> None:
    """Require a complete request/split/arm/rate paired observation grid.

    Arm-specific rate grids are supported because the independent PR13 arm can
    contain both the low and high product rates while nested arms contain only
    the low rates.  Within a request, however, every declared arm/rate pair
    must have exactly the same layer/observation identities.
    """

    observations = tuple(observation_cols)
    identity = (
        request_col,
        split_col,
        arm_col,
        rate_col,
        layer_col,
        *observations,
    )
    _require_columns(frame, identity, "paired frame")
    _validate_identity_values(frame, identity, name="paired frame")
    if len(set(identity)) != len(identity):
        raise ValueError("paired identity column names must be unique")
    if frame.duplicated(list(identity)).any():
        raise ValueError("paired frame contains duplicate observation identities")

    request_splits = frame[[request_col, split_col]].drop_duplicates()
    split_counts = request_splits.groupby(request_col, sort=False)[split_col].nunique()
    if split_counts.gt(1).any():
        offenders = _stable_tuple(split_counts[split_counts.gt(1)].index.tolist())
        raise ValueError(f"requests appear in more than one split: {offenders}")

    pairs = _expected_arm_rate_pairs(
        frame,
        arm_col=arm_col,
        rate_col=rate_col,
        expected_arms=expected_arms,
        expected_rates=expected_rates,
        expected_arm_rates=expected_arm_rates,
    )
    declared_layers = (
        _stable_tuple(tuple(expected_layers))
        if expected_layers is not None
        else _stable_tuple(frame[layer_col].tolist())
    )
    actual_layers = _stable_tuple(frame[layer_col].tolist())
    if actual_layers != declared_layers:
        raise ValueError(
            f"layer set differs from declaration: actual={actual_layers}, "
            f"expected={declared_layers}"
        )

    key_columns = (layer_col, *observations)
    expected_pair_set = set(pairs)
    for (split, request), request_rows in frame.groupby(
        [split_col, request_col], sort=False, dropna=False
    ):
        actual_pair_set = set(
            zip(request_rows[arm_col], request_rows[rate_col], strict=True)
        )
        if actual_pair_set != expected_pair_set:
            missing = expected_pair_set.difference(actual_pair_set)
            extra = actual_pair_set.difference(expected_pair_set)
            raise ValueError(
                f"request {request!r} in split {split!r} has an unaligned "
                f"arm/rate grid; missing={_stable_tuple(tuple(missing))}, "
                f"extra={_stable_tuple(tuple(extra))}"
            )
        template: set[tuple[object, ...]] | None = None
        for arm, rate in pairs:
            selected = request_rows[
                request_rows[arm_col].eq(arm) & request_rows[rate_col].eq(rate)
            ]
            keys = set(selected[list(key_columns)].itertuples(index=False, name=None))
            layers = _stable_tuple(selected[layer_col].tolist())
            if layers != declared_layers:
                raise ValueError(
                    f"request {request!r}, arm {arm!r}, rate {rate!r} "
                    f"has layers {layers}, expected {declared_layers}"
                )
            if template is None:
                template = keys
            elif keys != template:
                raise ValueError(
                    f"request {request!r} in split {split!r} has unequal "
                    "layer/observation identities across arm/rate pairs"
                )


def request_layer_equal_means(
    frame: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    request_col: str = "request_id",
    split_col: str = "split",
    arm_col: str = "arm",
    rate_col: str = "rate_pages_per_expert",
    layer_col: str = "injection_layer",
    observation_cols: Sequence[str] = (),
    expected_arms: Sequence[object] | None = None,
    expected_rates: Sequence[object] | None = None,
    expected_arm_rates: Mapping[object, Sequence[object]] | None = None,
    expected_layers: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Average observations within layer, then layers equally within request."""

    values = tuple(value_cols)
    if not values or len(set(values)) != len(values):
        raise ValueError("value_cols must contain unique column names")
    _require_columns(frame, values, "paired frame")
    validate_paired_alignment(
        frame,
        request_col=request_col,
        split_col=split_col,
        arm_col=arm_col,
        rate_col=rate_col,
        layer_col=layer_col,
        observation_cols=observation_cols,
        expected_arms=expected_arms,
        expected_rates=expected_rates,
        expected_arm_rates=expected_arm_rates,
        expected_layers=expected_layers,
    )
    _numeric_values(frame, values, name="paired frame")
    layer_keys = [split_col, request_col, arm_col, rate_col, layer_col]
    layer_rows = (
        frame.groupby(layer_keys, sort=True, dropna=False)[list(values)]
        .mean()
        .reset_index()
    )
    request_keys = [split_col, request_col, arm_col, rate_col]
    request_rows = (
        layer_rows.groupby(request_keys, sort=True, dropna=False)[list(values)]
        .mean()
        .reset_index()
    )
    layer_counts = (
        layer_rows.groupby(request_keys, sort=True, dropna=False)[layer_col]
        .nunique()
        .rename("layers_averaged")
        .reset_index()
    )
    return (
        request_rows.merge(layer_counts, on=request_keys, validate="one_to_one")
        .sort_values(request_keys, kind="stable")
        .reset_index(drop=True)
    )


def _validate_request_level_grid(
    frame: pd.DataFrame,
    *,
    request_col: str,
    split_col: str,
    arm_col: str,
    rate_col: str,
    expected_arm_rates: Mapping[object, Sequence[object]] | None = None,
) -> tuple[tuple[object, object], ...]:
    identity = (request_col, split_col, arm_col, rate_col)
    _require_columns(frame, identity, "request-level frame")
    _validate_identity_values(frame, identity, name="request-level frame")
    if frame.duplicated(list(identity)).any():
        raise ValueError("request-level frame has duplicate request/arm/rate rows")
    assignments = frame[[request_col, split_col]].drop_duplicates()
    if assignments.groupby(request_col, sort=False)[split_col].nunique().gt(1).any():
        raise ValueError("a request cannot appear in more than one split")
    pairs = _expected_arm_rate_pairs(
        frame,
        arm_col=arm_col,
        rate_col=rate_col,
        expected_arms=None,
        expected_rates=None,
        expected_arm_rates=expected_arm_rates,
    )
    expected = set(pairs)
    for (split, request), part in frame.groupby(
        [split_col, request_col], sort=False, dropna=False
    ):
        actual = set(zip(part[arm_col], part[rate_col], strict=True))
        if actual != expected:
            raise ValueError(
                f"request {request!r} in split {split!r} has an incomplete "
                "request-level arm/rate grid"
            )
    return pairs


def request_equal_arm_means(
    request_rows: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    request_col: str = "request_id",
    split_col: str = "split",
    arm_col: str = "arm",
    rate_col: str = "rate_pages_per_expert",
    expected_arm_rates: Mapping[object, Sequence[object]] | None = None,
) -> pd.DataFrame:
    """Average request-level rows with one equal contribution per request."""

    values = tuple(value_cols)
    if not values or len(set(values)) != len(values):
        raise ValueError("value_cols must contain unique column names")
    _require_columns(request_rows, values, "request-level frame")
    _validate_request_level_grid(
        request_rows,
        request_col=request_col,
        split_col=split_col,
        arm_col=arm_col,
        rate_col=rate_col,
        expected_arm_rates=expected_arm_rates,
    )
    _numeric_values(request_rows, values, name="request-level frame")
    keys = [split_col, arm_col, rate_col]
    means = (
        request_rows.groupby(keys, sort=True, dropna=False)[list(values)]
        .mean()
        .reset_index()
    )
    counts = (
        request_rows.groupby(keys, sort=True, dropna=False)[request_col]
        .nunique()
        .rename("requests_averaged")
        .reset_index()
    )
    return means.merge(counts, on=keys, validate="one_to_one").sort_values(
        keys, kind="stable"
    ).reset_index(drop=True)


def paired_request_differences(
    request_rows: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    candidate_arm: object,
    reference_arm: object,
    candidate_rate: object,
    reference_rate: object | None = None,
    split: object | None = None,
    request_col: str = "request_id",
    split_col: str = "split",
    arm_col: str = "arm",
    rate_col: str = "rate_pages_per_expert",
    expected_arm_rates: Mapping[object, Sequence[object]] | None = None,
) -> pd.DataFrame:
    """Form candidate-minus-reference differences after strict request pairing."""

    values = tuple(value_cols)
    if not values or len(set(values)) != len(values):
        raise ValueError("value_cols must contain unique column names")
    _require_columns(request_rows, values, "request-level frame")
    _validate_request_level_grid(
        request_rows,
        request_col=request_col,
        split_col=split_col,
        arm_col=arm_col,
        rate_col=rate_col,
        expected_arm_rates=expected_arm_rates,
    )
    _numeric_values(request_rows, values, name="request-level frame")
    reference_rate = candidate_rate if reference_rate is None else reference_rate
    if candidate_arm == reference_arm and candidate_rate == reference_rate:
        raise ValueError("candidate and reference arm/rate identities must differ")
    selected = request_rows
    if split is not None:
        selected = selected[selected[split_col].eq(split)]
        if selected.empty:
            raise ValueError(f"split {split!r} has no request-level rows")
    candidate = selected[
        selected[arm_col].eq(candidate_arm) & selected[rate_col].eq(candidate_rate)
    ][[split_col, request_col, *values]].copy()
    reference = selected[
        selected[arm_col].eq(reference_arm) & selected[rate_col].eq(reference_rate)
    ][[split_col, request_col, *values]].copy()
    if candidate.empty or reference.empty:
        raise ValueError("candidate or reference arm/rate has no rows")
    keys = [split_col, request_col]
    candidate_ids = set(candidate[keys].itertuples(index=False, name=None))
    reference_ids = set(reference[keys].itertuples(index=False, name=None))
    if candidate_ids != reference_ids:
        raise ValueError("candidate and reference request sets do not align")
    candidate = candidate.rename(columns={name: f"candidate_{name}" for name in values})
    reference = reference.rename(columns={name: f"reference_{name}" for name in values})
    paired = candidate.merge(reference, on=keys, validate="one_to_one")
    paired["candidate_arm"] = candidate_arm
    paired["reference_arm"] = reference_arm
    paired["candidate_rate"] = candidate_rate
    paired["reference_rate"] = reference_rate
    for name in values:
        paired[f"{name}_difference"] = (
            paired[f"candidate_{name}"] - paired[f"reference_{name}"]
        )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def _request_vector(
    frame: pd.DataFrame,
    *,
    value_col: str,
    request_col: str,
    split_col: str | None,
    minimum_requests: int,
) -> np.ndarray:
    columns = [request_col, value_col]
    if split_col is not None:
        columns.append(split_col)
    _require_columns(frame, columns, "paired request differences")
    _validate_identity_values(frame, [request_col], name="paired request differences")
    if frame.duplicated(request_col).any():
        raise ValueError(
            "paired inference requires exactly one row per independent request"
        )
    if split_col is not None:
        _validate_identity_values(frame, [split_col], name="paired request differences")
        splits = _stable_tuple(frame[split_col].tolist())
        if len(splits) != 1:
            raise ValueError(
                "paired inference cannot pool calibration and evaluation splits"
            )
    _numeric_values(frame, [value_col], name="paired request differences")
    if len(frame) < int(minimum_requests):
        raise ValueError(f"paired inference requires at least {minimum_requests} requests")
    ordering = np.argsort(
        np.asarray([repr(value) for value in frame[request_col]], dtype=object),
        kind="stable",
    )
    return frame[value_col].to_numpy(np.float64)[ordering]


def paired_request_cluster_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    value_col: str,
    request_col: str = "request_id",
    split_col: str | None = "split",
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile CI for a paired mean, resampling whole requests only."""

    values = _request_vector(
        frame,
        value_col=value_col,
        request_col=request_col,
        split_col=split_col,
        minimum_requests=2,
    )
    draws = int(resamples)
    if draws != resamples or draws < 1:
        raise ValueError("resamples must be a positive integer")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(draws, np.float64)
    request_count = values.size
    chunk_size = max(1, min(draws, 1_000_000 // request_count))
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        indices = rng.integers(
            0, request_count, size=(stop - start, request_count), endpoint=False
        )
        distribution[start:stop] = values[indices].mean(axis=1)
    alpha = 1.0 - level
    lower, upper = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=level,
        resamples=draws,
        requests=int(request_count),
        seed=int(seed),
    )


def _extreme_count(
    statistics: np.ndarray,
    observed: float,
    alternative: str,
) -> int:
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    if alternative == "two-sided":
        return int((np.abs(statistics) >= abs(observed) - tolerance).sum())
    if alternative == "less":
        return int((statistics <= observed + tolerance).sum())
    return int((statistics >= observed - tolerance).sum())


def paired_sign_flip_test(
    frame: pd.DataFrame,
    *,
    value_col: str,
    request_col: str = "request_id",
    split_col: str | None = "split",
    alternative: str = "two-sided",
    exact_max_nonzero_requests: int = 20,
    monte_carlo_draws: int = 100_000,
    seed: int = 0,
) -> SignFlipResult:
    """Paired sign-flip test over request-level differences.

    All sign patterns are enumerated when the number of nonzero request
    differences is small.  Otherwise a seeded Monte Carlo estimate uses the
    standard plus-one correction.
    """

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    exact_limit = int(exact_max_nonzero_requests)
    if exact_limit != exact_max_nonzero_requests or not 0 <= exact_limit <= 24:
        raise ValueError("exact_max_nonzero_requests must be an integer in [0, 24]")
    draws = int(monte_carlo_draws)
    if draws != monte_carlo_draws or draws < 1:
        raise ValueError("monte_carlo_draws must be a positive integer")
    values = _request_vector(
        frame,
        value_col=value_col,
        request_col=request_col,
        split_col=split_col,
        minimum_requests=1,
    )
    nonzero = values[values != 0.0]
    observed_sum = float(values.sum())
    observed = observed_sum / float(values.size)
    nonzero_count = int(nonzero.size)
    if nonzero_count <= exact_limit:
        permutations = 1 << nonzero_count
        extreme = 0
        bit_positions = np.arange(nonzero_count, dtype=np.uint64)
        chunk_size = 65_536
        for start in range(0, permutations, chunk_size):
            stop = min(permutations, start + chunk_size)
            patterns = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((patterns >> bit_positions) & 1).astype(np.float64)
            statistics = signs @ nonzero
            extreme += _extreme_count(statistics, observed_sum, alternative)
        p_value = extreme / float(permutations)
        method = "exact"
    else:
        rng = np.random.default_rng(int(seed))
        extreme = 0
        chunk_size = max(1, min(draws, 1_000_000 // nonzero_count))
        for start in range(0, draws, chunk_size):
            stop = min(draws, start + chunk_size)
            signs = rng.integers(
                0, 2, size=(stop - start, nonzero_count), dtype=np.int8
            ).astype(np.float64)
            signs = 2.0 * signs - 1.0
            extreme += _extreme_count(signs @ nonzero, observed_sum, alternative)
        permutations = draws
        p_value = (extreme + 1.0) / (draws + 1.0)
        method = "monte_carlo"
    return SignFlipResult(
        statistic=float(observed),
        p_value=float(p_value),
        alternative=alternative,
        method=method,
        permutations=int(permutations),
        requests=int(values.size),
        nonzero_requests=nonzero_count,
        seed=int(seed),
    )


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values in input order."""

    values = np.asarray(p_values, np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1]")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    scaled = (values.size - np.arange(values.size)) * ordered
    adjusted_ordered = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def paired_distribution_summary(
    frame: pd.DataFrame,
    *,
    value_col: str,
    request_col: str = "request_id",
    split_col: str | None = "split",
    improvement: str = "lower",
) -> PairedDistributionSummary:
    """Summarize one paired request-level difference distribution."""

    if improvement not in {"lower", "higher"}:
        raise ValueError("improvement must be 'lower' or 'higher'")
    values = _request_vector(
        frame,
        value_col=value_col,
        request_col=request_col,
        split_col=split_col,
        minimum_requests=1,
    )
    wins = values < 0.0 if improvement == "lower" else values > 0.0
    return PairedDistributionSummary(
        requests=int(values.size),
        mean=float(values.mean()),
        median=float(np.median(values)),
        win_fraction=float(wins.mean()),
        p90=float(np.quantile(values, 0.90)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(values.max()),
    )
