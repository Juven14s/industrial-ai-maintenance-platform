"""Tests for safety-oriented model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from industrial_maintenance.safety import (
    safety_metrics,
    select_safety_candidate,
    weighted_quantile,
)


def test_weighted_quantile_respects_weights():
    values = np.array([1.0, 10.0, 100.0])
    assert weighted_quantile(values, 0.5, np.array([1.0, 8.0, 1.0])) == 10.0


def test_safety_metrics_penalize_overestimation_five_times_more():
    truth = np.array([50.0])
    weights = np.ones(1)
    early = safety_metrics(
        truth,
        np.array([40.0]),
        weights,
        overestimate_cost=5.0,
        underestimate_cost=1.0,
        dangerous_margin=10.0,
    )
    late = safety_metrics(
        truth,
        np.array([60.0]),
        weights,
        overestimate_cost=5.0,
        underestimate_cost=1.0,
        dangerous_margin=10.0,
    )
    assert early["mae"] == late["mae"] == 10.0
    assert late["asymmetric_mae"] == 5 * early["asymmetric_mae"]
    assert late["overestimate_rate"] == 1.0
    assert early["overestimate_rate"] == 0.0


def test_selection_applies_safety_constraint_before_cost():
    comparison = pd.DataFrame(
        [
            {
                "model": "accurate_but_unsafe",
                "overestimate_rate": 0.40,
                "asymmetric_mae": 10.0,
                "dangerous_overestimate_rate": 0.10,
                "rmse": 8.0,
            },
            {
                "model": "conservative",
                "overestimate_rate": 0.15,
                "asymmetric_mae": 12.0,
                "dangerous_overestimate_rate": 0.02,
                "rmse": 11.0,
            },
        ]
    )
    selected, constraint_met = select_safety_candidate(comparison, 0.20)
    assert constraint_met is True
    assert selected["model"] == "conservative"


@pytest.mark.parametrize(
    ("values", "weights"),
    [([], []), ([1.0], [-1.0]), ([np.nan], [1.0]), ([1.0, 2.0], [1.0])],
)
def test_weighted_quantile_rejects_invalid_inputs(values, weights):
    with pytest.raises(ValueError):
        weighted_quantile(np.asarray(values), 0.5, np.asarray(weights))
