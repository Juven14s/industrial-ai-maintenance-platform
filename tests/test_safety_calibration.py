"""Tests for leakage-safe cross-fitted safety calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_maintenance.safety_calibration import (
    apply_safety_margin,
    calibration_candidate_metrics,
    cross_fitted_margin_predictions,
    select_calibrated_candidate,
)


def _predictions() -> pd.DataFrame:
    rows = []
    errors = {1: [1.0, 2.0], 2: [10.0, 20.0], 3: [-2.0, -1.0]}
    for fold, fold_errors in errors.items():
        for index, error in enumerate(fold_errors):
            truth = 50.0
            rows.append(
                {
                    "fold": fold,
                    "model": "quantile",
                    "unit_id": fold * 10 + index,
                    "end_cycle": 30,
                    "true_rul": truth,
                    "predicted_rul": truth + error,
                    "sample_weight": 1.0,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def test_each_fold_margin_excludes_that_folds_errors():
    adjusted, margins = cross_fitted_margin_predictions(
        _predictions(), model_name="quantile", target_overestimate_rate=0.25
    )
    assert margins[2] == 1.0
    assert margins[1] == 10.0
    assert set(adjusted["safety_margin"]) == set(margins.values())


def test_safety_margin_never_returns_negative_rul():
    predictions = apply_safety_margin(np.array([2.0, 8.0, 20.0]), 5.0)
    np.testing.assert_array_equal(predictions, [0.0, 3.0, 15.0])


def test_calibrated_metrics_report_worst_groups():
    adjusted, _ = cross_fitted_margin_predictions(
        _predictions(), model_name="quantile", target_overestimate_rate=0.25
    )
    metrics = calibration_candidate_metrics(
        adjusted,
        overestimate_cost=5.0,
        underestimate_cost=1.0,
        dangerous_margin=10.0,
    )
    assert 0 <= metrics["overestimate_rate"] <= 1
    assert metrics["maximum_fold_overestimate_rate"] >= metrics["overestimate_rate"]
    assert metrics["maximum_horizon_overestimate_rate"] >= metrics["overestimate_rate"]


def test_selection_requires_global_fold_and_horizon_constraints():
    comparison = pd.DataFrame(
        [
            {
                "candidate": "globally_safe_but_unstable",
                "asymmetric_mae": 5.0,
                "rmse": 5.0,
                "overestimate_rate": 0.15,
                "dangerous_overestimate_rate": 0.01,
                "maximum_fold_overestimate_rate": 0.30,
                "maximum_horizon_overestimate_rate": 0.20,
            },
            {
                "candidate": "stable",
                "asymmetric_mae": 8.0,
                "rmse": 7.0,
                "overestimate_rate": 0.12,
                "dangerous_overestimate_rate": 0.01,
                "maximum_fold_overestimate_rate": 0.18,
                "maximum_horizon_overestimate_rate": 0.22,
            },
        ]
    )
    selected, met = select_calibrated_candidate(
        comparison,
        maximum_overestimate_rate=0.20,
        maximum_fold_overestimate_rate=0.20,
        maximum_horizon_overestimate_rate=0.25,
    )
    assert met is True
    assert selected["candidate"] == "stable"
