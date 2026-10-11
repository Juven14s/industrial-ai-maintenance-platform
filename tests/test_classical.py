"""Tests for the constant baseline and padding-aware Random Forest inputs."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from industrial_maintenance.classical import (
    MeanRULBaseline,
    PreparedWindows,
    engineer_inference_features,
    engineer_window_features,
    fit_random_forest,
    load_prepared_windows,
    regression_metrics,
    validate_prepared_windows,
)


def prepared_windows() -> PreparedWindows:
    X = np.array(
        [
            [[99.0, -99.0], [99.0, -99.0], [1.0, 8.0], [3.0, 6.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
        ],
        dtype=np.float32,
    )
    return validate_prepared_windows(
        X=X,
        y=np.array([20.0, 10.0]),
        unit_ids=np.array([1, 2]),
        end_cycles=np.array([2, 4]),
        observed_lengths=np.array([2, 4]),
        feature_names=np.array(["sensor_a", "sensor_b"]),
    )


def test_engineered_features_ignore_padding_and_have_stable_names():
    features = engineer_window_features(prepared_windows())
    assert features.values.shape == (2, 15)
    assert features.names == (
        "sensor_a__last",
        "sensor_a__mean",
        "sensor_a__std",
        "sensor_a__min",
        "sensor_a__max",
        "sensor_a__delta",
        "sensor_a__slope",
        "sensor_b__last",
        "sensor_b__mean",
        "sensor_b__std",
        "sensor_b__min",
        "sensor_b__max",
        "sensor_b__delta",
        "sensor_b__slope",
        "observed_length",
    )
    np.testing.assert_allclose(
        features.values[0],
        [3.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0, 6.0, 7.0, 1.0, 6.0, 8.0, -2.0, -2.0, 2.0],
    )


def test_changing_only_left_padding_cannot_change_engineered_features():
    original = prepared_windows()
    changed_X = original.X.copy()
    changed_X[0, :2] = [[-1e6, 1e6], [3e6, -4e6]]
    changed = validate_prepared_windows(
        changed_X,
        original.y,
        original.unit_ids,
        original.end_cycles,
        original.observed_lengths,
        original.feature_names,
    )
    np.testing.assert_array_equal(
        engineer_window_features(original).values,
        engineer_window_features(changed).values,
    )


def test_unlabeled_inference_features_exactly_match_supervised_features():
    windows = prepared_windows()
    supervised = engineer_window_features(windows)
    inference = engineer_inference_features(
        windows.X, windows.observed_lengths, windows.feature_names
    )
    assert inference.names == supervised.names
    np.testing.assert_array_equal(inference.values, supervised.values)


def test_one_observation_has_zero_std_delta_and_slope():
    windows = validate_prepared_windows(
        X=np.array([[[99.0], [99.0], [5.0]]]),
        y=np.array([7.0]),
        unit_ids=np.array([1]),
        end_cycles=np.array([1]),
        observed_lengths=np.array([1]),
        feature_names=["sensor"],
    )
    values = engineer_window_features(windows).values[0]
    np.testing.assert_array_equal(values, [5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0, 1.0])


def test_npz_loader_reuses_milestone_one_schema(tmp_path):
    expected = prepared_windows()
    path = tmp_path / "learning_windows.npz"
    np.savez_compressed(
        path,
        X=expected.X,
        y=expected.y,
        unit_ids=expected.unit_ids,
        end_cycles=expected.end_cycles,
        observed_lengths=expected.observed_lengths,
        feature_names=np.asarray(expected.feature_names),
    )
    loaded = load_prepared_windows(path)
    np.testing.assert_array_equal(loaded.X, expected.X)
    np.testing.assert_array_equal(loaded.y, expected.y)
    assert loaded.feature_names == expected.feature_names


def test_npz_loader_rejects_an_incomplete_artifact(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, X=np.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="Missing arrays"):
        load_prepared_windows(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("X", np.array([[[np.nan]]])),
        ("y", np.array([-1.0, 2.0])),
        ("observed_lengths", np.array([0, 4])),
        ("feature_names", ["duplicate", "duplicate"]),
    ],
)
def test_prepared_windows_reject_invalid_arrays(field, replacement):
    source = prepared_windows()
    values = {
        "X": source.X,
        "y": source.y,
        "unit_ids": source.unit_ids,
        "end_cycles": source.end_cycles,
        "observed_lengths": source.observed_lengths,
        "feature_names": source.feature_names,
    }
    values[field] = replacement
    with pytest.raises(ValueError):
        validate_prepared_windows(**values)


def test_mean_baseline_uses_training_targets_only():
    baseline = MeanRULBaseline().fit(np.ones((3, 2)), np.array([3.0, 6.0, 12.0]))
    np.testing.assert_array_equal(baseline.predict(np.zeros((2, 2))), [7.0, 7.0])
    with pytest.raises(ValueError, match="Expected 2"):
        baseline.predict(np.zeros((1, 3)))


def test_mean_baseline_uses_a_weighted_training_mean():
    baseline = MeanRULBaseline().fit(
        np.ones((2, 1)),
        np.array([0.0, 10.0]),
        sample_weight=np.array([9.0, 1.0]),
    )
    np.testing.assert_array_equal(baseline.predict(np.ones((2, 1))), [1.0, 1.0])


def test_mean_baseline_requires_fit():
    with pytest.raises(ValueError, match="fitted"):
        MeanRULBaseline().predict(np.ones((1, 2)))


def test_random_forest_is_reproducible_and_learns_training_signal():
    X = np.arange(40, dtype=np.float64).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 5.0
    first = fit_random_forest(X, y, seed=17, n_estimators=25, max_depth=6, min_samples_leaf=1)
    second = fit_random_forest(X, y, seed=17, n_estimators=25, max_depth=6, min_samples_leaf=1)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.n_jobs == 1
    assert regression_metrics(y, first.predict(X))["mae"] < 2.0


def test_random_forest_passes_validated_sample_weights(monkeypatch):
    captured = {}

    def fake_fit(self, X, y, sample_weight=None):
        captured["X"] = X
        captured["y"] = y
        captured["sample_weight"] = sample_weight
        return self

    monkeypatch.setattr(RandomForestRegressor, "fit", fake_fit)
    weights = np.array([0.25, 0.75])
    fit_random_forest(np.array([[1.0], [2.0]]), np.array([3.0, 4.0]), sample_weight=weights)
    np.testing.assert_array_equal(captured["sample_weight"], weights)


@pytest.mark.parametrize(
    "weights",
    [np.array([1.0]), np.array([1.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, np.nan])],
)
def test_models_reject_invalid_sample_weights(weights):
    X = np.ones((2, 1))
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="sample_weight"):
        MeanRULBaseline().fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="sample_weight"):
        fit_random_forest(X, y, n_estimators=1, sample_weight=weights)


def test_regression_metrics_are_in_cycles():
    metrics = regression_metrics(np.array([10.0, 20.0]), np.array([13.0, 16.0]))
    assert metrics["mae"] == pytest.approx(3.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(12.5))


@pytest.mark.parametrize(
    ("truth", "predictions"),
    [
        ([], []),
        ([1.0, 2.0], [1.0]),
        ([1.0], [[1.0]]),
        ([1.0], [np.inf]),
    ],
)
def test_regression_metrics_reject_invalid_inputs(truth, predictions):
    with pytest.raises(ValueError):
        regression_metrics(np.asarray(truth), np.asarray(predictions))
