"""Protocol and integration checks for the milestone-2 experiment runner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from industrial_maintenance.classical import PreparedWindows
from industrial_maintenance.train import (
    eligible_learning_windows,
    engine_balancing_weights,
    nasa_score,
    read_config,
    run_experiment,
)


def test_learning_filter_matches_validation_cutoff_support():
    windows = PreparedWindows(
        X=np.zeros((6, 3, 1), dtype=np.float32),
        y=np.array([5, 4, 3, 2, 1, 0], dtype=np.float64),
        unit_ids=np.ones(6, dtype=np.int64),
        end_cycles=np.arange(1, 7, dtype=np.int64),
        observed_lengths=np.minimum(np.arange(1, 7), 3),
        feature_names=("sensor_2",),
    )
    selected = eligible_learning_windows(windows, minimum_end_cycle=3, exclude_failure_windows=True)
    np.testing.assert_array_equal(selected.end_cycles, [3, 4, 5])
    np.testing.assert_array_equal(selected.y, [3, 2, 1])


def test_engine_balancing_gives_each_engine_equal_total_weight():
    unit_ids = np.array([1, 1, 2, 2, 2, 2, 3], dtype=np.int64)
    weights = engine_balancing_weights(unit_ids)
    totals = [weights[unit_ids == unit_id].sum() for unit_id in [1, 2, 3]]
    np.testing.assert_allclose(totals, totals[0])
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_nasa_score_penalizes_late_predictions_more_than_early_ones():
    truth = np.array([50.0])
    early = nasa_score(truth, np.array([40.0]))
    late = nasa_score(truth, np.array([60.0]))
    assert late > early > 0.0
    assert nasa_score(truth, truth) == 0.0


def _save_windows(path, X, y, unit_ids, end_cycles, observed_lengths):
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        end_cycles=np.asarray(end_cycles, dtype=np.int64),
        observed_lengths=np.asarray(observed_lengths, dtype=np.int64),
        feature_names=np.asarray(["sensor_2", "sensor_11"], dtype=str),
    )


@pytest.fixture
def tiny_experiment(tmp_path):
    (tmp_path / "configs").mkdir()
    data = tmp_path / "data/processed/FD001"
    data.mkdir(parents=True)
    (data / "preprocessor.joblib").write_bytes(b"synthetic fitted preprocessor")
    rng = np.random.default_rng(91)
    train_x = rng.normal(size=(12, 3, 2)).astype(np.float32)
    train_y = np.array([8, 7, 6, 5, 4, 3, 9, 8, 7, 6, 5, 4], dtype=np.float32)
    _save_windows(
        data / "learning_windows.npz",
        train_x,
        train_y,
        [1] * 6 + [2] * 6,
        [2, 3, 4, 5, 6, 7] * 2,
        [2, 3, 3, 3, 3, 3] * 2,
    )
    validation_x = rng.normal(size=(2, 3, 2)).astype(np.float32)
    _save_windows(
        data / "validation_windows.npz",
        validation_x,
        [5, 7],
        [3, 4],
        [5, 6],
        [3, 3],
    )
    config = tmp_path / "configs/fd001_models.toml"
    config.write_text(
        """[data]
directory = "data/processed/FD001"
learning_windows = "learning_windows.npz"
validation_windows = "validation_windows.npz"
preprocessor = "preprocessor.joblib"
[output]
artifact_directory = "artifacts/models/FD001"
report_directory = "reports/fd001/models"
[tracking]
directory = "mlruns"
experiment_name = "tiny-fd001"
[experiment]
seed = 7
balance_engines = true
primary_metric = "rmse"
minimum_end_cycle = 2
exclude_failure_windows = true
[[random_forest.candidates]]
name = "rf_tiny"
n_estimators = 5
max_depth = 3
min_samples_leaf = 1
max_features = 1.0
[lstm]
hidden_size = 4
num_layers = 1
dropout = 0.0
batch_size = 4
learning_rate = 0.01
weight_decay = 0.0
max_epochs = 1
patience = 1
min_delta = 0.0
gradient_clip = 1.0
num_threads = 1
""",
        encoding="utf-8",
    )
    return tmp_path, config


def test_complete_tiny_experiment_is_reloadable_and_never_needs_raw_test_files(tiny_experiment):
    root, config = tiny_experiment
    summary = run_experiment(config)
    assert summary["official_test_read"] is False
    assert summary["data"]["learning_units"] == 2
    assert summary["data"]["validation_units"] == 2
    assert {row["model"] for row in summary["metrics"]} == {
        "baseline_mean",
        "rf_tiny",
        "lstm",
    }
    assert not (root / "CMAPSSData").exists()
    assert (root / "artifacts/models/FD001/baseline.joblib").is_file()
    assert (root / "artifacts/models/FD001/random_forest.joblib").is_file()
    assert (root / "artifacts/models/FD001/lstm.pt").is_file()
    assert (root / "mlruns/mlflow.db").is_file()
    manifest = json.loads(
        (root / "artifacts/models/FD001/selected_model.json").read_text(encoding="utf-8")
    )
    assert manifest["official_test_evaluated"] is False
    assert manifest["input_contract"]["observed_length_required"] is (
        manifest["model"] != "baseline_mean"
    )
    assert manifest["preprocessor"]["path"] == "data/processed/FD001/preprocessor.joblib"
    assert (root / "artifacts/models/FD001" / manifest["artifact"]).is_file()
    report = root / "reports/fd001/models"
    assert (report / "MODEL_COMPARISON.md").is_file()
    recorded = json.loads((report / "experiment_summary.json").read_text(encoding="utf-8"))
    assert recorded["selection"] == summary["selection"]
    assert set(recorded["metrics"][0]) == {
        "model",
        "mae",
        "rmse",
        "bias",
        "p90_absolute_error",
        "overestimate_rate",
        "nasa_score",
    }


@pytest.mark.parametrize("destination", [".", "../outside", "data/processed/FD001/output"])
def test_config_rejects_unsafe_or_overlapping_output_paths(tiny_experiment, destination):
    _, config = tiny_experiment
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'artifact_directory = "artifacts/models/FD001"',
            f'artifact_directory = "{destination}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_config(config)


@pytest.mark.parametrize("source", ["../outside.npz", "C:/outside.npz"])
def test_config_rejects_windows_outside_prepared_data_directory(tiny_experiment, source):
    _, config = tiny_experiment
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'learning_windows = "learning_windows.npz"',
            f'learning_windows = "{source}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_config(config)
