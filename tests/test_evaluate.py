"""Synthetic checks for the frozen official-test evaluation workflow."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from industrial_maintenance.data import DATA_COLUMNS, SENSOR_COLUMNS
from industrial_maintenance.evaluate import run_official_evaluation


def _raw_trajectories(lengths: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for unit_id, length in enumerate(lengths, start=1):
        for cycle in range(1, length + 1):
            row = {name: 10.0 for name in DATA_COLUMNS}
            row["unit_id"] = float(unit_id)
            row["cycle"] = float(cycle)
            row[SENSOR_COLUMNS[0]] = 100.0 * unit_id + cycle
            row[SENSOR_COLUMNS[1]] = unit_id + 0.25 * cycle**2
            rows.append(row)
    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def _write_raw(path, lengths: tuple[int, ...]) -> None:
    _raw_trajectories(lengths).to_csv(path, sep=" ", header=False, index=False)


@pytest.fixture
def tiny_final_evaluation(tmp_path):
    (tmp_path / "configs").mkdir()
    raw = tmp_path / "CMAPSSData"
    raw.mkdir()
    _write_raw(raw / "train_FD001.txt", (5, 6, 7))
    _write_raw(raw / "test_FD001.txt", (3, 4))
    (raw / "RUL_FD001.txt").write_text("2\n4\n", encoding="utf-8")

    selection_directory = tmp_path / "reports/fd001/models"
    selection_directory.mkdir(parents=True)
    selection = {
        "official_test_read": False,
        "selection": {
            "model": "lstm",
            "primary_metric": "rmse",
            "mae": 3.0,
            "rmse": 4.0,
            "bias": 1.0,
            "p90_absolute_error": 5.0,
            "overestimate_rate": 0.5,
            "nasa_score": 2.0,
        },
        "data": {"window_length": 3},
        "config": {
            "experiment": {
                "seed": 7,
                "balance_engines": True,
                "minimum_end_cycle": 2,
                "exclude_failure_windows": True,
            },
            "lstm": {
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "patience": 1,
                "min_delta": 0.0,
                "gradient_clip": 1.0,
            },
        },
        "lstm": {"best_epoch": 1},
    }
    selection_path = selection_directory / "experiment_summary.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()

    config = tmp_path / "configs/fd001_final.toml"
    config.write_text(
        f"""[data]
train_trajectories = "CMAPSSData/train_FD001.txt"
test_trajectories = "CMAPSSData/test_FD001.txt"
test_targets = "CMAPSSData/RUL_FD001.txt"
[selection]
summary = "reports/fd001/models/experiment_summary.json"
summary_sha256 = "{selection_hash}"
model = "lstm"
fixed_epochs = 1
[output]
artifact_directory = "artifacts/models/FD001/final"
report_directory = "reports/fd001/test"
[tracking]
directory = "mlruns"
experiment_name = "tiny-official-test"
[runtime]
num_threads = 1
""",
        encoding="utf-8",
    )
    return tmp_path, config, selection_path


def test_frozen_evaluation_retrains_all_engines_and_writes_reloadable_bundle(
    tiny_final_evaluation,
):
    root, config, _ = tiny_final_evaluation
    summary = run_official_evaluation(config)
    assert summary["official_test_read"] is True
    assert summary["official_test_used_for_selection"] is False
    assert summary["hyperparameters_frozen_before_test"] is True
    assert summary["selection"]["fixed_epochs"] == 1
    assert summary["data"]["training_units"] == 3
    assert summary["data"]["test_units"] == 2
    assert np.isfinite(list(summary["official_test_metrics"].values())).all()

    artifact_directory = root / "artifacts/models/FD001/final"
    assert (artifact_directory / "lstm.pt").is_file()
    assert (artifact_directory / "preprocessor.joblib").is_file()
    manifest = json.loads((artifact_directory / "final_model.json").read_text(encoding="utf-8"))
    assert manifest["training"]["official_test_used"] is False
    assert manifest["training"]["units"] == 3

    report_directory = root / "reports/fd001/test"
    predictions = pd.read_csv(report_directory / "official_test_predictions.csv")
    assert predictions["unit_id"].tolist() == [1, 2]
    assert predictions["true_rul"].tolist() == [2.0, 4.0]
    assert (report_directory / "OFFICIAL_TEST_EVALUATION.md").is_file()
    assert (root / "mlruns/mlflow.db").is_file()


def test_frozen_evaluation_rejects_changed_selection_evidence(tiny_final_evaluation):
    _, config, selection_path = tiny_final_evaluation
    selection_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        run_official_evaluation(config)
