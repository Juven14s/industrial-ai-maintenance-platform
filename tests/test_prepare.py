"""Exercise the complete preparation with a tiny, train-only synthetic project."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from industrial_maintenance.data import DATA_COLUMNS, add_rul, load_cmapss, truncate_validation
from industrial_maintenance.prepare import prepare, read_config


@pytest.fixture
def tiny_project(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "CMAPSSData").mkdir()
    rows = []
    for unit in range(1, 6):
        for cycle in range(1, 8 + unit):
            record = {name: 1.0 for name in DATA_COLUMNS}
            record.update(unit_id=unit, cycle=cycle)
            record["sensor_2"] = unit * 10.0 + cycle
            record["sensor_11"] = unit + cycle / 10.0
            rows.append(record)
    source = tmp_path / "CMAPSSData" / "train_FD001.txt"
    pd.DataFrame(rows, columns=DATA_COLUMNS).to_csv(source, sep=" ", index=False, header=False)
    config = tmp_path / "configs" / "fd001.toml"
    config.write_text(
        '[data]\nraw_dir = "CMAPSSData"\nsubset = "FD001"\n'
        "[split]\nvalidation_fraction = 0.2\nseed = 42\n"
        "[validation]\nseed = 43\nmin_observed_cycles = 2\n"
        "[windows]\nlength = 3\n"
        '[output]\ndirectory = "data/processed/FD001"\nreport_dir = "reports/fd001"\n',
        encoding="utf-8",
    )
    return tmp_path, source, config


def test_prepare_rebuilds_artifacts_without_any_official_test_file(tiny_project):
    root, source, config = tiny_project
    original_bytes = source.read_bytes()
    summary = prepare(config)
    assert summary["data"]["learning_units"] == 4
    assert summary["data"]["validation_units"] == 1
    assert summary["official_test_read"] is False
    assert source.read_bytes() == original_bytes
    assert summary["source"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert len(list((root / "CMAPSSData").iterdir())) == 1
    for artifact in summary["artifacts"]:
        path = root / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    report = root / "reports" / "fd001"
    splits = json.loads((report / "splits.json").read_text(encoding="utf-8"))
    assert set(splits["learning_unit_ids"]).isdisjoint(splits["validation_unit_ids"])
    with np.load(root / "data/processed/FD001/validation_windows.npz", allow_pickle=False) as data:
        cutoffs = pd.read_csv(report / "validation_cutoffs.csv").set_index("unit_id")
        assert data["X"].shape == (1, 3, 2)
        np.testing.assert_array_equal(data["y"], cutoffs.loc[data["unit_ids"], "rul"])
        assert np.all(data["y"] > 0)


def test_identical_config_rebuilds_identical_arrays_and_split(tiny_project):
    root, _, config = tiny_project
    prepare(config)
    with np.load(root / "data/processed/FD001/learning_windows.npz", allow_pickle=False) as arrays:
        before = {key: arrays[key].copy() for key in arrays.files}
    split_before = (root / "reports/fd001/splits.json").read_bytes()
    prepare(config)
    with np.load(root / "data/processed/FD001/learning_windows.npz", allow_pickle=False) as arrays:
        for key, values in before.items():
            np.testing.assert_array_equal(values, arrays[key])
    assert split_before == (root / "reports/fd001/splits.json").read_bytes()


def test_truncation_rejects_already_truncated_histories(tiny_project):
    _, source, _ = tiny_project
    observed, _ = truncate_validation(add_rul(load_cmapss(source)), min_observed_cycles=2)
    with pytest.raises(ValueError, match="complete trajectory"):
        truncate_validation(observed, min_observed_cycles=1)


def test_loader_rejects_blank_rows_instead_of_silently_dropping_them(tiny_project):
    _, source, _ = tiny_project
    text = source.read_text(encoding="utf-8")
    first, remainder = text.split("\n", maxsplit=1)
    source.write_text(first + "\n\n" + remainder, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cmapss(source)


@pytest.mark.parametrize("destination", ["CMAPSSData/prepared", "CMAPSSData", ".", "../outside"])
def test_configuration_rejects_output_overlapping_raw_data_or_leaving_project(
    tiny_project, destination
):
    _, _, config = tiny_project
    config.write_text(
        config.read_text(encoding="utf-8").replace('"data/processed/FD001"', f'"{destination}"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_config(config)


def test_cli_fails_clearly_for_a_missing_training_file(tiny_project):
    root, source, config = tiny_project
    # Keep the source intact; point the configuration at an absent folder.
    config.write_text(
        config.read_text(encoding="utf-8").replace('"CMAPSSData"', '"absent-data"'),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "industrial_maintenance.prepare", "--config", str(config)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "train_FD001.txt" in result.stderr
    assert source.is_file()
    assert not (root / "data").exists()
