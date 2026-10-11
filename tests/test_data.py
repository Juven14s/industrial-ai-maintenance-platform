"""Vérifications du protocole FD001 sur de petites trajectoires synthétiques.

Les jeux officiels ne sont pas nécessaires pour lancer ces tests. Les assertions
portent surtout sur les cibles, la séparation des moteurs et l'absence de fuite.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from industrial_maintenance.data import (
    DATA_COLUMNS,
    SENSOR_COLUMNS,
    SensorPreprocessor,
    add_rul,
    load_cmapss,
    load_rul_targets,
    make_windows,
    split_by_engine,
    truncate_validation,
)


def trajectories(lengths: tuple[int, ...] = (5, 7, 6)) -> pd.DataFrame:
    """Deux capteurs variables et toutes les autres mesures constantes."""
    records = []
    for unit_id, length in enumerate(lengths, start=1):
        for cycle in range(1, length + 1):
            record = {name: 10.0 for name in DATA_COLUMNS}
            record.update(unit_id=unit_id, cycle=cycle)
            record[SENSOR_COLUMNS[0]] = 100.0 * unit_id + cycle
            record[SENSOR_COLUMNS[1]] = unit_id + 0.5 * cycle
            records.append(record)
    return pd.DataFrame(records, columns=DATA_COLUMNS)


def write_raw(tmp_path, frame: pd.DataFrame):
    path = tmp_path / "train_synthetic.txt"
    frame.to_csv(path, sep=" ", header=False, index=False)
    return path


def test_load_cmapss_preserves_all_measurements_and_row_order(tmp_path):
    original = trajectories()
    loaded = load_cmapss(write_raw(tmp_path, original))
    assert list(loaded.columns) == list(DATA_COLUMNS)
    assert loaded.shape == (18, 26)
    assert_frame_equal(loaded, original, check_dtype=False)


def test_load_rul_targets_preserves_engine_order_and_count(tmp_path):
    path = tmp_path / "RUL_synthetic.txt"
    path.write_text("12 \n0 \n37 \n", encoding="utf-8")
    targets = load_rul_targets(path, expected_count=3)
    np.testing.assert_array_equal(targets, [12.0, 0.0, 37.0])
    assert targets.dtype == np.float32
    with pytest.raises(ValueError, match="Expected 2"):
        load_rul_targets(path, expected_count=2)


@pytest.mark.parametrize("content", ["1 2\n", "1.5\n", "-1\n", "nan\n", ""])
def test_load_rul_targets_rejects_invalid_files(tmp_path, content):
    path = tmp_path / "RUL_invalid.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_rul_targets(path)


@pytest.mark.parametrize("column_count", [25, 27])
def test_load_cmapss_rejects_missing_or_extra_columns(tmp_path, column_count):
    frame = trajectories()
    if column_count == 25:
        frame = frame.iloc[:, :-1]
    else:
        frame["extra"] = 0
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf", "invalid"])
def test_load_cmapss_rejects_nonfinite_or_nonnumeric_values(tmp_path, invalid_value):
    path = write_raw(tmp_path, trajectories())
    lines = path.read_text(encoding="utf-8").splitlines()
    tokens = lines[0].split()
    tokens[-1] = invalid_value
    lines[0] = " ".join(tokens)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cmapss(path)


@pytest.mark.parametrize("column", ["unit_id", "cycle"])
@pytest.mark.parametrize("invalid_value", [0, -1, 1.5])
def test_load_cmapss_requires_positive_integer_identifiers_and_cycles(
    tmp_path, column, invalid_value
):
    frame = trajectories().astype(float)
    frame.loc[0, column] = invalid_value
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_rejects_duplicate_cycles(tmp_path):
    frame = trajectories()
    frame.loc[1, "cycle"] = 1
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_rejects_a_gap_in_cycles(tmp_path):
    frame = trajectories().drop(index=2)
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_rejects_a_trajectory_not_starting_at_one(tmp_path):
    frame = trajectories()
    frame.loc[frame.unit_id == 1, "cycle"] += 1
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_does_not_silently_sort_invalid_rows(tmp_path):
    frame = trajectories().iloc[[1, 0, *range(2, 18)]]
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_requires_contiguous_raw_engine_ids(tmp_path):
    frame = trajectories()
    frame.loc[frame.unit_id == 2, "unit_id"] = 4
    frame = frame.sort_values(["unit_id", "cycle"])
    with pytest.raises(ValueError):
        load_cmapss(write_raw(tmp_path, frame))


def test_load_cmapss_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cmapss(path)


def test_add_rul_uses_each_engines_failure_cycle_and_preserves_input():
    frame = trajectories((5, 3))
    original = frame.copy(deep=True)
    labelled = add_rul(frame)
    assert labelled.loc[labelled.unit_id == 1, "rul"].tolist() == [4, 3, 2, 1, 0]
    assert labelled.loc[labelled.unit_id == 2, "rul"].tolist() == [2, 1, 0]
    assert_frame_equal(frame, original)


def test_engine_split_is_disjoint_complete_and_reproducible():
    frame = add_rul(trajectories((3,) * 10))
    original = frame.copy(deep=True)
    train, validation = split_by_engine(frame, validation_fraction=0.2, seed=42)
    repeat_train, repeat_validation = split_by_engine(frame, validation_fraction=0.2, seed=42)
    assert len(train.unit_id.unique()) == 8
    assert len(validation.unit_id.unique()) == 2
    assert set(train.unit_id).isdisjoint(validation.unit_id)
    reconstructed = pd.concat([train, validation]).sort_values(["unit_id", "cycle"])
    assert_frame_equal(reconstructed.reset_index(drop=True), frame.reset_index(drop=True))
    assert_frame_equal(train, repeat_train)
    assert_frame_equal(validation, repeat_validation)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.2])
def test_engine_split_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError):
        split_by_engine(trajectories(), validation_fraction=fraction)


def test_engine_split_requires_at_least_two_engines():
    with pytest.raises(ValueError):
        split_by_engine(trajectories((5,)))


def test_validation_truncation_preserves_true_rul_and_input():
    full = add_rul(trajectories((30, 35, 40)))
    original = full.copy(deep=True)
    observed, cutoffs = truncate_validation(full, seed=43, min_observed_cycles=20)
    assert set(cutoffs.columns) >= {"unit_id", "cutoff_cycle", "failure_cycle", "rul"}
    assert set(observed.unit_id) == set(full.unit_id)
    assert cutoffs.unit_id.is_unique
    for row in cutoffs.itertuples(index=False):
        actual = observed.loc[observed.unit_id == row.unit_id]
        source = full.loc[full.unit_id == row.unit_id]
        assert row.failure_cycle == source.cycle.max()
        assert 20 <= row.cutoff_cycle < row.failure_cycle
        assert actual.cycle.max() == row.cutoff_cycle
        assert actual.cycle.tolist() == list(range(1, row.cutoff_cycle + 1))
        assert row.rul == row.failure_cycle - row.cutoff_cycle
        assert actual.iloc[-1]["rul"] == row.rul > 0
        expected = source.loc[source.cycle <= row.cutoff_cycle]
        assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert_frame_equal(full, original)


def test_validation_truncation_is_reproducible():
    full = add_rul(trajectories((30, 35, 40)))
    first_observed, first_cutoffs = truncate_validation(full, seed=43)
    second_observed, second_cutoffs = truncate_validation(full, seed=43)
    assert_frame_equal(first_observed, second_observed)
    assert_frame_equal(first_cutoffs, second_cutoffs)


def test_validation_truncation_adapts_minimum_to_short_engines():
    full = add_rul(trajectories((2,)))
    observed, cutoffs = truncate_validation(full, seed=43, min_observed_cycles=20)
    assert observed.cycle.tolist() == [1]
    assert observed.rul.tolist() == [1]
    assert cutoffs.cutoff_cycle.tolist() == [1]


def test_validation_truncation_requires_a_future_cycle():
    with pytest.raises(ValueError):
        truncate_validation(add_rul(trajectories((1,))))


def test_preprocessor_selects_features_from_training_data_only():
    train = add_rul(trajectories((5, 6)))
    validation = add_rul(trajectories((4,)))
    validation[SENSOR_COLUMNS[2]] = [1.0, 2.0, 3.0, 4.0]
    processor = SensorPreprocessor().fit(train)
    features_before = list(processor.feature_names_)
    mean_before = processor.scaler_.mean_.copy()
    processor.transform(validation)
    assert set(processor.feature_names_) == set(SENSOR_COLUMNS[:2])
    assert SENSOR_COLUMNS[2] in processor.dropped_features_
    assert list(processor.feature_names_) == features_before
    np.testing.assert_array_equal(processor.scaler_.mean_, mean_before)


def test_preprocessor_keeps_small_but_nonzero_training_variation():
    train = trajectories((5,))
    train[SENSOR_COLUMNS[2]] = 10.0 + train.cycle * 1e-5
    processor = SensorPreprocessor().fit(train)
    assert SENSOR_COLUMNS[2] in processor.feature_names_
    assert SENSOR_COLUMNS[2] not in processor.dropped_features_


def test_preprocessor_uses_training_statistics_and_preserves_metadata_and_inputs():
    train = add_rul(trajectories((5, 6)))
    validation = add_rul(trajectories((4,)))
    validation[SENSOR_COLUMNS[0]] += 1000.0
    train_original = train.copy(deep=True)
    validation_original = validation.copy(deep=True)
    processor = SensorPreprocessor().fit(train)
    transformed_train = processor.transform(train)
    transformed_validation = processor.transform(validation)
    names = list(processor.feature_names_)
    np.testing.assert_allclose(transformed_train[names].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed_train[names].std(ddof=0), 1.0, atol=1e-6)
    assert transformed_validation[SENSOR_COLUMNS[0]].mean() > 10.0
    np.testing.assert_allclose(
        transformed_validation[names],
        processor.scaler_.transform(validation[names]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert_frame_equal(
        transformed_train[["unit_id", "cycle", "rul"]], train[["unit_id", "cycle", "rul"]]
    )
    assert_frame_equal(train, train_original)
    assert_frame_equal(validation, validation_original)


def test_preprocessor_can_be_saved_and_reloaded_without_changing_predictions(tmp_path):
    train = trajectories((5, 6))
    validation = trajectories((4,))
    processor = SensorPreprocessor().fit(train)
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(processor, path)
    restored = joblib.load(path)
    assert list(restored.feature_names_) == list(processor.feature_names_)
    assert_frame_equal(restored.transform(validation), processor.transform(validation))


def test_preprocessor_requires_fit_before_transform():
    with pytest.raises(ValueError):
        SensorPreprocessor().transform(trajectories())


def test_preprocessor_rejects_no_informative_measurements():
    frame = trajectories()
    frame.loc[:, list(DATA_COLUMNS[2:])] = 10.0
    with pytest.raises(ValueError):
        SensorPreprocessor().fit(frame)


def test_windows_pad_left_and_use_the_true_endpoint_label():
    frame = add_rul(trajectories((2,)))
    original = frame.copy(deep=True)
    batch = make_windows(frame, feature_names=list(SENSOR_COLUMNS[:2]), window_length=4)
    assert batch.X.shape == (2, 4, 2)
    assert batch.X.dtype == np.float32
    values = frame[list(SENSOR_COLUMNS[:2])].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(batch.X[0], np.repeat(values[:1], 4, axis=0))
    np.testing.assert_array_equal(batch.X[1], values[[0, 0, 0, 1]])
    np.testing.assert_array_equal(batch.y, [1, 0])
    np.testing.assert_array_equal(batch.unit_ids, [1, 1])
    np.testing.assert_array_equal(batch.end_cycles, [1, 2])
    np.testing.assert_array_equal(batch.observed_lengths, [1, 2])
    assert_frame_equal(frame, original)


def test_windows_never_mix_engines_and_cover_every_observed_cycle():
    frame = add_rul(trajectories((2, 5)))
    features = list(SENSOR_COLUMNS[:2])
    batch = make_windows(frame, feature_names=features, window_length=3)
    assert batch.X.shape == (7, 3, 2)
    np.testing.assert_array_equal(batch.unit_ids, [1, 1, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(batch.end_cycles, [1, 2, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(batch.observed_lengths, [1, 2, 1, 2, 3, 3, 3])
    for index, (unit_id, end_cycle) in enumerate(zip(batch.unit_ids, batch.end_cycles)):
        source = frame.loc[(frame.unit_id == unit_id) & (frame.cycle <= end_cycle)]
        recent = source[features].tail(3).to_numpy(dtype=np.float32)
        observed_length = len(recent)
        np.testing.assert_array_equal(batch.X[index, -observed_length:], recent)
        if observed_length < 3:
            np.testing.assert_array_equal(
                batch.X[index, : 3 - observed_length],
                np.repeat(recent[:1], 3 - observed_length, axis=0),
            )
        assert batch.y[index] == source.iloc[-1]["rul"]


def test_last_only_returns_one_window_per_engine_including_short_histories():
    frame = add_rul(trajectories((2, 5)))
    batch = make_windows(
        frame, feature_names=list(SENSOR_COLUMNS[:2]), window_length=3, last_only=True
    )
    assert batch.X.shape == (2, 3, 2)
    np.testing.assert_array_equal(batch.unit_ids, [1, 2])
    np.testing.assert_array_equal(batch.end_cycles, [2, 5])
    np.testing.assert_array_equal(batch.observed_lengths, [2, 3])
    np.testing.assert_array_equal(batch.y, [0, 0])


def test_changing_future_measurements_cannot_change_past_windows():
    frame = add_rul(trajectories((7,)))
    changed = frame.copy(deep=True)
    changed.loc[changed.cycle > 4, list(SENSOR_COLUMNS[:2])] += 9999.0
    features = list(SENSOR_COLUMNS[:2])
    original_batch = make_windows(frame, feature_names=features, window_length=3)
    changed_batch = make_windows(changed, feature_names=features, window_length=3)
    np.testing.assert_array_equal(original_batch.X[:4], changed_batch.X[:4])
    assert not np.array_equal(original_batch.X[-1], changed_batch.X[-1])


def test_windows_on_truncated_validation_keep_positive_endpoint_rul():
    full = add_rul(trajectories((30, 35)))
    observed, cutoffs = truncate_validation(full, seed=43, min_observed_cycles=20)
    batch = make_windows(
        observed, feature_names=list(SENSOR_COLUMNS[:2]), window_length=30, last_only=True
    )
    expected = cutoffs.set_index("unit_id").loc[batch.unit_ids]
    np.testing.assert_array_equal(batch.y, expected["rul"].to_numpy())
    assert np.all(batch.y > 0)


def test_windows_without_labels_are_valid_for_inference():
    batch = make_windows(
        trajectories((2,)), feature_names=list(SENSOR_COLUMNS[:2]), window_length=3
    )
    assert batch.y is None
    assert batch.X.shape == (2, 3, 2)


@pytest.mark.parametrize("forbidden_feature", ["unit_id", "cycle", "rul"])
def test_windows_reject_identifiers_age_or_targets_as_measurement_features(forbidden_feature):
    with pytest.raises(ValueError):
        make_windows(
            add_rul(trajectories()),
            feature_names=[SENSOR_COLUMNS[0], forbidden_feature],
        )


@pytest.mark.parametrize("window_length", [0, -1, 2.5])
def test_windows_reject_invalid_lengths(window_length):
    with pytest.raises(ValueError):
        make_windows(
            trajectories(), feature_names=list(SENSOR_COLUMNS[:2]), window_length=window_length
        )


def test_windows_reject_empty_trajectories():
    with pytest.raises(ValueError):
        make_windows(trajectories().iloc[:0], feature_names=list(SENSOR_COLUMNS[:2]))


@pytest.mark.parametrize("features", [[], [SENSOR_COLUMNS[0]] * 2, ["missing_sensor"]])
def test_windows_reject_empty_duplicate_or_unknown_features(features):
    with pytest.raises(ValueError):
        make_windows(trajectories(), feature_names=features)
