"""Fast synthetic checks for the deterministic LSTM component."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from industrial_maintenance.lstm import (  # noqa: E402
    LSTMConfig,
    LSTMRegressor,
    RULSequenceDataset,
    TargetScaler,
    fit_lstm,
    fit_lstm_fixed_epochs,
    load_lstm_checkpoint,
    predict_lstm,
    save_lstm_checkpoint,
)


def synthetic_sequences(sample_count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    sequences = rng.normal(size=(sample_count, 5, 2)).astype(np.float32)
    targets = (40.0 + 8.0 * sequences[:, -1, 0] - 3.0 * sequences[:, :, 1].mean(axis=1)).astype(
        np.float32
    )
    return sequences, targets


def fast_config(**overrides) -> LSTMConfig:
    values = {
        "input_size": 2,
        "hidden_size": 6,
        "batch_size": 8,
        "learning_rate": 0.02,
        "max_epochs": 4,
        "patience": 2,
        "seed": 17,
    }
    values.update(overrides)
    return LSTMConfig(**values)


def test_target_scaler_uses_learning_targets_and_handles_constant_values():
    scaler = TargetScaler.fit(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(scaler.transform(np.array([10.0, 20.0, 30.0])).mean(), 0.0)
    np.testing.assert_allclose(scaler.inverse_transform(np.array([0.0], dtype=np.float32)), [20.0])
    assert TargetScaler.fit(np.array([7.0, 7.0], dtype=np.float32)).scale == 1.0


def test_target_scaler_uses_optional_engine_balancing_weights():
    scaler = TargetScaler.fit(
        np.array([0.0, 10.0], dtype=np.float32),
        sample_weights=np.array([9.0, 1.0], dtype=np.float32),
    )
    assert scaler.mean == pytest.approx(1.0)
    assert scaler.scale == pytest.approx(3.0)


def test_dataset_validates_shapes_features_and_sample_weights():
    sequences, targets = synthetic_sequences(5)
    scaler = TargetScaler.fit(targets)
    dataset = RULSequenceDataset(
        sequences,
        targets,
        input_size=2,
        target_scaler=scaler,
        sample_weights=np.arange(1, 6, dtype=np.float32),
        observed_lengths=np.arange(1, 6, dtype=np.int64),
    )
    assert len(dataset) == 5
    assert dataset[0][0].shape == (5, 2)
    assert dataset[0][1].dtype == torch.float32
    assert dataset.observed_lengths.tolist() == [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="Expected 3 features"):
        RULSequenceDataset(sequences, targets, input_size=3, target_scaler=scaler)
    with pytest.raises(ValueError, match="strictly positive"):
        RULSequenceDataset(
            sequences,
            targets,
            input_size=2,
            target_scaler=scaler,
            sample_weights=np.zeros(5),
        )
    with pytest.raises(ValueError, match="between 1 and 5"):
        RULSequenceDataset(
            sequences,
            targets,
            input_size=2,
            target_scaler=scaler,
            observed_lengths=np.array([0, 2, 3, 4, 5]),
        )


def test_forward_ignores_values_in_repeated_left_padding():
    torch.manual_seed(5)
    model = LSTMRegressor(fast_config(max_epochs=1)).eval()
    sequences = torch.tensor(
        [
            [[100.0, 100.0], [100.0, 100.0], [1.0, 2.0], [3.0, 4.0]],
            [[-100.0, -100.0], [-100.0, -100.0], [1.0, 2.0], [3.0, 4.0]],
        ]
    )
    predictions = model(sequences, torch.tensor([2, 2]))
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0.0, atol=0.0)


def test_fit_is_deterministic_and_accepts_engine_balancing_weights():
    sequences, targets = synthetic_sequences()
    train_x, validation_x = sequences[:18], sequences[18:]
    train_y, validation_y = targets[:18], targets[18:]
    weights = np.linspace(0.5, 1.5, len(train_x), dtype=np.float32)
    config = fast_config()

    first = fit_lstm(
        train_x,
        train_y,
        validation_x,
        validation_y,
        config,
        train_sample_weights=weights,
    )
    second = fit_lstm(
        train_x,
        train_y,
        validation_x,
        validation_y,
        config,
        train_sample_weights=weights,
    )

    assert first.target_scaler == second.target_scaler
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    np.testing.assert_array_equal(
        predict_lstm(first.model, validation_x, first.target_scaler),
        predict_lstm(second.model, validation_x, second.target_scaler),
    )


def test_early_stopping_restores_best_epoch():
    sequences, targets = synthetic_sequences()
    result = fit_lstm(
        sequences[:18],
        targets[:18],
        sequences[18:],
        targets[18:],
        fast_config(max_epochs=10, patience=2, min_delta=1e6),
    )
    assert result.best_epoch == 1
    assert len(result.history) == 3
    assert result.stopped_early


def test_fixed_epoch_fit_is_deterministic_and_never_uses_validation():
    sequences, targets = synthetic_sequences(18)
    lengths = np.resize(np.arange(1, 6), len(sequences))
    config = fast_config(max_epochs=3)
    first = fit_lstm_fixed_epochs(
        sequences,
        targets,
        config,
        train_sample_weights=np.linspace(0.5, 1.5, len(sequences)),
        train_observed_lengths=lengths,
    )
    second = fit_lstm_fixed_epochs(
        sequences,
        targets,
        config,
        train_sample_weights=np.linspace(0.5, 1.5, len(sequences)),
        train_observed_lengths=lengths,
    )
    assert len(first.history) == 3
    assert first.history == second.history
    np.testing.assert_array_equal(
        predict_lstm(
            first.model,
            sequences,
            first.target_scaler,
            observed_lengths=lengths,
        ),
        predict_lstm(
            second.model,
            sequences,
            second.target_scaler,
            observed_lengths=lengths,
        ),
    )


def test_fit_requires_both_observed_length_arrays():
    sequences, targets = synthetic_sequences()
    with pytest.raises(ValueError, match="must be provided together"):
        fit_lstm(
            sequences[:18],
            targets[:18],
            sequences[18:],
            targets[18:],
            fast_config(max_epochs=1),
            train_observed_lengths=np.full(18, 5),
        )


def test_prediction_inverts_target_scaling_and_clips_negative_rul():
    model = LSTMRegressor(fast_config(max_epochs=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    sequences, _ = synthetic_sequences(3)
    scaler = TargetScaler(mean=-2.0, scale=5.0)
    np.testing.assert_array_equal(predict_lstm(model, sequences, scaler), np.zeros(3))
    np.testing.assert_array_equal(
        predict_lstm(model, sequences, scaler, minimum_rul=None),
        np.full(3, -2.0, dtype=np.float32),
    )


def test_checkpoint_round_trip_keeps_cpu_predictions_and_metadata(tmp_path):
    sequences, targets = synthetic_sequences()
    train_lengths = np.resize(np.arange(1, 6), 18)
    validation_lengths = np.arange(1, 7).clip(max=5)
    result = fit_lstm(
        sequences[:18],
        targets[:18],
        sequences[18:],
        targets[18:],
        fast_config(),
        train_observed_lengths=train_lengths,
        validation_observed_lengths=validation_lengths,
    )
    with pytest.raises(ValueError, match="expects observed_lengths"):
        predict_lstm(result.model, sequences[18:], result.target_scaler)
    expected = predict_lstm(
        result.model,
        sequences[18:],
        result.target_scaler,
        observed_lengths=validation_lengths,
    )
    checkpoint_path = save_lstm_checkpoint(
        tmp_path / "lstm.pt",
        result.model,
        result.target_scaler,
        ("sensor_2", "sensor_3"),
        metadata={"best_epoch": result.best_epoch},
    )

    restored = load_lstm_checkpoint(checkpoint_path)
    actual = predict_lstm(
        restored.model,
        sequences[18:],
        restored.target_scaler,
        observed_lengths=validation_lengths,
    )
    np.testing.assert_array_equal(actual, expected)
    assert restored.feature_names == ("sensor_2", "sensor_3")
    assert restored.metadata == {"best_epoch": result.best_epoch}
    assert restored.expects_observed_lengths
    assert all(parameter.device.type == "cpu" for parameter in restored.model.parameters())

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 2
    assert payload["expects_observed_lengths"] is True
    assert all(tensor.device.type == "cpu" for tensor in payload["state_dict"].values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_size": 0}, "input_size"),
        ({"input_size": 2, "dropout": 1.0}, "dropout"),
        ({"input_size": 2, "learning_rate": 0.0}, "learning_rate"),
        ({"input_size": 2, "patience": 0}, "patience"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LSTMConfig(**kwargs)
