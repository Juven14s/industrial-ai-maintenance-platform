"""Compact, reproducible LSTM regression for causal C-MAPSS windows.

The module deliberately keeps model training independent from the FD001 CLI.
Targets are standardized from the learning set only and converted back to RUL
cycles by :func:`predict_lstm`. Checkpoints contain everything required for
CPU inference: architecture, weights, target scaling, and feature order.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class LSTMConfig:
    """Architecture and deterministic optimization settings.

    ``min_delta`` is the minimum improvement in clipped validation RMSE,
    expressed in RUL cycles, required to reset early-stopping patience.
    """

    input_size: int
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    patience: int = 7
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "input_size",
            "hidden_size",
            "num_layers",
            "batch_size",
            "max_epochs",
            "patience",
        ):
            _positive_integer(getattr(self, name), name)
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be a non-negative integer")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        dropout = _finite_float(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        learning_rate = _finite_float(self.learning_rate, "learning_rate")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive")
        weight_decay = _finite_float(self.weight_decay, "weight_decay")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        min_delta = _finite_float(self.min_delta, "min_delta")
        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.gradient_clip_norm is not None:
            gradient_clip_norm = _finite_float(self.gradient_clip_norm, "gradient_clip_norm")
            if gradient_clip_norm <= 0.0:
                raise ValueError("gradient_clip_norm must be strictly positive or None")


@dataclass(frozen=True)
class TargetScaler:
    """Mean and standard deviation fitted on learning targets only."""

    mean: float
    scale: float

    def __post_init__(self) -> None:
        mean = _finite_float(self.mean, "target mean")
        scale = _finite_float(self.scale, "target scale")
        if scale <= 0.0:
            raise ValueError("target scale must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        targets: np.ndarray | Tensor,
        sample_weights: np.ndarray | Tensor | None = None,
    ) -> TargetScaler:
        values = _target_array(targets, "targets")
        weights = _weight_array(sample_weights, len(values), "sample_weights").astype(np.float64)
        values_float64 = values.astype(np.float64)
        mean = float(np.average(values_float64, weights=weights))
        variance = float(np.average((values_float64 - mean) ** 2, weights=weights))
        scale = sqrt(variance)
        if scale <= np.finfo(np.float32).eps:
            scale = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, targets: np.ndarray | Tensor) -> np.ndarray:
        values = _target_array(targets, "targets")
        transformed = (values.astype(np.float64) - self.mean) / self.scale
        return transformed.astype(np.float32)

    def inverse_transform(self, targets: np.ndarray | Tensor) -> np.ndarray:
        values = _target_array(targets, "standardized targets", non_negative=False)
        restored = values.astype(np.float64) * self.scale + self.mean
        if not np.isfinite(restored).all():
            raise ValueError("Inverse target scaling produced non-finite values")
        return restored.astype(np.float32)


@dataclass(frozen=True)
class EpochMetrics:
    """Losses for one epoch; RMSE values are expressed in RUL cycles."""

    epoch: int
    train_mse_standardized: float
    validation_mse_standardized: float
    train_rmse: float
    validation_rmse: float


@dataclass
class TrainingResult:
    """Best fitted model and the evidence used by early stopping."""

    model: LSTMRegressor
    target_scaler: TargetScaler
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    stopped_early: bool


@dataclass(frozen=True)
class FixedEpochMetrics:
    """Training losses for one fixed epoch, without validation feedback."""

    epoch: int
    train_mse_standardized: float
    train_rmse: float


@dataclass
class FixedTrainingResult:
    """Model fitted for a frozen number of epochs on all available training engines."""

    model: LSTMRegressor
    target_scaler: TargetScaler
    history: tuple[FixedEpochMetrics, ...]


@dataclass
class LoadedLSTMCheckpoint:
    """A checkpoint restored on CPU for portable inference."""

    model: LSTMRegressor
    target_scaler: TargetScaler
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    expects_observed_lengths: bool


def _sequence_array(sequences: np.ndarray | Tensor, input_size: int | None = None) -> np.ndarray:
    if isinstance(sequences, Tensor):
        values = sequences.detach().cpu().numpy()
    else:
        try:
            values = np.asarray(sequences)
        except (TypeError, ValueError) as exc:
            raise ValueError("sequences must be a numeric three-dimensional array") from exc
    if values.ndim != 3 or not all(size > 0 for size in values.shape):
        raise ValueError("sequences must have shape (samples, timesteps, features)")
    if input_size is not None and values.shape[2] != input_size:
        raise ValueError(f"Expected {input_size} features, got {values.shape[2]}")
    try:
        result = np.ascontiguousarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("sequences must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError("sequences must contain only finite values")
    return result


def _target_array(
    targets: np.ndarray | Tensor,
    name: str,
    *,
    non_negative: bool = True,
) -> np.ndarray:
    if isinstance(targets, Tensor):
        values = targets.detach().cpu().numpy()
    else:
        try:
            values = np.asarray(targets)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        result = np.ascontiguousarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if non_negative and (result < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    return result


def _weight_array(
    sample_weights: np.ndarray | Tensor | None,
    sample_count: int,
    name: str,
) -> np.ndarray:
    if sample_weights is None:
        return np.ones(sample_count, dtype=np.float32)
    values = _target_array(sample_weights, name, non_negative=False)
    if len(values) != sample_count:
        raise ValueError(f"{name} must contain one value per sample")
    if (values <= 0.0).any():
        raise ValueError(f"{name} must contain strictly positive values")
    return values


def _length_array(
    observed_lengths: np.ndarray | Tensor | None,
    sample_count: int,
    sequence_length: int,
    name: str,
) -> np.ndarray:
    if observed_lengths is None:
        return np.full(sample_count, sequence_length, dtype=np.int64)
    if isinstance(observed_lengths, Tensor):
        values = observed_lengths.detach().cpu().numpy()
    else:
        try:
            values = np.asarray(observed_lengths)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a one-dimensional integer array") from exc
    if values.ndim != 1 or len(values) != sample_count:
        raise ValueError(f"{name} must contain one value per sample")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer values") from exc
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"{name} must contain integer values")
    if (numeric < 1).any() or (numeric > sequence_length).any():
        raise ValueError(f"{name} values must be between 1 and {sequence_length}")
    return np.ascontiguousarray(numeric, dtype=np.int64)


class RULSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Validated float32 sequences, standardized targets, and sample weights."""

    def __init__(
        self,
        sequences: np.ndarray | Tensor,
        targets: np.ndarray | Tensor,
        *,
        input_size: int,
        target_scaler: TargetScaler,
        sample_weights: np.ndarray | Tensor | None = None,
        observed_lengths: np.ndarray | Tensor | None = None,
    ) -> None:
        inputs = _sequence_array(sequences, input_size)
        raw_targets = _target_array(targets, "targets")
        if len(inputs) != len(raw_targets):
            raise ValueError("sequences and targets must contain the same number of samples")
        weights = _weight_array(sample_weights, len(inputs), "sample_weights")
        lengths = _length_array(observed_lengths, len(inputs), inputs.shape[1], "observed_lengths")
        self.sequences = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(target_scaler.transform(raw_targets))
        self.sample_weights = torch.from_numpy(weights)
        self.observed_lengths = torch.from_numpy(lengths)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.sequences[index], self.targets[index], self.sample_weights[index]


class LSTMRegressor(nn.Module):
    """Unidirectional LSTM followed by a scalar linear regression head."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.regression_head = nn.Linear(config.hidden_size, 1)
        self.expects_observed_lengths_ = False

    def forward(self, sequences: Tensor, observed_lengths: Tensor | None = None) -> Tensor:
        if sequences.ndim != 3:
            raise ValueError("LSTM input must have shape (batch, timesteps, features)")
        if sequences.shape[2] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} input features, got {sequences.shape[2]}"
            )
        if observed_lengths is None:
            _, (hidden_state, _) = self.lstm(sequences)
        else:
            lengths = observed_lengths.to(device=sequences.device)
            if lengths.ndim != 1 or len(lengths) != len(sequences):
                raise ValueError("observed_lengths must contain one value per sequence")
            if lengths.dtype == torch.bool:
                raise ValueError("observed_lengths must contain integers")
            if torch.is_floating_point(lengths):
                if not torch.isfinite(lengths).all() or not torch.equal(lengths, lengths.floor()):
                    raise ValueError("observed_lengths must contain integers")
            lengths = lengths.to(dtype=torch.long)
            timestep_count = sequences.shape[1]
            if bool(((lengths < 1) | (lengths > timestep_count)).any()):
                raise ValueError(f"observed_lengths values must be between 1 and {timestep_count}")

            # C-MAPSS short windows are padded on the left. Move each real suffix
            # to the start before packing; values to its right are ignored.
            positions = torch.arange(timestep_count, device=sequences.device).unsqueeze(0)
            source_positions = positions + (timestep_count - lengths).unsqueeze(1)
            source_positions = source_positions.clamp(max=timestep_count - 1)
            gather_index = source_positions.unsqueeze(2).expand(-1, -1, sequences.shape[2])
            right_padded = sequences.gather(1, gather_index)
            packed = pack_padded_sequence(
                right_padded,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden_state, _) = self.lstm(packed)
        final_state = self.dropout(hidden_state[-1])
        return self.regression_head(final_state).squeeze(-1)


def set_deterministic(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid PyTorch device: {value}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _run_epoch(
    model: LSTMRegressor,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
    target_scaler: TargetScaler,
    use_observed_lengths: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    standardized_squared_error = 0.0
    clipped_rul_squared_error = 0.0
    weight_sum = 0.0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for sequences, targets, weights, observed_lengths in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            observed_lengths = observed_lengths.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(sequences, observed_lengths if use_observed_lengths else None)
            squared_errors = (predictions - targets).square()
            batch_weight = weights.sum()
            loss = (squared_errors * weights).sum() / batch_weight
            if not torch.isfinite(loss):
                raise RuntimeError("LSTM training produced a non-finite loss")
            if optimizer is not None:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            predicted_rul = (predictions * target_scaler.scale + target_scaler.mean).clamp(min=0.0)
            target_rul = targets * target_scaler.scale + target_scaler.mean
            rul_squared_errors = (predicted_rul - target_rul).square()
            standardized_squared_error += float((squared_errors * weights).sum().detach().cpu())
            clipped_rul_squared_error += float((rul_squared_errors * weights).sum().detach().cpu())
            weight_sum += float(batch_weight.detach().cpu())
    standardized_mse = standardized_squared_error / weight_sum
    clipped_rul_rmse = sqrt(clipped_rul_squared_error / weight_sum)
    return standardized_mse, clipped_rul_rmse


def fit_lstm(
    train_sequences: np.ndarray | Tensor,
    train_targets: np.ndarray | Tensor,
    validation_sequences: np.ndarray | Tensor,
    validation_targets: np.ndarray | Tensor,
    config: LSTMConfig,
    *,
    train_sample_weights: np.ndarray | Tensor | None = None,
    validation_sample_weights: np.ndarray | Tensor | None = None,
    train_observed_lengths: np.ndarray | Tensor | None = None,
    validation_observed_lengths: np.ndarray | Tensor | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Fit a fresh model and restore the epoch with best validation RMSE.

    Sample weights affect the optimization and reported losses. They are useful
    when every engine should contribute the same total weight despite producing
    a different number of learning windows. When the input windows contain
    repeated left padding, provide both observed-length arrays so only their
    real suffixes enter the recurrent model. Early stopping uses weighted RMSE
    in RUL cycles after clipping predictions at zero; ``min_delta`` is in cycles.
    """
    train_x = _sequence_array(train_sequences, config.input_size)
    train_y = _target_array(train_targets, "train_targets")
    validation_x = _sequence_array(validation_sequences, config.input_size)
    validation_y = _target_array(validation_targets, "validation_targets")
    if len(train_x) != len(train_y):
        raise ValueError("train_sequences and train_targets must have equal length")
    if len(validation_x) != len(validation_y):
        raise ValueError("validation_sequences and validation_targets must have equal length")
    uses_observed_lengths = train_observed_lengths is not None
    if uses_observed_lengths != (validation_observed_lengths is not None):
        raise ValueError(
            "train_observed_lengths and validation_observed_lengths must be provided together"
        )

    train_weights = _weight_array(train_sample_weights, len(train_y), "train_sample_weights")
    validation_weights = _weight_array(
        validation_sample_weights, len(validation_y), "validation_sample_weights"
    )
    train_lengths = _length_array(
        train_observed_lengths,
        len(train_y),
        train_x.shape[1],
        "train_observed_lengths",
    )
    validation_lengths = _length_array(
        validation_observed_lengths,
        len(validation_y),
        validation_x.shape[1],
        "validation_observed_lengths",
    )
    target_scaler = TargetScaler.fit(train_y, train_weights)
    train_dataset = RULSequenceDataset(
        train_x,
        train_y,
        input_size=config.input_size,
        target_scaler=target_scaler,
        sample_weights=train_weights,
        observed_lengths=train_lengths,
    )
    validation_dataset = RULSequenceDataset(
        validation_x,
        validation_y,
        input_size=config.input_size,
        target_scaler=target_scaler,
        sample_weights=validation_weights,
        observed_lengths=validation_lengths,
    )

    set_deterministic(config.seed)
    torch_device = _device(device)
    model = LSTMRegressor(config).to(torch_device)
    model.expects_observed_lengths_ = uses_observed_lengths
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(
            train_dataset.sequences,
            train_dataset.targets,
            train_dataset.sample_weights,
            train_dataset.observed_lengths,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(
            validation_dataset.sequences,
            validation_dataset.targets,
            validation_dataset.sample_weights,
            validation_dataset.observed_lengths,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_rmse = _run_epoch(
            model,
            train_loader,
            torch_device,
            optimizer,
            config.gradient_clip_norm,
            target_scaler,
            uses_observed_lengths,
        )
        validation_loss, validation_rmse = _run_epoch(
            model,
            validation_loader,
            torch_device,
            optimizer=None,
            gradient_clip_norm=None,
            target_scaler=target_scaler,
            use_observed_lengths=uses_observed_lengths,
        )
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_mse_standardized=train_loss,
                validation_mse_standardized=validation_loss,
                train_rmse=train_rmse,
                validation_rmse=validation_rmse,
            )
        )
        if validation_rmse < best_validation_rmse - config.min_delta:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:  # pragma: no cover - every finite first epoch improves over infinity
        raise RuntimeError("LSTM training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    return TrainingResult(
        model=model,
        target_scaler=target_scaler,
        history=tuple(history),
        best_epoch=best_epoch,
        stopped_early=len(history) < config.max_epochs,
    )


def fit_lstm_fixed_epochs(
    train_sequences: np.ndarray | Tensor,
    train_targets: np.ndarray | Tensor,
    config: LSTMConfig,
    *,
    train_sample_weights: np.ndarray | Tensor | None = None,
    train_observed_lengths: np.ndarray | Tensor | None = None,
    device: str | torch.device = "cpu",
) -> FixedTrainingResult:
    """Fit for exactly ``config.max_epochs`` without consulting validation or test data."""
    train_x = _sequence_array(train_sequences, config.input_size)
    train_y = _target_array(train_targets, "train_targets")
    if len(train_x) != len(train_y):
        raise ValueError("train_sequences and train_targets must have equal length")
    weights = _weight_array(train_sample_weights, len(train_y), "train_sample_weights")
    lengths = _length_array(
        train_observed_lengths,
        len(train_y),
        train_x.shape[1],
        "train_observed_lengths",
    )
    use_observed_lengths = train_observed_lengths is not None
    target_scaler = TargetScaler.fit(train_y, weights)
    dataset = RULSequenceDataset(
        train_x,
        train_y,
        input_size=config.input_size,
        target_scaler=target_scaler,
        sample_weights=weights,
        observed_lengths=lengths,
    )

    set_deterministic(config.seed)
    torch_device = _device(device)
    model = LSTMRegressor(config).to(torch_device)
    model.expects_observed_lengths_ = use_observed_lengths
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(
            dataset.sequences,
            dataset.targets,
            dataset.sample_weights,
            dataset.observed_lengths,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    history: list[FixedEpochMetrics] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_rmse = _run_epoch(
            model,
            loader,
            torch_device,
            optimizer,
            config.gradient_clip_norm,
            target_scaler,
            use_observed_lengths,
        )
        history.append(
            FixedEpochMetrics(
                epoch=epoch,
                train_mse_standardized=train_loss,
                train_rmse=train_rmse,
            )
        )

    model.to("cpu")
    model.eval()
    return FixedTrainingResult(
        model=model,
        target_scaler=target_scaler,
        history=tuple(history),
    )


def predict_lstm(
    model: LSTMRegressor,
    sequences: np.ndarray | Tensor,
    target_scaler: TargetScaler,
    *,
    observed_lengths: np.ndarray | Tensor | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    minimum_rul: float | None = 0.0,
) -> np.ndarray:
    """Predict RUL in cycles, ignoring repeated left padding when lengths exist."""
    size = _positive_integer(batch_size, "batch_size")
    inputs = _sequence_array(sequences, model.config.input_size)
    expects_lengths = bool(getattr(model, "expects_observed_lengths_", False))
    if expects_lengths and observed_lengths is None:
        raise ValueError("This LSTM checkpoint expects observed_lengths for left-padded windows")
    lengths = _length_array(observed_lengths, len(inputs), inputs.shape[1], "observed_lengths")
    use_observed_lengths = observed_lengths is not None
    torch_device = _device(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(lengths)),
        batch_size=size,
        shuffle=False,
        num_workers=0,
    )
    model.to(torch_device)
    model.eval()
    standardized_predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch, batch_lengths in loader:
            predictions = model(
                batch.to(torch_device),
                batch_lengths.to(torch_device) if use_observed_lengths else None,
            )
            standardized_predictions.append(predictions.detach().cpu().numpy())
    model.to("cpu")
    result = target_scaler.inverse_transform(np.concatenate(standardized_predictions))
    if minimum_rul is not None:
        minimum = _finite_float(minimum_rul, "minimum_rul")
        result = np.maximum(result, minimum).astype(np.float32, copy=False)
    return result


def _validated_feature_names(feature_names: Sequence[str], input_size: int) -> tuple[str, ...]:
    if isinstance(feature_names, str):
        raise ValueError("feature_names must be a sequence, not a string")
    names = tuple(feature_names)
    if len(names) != input_size or any(not name for name in names):
        raise ValueError(f"feature_names must contain exactly {input_size} non-empty strings")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("feature_names must contain strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names


def save_lstm_checkpoint(
    path: str | Path,
    model: LSTMRegressor,
    target_scaler: TargetScaler,
    feature_names: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a self-contained checkpoint whose tensors all live on CPU."""
    names = _validated_feature_names(feature_names, model.config.input_size)
    metadata_dict = dict(metadata or {})
    try:
        json.dumps(metadata_dict, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable with finite values") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "config": asdict(model.config),
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "target_scaler": asdict(target_scaler),
        "feature_names": list(names),
        "metadata": metadata_dict,
        "expects_observed_lengths": bool(getattr(model, "expects_observed_lengths_", False)),
    }
    torch.save(payload, destination)
    return destination


def load_lstm_checkpoint(path: str | Path) -> LoadedLSTMCheckpoint:
    """Load a trusted project checkpoint on CPU using PyTorch's restricted loader."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported or invalid LSTM checkpoint")
    required = {"config", "state_dict", "target_scaler", "feature_names", "metadata"}
    if not required.issubset(payload):
        raise ValueError("Incomplete LSTM checkpoint")
    try:
        config = LSTMConfig(**payload["config"])
        target_scaler = TargetScaler(**payload["target_scaler"])
        feature_names = _validated_feature_names(payload["feature_names"], config.input_size)
        metadata = dict(payload["metadata"])
        expects_observed_lengths = payload.get("expects_observed_lengths", False)
        if not isinstance(expects_observed_lengths, bool):
            raise ValueError("expects_observed_lengths must be a boolean")
        model = LSTMRegressor(config)
        model.load_state_dict(payload["state_dict"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid LSTM checkpoint contents") from exc
    model.to("cpu")
    model.eval()
    model.expects_observed_lengths_ = expects_observed_lengths
    return LoadedLSTMCheckpoint(
        model=model,
        target_scaler=target_scaler,
        feature_names=feature_names,
        metadata=metadata,
        expects_observed_lengths=expects_observed_lengths,
    )
