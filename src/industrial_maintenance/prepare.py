"""Rebuild milestone 1 from train_FD001.txt without opening official test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from industrial_maintenance.data import (
    SENSOR_COLUMNS,
    SensorPreprocessor,
    WindowBatch,
    add_rul,
    load_cmapss,
    make_windows,
    split_by_engine,
    truncate_validation,
)
from industrial_maintenance.plots import save_exploration_figures


@dataclass(frozen=True)
class PreparationConfig:
    """Resolved preparation settings; paths in TOML are project-relative."""

    project_root: Path
    source: Path
    output_dir: Path
    report_dir: Path
    validation_fraction: float
    split_seed: int
    validation_seed: int
    min_observed_cycles: int
    window_length: int
    parameters: dict[str, Any]


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def read_config(path: Path | str) -> PreparationConfig:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        params = tomllib.load(handle)
    root = path.parent.parent
    try:
        if params["data"]["subset"] != "FD001":
            raise ValueError("This first-stage configuration supports FD001 only.")
        raw_dir = (root / params["data"]["raw_dir"]).resolve()
        output_dir = (root / params["output"]["directory"]).resolve()
        report_dir = (root / params["output"]["report_dir"]).resolve()
        fraction = params["split"]["validation_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 < fraction < 1
        ):
            raise ValueError("validation_fraction must be strictly between 0 and 1.")
        split_seed = _positive_integer(params["split"]["seed"], "split.seed", 0)
        validation_seed = _positive_integer(params["validation"]["seed"], "validation.seed", 0)
        minimum = _positive_integer(
            params["validation"]["min_observed_cycles"], "validation.min_observed_cycles"
        )
        length = _positive_integer(params["windows"]["length"], "windows.length")
    except KeyError as exc:
        raise ValueError(f"Missing configuration key: {exc}") from exc
    for destination in (output_dir, report_dir):
        if not destination.is_relative_to(root) or destination == root:
            raise ValueError("Output directories must be inside the project, below its root.")
        if destination.is_relative_to(raw_dir) or raw_dir.is_relative_to(destination):
            raise ValueError("Output directories must not contain or overlap the raw-data folder.")
    return PreparationConfig(
        root,
        raw_dir / "train_FD001.txt",
        output_dir,
        report_dir,
        float(fraction),
        split_seed,
        validation_seed,
        minimum,
        length,
        params,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_batch(path: Path, batch: WindowBatch, features: list[str]) -> None:
    if batch.y is None:
        raise ValueError("Preparation outputs require known labels.")
    np.savez_compressed(
        path,
        X=batch.X,
        y=batch.y,
        unit_ids=batch.unit_ids,
        end_cycles=batch.end_cycles,
        observed_lengths=batch.observed_lengths,
        feature_names=np.asarray(features, dtype=str),
    )


def _report_markdown(summary: dict[str, Any]) -> str:
    data = summary["data"]
    preprocessing = summary["preprocessing"]
    windows = summary["windows"]
    dropped = ", ".join(f"`{name}`" for name in preprocessing["dropped_features"])
    return f"""# FD001 — Préparation et exploration

Rapport produit par `python -m industrial_maintenance.prepare --config configs/fd001.toml`.
Il décrit exclusivement la préparation des données. La comparaison ML/DL exécutée ensuite est
documentée séparément dans [models/MODEL_COMPARISON.md](models/MODEL_COMPARISON.md).

## Séparation des moteurs

| Groupe | Moteurs | Observations |
| --- | ---: | ---: |
| Apprentissage | {data["learning_units"]} | {data["learning_rows"]} |
| Validation, historiques complets avant troncature | {data["validation_units"]} | {data["validation_full_rows"]} |
| Validation, historiques observables après troncature | {data["validation_units"]} | {data["validation_observed_rows"]} |

Les identifiants des deux groupes sont disjoints et enregistrés dans [splits.json](splits.json).
La séparation utilise la graine {summary["parameters"]["split"]["seed"]} avant tout
ajustement du prétraitement. Seul `train_FD001.txt` a été ouvert pour cette préparation.
Les fichiers test et RUL officiels restent réservés au bilan final.

La RUL brute est calculée par moteur à partir de la fin de vie connue. Pour la validation,
un seul instant d'observation est tiré avant la fin de vie de chaque moteur, avec la graine
{summary["parameters"]["validation"]["seed"]}. Les RUL calculées avant troncature sont conservées.
Les instants et les cibles figurent dans [validation_cutoffs.csv](validation_cutoffs.csv).
Ce protocole de validation simulée ne reproduit pas nécessairement la distribution du test officiel.

## Capteurs et prétraitement

Le prétraitement utilise les capteurs ; les réglages et l'âge sont conservés dans les CSV bruts
préparés mais ne sont pas des entrées dans cette première configuration FD001 à régime unique.
L'identifiant du moteur et la RUL ne sont jamais des caractéristiques d'entrée.

{len(preprocessing["feature_names"])} capteurs sont conservés. Les constantes exactes,
détectées sur les seuls moteurs d'apprentissage, sont retirées : {dropped or "aucune"}.
Les capteurs de faible variance restent présents à ce stade. Les statistiques de chaque
capteur figurent dans [learning_sensor_statistics.csv](learning_sensor_statistics.csv).
Ces variances brutes ne sont pas comparables physiquement entre grandeurs d'unités différentes.

Les moyennes et écarts-types sont ajustés sur l'apprentissage avec `StandardScaler` puis
réutilisés pour la validation. L'ordre des capteurs et les paramètres sont enregistrés
dans [preprocessing.json](preprocessing.json) ainsi que dans `preprocessor.joblib`.
Une validation normalisée n'a aucune raison d'avoir une moyenne exactement nulle.

## Fenêtres causales

- Apprentissage : `{windows["learning_shape"]}`, une fenêtre par observation.
- Validation : `{windows["validation_shape"]}`, une dernière fenêtre par moteur tronqué.
- Longueur choisie : {windows["length"]} cycles ; cibles non plafonnées, en cycles.
- Les historiques courts sont complétés à gauche par répétition de leur première observation.
  `observed_lengths` conserve le nombre réel d'observations présentes, distinct du remplissage.

Cette règle est identique pour les deux groupes et sera réutilisée à l'inférence. Elle ne crée
pas de nouveaux cycles réels. Les futures caractéristiques Random Forest devront être construites
sur la même fenêtre observable, en tenant compte de cette convention et des longueurs réelles.

## Exploration du groupe d'apprentissage

![Longueurs des trajectoires d'apprentissage](figures/learning_lifetimes.png)

![Exemples de mesures et cibles RUL](figures/learning_trajectories.png)

Les courbes de RUL sont des cibles reconstruites, pas des prédictions. Les mesures peuvent
fluctuer ; un capteur n'est pas tenu de suivre une évolution monotone avec la dégradation.

## Reproductibilité

Le [résumé structuré](preparation_summary.json) contient la configuration, le hash du fichier
source, les empreintes du code de préparation, les versions des bibliothèques et les empreintes
des artefacts. Le [snapshot Windows Python 3.12](../../requirements-lock-win-py312.txt) fige
les dépendances Python de cette exécution. Les identifiants, les coupures et les valeurs des
tableaux sont déterministes avec la configuration et les versions enregistrées. Le manifeste
JSON du prétraitement est lisible indépendamment de joblib.

Cette étape prépare la comparaison des modèles du jalon 2. Aucune performance de RUL
sur validation ou test n'est revendiquée dans ce rapport.
"""


def prepare(config_path: Path | str = "configs/fd001.toml") -> dict[str, Any]:
    """Prepare learning and validation artifacts using only the official training file."""
    config = read_config(config_path)
    if not config.source.is_file():
        raise FileNotFoundError(f"Place train_FD001.txt at: {config.source}")
    source_hash = _sha256(config.source)
    raw = load_cmapss(config.source)
    learning_raw, validation_raw = split_by_engine(
        raw, validation_fraction=config.validation_fraction, seed=config.split_seed
    )
    learning = add_rul(learning_raw)
    validation_full = add_rul(validation_raw)
    observed, cutoffs = truncate_validation(
        validation_full,
        seed=config.validation_seed,
        min_observed_cycles=config.min_observed_cycles,
    )
    preprocessor = SensorPreprocessor().fit(learning)
    learning_scaled = preprocessor.transform(learning)
    validation_scaled = preprocessor.transform(observed)
    features = list(preprocessor.feature_names_)
    learning_batch = make_windows(learning_scaled, features, window_length=config.window_length)
    validation_batch = make_windows(
        validation_scaled, features, window_length=config.window_length, last_only=True
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    learning.to_csv(config.output_dir / "learning.csv", index=False)
    observed.to_csv(config.output_dir / "validation_observed.csv", index=False)
    cutoffs.to_csv(config.report_dir / "validation_cutoffs.csv", index=False)
    _save_batch(config.output_dir / "learning_windows.npz", learning_batch, features)
    _save_batch(config.output_dir / "validation_windows.npz", validation_batch, features)
    joblib.dump(preprocessor, config.output_dir / "preprocessor.joblib")

    splits = {
        "seed": config.split_seed,
        "validation_fraction": config.validation_fraction,
        "learning_unit_ids": sorted(int(x) for x in learning["unit_id"].unique()),
        "validation_unit_ids": sorted(int(x) for x in validation_full["unit_id"].unique()),
    }
    _write_json(config.report_dir / "splits.json", splits)
    preprocessing = {
        "feature_names": features,
        "dropped_features": list(preprocessor.dropped_features_),
        "selection": "Remove exact constants on learning engines only",
        "normalization": "StandardScaler, fitted on learning engines only (ddof=0)",
        "mean": preprocessor.scaler_.mean_.tolist(),
        "scale": preprocessor.scaler_.scale_.tolist(),
        "variance": preprocessor.scaler_.var_.tolist(),
        "fit_rows": len(learning),
        "padding": "Left-repeat the first observed measurement; record true observed length",
    }
    _write_json(config.report_dir / "preprocessing.json", preprocessing)

    statistics = learning.loc[:, list(SENSOR_COLUMNS)].agg(["min", "max", "mean", "std"]).T
    statistics["unique_values"] = learning.loc[:, list(SENSOR_COLUMNS)].nunique()
    statistics["retained"] = statistics.index.isin(features)
    statistics.index.name = "feature"
    statistics.to_csv(config.report_dir / "learning_sensor_statistics.csv")
    figure_paths = save_exploration_figures(learning, config.report_dir / "figures")

    module_dir = Path(__file__).parent
    implementation_files = [module_dir / name for name in ["data.py", "plots.py", "prepare.py"]]

    summary = {
        "schema_version": 1,
        "stage": "FD001 preparation; no predictive model fitted",
        "source": {"file": config.source.name, "sha256": source_hash},
        "official_test_read": False,
        "parameters": config.parameters,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "packages": {
                name: version(name)
                for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]
            },
        },
        "implementation": {
            "package_version": version("industrial-maintenance"),
            "files": [
                {
                    "path": f"src/industrial_maintenance/{path.name}",
                    "sha256": _sha256(path),
                }
                for path in implementation_files
            ],
        },
        "data": {
            "source_rows": len(raw),
            "learning_units": learning["unit_id"].nunique(),
            "validation_units": validation_full["unit_id"].nunique(),
            "learning_rows": len(learning),
            "validation_full_rows": len(validation_full),
            "validation_observed_rows": len(observed),
            "learning_lifetime_range": [
                int(learning.groupby("unit_id")["cycle"].max().min()),
                int(learning.groupby("unit_id")["cycle"].max().max()),
            ],
        },
        "preprocessing": preprocessing,
        "windows": {
            "length": config.window_length,
            "learning_shape": list(learning_batch.X.shape),
            "validation_shape": list(validation_batch.X.shape),
            "learning_padded_samples": int(
                (learning_batch.observed_lengths < config.window_length).sum()
            ),
            "validation_padded_samples": int(
                (validation_batch.observed_lengths < config.window_length).sum()
            ),
        },
        "artifacts": [],
    }
    (config.report_dir / "PREPARATION_FD001.md").write_text(
        _report_markdown(summary), encoding="utf-8"
    )
    artifact_paths = (
        [
            config.output_dir / name
            for name in [
                "learning.csv",
                "validation_observed.csv",
                "learning_windows.npz",
                "validation_windows.npz",
                "preprocessor.joblib",
            ]
        ]
        + [
            config.report_dir / name
            for name in [
                "splits.json",
                "preprocessing.json",
                "validation_cutoffs.csv",
                "learning_sensor_statistics.csv",
                "PREPARATION_FD001.md",
            ]
        ]
        + figure_paths
    )
    summary["artifacts"] = [
        {"path": path.relative_to(config.project_root).as_posix(), "sha256": _sha256(path)}
        for path in artifact_paths
    ]
    _write_json(config.report_dir / "preparation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fd001.toml"))
    args = parser.parse_args()
    try:
        summary = prepare(args.config)
    except (ValueError, OSError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"data": summary["data"], "windows": summary["windows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
