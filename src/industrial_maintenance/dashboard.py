"""Pure dashboard helpers and a small HTTP client for the maintenance API."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd

REQUIRED_SENSORS = (
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)
MINIMUM_END_CYCLE = 20
WINDOW_LENGTH = 30
MAXIMUM_OBSERVATIONS = 1000
_INCIDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")


class DashboardInputError(ValueError):
    """Raised when a CSV cannot form a valid inference history."""


class DashboardServiceError(RuntimeError):
    """Raised when the dashboard cannot obtain a valid API response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize one or more complete engine histories."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DashboardInputError("Le fichier ne contient aucune observation.")

    required = ("unit_id", "cycle", *REQUIRED_SENSORS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DashboardInputError("Colonnes requises absentes : " + ", ".join(missing))

    canonical = frame.copy()
    for column in required:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")

    invalid_columns = [
        column
        for column in required
        if not np.isfinite(canonical[column].to_numpy(dtype=np.float64)).all()
    ]
    if invalid_columns:
        raise DashboardInputError(
            "Valeurs non numériques ou non finies dans : " + ", ".join(invalid_columns)
        )

    identifiers = canonical.loc[:, ["unit_id", "cycle"]].to_numpy(dtype=np.float64)
    if (identifiers <= 0).any() or (identifiers != np.floor(identifiers)).any():
        raise DashboardInputError("unit_id et cycle doivent être des entiers strictement positifs.")

    canonical["unit_id"] = canonical["unit_id"].astype(np.int64)
    canonical["cycle"] = canonical["cycle"].astype(np.int64)
    for sensor in REQUIRED_SENSORS:
        canonical[sensor] = canonical[sensor].astype(np.float64)

    canonical = canonical.sort_values(["unit_id", "cycle"], kind="stable").reset_index(drop=True)
    duplicates = canonical.duplicated(["unit_id", "cycle"])
    if duplicates.any():
        raise DashboardInputError("Un moteur contient des cycles dupliqués.")

    for unit_id, group in canonical.groupby("unit_id", sort=False):
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        if len(cycles) > 1 and np.any(np.diff(cycles) != 1):
            raise DashboardInputError(
                f"Les cycles du moteur {int(unit_id)} ne sont pas consécutifs."
            )
    return canonical


def prediction_cutoffs(frame: pd.DataFrame, unit_id: int) -> list[int]:
    """Return observed cycles at which the API contract permits a prediction."""

    canonical = validate_source_frame(frame)
    engine = canonical.loc[canonical["unit_id"] == int(unit_id)]
    if engine.empty:
        raise DashboardInputError(f"Le moteur {unit_id} est absent du fichier.")

    start_cycle = int(engine["cycle"].iloc[0])
    cutoffs: list[int] = []
    for count, cycle in enumerate(engine["cycle"].tolist(), start=1):
        if cycle < MINIMUM_END_CYCLE:
            continue
        if count >= WINDOW_LENGTH or start_cycle == 1:
            cutoffs.append(int(cycle))
    return cutoffs


def prepare_engine_history(
    frame: pd.DataFrame,
    unit_id: int,
    end_cycle: int,
) -> pd.DataFrame:
    """Select one engine history up to an observed cycle and enforce the API limits."""

    canonical = validate_source_frame(frame)
    history = canonical.loc[
        (canonical["unit_id"] == int(unit_id)) & (canonical["cycle"] <= int(end_cycle))
    ].reset_index(drop=True)
    if history.empty:
        raise DashboardInputError(
            f"Aucune observation disponible pour le moteur {unit_id} avant le cycle {end_cycle}."
        )
    if int(history["cycle"].iloc[-1]) != int(end_cycle):
        raise DashboardInputError(f"Le cycle {end_cycle} n'existe pas pour le moteur {unit_id}.")
    if int(history["cycle"].iloc[-1]) < MINIMUM_END_CYCLE:
        raise DashboardInputError(
            f"La prédiction est disponible à partir du cycle {MINIMUM_END_CYCLE}."
        )
    if len(history) < WINDOW_LENGTH and int(history["cycle"].iloc[0]) != 1:
        raise DashboardInputError(
            f"Un historique de moins de {WINDOW_LENGTH} cycles doit commencer au cycle 1."
        )
    if len(history) > MAXIMUM_OBSERVATIONS:
        raise DashboardInputError(
            f"L'API accepte au maximum {MAXIMUM_OBSERVATIONS} observations par requête."
        )
    return history


def build_prediction_payload(history: pd.DataFrame) -> dict[str, Any]:
    """Build the closed API request schema without forwarding targets or settings."""

    canonical = validate_source_frame(history)
    units = canonical["unit_id"].unique()
    if len(units) != 1:
        raise DashboardInputError("Une prédiction doit concerner exactement un moteur.")

    unit_id = int(units[0])
    selected = prepare_engine_history(
        canonical,
        unit_id=unit_id,
        end_cycle=int(canonical["cycle"].iloc[-1]),
    )
    observations: list[dict[str, int | float]] = []
    for row in selected.itertuples(index=False):
        observation: dict[str, int | float] = {"cycle": int(row.cycle)}
        for sensor in REQUIRED_SENSORS:
            observation[sensor] = float(getattr(row, sensor))
        observations.append(observation)
    return {"unit_id": unit_id, "observations": observations}


def build_agent_history_payload(history: pd.DataFrame) -> dict[str, Any]:
    """Translate a dashboard history into the agent's nested sensor contract.

    The regular prediction endpoint uses one flat mapping per cycle whereas the
    supervised agent deliberately groups measurements under ``sensors``.  Both
    representations are derived from the same validated history so the UI cannot
    silently send a different trajectory to the two workflows.
    """

    prediction_payload = build_prediction_payload(history)
    observations = []
    for observation in prediction_payload["observations"]:
        observations.append(
            {
                "cycle": int(observation["cycle"]),
                "sensors": {sensor: float(observation[sensor]) for sensor in REQUIRED_SENSORS},
            }
        )
    return {
        "unit_id": int(prediction_payload["unit_id"]),
        "observations": observations,
    }


def normalize_incident_id(value: str) -> str:
    """Validate the portable identifier used in both URLs and SQLite threads."""

    normalized = value.strip()
    if _INCIDENT_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "L'identifiant doit commencer par une lettre ou un chiffre et contenir "
            "uniquement lettres, chiffres, point, tiret, deux-points ou underscore "
            "(100 caractères maximum)."
        )
    return normalized


def find_default_demo_data(project_root: str | Path | None = None) -> Path | None:
    """Find the full local validation data first, then the bundled demo trajectory."""

    roots: list[Path] = []
    if project_root is not None:
        roots.append(Path(project_root).resolve())
    roots.extend([Path.cwd().resolve(), Path(__file__).resolve().parents[2]])

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates = (
            root / "data" / "processed" / "FD001" / "validation_observed.csv",
            root / "examples" / "fd001_demo_engine_3.csv",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _error_message(body: bytes, fallback: str) -> str:
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    detail = document.get("detail") if isinstance(document, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, str):
        return detail
    return fallback


class MaintenanceApiClient:
    """Minimal standard-library client used by the Streamlit application."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        normalized = base_url.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("L'adresse de l'API doit être une URL HTTP valide.")
        if timeout <= 0:
            raise ValueError("Le délai d'attente doit être strictement positif.")
        self.base_url = normalized
        self.timeout = float(timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                status_code = int(response.status)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            fallback = f"L'API a répondu avec le statut HTTP {exc.code}."
            raise DashboardServiceError(
                _error_message(body, fallback), status_code=int(exc.code)
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise DashboardServiceError(
                "API indisponible. Vérifie qu'elle est lancée et que son adresse est correcte."
            ) from exc

        if not 200 <= status_code < 300:
            raise DashboardServiceError(
                f"L'API a répondu avec le statut HTTP {status_code}.",
                status_code=status_code,
            )
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DashboardServiceError("L'API a renvoyé une réponse JSON invalide.") from exc
        if not isinstance(document, dict):
            raise DashboardServiceError("L'API a renvoyé un format de réponse inattendu.")
        return document

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/models")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict/rul", payload)

    def ask_documents(self, question: str) -> dict[str, Any]:
        normalized = question.strip()
        if len(normalized) < 3:
            raise ValueError("La question doit contenir au moins trois caractères.")
        return self._request("POST", "/documents/ask", {"question": normalized})

    def plan_maintenance(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/maintenance/plan", payload)

    @staticmethod
    def _incident_path(incident_id: str) -> str:
        normalized = normalize_incident_id(incident_id)
        return quote(normalized, safe="")

    def start_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start a supervised maintenance analysis."""

        normalized = dict(payload)
        identifier = normalized.get("incident_id")
        if not isinstance(identifier, str):
            raise ValueError("L'identifiant de l'incident est obligatoire.")
        normalized["incident_id"] = normalize_incident_id(identifier)
        return self._request("POST", "/agent/runs", normalized)

    def get_agent(self, incident_id: str) -> dict[str, Any]:
        """Read the latest persisted snapshot for one incident."""

        identifier = self._incident_path(incident_id)
        return self._request("GET", f"/agent/runs/{identifier}")

    def review_agent(
        self,
        incident_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Approve, edit, or reject a recommendation awaiting human review."""

        identifier = self._incident_path(incident_id)
        return self._request("POST", f"/agent/runs/{identifier}/review", payload)
