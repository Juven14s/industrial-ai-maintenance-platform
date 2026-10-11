"""Streamlit user interface for the Industrial AI Maintenance API."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from industrial_maintenance.dashboard import (
    REQUIRED_SENSORS,
    DashboardInputError,
    DashboardServiceError,
    MaintenanceApiClient,
    build_prediction_payload,
    find_default_demo_data,
    prediction_cutoffs,
    prepare_engine_history,
    validate_source_frame,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_API_URL = "http://127.0.0.1:8000"


@st.cache_data(show_spinner=False)
def _read_demo_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=5, show_spinner=False)
def _read_health(api_url: str) -> dict[str, Any]:
    return MaintenanceApiClient(api_url, timeout=2.0).health()


@st.cache_data(ttl=30, show_spinner=False)
def _read_models(api_url: str) -> dict[str, Any]:
    return MaintenanceApiClient(api_url, timeout=3.0).models()


def _history_key(api_url: str, history: pd.DataFrame) -> str:
    columns = ["unit_id", "cycle", *REQUIRED_SENSORS]
    digest = hashlib.sha256()
    digest.update(api_url.encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(history[columns], index=False).values.tobytes())
    return digest.hexdigest()


def _read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    if int(uploaded_file.size) > MAX_UPLOAD_BYTES:
        raise DashboardInputError("Le fichier dépasse la limite de 5 Mo.")
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DashboardInputError("Le fichier ne peut pas être lu comme un CSV.") from exc


def _render_prediction(result: dict[str, Any]) -> None:
    try:
        conservative = float(result["rul_conservative"])
        central = float(result["rul_estimate"])
        margin = float(result["safety_margin"])
        base = float(result["safety_base_prediction"])
        used = int(result["window_cycles_used"])
        last_cycle = int(result["last_cycle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardServiceError("La réponse de prédiction est incomplète.") from exc

    st.subheader("Résultat de la prédiction")
    decision_column, central_column, window_column = st.columns(3)
    decision_column.metric("RUL de décision", f"{conservative:.1f} cycles")
    central_column.metric("Estimation centrale", f"{central:.1f} cycles")
    window_column.metric("Fenêtre analysée", f"{used} cycles")

    st.info(
        f"**Échéance prudente proposée : {conservative:.1f} cycles d'utilisation "
        "restants.** Cette valeur alimente la future planification de maintenance."
    )
    st.caption(
        "Cette estimation est exprimée en cycles C-MAPSS. Elle ne correspond pas "
        "automatiquement à un nombre de jours et ne constitue pas un diagnostic de panne."
    )

    with st.expander("Comprendre le résultat et voir les détails"):
        st.markdown(
            "- **RUL de décision** : sortie du modèle conservateur V2 après application "
            "de sa marge de sécurité.\n"
            "- **Estimation centrale** : sortie indépendante du LSTM V1.\n"
            "- Ces deux valeurs ne forment pas un intervalle de confiance et leur ordre "
            "n'est pas forcé."
        )
        detail_one, detail_two, detail_three = st.columns(3)
        detail_one.metric("Prédiction V2 avant marge", f"{base:.2f}")
        detail_two.metric("Marge V2", f"{margin:.2f}")
        detail_three.metric("Dernier cycle observé", str(last_cycle))
        st.code(
            f"max({base:.6f} - {margin:.6f}, 0) = {conservative:.6f}",
            language="text",
        )
        st.json(result)


def _render_document_answer(result: dict[str, Any]) -> None:
    try:
        answer = str(result["answer"])
        answerable = bool(result["answerable"])
        citations = result["citations"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardServiceError("La réponse documentaire est incomplète.") from exc
    if not isinstance(citations, list):
        raise DashboardServiceError("Les citations documentaires sont invalides.")

    if answerable:
        st.success(answer)
    else:
        st.info(answer)
    for index, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            raise DashboardServiceError("Une citation documentaire est invalide.")
        label = str(citation.get("citation", f"Source {index}"))
        with st.expander(f"Source {index} — {label}"):
            st.write(str(citation.get("text", "Passage indisponible")))
            source_url = citation.get("source_url")
            if isinstance(source_url, str) and source_url:
                st.markdown(f"[Consulter la source publique]({source_url})")
            limitation = citation.get("limitations")
            if isinstance(limitation, str) and limitation:
                st.caption(f"Limite d'usage : {limitation}")


def _render_plan_result(result: dict[str, Any]) -> None:
    try:
        heuristic = result["heuristic"]
        optimized = result["optimized"]
        savings = result["absolute_savings"]
        relative = result["relative_savings"]
    except (KeyError, TypeError) as exc:
        raise DashboardServiceError("La réponse de planification est incomplète.") from exc

    st.subheader("Comparaison des plannings")
    columns = st.columns(4)
    columns[0].metric("Statut CP-SAT", str(optimized.get("status", "inconnu")))
    columns[1].metric("Coût heuristique", str(heuristic.get("objective_cost", "—")))
    columns[2].metric("Coût optimisé", str(optimized.get("objective_cost", "—")))
    relative_label = "—" if relative is None else f"{100 * float(relative):.1f} %"
    columns[3].metric("Réduction simulée", relative_label)

    if optimized.get("status") in {"infeasible", "unknown"}:
        st.warning(
            "Le solveur n'a pas trouvé de planning complet dans l'horizon avec les "
            "ressources fournies."
        )
    elif savings is not None:
        st.success(f"Réduction de coût simulée : {int(savings)} unités.")

    for label, plan in (("Planning CP-SAT", optimized), ("Heuristique", heuristic)):
        with st.expander(label, expanded=label == "Planning CP-SAT"):
            assignments = plan.get("assignments", [])
            if assignments:
                frame = pd.DataFrame(assignments)
                display_columns = [
                    "unit_id",
                    "start_cycle",
                    "end_cycle",
                    "deadline_cycle",
                    "criticality",
                    "technicians_required",
                    "tardiness_cycles",
                    "cost",
                ]
                st.dataframe(frame.loc[:, display_columns], hide_index=True, width="stretch")
                chart = frame.assign(
                    engine=frame["unit_id"].map(lambda value: f"Moteur {value}")
                )
                st.vega_lite_chart(
                    chart,
                    {
                        "mark": {"type": "bar", "cornerRadius": 3},
                        "encoding": {
                            "y": {"field": "engine", "type": "nominal", "title": None},
                            "x": {
                                "field": "start_cycle",
                                "type": "quantitative",
                                "title": "Cycle de planification",
                            },
                            "x2": {"field": "end_cycle"},
                            "color": {
                                "field": "criticality",
                                "type": "nominal",
                                "title": "Criticité",
                            },
                            "tooltip": [
                                {"field": "unit_id", "title": "Moteur"},
                                {"field": "start_cycle", "title": "Début"},
                                {"field": "end_cycle", "title": "Fin"},
                                {"field": "deadline_cycle", "title": "Échéance"},
                                {"field": "tardiness_cycles", "title": "Retard"},
                            ],
                        },
                    },
                    width="stretch",
                )
            else:
                st.write("Aucune intervention planifiée.")
            if plan.get("unscheduled_unit_ids"):
                st.warning(
                    "Moteurs non planifiés : "
                    + ", ".join(map(str, plan["unscheduled_unit_ids"]))
                )
            if plan.get("deferred_unit_ids"):
                st.info(
                    "Hors de l'horizon actuel : "
                    + ", ".join(map(str, plan["deferred_unit_ids"]))
                )


def render() -> None:
    st.set_page_config(
        page_title="Industrial AI Maintenance",
        page_icon="🛠️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
        [data-testid="stMetric"] {
            background: rgba(28, 131, 225, 0.07);
            border: 1px solid rgba(28, 131, 225, 0.18);
            padding: 1rem;
            border-radius: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Industrial AI Maintenance")
    st.caption(
        "Explore un historique moteur et estime sa durée de vie restante sans manipuler "
        "de JSON ni appliquer manuellement le prétraitement."
    )

    configured_api_url = os.environ.get("MAINTENANCE_API_URL", DEFAULT_API_URL)
    with st.sidebar:
        st.header("Configuration")
        api_url = st.text_input("Adresse de l'API", value=configured_api_url)
        source = st.radio(
            "Source des observations",
            ("Démonstration FD001", "Importer un CSV"),
        )

        try:
            MaintenanceApiClient(api_url)
            health = _read_health(api_url)
            api_ready = health.get("status") == "ok"
        except (ValueError, DashboardServiceError) as exc:
            health = None
            api_ready = False
            st.error(str(exc))
        else:
            st.success("API connectée")
            st.caption(
                f"LSTM : `{health.get('central_model_version', 'inconnu')}`  \n"
                f"V2 : `{health.get('safety_model_version', 'inconnu')}`"
            )

        if not api_ready:
            st.code(
                ".\\.venv\\Scripts\\python.exe -m industrial_maintenance.api "
                "--config configs/api.toml",
                language="powershell",
            )

    frame: pd.DataFrame | None = None
    source_name = ""
    if source == "Démonstration FD001":
        configured_demo = os.environ.get("MAINTENANCE_DEMO_DATA")
        demo_path = (
            Path(configured_demo).resolve()
            if configured_demo
            else find_default_demo_data()
        )
        if demo_path is None or not demo_path.is_file():
            st.error(
                "Aucune trajectoire de démonstration n'est disponible. "
                "Choisis l'import CSV dans le panneau latéral."
            )
            st.stop()
        frame = _read_demo_csv(str(demo_path))
        source_name = demo_path.name
    else:
        uploaded = st.file_uploader(
            "Dépose un fichier CSV",
            type=["csv"],
            help=(
                "Le fichier doit contenir unit_id, cycle et les 15 capteurs attendus. "
                "Les colonnes supplémentaires ne sont pas envoyées à l'API."
            ),
        )
        if uploaded is None:
            st.info("Ajoute un fichier CSV pour commencer l'analyse.")
            st.stop()
        try:
            frame = _read_uploaded_csv(uploaded)
        except DashboardInputError as exc:
            st.error(str(exc))
            st.stop()
        source_name = uploaded.name

    try:
        canonical = validate_source_frame(frame)
    except DashboardInputError as exc:
        st.error(f"Données non valides : {exc}")
        st.stop()

    st.subheader("1. Choisir l'historique à analyser")
    st.caption(f"Source chargée : `{source_name}`")
    unit_options = [int(value) for value in canonical["unit_id"].unique()]
    selection_column, cutoff_column = st.columns(2)
    with selection_column:
        unit_id = st.selectbox(
            "Moteur",
            options=unit_options,
            format_func=lambda value: f"Moteur {value}",
        )
    try:
        cutoffs = prediction_cutoffs(canonical, int(unit_id))
    except DashboardInputError as exc:
        st.error(str(exc))
        st.stop()
    if not cutoffs:
        st.error(
            "Ce moteur ne possède pas encore assez de cycles pour respecter le contrat du modèle."
        )
        st.stop()
    with cutoff_column:
        end_cycle = st.select_slider(
            "Dernier cycle observé",
            options=cutoffs,
            value=cutoffs[-1],
        )

    try:
        history = prepare_engine_history(canonical, int(unit_id), int(end_cycle))
    except DashboardInputError as exc:
        st.error(str(exc))
        st.stop()

    summary_columns = st.columns(4)
    summary_columns[0].metric("Moteur", str(unit_id))
    summary_columns[1].metric("Dernier cycle", str(end_cycle))
    summary_columns[2].metric("Observations disponibles", str(len(history)))
    summary_columns[3].metric("Cycles utilisés par le modèle", str(min(len(history), 30)))

    st.subheader("2. Observer les capteurs")
    chart_controls, chart_area = st.columns([1, 3])
    with chart_controls:
        default_sensors = [sensor for sensor in ("sensor_2", "sensor_11") if sensor in history]
        selected_sensors = st.multiselect(
            "Capteurs affichés",
            options=list(REQUIRED_SENSORS),
            default=default_sensors,
        )
        normalized = st.checkbox(
            "Normaliser pour comparer les tendances",
            value=True,
            help="Cette normalisation sert uniquement au graphique, jamais à la prédiction.",
        )
    with chart_area:
        if selected_sensors:
            chart = history.set_index("cycle")[selected_sensors].astype(float)
            if normalized:
                deviations = chart.std(ddof=0).replace(0.0, np.nan)
                chart = ((chart - chart.mean()) / deviations).fillna(0.0)
            st.line_chart(chart, height=330)
        else:
            st.info("Sélectionne au moins un capteur pour afficher une trajectoire.")

    with st.expander("Aperçu des 30 cycles transmis au modèle"):
        st.dataframe(
            history.loc[:, ["cycle", *REQUIRED_SENSORS]].tail(30),
            hide_index=True,
            width="stretch",
        )

    st.subheader("3. Estimer la durée de vie restante")
    st.write(
        "La prédiction utilise les 30 derniers cycles affichés. L'API applique elle-même "
        "les prétraitements sauvegardés avec les modèles."
    )
    prediction_key = _history_key(api_url, history)
    if st.button(
        "Estimer la RUL",
        type="primary",
        disabled=not api_ready,
        width="stretch",
    ):
        try:
            payload = build_prediction_payload(history)
            with st.spinner("Analyse de la trajectoire en cours..."):
                prediction = MaintenanceApiClient(api_url).predict(payload)
        except (DashboardInputError, DashboardServiceError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["maintenance_prediction"] = {
                "key": prediction_key,
                "value": prediction,
            }

    stored = st.session_state.get("maintenance_prediction")
    if isinstance(stored, dict) and stored.get("key") == prediction_key:
        try:
            _render_prediction(stored["value"])
        except DashboardServiceError as exc:
            st.error(str(exc))

    with st.expander("Informations sur les modèles"):
        if api_ready:
            try:
                st.json(_read_models(api_url))
            except DashboardServiceError as exc:
                st.warning(str(exc))
        else:
            st.write("Les informations seront disponibles lorsque l'API sera connectée.")

    st.divider()
    st.subheader("4. Interroger la documentation technique")
    st.write(
        "Pose une question sur C-MAPSS, la RUL ou la maintenance prédictive. "
        "La réponse est produite localement et affiche les passages utilisés."
    )
    documentation_question = st.text_input(
        "Question",
        value="Quel critère arrête une simulation run-to-failure dans C-MAPSS ?",
    )
    if st.button(
        "Chercher et répondre",
        disabled=not api_ready or len(documentation_question.strip()) < 3,
        width="stretch",
    ):
        try:
            with st.spinner("Recherche et génération en cours sur le CPU local..."):
                document_answer = MaintenanceApiClient(
                    api_url,
                    timeout=120.0,
                ).ask_documents(documentation_question)
        except (DashboardServiceError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["maintenance_document_answer"] = {
                "api_url": api_url,
                "question": documentation_question.strip(),
                "value": document_answer,
            }

    stored_answer = st.session_state.get("maintenance_document_answer")
    if (
        isinstance(stored_answer, dict)
        and stored_answer.get("api_url") == api_url
        and stored_answer.get("question") == documentation_question.strip()
    ):
        try:
            _render_document_answer(stored_answer["value"])
        except DashboardServiceError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("5. Planifier les interventions de la flotte")
    st.write(
        "Modifie cette flotte simulée, puis compare une règle de priorité simple au "
        "planning optimisé. Les coûts et les ressources sont des hypothèses de démonstration."
    )
    current_prediction = None
    if isinstance(stored, dict) and stored.get("key") == prediction_key:
        current_prediction = stored.get("value", {}).get("rul_conservative")
    default_rul = float(current_prediction) if current_prediction is not None else 8.0
    planning_data = pd.DataFrame(
        [
            {
                "unit_id": int(unit_id),
                "conservative_rul": default_rul,
                "maintenance_duration": 2,
                "technicians_required": 2,
                "criticality": "high",
                "preventive_cost": 650,
            },
            {"unit_id": 1002, "conservative_rul": 12.0, "maintenance_duration": 3, "technicians_required": 1, "criticality": "medium", "preventive_cost": 500},
            {"unit_id": 1003, "conservative_rul": 18.0, "maintenance_duration": 4, "technicians_required": 2, "criticality": "high", "preventive_cost": 680},
            {"unit_id": 1004, "conservative_rul": 24.0, "maintenance_duration": 2, "technicians_required": 1, "criticality": "low", "preventive_cost": 430},
        ]
    )
    edited_fleet = st.data_editor(
        planning_data,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "criticality": st.column_config.SelectboxColumn(
                "Criticité",
                options=["low", "medium", "high"],
                required=True,
            )
        },
        key="maintenance_fleet_editor",
    )
    resource_columns = st.columns(3)
    horizon = resource_columns[0].number_input(
        "Horizon (cycles)", min_value=1, max_value=1000, value=30
    )
    technicians = resource_columns[1].number_input(
        "Techniciens disponibles", min_value=1, max_value=100, value=3
    )
    capacity = resource_columns[2].number_input(
        "Capacité de l'atelier", min_value=1, max_value=100, value=2
    )
    if st.button("Comparer les plannings", disabled=not api_ready, width="stretch"):
        try:
            if edited_fleet.empty or edited_fleet.isna().any().any():
                raise DashboardInputError("La flotte doit contenir des lignes complètes.")
            engines = []
            for row in edited_fleet.to_dict(orient="records"):
                engines.append(
                    {
                        "unit_id": int(row["unit_id"]),
                        "conservative_rul": float(row["conservative_rul"]),
                        "maintenance_duration": int(row["maintenance_duration"]),
                        "technicians_required": int(row["technicians_required"]),
                        "criticality": str(row["criticality"]),
                        "preventive_cost": int(row["preventive_cost"]),
                    }
                )
            payload = {
                "scenario_id": "dashboard_fleet",
                "description": "Scénario simulé depuis le dashboard",
                "planning_horizon": int(horizon),
                "technicians": int(technicians),
                "workshop_capacity": int(capacity),
                "engines": engines,
            }
            with st.spinner("Optimisation du planning..."):
                plan_result = MaintenanceApiClient(
                    api_url,
                    timeout=20.0,
                ).plan_maintenance(payload)
        except (DashboardInputError, DashboardServiceError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["maintenance_plan"] = plan_result

    stored_plan = st.session_state.get("maintenance_plan")
    if isinstance(stored_plan, dict):
        try:
            _render_plan_result(stored_plan)
        except DashboardServiceError as exc:
            st.error(str(exc))


render()
