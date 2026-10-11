FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install ".[api,ui,rag]"

COPY configs/api.toml configs/rag.toml configs/rag_corpus.toml configs/rag_query_terms.toml configs/maintenance_planning.toml ./configs/
COPY docs ./docs
COPY ["CMAPSSData/readme.txt", "CMAPSSData/Damage Propagation Modeling.pdf", "./CMAPSSData/"]
RUN python -m industrial_maintenance.rag --config configs/rag.toml prepare && \
    python -m industrial_maintenance.rag --config configs/rag.toml index && \
    python -m industrial_maintenance.rag_generation --config configs/rag.toml download-model

COPY examples ./examples
COPY artifacts/models/FD001/final ./artifacts/models/FD001/final
COPY artifacts/models/FD001/safety_v2 ./artifacts/models/FD001/safety_v2
COPY reports/fd001/safety_v2/calibration_summary.json ./reports/fd001/safety_v2/calibration_summary.json

EXPOSE 8000 8501

CMD ["serve-maintenance-api", "--config", "configs/api.toml", "--host", "0.0.0.0", "--port", "8000"]
