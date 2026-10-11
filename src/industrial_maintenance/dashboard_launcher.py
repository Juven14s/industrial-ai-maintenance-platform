"""Command-line launcher for the Streamlit maintenance dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("MAINTENANCE_API_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--demo-data", default=os.environ.get("MAINTENANCE_DEMO_DATA"))
    parser.add_argument("--headless", action="store_true")
    arguments = parser.parse_args()

    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    os.environ["MAINTENANCE_API_URL"] = arguments.api_url
    if arguments.demo_data:
        os.environ["MAINTENANCE_DEMO_DATA"] = str(Path(arguments.demo_data).resolve())

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("dashboard_app.py")
    streamlit_arguments = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        arguments.host,
        "--server.port",
        str(arguments.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    if arguments.headless:
        streamlit_arguments.extend(["--server.headless", "true"])
    sys.argv = streamlit_arguments
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
