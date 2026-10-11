"""Execute the FD001 exploration notebook with this project's Python interpreter."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path("notebooks/01_fd001_exploration.ipynb"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    notebook_path = (project_root / args.notebook).resolve()
    if not notebook_path.is_relative_to(project_root) or not notebook_path.is_file():
        raise SystemExit(f"Notebook introuvable dans le projet : {notebook_path}")

    runtime_root = project_root / ".runtime"
    runtime_settings = {
        "JUPYTER_RUNTIME_DIR": runtime_root / "jupyter",
        "IPYTHONDIR": runtime_root / "ipython",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
    }
    for variable, directory in runtime_settings.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)

    notebook = nbformat.read(notebook_path, as_version=4)
    kernel_manager = KernelManager(kernel_name="python3")
    kernel_manager.kernel_spec.argv[0] = sys.executable
    kernel_manager.session.key = secrets.token_hex(32).encode("ascii")
    client = NotebookClient(
        notebook,
        km=kernel_manager,
        timeout=180,
        extra_arguments=["--Application.log_level=ERROR"],
        resources={"metadata": {"path": str(project_root)}},
    )
    executed = client.execute()
    nbformat.write(executed, notebook_path)

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise SystemExit("Le notebook contient au moins une cellule de code non exécutée.")
    print(
        f"Notebook execute : {notebook_path.relative_to(project_root)} ({len(code_cells)} cellules)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
