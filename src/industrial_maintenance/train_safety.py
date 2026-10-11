"""Command-line entry point for FD001 safety-oriented model selection."""

from __future__ import annotations

import argparse

from industrial_maintenance.safety import run_safety_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fd001_safety_v2.toml")
    arguments = parser.parse_args()
    summary = run_safety_experiment(arguments.config)
    metrics = summary["selection"]["metrics"]
    print(f"Selected model: {summary['selection']['model']}")
    print(f"OOF asymmetric MAE: {metrics['asymmetric_mae']:.3f}")
    print(f"OOF overestimate rate: {100 * metrics['overestimate_rate']:.1f}%")
    print(f"Safety constraint met: {summary['selection']['safety_constraint_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
