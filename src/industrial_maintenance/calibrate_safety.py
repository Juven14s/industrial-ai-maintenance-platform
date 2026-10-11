"""Command-line entry point for cross-fitted RUL safety calibration."""

from __future__ import annotations

import argparse

from industrial_maintenance.safety_calibration import run_safety_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fd001_safety_calibration.toml")
    arguments = parser.parse_args()
    summary = run_safety_calibration(arguments.config)
    selection = summary["selection"]
    print(f"Selected calibrated candidate: {selection['candidate']}")
    print(f"Final safety margin: {selection['final_safety_margin']:.3f} cycles")
    print(f"OOF overestimate rate: {100 * selection['metrics']['overestimate_rate']:.1f}%")
    print(
        "Worst-fold overestimate rate: "
        f"{100 * selection['metrics']['maximum_fold_overestimate_rate']:.1f}%"
    )
    print(f"Safety constraints met: {selection['constraints_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
