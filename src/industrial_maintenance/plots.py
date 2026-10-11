"""Small, exportable figures built only from the learning engines."""

from pathlib import Path

import pandas as pd


def save_exploration_figures(training: pd.DataFrame, destination: Path) -> list[Path]:
    """Save descriptive figures without opening a GUI or reading test observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    lifetimes = training.groupby("unit_id")["cycle"].max()
    paths = [destination / "learning_lifetimes.png", destination / "learning_trajectories.png"]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.hist(lifetimes, bins=16, color=colors[0], edgecolor="white")
        ax.set(
            xlabel="Cycle de fin de vie observée",
            ylabel="Nombre de moteurs",
            title=f"FD001 — Durées des {len(lifetimes)} trajectoires d’apprentissage",
        )
        ax.yaxis.get_major_locator().set_params(integer=True)
        fig.savefig(paths[0], dpi=160)
        plt.close(fig)

        units = sorted(training["unit_id"].unique())[:3]
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
        for unit, color in zip(units, colors, strict=False):
            trajectory = training.loc[training["unit_id"] == unit]
            for ax, feature in zip(axes, ["sensor_2", "sensor_11", "rul"], strict=True):
                ax.plot(
                    trajectory["cycle"],
                    trajectory[feature],
                    color=color,
                    linewidth=1.1,
                    label=f"Moteur {unit}",
                )
        axes[0].set(ylabel="T24 (°R)", title="FD001 — Trois moteurs du groupe d’apprentissage")
        axes[1].set_ylabel("Ps30 (psia)")
        axes[2].set(ylabel="RUL réelle (cycles)", xlabel="Cycle observé")
        axes[0].legend(loc="upper left", ncol=min(3, len(units)), frameon=False)
        fig.savefig(paths[1], dpi=160)
        plt.close(fig)
    return paths
