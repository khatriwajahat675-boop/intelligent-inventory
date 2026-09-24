"""Chart palette - the validated colorblind-safe set from the dataviz skill
(references/palette.md), reused as-is for every static matplotlib chart in
this project so the comparison report reads as one consistent system."""
from __future__ import annotations

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SURFACE = "#fcfcfb"

MODEL_COLORS = {
    "baseline_majority": INK["muted"],
    "logistic_regression": CATEGORICAL[0],
    "decision_tree": CATEGORICAL[1],
    "random_forest": CATEGORICAL[2],
    "hist_gradient_boosting": CATEGORICAL[6],
    "mlp_neural_net": CATEGORICAL[4],
}


def apply_style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)
    ax.yaxis.grid(True, color=INK["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK["primary"])
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])
