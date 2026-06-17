from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


def setup_matplotlib():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def add_box(ax, xy, width, height, text, color, edge=None, fontsize=10.5):
    import matplotlib.patches as patches

    x, y = xy
    edge = edge or color
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        weight="semibold",
        linespacing=1.18,
    )


def add_arrow(ax, start, end, color="#334155", dashed=False, rad=0.0):
    import matplotlib.patches as patches

    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (4, 3)) if dashed else "solid",
    )
    ax.add_patch(arrow)


def annotate_bar(ax, bars: Iterable, fmt: str = "{:.3f}", dy: float = 0.02) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#334155",
        )


def render_oracle_pipeline(output_dir: Path) -> Path:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("AI2-THOR Oracle Data Generation", y=0.98, fontsize=17, weight="bold")
    ax.text(
        0.5,
        0.91,
        "Each record stores RGB-D observation, camera pose, candidate action gains, and the oracle label.",
        ha="center",
        va="center",
        color="#475569",
        fontsize=11.5,
    )

    blue = "#dbeafe"
    teal = "#ccfbf1"
    amber = "#fef3c7"
    violet = "#ede9fe"
    green = "#dcfce7"
    gray = "#f1f5f9"

    boxes = [
        ((0.04, 0.62), 0.15, 0.13, "Sample\nscene & pose", blue),
        ((0.24, 0.62), 0.15, 0.13, "Render\nRGB-D + pose", teal),
        ((0.44, 0.62), 0.15, 0.13, "Select\ntarget object", amber),
        ((0.64, 0.62), 0.15, 0.13, "Enumerate\ncandidate actions", violet),
        ((0.84, 0.62), 0.12, 0.13, "Step(a)", gray),
        ((0.65, 0.34), 0.15, 0.13, "Count target\nmask pixels", teal),
        ((0.44, 0.34), 0.15, 0.13, "Compute gain\ns(a)=V_after-V_before", green, None, 9.8),
        ((0.24, 0.34), 0.15, 0.13, "Restore\noriginal pose", gray),
        ((0.04, 0.34), 0.15, 0.13, "Write JSONL\nscores + label", green),
    ]
    for box in boxes:
        add_box(ax, *box)

    arrow_color = "#2563eb"
    add_arrow(ax, (0.19, 0.685), (0.24, 0.685), arrow_color)
    add_arrow(ax, (0.39, 0.685), (0.44, 0.685), arrow_color)
    add_arrow(ax, (0.59, 0.685), (0.64, 0.685), arrow_color)
    add_arrow(ax, (0.79, 0.685), (0.84, 0.685), arrow_color)
    add_arrow(ax, (0.90, 0.62), (0.725, 0.47), "#0f766e", rad=-0.18)
    add_arrow(ax, (0.65, 0.405), (0.59, 0.405), "#16a34a")
    add_arrow(ax, (0.44, 0.405), (0.39, 0.405), "#64748b")
    add_arrow(ax, (0.24, 0.405), (0.19, 0.405), "#16a34a")

    add_arrow(ax, (0.315, 0.34), (0.865, 0.62), "#64748b", dashed=True, rad=-0.22)
    ax.text(
        0.62,
        0.18,
        "Temporary action execution is reset after every candidate action,\nso all gains are measured from the same initial state.",
        ha="center",
        va="center",
        color="#475569",
        fontsize=10.5,
    )
    ax.text(
        0.72,
        0.54,
        "A = {MoveAhead, MoveBack, MoveLeft,\nMoveRight, RotateLeft, RotateRight,\nLookUp, LookDown, Stay}",
        ha="center",
        va="center",
        color="#4c1d95",
        fontsize=9.5,
    )

    path = output_dir / "oracle_generation_pipeline.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_stay_threshold(output_dir: Path) -> Path:
    plt = setup_matplotlib()
    import numpy as np

    thresholds = np.array([0, 100, 200, 500])
    positive = np.array([0.685, 0.683, 0.679, 0.611])
    negative = np.array([0.307, 0.304, 0.296, 0.229])
    safe = np.array([0.543, 0.543, 0.545, 0.528])
    stay_count = np.array([11, 38, 98, 781])

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax2 = ax.twinx()
    ax.plot(thresholds, positive, marker="o", linewidth=2.5, color="#16a34a", label="Positive gain rate")
    ax.plot(thresholds, negative, marker="s", linewidth=2.5, color="#dc2626", label="Negative gain rate")
    ax.plot(thresholds, safe, marker="D", linewidth=2.2, color="#7c3aed", label="Safe balanced score")
    bars = ax2.bar(thresholds, stay_count, width=38, color="#93c5fd", alpha=0.35, label="Stay predictions")

    ax.axvline(200, color="#334155", linestyle=(0, (4, 3)), linewidth=1.5)
    ax.text(205, 0.225, "selected\nthreshold=200", color="#334155", fontsize=9.5, va="bottom")

    ax.set_title("Stay Threshold Trade-off")
    ax.set_xlabel("Stay threshold")
    ax.set_ylabel("Rate / score")
    ax2.set_ylabel("Stay predicted count")
    ax.set_ylim(0.18, 0.72)
    ax2.set_ylim(0, 900)
    ax.set_xticks(thresholds)
    ax.grid(axis="y", color="#e2e8f0", linewidth=1.0)

    for x, y in zip(thresholds, positive):
        ax.text(x, y + 0.012, f"{y:.3f}", ha="center", fontsize=8.5, color="#166534")
    for x, y in zip(thresholds, negative):
        ax.text(x, y - 0.028, f"{y:.3f}", ha="center", fontsize=8.5, color="#991b1b")
    for bar, count in zip(bars, stay_count):
        ax2.text(bar.get_x() + bar.get_width() / 2, count + 18, str(count), ha="center", fontsize=8.5, color="#1e3a8a")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    fig.tight_layout()

    path = output_dir / "stay_threshold_tradeoff.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def render_ablation_results(output_dir: Path) -> Path:
    plt = setup_matplotlib()
    import numpy as np

    models = ["Full", "No Depth", "No Pose", "No Qwen", "SigLIP"]
    mean_gain = np.array([1777.62, 1659.26, 1718.64, 776.45, 1479.99])
    positive = np.array([0.6790, 0.6640, 0.6668, 0.5652, 0.6352])
    negative = np.array([0.2960, 0.3120, 0.3124, 0.4284, 0.3256])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    colors = ["#2563eb", "#0f766e", "#d97706", "#64748b", "#7c3aed"]
    x = np.arange(len(models))

    bars0 = axes[0].bar(x, mean_gain, color=colors, alpha=0.9)
    axes[0].set_title("Mean predicted gain")
    axes[0].set_ylabel("Visible-pixel gain")
    axes[0].set_xticks(x, models, rotation=20, ha="right")
    axes[0].grid(axis="y", color="#e2e8f0")
    annotate_bar(axes[0], bars0, "{:.0f}", dy=38)
    axes[0].set_ylim(0, 2050)

    bars1 = axes[1].bar(x, positive, color=colors, alpha=0.9)
    axes[1].set_title("Positive gain rate")
    axes[1].set_ylim(0.50, 0.72)
    axes[1].set_xticks(x, models, rotation=20, ha="right")
    axes[1].grid(axis="y", color="#e2e8f0")
    annotate_bar(axes[1], bars1, "{:.3f}", dy=0.006)

    bars2 = axes[2].bar(x, negative, color=colors, alpha=0.9)
    axes[2].set_title("Negative gain rate")
    axes[2].set_ylim(0.20, 0.46)
    axes[2].set_xticks(x, models, rotation=20, ha="right")
    axes[2].grid(axis="y", color="#e2e8f0")
    annotate_bar(axes[2], bars2, "{:.3f}", dy=0.007)

    fig.suptitle("Ablation and Strong Baseline Results", fontsize=17, weight="bold", y=1.04)
    fig.text(
        0.5,
        -0.02,
        "Full uses Qwen3-VL, depth, and pose. SigLIP uses the same policy head with SigLIP frozen features.",
        ha="center",
        color="#475569",
        fontsize=10.5,
    )
    fig.tight_layout()

    path = output_dir / "ablation_results.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        render_oracle_pipeline(args.output_dir),
        render_stay_threshold(args.output_dir),
        render_ablation_results(args.output_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
