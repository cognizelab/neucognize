"""Reproduce Figure 4c: generalisability of factor geometry to THINGS."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATA = (
    REPO_ROOT / "Figure_codes" / "Figure_data" / "fig4c_things_plot_points.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "Figure4C"
OUTPUT_NAME = "fig4c_generalisability_things.png"

CATEGORY_ORDER = [
    "animal",
    "body part",
    "clothing",
    "decoration",
    "weapon",
    "food",
    "plant",
    "musical instrument",
    "sports equipment",
    "toy",
    "tool",
    "container",
    "vehicle",
    "furniture",
]

LABEL_POSITIONS = {
    "body part": (-2.15, -1.13),
    "animal": (-1.48, -0.58),
    "sports equipment": (-1.22, -0.92),
    "clothing": (-1.05, -0.12),
    "decoration": (-0.50, -0.47),
    "weapon": (-0.57, 0.03),
    "toy": (-0.47, 0.36),
    "musical instrument": (-0.20, 0.67),
    "tool": (0.05, -0.11),
    "container": (0.34, 0.56),
    "vehicle": (0.20, 1.14),
    "furniture": (0.30, 1.90),
    "plant": (0.50, -0.86),
    "food": (0.48, -1.12),
}


def build_category_colors() -> dict[str, np.ndarray]:
    color_map = LinearSegmentedColormap.from_list(
        "ModifiedCoolwarm",
        ["#3B4CC0", "#00FFEE", "#B40426"],
        N=14,
    )
    colors = color_map(np.linspace(0, 1, 14))
    color_index = {
        "animal": 0,
        "body part": 1,
        "clothing": 2,
        "toy": 3,
        "sports equipment": 4,
        "decoration": 5,
        "food": 6,
        "plant": 7,
        "weapon": 8,
        "tool": 9,
        "container": 10,
        "musical instrument": 11,
        "vehicle": 12,
        "furniture": 13,
    }
    return {category: colors[index] for category, index in color_index.items()}


CATEGORY_COLORS = build_category_colors()


def load_plot_points(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = ["point_id", "x", "y", "category"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{path} must contain exactly these columns: {expected_columns}"
        )
    if frame.shape != (425, 4):
        raise ValueError(f"expected 425 plotting points, found {len(frame)}")
    if frame.isna().any().any() or frame["point_id"].duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate values")
    if not np.array_equal(frame["point_id"], np.arange(len(frame))):
        raise ValueError("point_id must be consecutive and start at zero")
    if set(frame["category"]) != set(CATEGORY_ORDER):
        raise ValueError("category labels do not match the expected 14 categories")
    return frame


def draw_density_regions(
    axis: plt.Axes,
    points: pd.DataFrame,
    *,
    layers: int = 100,
    maximum_alpha: float = 0.9,
) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    for category in sorted(CATEGORY_ORDER):
        category_points = points.loc[
            points["category"] == category,
            ["x", "y"],
        ].to_numpy()
        means = category_points.mean(axis=0)
        sem = category_points.std(axis=0, ddof=1) / np.sqrt(len(category_points))
        confidence_interval = 1.96 * sem

        for layer in range(1, layers + 1):
            scale = layer / layers
            alpha = maximum_alpha * (1 - (layer - 1) / layers)
            axis.fill(
                means[0] + scale * confidence_interval[0] * np.cos(theta),
                means[1] + scale * confidence_interval[1] * np.sin(theta),
                color=CATEGORY_COLORS[category],
                alpha=alpha,
                edgecolor="none",
                zorder=1,
            )


def save_figure(points: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    draw_density_regions(axis, points)

    for category in CATEGORY_ORDER:
        x, y = LABEL_POSITIONS[category]
        axis.text(
            x,
            y,
            category.capitalize(),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            zorder=3,
        )

    axis.set_xlim(-3, 1)
    axis.set_ylim(-1.5, 2)
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    for spine in axis.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(output_path, dpi=300, transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    points = load_plot_points(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / OUTPUT_NAME
    save_figure(points, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
