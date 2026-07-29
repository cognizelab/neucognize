"""Reproduce the two Figure 3A plotting assets from exported 2-D points only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_DATA = PROJECT_ROOT / "code" / "data" / "fig3a_plot_points.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "fig3a"

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


def build_category_colors() -> dict[str, np.ndarray]:
    """Return the modified-coolwarm colors used by the source figure."""
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
    """Load and validate the plotting-only point table."""
    frame = pd.read_csv(path)
    expected_columns = ["point_id", "x", "y", "category"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{path} must contain exactly these columns: {expected_columns}"
        )
    if frame[expected_columns].isna().any().any():
        raise ValueError(f"{path} contains missing values")
    if frame["point_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate point_id values")
    unknown = sorted(set(frame["category"]) - set(CATEGORY_ORDER))
    if unknown:
        raise ValueError(f"{path} contains unknown categories: {unknown}")
    return frame


def plot_2d_space_error(
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    ax: plt.Axes,
    *,
    colors: list[np.ndarray],
    doarea: int = 100,
    area_alpha: float = 0.9,
    ci: float = 95,
) -> plt.Axes:
    """Draw category-wise, layered confidence ellipses."""
    x = np.asarray(x)
    y = np.asarray(y)
    categories = np.asarray(categories)
    unique_categories, category_indices = np.unique(
        categories, return_inverse=True
    )
    one_hot = np.eye(len(unique_categories))[category_indices]
    ci_scale = {95: 1.96, 90: 1.645, 50: 0.674, 30: 0.385, 10: 0.13}[ci]
    theta = np.linspace(0, 2 * np.pi, 100)

    for index in range(len(unique_categories)):
        selected = one_hot[:, index].astype(bool)
        x_category = x[selected]
        y_category = y[selected]
        x_mean = np.mean(x_category)
        y_mean = np.mean(y_category)
        x_sem = np.std(x_category, ddof=1) / np.sqrt(len(x_category))
        y_sem = np.std(y_category, ddof=1) / np.sqrt(len(y_category))
        x_ci = ci_scale * x_sem
        y_ci = ci_scale * y_sem

        for layer in range(1, doarea + 1):
            scale = layer / doarea
            alpha = area_alpha * (1 - (layer - 1) / doarea)
            ax.fill(
                x_mean + scale * x_ci * np.cos(theta),
                y_mean + scale * y_ci * np.sin(theta),
                color=colors[index],
                alpha=alpha,
                edgecolor="none",
                zorder=1,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.2, length=6)
    return ax


def save_density_panel(points: pd.DataFrame, output_path: Path) -> None:
    """Create the Figure 3A category-density visualization."""
    figure, axis = plt.subplots(1, 1, figsize=(5.5, 6))
    unique_categories = np.unique(points["category"].values)
    plot_2d_space_error(
        points["x"].to_numpy(),
        points["y"].to_numpy(),
        points["category"].to_numpy(),
        axis,
        colors=[CATEGORY_COLORS[item] for item in unique_categories],
    )

    axis.set(xlabel="", ylabel="", xlim=(-1.8, 1.1), ylim=(-2.2, 1.1))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0.0, right=0.95, bottom=0.05, top=0.95)
    figure.savefig(output_path, dpi=300, transparent=True)
    plt.close(figure)


def select_extreme_points(
    points: pd.DataFrame, top_k: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the top-right, low-x, and low-y point IDs used as markers."""
    coordinates = points[["x", "y"]].to_numpy()
    top_right = coordinates.max(axis=0)
    distances = np.sqrt(np.sum((coordinates - top_right) ** 2, axis=1))
    top_right_indices = np.argsort(distances)[:top_k]
    low_x_indices = np.argsort(coordinates[:, 0])[:top_k]
    low_y_indices = np.argsort(coordinates[:, 1])[:top_k]
    return top_right_indices, low_x_indices, low_y_indices


def _scatter_by_category(
    axis: plt.Axes,
    points: pd.DataFrame,
    *,
    size: float,
    alpha: float,
) -> None:
    for category in CATEGORY_ORDER:
        selected = points["category"] == category
        axis.scatter(
            points.loc[selected, "x"],
            points.loc[selected, "y"],
            s=size,
            color=CATEGORY_COLORS[category],
            alpha=alpha,
            edgecolors="none",
            zorder=3,
        )


def save_extreme_points_panel(points: pd.DataFrame, output_path: Path) -> None:
    """Create the Figure 3A point cloud with highlighted extreme points."""
    top_right, low_x, low_y = select_extreme_points(points, top_k=9)
    special_indices = np.unique(np.concatenate([top_right, low_x, low_y]))
    background = points.drop(index=special_indices)

    figure, axis = plt.subplots(1, 1, figsize=(12, 12))
    _scatter_by_category(axis, background, size=120, alpha=0.3)

    marker_groups = [
        (top_right, "*"),
        (low_x, "P"),
        (low_y, "X"),
    ]
    for indices, marker in marker_groups:
        for index in indices:
            row = points.iloc[index]
            axis.scatter(
                row["x"],
                row["y"],
                s=1500,
                zorder=3,
                facecolor=CATEGORY_COLORS[row["category"]],
                edgecolor="gray",
                linewidth=2,
                marker=marker,
            )

    axis.set(xlabel="", ylabel="", xlim=(-5, 3), ylim=(-5, 3))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([-5, 3])
    axis.set_yticks([-5, 3])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        length=16,
        width=4,
        colors="gray",
    )
    axis.spines["left"].set_position("zero")
    axis.spines["bottom"].set_position("zero")
    axis.spines["left"].set_color("gray")
    axis.spines["bottom"].set_color("gray")
    axis.spines["right"].set_color("none")
    axis.spines["top"].set_color("none")
    axis.spines["left"].set_linewidth(4)
    axis.spines["bottom"].set_linewidth(4)

    plt.subplots_adjust(left=0.06, right=0.93, bottom=0.05, top=0.95)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    points = load_plot_points(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    density_path = args.output_dir / "fig3a_factor_space_density.png"
    extremes_path = (
        args.output_dir
        / "fig3a_factor_space_extreme_points_top9.png"
    )
    save_density_panel(points, density_path)
    save_extreme_points_panel(points, extremes_path)
    print(density_path)
    print(extremes_path)


if __name__ == "__main__":
    main()
