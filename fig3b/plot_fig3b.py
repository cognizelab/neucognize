"""Reproduce the Figure 3B factor-correlation bar plot from plot-ready data."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_DATA = PROJECT_ROOT / "code" / "data" / "fig3b_factor_correlations.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "fig3b"
DEFAULT_TOP_K = 10

EXPECTED_COLUMNS = [
    "dimension_id",
    "dimension",
    "factor1_correlation",
    "factor1_p_value",
    "factor1_significant",
    "factor2_correlation",
    "factor2_p_value",
    "factor2_significant",
]


def load_plot_data(path: Path) -> pd.DataFrame:
    """Load and validate the 66-row table containing final plotting statistics."""
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path} must contain exactly these columns: {EXPECTED_COLUMNS}"
        )
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    if frame["dimension_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate dimension_id values")
    if not np.array_equal(
        frame["dimension_id"].to_numpy(),
        np.arange(len(frame)),
    ):
        raise ValueError("dimension_id must be consecutive and start at zero")
    if len(frame) != 66:
        raise ValueError(f"expected 66 dimensions, found {len(frame)}")

    for factor in (1, 2):
        correlations = frame[f"factor{factor}_correlation"]
        if not correlations.between(-1, 1).all():
            raise ValueError(f"factor{factor} correlations must be in [-1, 1]")
        expected_significance = frame[f"factor{factor}_p_value"] < 0.05
        actual_significance = frame[f"factor{factor}_significant"].astype(bool)
        if not np.array_equal(expected_significance, actual_significance):
            raise ValueError(f"factor{factor} significance flags are inconsistent")
    return frame


def select_plot_rows(
    correlations: np.ndarray,
    *,
    factor_index: int,
    top_k: int,
) -> np.ndarray:
    """Match the source ordering of the strongest positive and negative rows."""
    positive = np.argsort(correlations)[-top_k:][::-1]
    negative = np.argsort(correlations)[:top_k]
    if factor_index == 0:
        return np.concatenate([positive, negative[::-1]])
    return np.concatenate([negative, positive[::-1]])


def save_figure(
    data: pd.DataFrame,
    output_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Render the two vertically stacked horizontal bar panels."""
    if top_k < 1 or top_k * 2 > len(data):
        raise ValueError(f"top_k must be between 1 and {len(data) // 2}")

    all_correlations = data[
        ["factor1_correlation", "factor2_correlation"]
    ].to_numpy()
    value_min = all_correlations.min()
    value_max = all_correlations.max()
    norm = mcolors.TwoSlopeNorm(vmin=value_min, vcenter=0, vmax=value_max)
    color_map = plt.get_cmap("coolwarm")

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6, 10),
        sharey=False,
        sharex=True,
    )

    for factor_index, axis in enumerate(axes):
        factor_number = factor_index + 1
        correlations = data[f"factor{factor_number}_correlation"].to_numpy()
        significance = data[f"factor{factor_number}_significant"].astype(bool).to_numpy()
        selected = select_plot_rows(
            correlations,
            factor_index=factor_index,
            top_k=top_k,
        )

        weights = correlations[selected]
        significant = significance[selected]
        labels = data["dimension"].to_numpy()[selected]
        colors = color_map(norm(weights))
        colors[:, 3] = np.where(significant, 1.0, 0.2)
        y_positions = np.arange(len(weights))

        axis.barh(
            y_positions,
            weights,
            color=colors,
            edgecolor="none",
            height=0.6,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks([])

        for y_position, label in zip(y_positions, labels):
            axis.text(
                0.52,
                y_position,
                textwrap.fill(label, width=50),
                va="center",
                ha="left",
                fontsize=9,
            )

        axis.set_xlim(-0.8, 0.5)
        axis.set_ylim(-0.5, len(weights) - 0.5)
        if factor_index == 1:
            axis.set_xlabel("Correlation")
        axis.xaxis.grid(True, linestyle="--", alpha=0.3)
        for spine in ["top", "right", "left"]:
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelsize=8)

    plt.subplots_adjust(
        left=0.02,
        right=0.5,
        bottom=0.04,
        top=0.99,
        hspace=0.1,
    )
    figure.savefig(output_path, dpi=300, transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    data = load_plot_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"fig3b_factor_correlation_top{args.top_k}.jpg"
    )
    save_figure(data, output_path, top_k=args.top_k)
    print(output_path)


if __name__ == "__main__":
    main()
