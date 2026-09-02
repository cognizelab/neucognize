"""Fit the manuscript's core Varimax factor-analysis model.

The input is an external numeric matrix with observations in rows and features
in columns. Repeated observations can optionally be averaged by concept before
the model is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from core_fa import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    average_repeated_measurements,
    cross_validated_log_likelihood,
    factor_variance,
    fit_varimax_fa,
    match_components,
    reorder_components,
    standardize_scores,
)


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    """Load a matrix or vector from NPY, NPZ, CSV, or TSV."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(
                        f"{path} contains multiple arrays; specify --input-key"
                    )
                key = archive.files[0]
            if key not in archive.files:
                raise ValueError(f"{path} does not contain an array named {key!r}")
            return archive[key]
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=separator).to_numpy()
    raise ValueError(f"unsupported file type: {path.suffix}")


def save_cv_results(results: dict[int, np.ndarray], path: Path) -> None:
    rows = []
    for component_count, values in results.items():
        for repeat, score in enumerate(values):
            rows.append(
                {
                    "n_components": component_count,
                    "repeat": repeat,
                    "mean_log_likelihood": score,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-key")
    parser.add_argument("--group-labels", type=Path)
    parser.add_argument("--group-labels-key")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-key", default="components")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--align-signs",
        action="store_true",
        help="Flip matched factors to positive reference correlations.",
    )
    parser.add_argument(
        "--evaluate-components",
        type=int,
        nargs="+",
        metavar="N",
        help="Optional component counts for repeated likelihood evaluation.",
    )
    parser.add_argument("--cv-repeats", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=2)
    args = parser.parse_args()

    source_matrix = load_array(args.input, args.input_key)
    row_ids: np.ndarray = np.arange(len(source_matrix))
    matrix = source_matrix
    if args.group_labels is not None:
        group_ids = load_array(args.group_labels, args.group_labels_key).squeeze()
        matrix, row_ids = average_repeated_measurements(source_matrix, group_ids)

    model, scores = fit_varimax_fa(
        matrix,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    components = model.components_.copy()
    match_correlations = np.array([], dtype=np.float64)
    match_permutation = np.array([], dtype=np.int64)
    match_signs = np.array([], dtype=np.float64)

    if args.reference is not None:
        reference_components = load_array(args.reference, args.reference_key)
        match = match_components(reference_components, components)
        scores, components = reorder_components(
            scores,
            components,
            match,
            align_signs=args.align_signs,
        )
        match_correlations = match.correlations
        match_permutation = match.permutation
        match_signs = match.signs

    normalized_scores = standardize_scores(scores)
    variance, variance_ratio, cumulative_variance_ratio = factor_variance(
        components,
        matrix,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "factor_analysis_results.npz"
    np.savez_compressed(
        result_path,
        scores=scores,
        standardized_scores=normalized_scores,
        components=components,
        loadings=components.T,
        mean=model.mean_,
        noise_variance=model.noise_variance_,
        factor_variance=variance,
        factor_variance_ratio=variance_ratio,
        cumulative_factor_variance_ratio=cumulative_variance_ratio,
        row_ids=row_ids,
        match_permutation=match_permutation,
        match_correlations=match_correlations,
        match_signs=match_signs,
    )

    metadata = {
        "input": str(args.input),
        "input_shape": list(np.asarray(source_matrix).shape),
        "fitted_shape": list(matrix.shape),
        "group_averaging_applied": args.group_labels is not None,
        "n_components": args.n_components,
        "rotation": "varimax",
        "svd_method": "lapack",
        "random_state": args.random_state,
        "reference": str(args.reference) if args.reference else None,
        "component_signs_aligned": args.align_signs,
        "scikit_learn_version": sklearn.__version__,
    }
    metadata_path = args.output_dir / "factor_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.evaluate_components:
        cv_results = cross_validated_log_likelihood(
            matrix,
            args.evaluate_components,
            repeats=args.cv_repeats,
            folds=args.cv_folds,
            random_state=args.random_state,
        )
        save_cv_results(
            cv_results,
            args.output_dir / "factor_analysis_component_selection.csv",
        )

    print(result_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
