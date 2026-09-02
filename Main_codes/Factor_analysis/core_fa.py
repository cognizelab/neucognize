"""Core factor-analysis operations used by the manuscript analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold, cross_val_score


DEFAULT_N_COMPONENTS = 50
DEFAULT_RANDOM_STATE = 1234


@dataclass(frozen=True)
class ComponentMatch:
    """Mapping from reference-component order to target-component order."""

    permutation: np.ndarray
    correlations: np.ndarray
    signs: np.ndarray


def validate_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Return a finite, two-dimensional floating-point matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, found shape {values.shape}")
    if min(values.shape) < 2:
        raise ValueError(f"{name} must contain at least two rows and two columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def average_repeated_measurements(
    measurements: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average repeated rows within each concept or other grouping variable."""
    values = validate_matrix(measurements, "measurements")
    labels = np.asarray(group_ids)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("group_ids must be one-dimensional with one value per row")

    unique_ids = np.unique(labels)
    means = np.stack([values[labels == group_id].mean(axis=0) for group_id in unique_ids])
    return means, unique_ids


def fit_varimax_fa(
    matrix: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[FactorAnalysis, np.ndarray]:
    """Fit the manuscript's Varimax-rotated FA model and return factor scores."""
    values = validate_matrix(matrix)
    maximum = min(values.shape)
    if not 1 <= n_components <= maximum:
        raise ValueError(
            f"n_components must be between 1 and {maximum}, found {n_components}"
        )

    model = FactorAnalysis(
        n_components=n_components,
        rotation="varimax",
        svd_method="lapack",
        random_state=random_state,
    )
    scores = model.fit_transform(values)
    return model, scores


def standardize_scores(scores: np.ndarray) -> np.ndarray:
    """Z-score factor scores using the sample standard deviation (ddof=1)."""
    values = validate_matrix(scores, "scores")
    standard_deviation = values.std(axis=0, ddof=1)
    if np.any(standard_deviation == 0):
        raise ValueError("at least one factor score has zero variance")
    return (values - values.mean(axis=0)) / standard_deviation


def match_components(
    reference_components: np.ndarray,
    target_components: np.ndarray,
) -> ComponentMatch:
    """Match target factors to a reference with absolute loading correlations.

    This reproduces the manuscript workflow: the absolute Pearson-correlation
    matrix is optimized with the Hungarian assignment algorithm. ``signs`` is
    returned separately because the original cross-dataset analysis reordered
    components without flipping their signs.
    """
    reference = validate_matrix(reference_components, "reference_components")
    target = validate_matrix(target_components, "target_components")
    if reference.shape != target.shape:
        raise ValueError(
            "reference_components and target_components must have identical shapes"
        )

    count = reference.shape[0]
    signed_correlations = np.corrcoef(reference, target)[:count, count:]
    if not np.isfinite(signed_correlations).all():
        raise ValueError("component correlations contain NaN or infinite values")

    row_indices, column_indices = linear_sum_assignment(
        np.abs(signed_correlations),
        maximize=True,
    )
    permutation = np.empty(count, dtype=np.int64)
    correlations = np.empty(count, dtype=np.float64)
    for reference_index, target_index in zip(row_indices, column_indices):
        permutation[reference_index] = target_index
        correlations[reference_index] = signed_correlations[
            reference_index,
            target_index,
        ]

    signs = np.where(correlations < 0, -1.0, 1.0)
    return ComponentMatch(permutation, correlations, signs)


def reorder_components(
    scores: np.ndarray,
    components: np.ndarray,
    match: ComponentMatch,
    *,
    align_signs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a component match to factor scores and component weights."""
    reordered_scores = scores[:, match.permutation]
    reordered_components = components[match.permutation]
    if align_signs:
        reordered_scores = reordered_scores * match.signs
        reordered_components = reordered_components * match.signs[:, None]
    return reordered_scores, reordered_components


def factor_variance(
    components: np.ndarray,
    input_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute variance, proportional variance, and cumulative proportion."""
    weights = validate_matrix(components, "components")
    values = validate_matrix(input_matrix, "input_matrix")
    if weights.shape[1] != values.shape[1]:
        raise ValueError("components and input_matrix have different feature counts")

    variance = np.sum(weights.T**2, axis=0)
    total_variance = values.var(axis=0).sum()
    if total_variance <= 0:
        raise ValueError("input_matrix has no variance")
    proportion = variance / total_variance
    return variance, proportion, np.cumsum(proportion)


def cross_validated_log_likelihood(
    matrix: np.ndarray,
    component_counts: Iterable[int],
    *,
    repeats: int = 100,
    folds: int = 2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[int, np.ndarray]:
    """Evaluate FA dimensionalities with repeated shuffled K-fold likelihood."""
    values = validate_matrix(matrix)
    counts = [int(count) for count in component_counts]
    if not counts:
        raise ValueError("component_counts cannot be empty")
    if repeats < 1 or folds < 2:
        raise ValueError("repeats must be positive and folds must be at least two")

    maximum = min(values.shape)
    invalid = [count for count in counts if not 1 <= count <= maximum]
    if invalid:
        raise ValueError(f"invalid component counts for this matrix: {invalid}")

    results: dict[int, np.ndarray] = {}
    for count in counts:
        repeated_scores = np.empty(repeats, dtype=np.float64)
        estimator = FactorAnalysis(
            n_components=count,
            rotation="varimax",
            svd_method="lapack",
            random_state=random_state,
        )
        for repeat in range(repeats):
            splitter = KFold(
                n_splits=folds,
                shuffle=True,
                random_state=repeat,
            )
            repeated_scores[repeat] = cross_val_score(
                estimator,
                values,
                cv=splitter,
            ).mean()
        results[count] = repeated_scores
    return results
