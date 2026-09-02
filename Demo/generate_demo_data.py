"""Generate a fully synthetic matrix for the public factor-analysis demo.

The generator does not load, summarize, transform, or estimate parameters from
any study participant, stimulus, brain image, or original response matrix.
All values are produced from a documented pseudorandom model with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SEED = 20260902
DEFAULT_SAMPLES = 240
DEFAULT_FEATURES = 60
DEFAULT_FACTORS = 5
DEFAULT_NOISE_SCALE = 0.35


def generate_synthetic_matrix(
    *,
    samples: int = DEFAULT_SAMPLES,
    features: int = DEFAULT_FEATURES,
    factors: int = DEFAULT_FACTORS,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic low-rank matrix and its artificial loadings."""
    if samples < 20:
        raise ValueError("samples must be at least 20")
    if features < factors * 2:
        raise ValueError("features must be at least twice the number of factors")
    if factors < 2:
        raise ValueError("factors must be at least 2")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")

    rng = np.random.default_rng(seed)
    latent_scores = rng.normal(size=(samples, factors))
    loadings = rng.normal(scale=0.08, size=(factors, features))

    # Create transparent block structure so each artificial factor has a
    # distinct group of strongly associated artificial features.
    for factor_index, feature_indices in enumerate(
        np.array_split(np.arange(features), factors)
    ):
        signs = np.where(np.arange(len(feature_indices)) % 2 == 0, 1.0, -1.0)
        magnitudes = rng.uniform(0.75, 1.25, size=len(feature_indices))
        loadings[factor_index, feature_indices] = signs * magnitudes

    noise = rng.normal(scale=noise_scale, size=(samples, features))
    matrix = latent_scores @ loadings + noise
    matrix -= matrix.mean(axis=0, keepdims=True)
    matrix /= matrix.std(axis=0, ddof=1, keepdims=True)
    return matrix, loadings


def write_demo_inputs(
    output_dir: Path,
    *,
    samples: int = DEFAULT_SAMPLES,
    features: int = DEFAULT_FEATURES,
    factors: int = DEFAULT_FACTORS,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path, Path]:
    """Generate and save the synthetic input plus provenance metadata."""
    matrix, loadings = generate_synthetic_matrix(
        samples=samples,
        features=features,
        factors=factors,
        noise_scale=noise_scale,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "synthetic_response_matrix.npy"
    loadings_path = output_dir / "synthetic_ground_truth_loadings.npy"
    metadata_path = output_dir / "synthetic_data_metadata.json"
    np.save(matrix_path, matrix, allow_pickle=False)
    np.save(loadings_path, loadings, allow_pickle=False)
    metadata = {
        "synthetic_only": True,
        "contains_human_data": False,
        "derived_from_study_data": False,
        "generator": "independent Gaussian latent-factor model",
        "seed": seed,
        "samples": samples,
        "features": features,
        "factors": factors,
        "noise_scale": noise_scale,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix_path, loadings_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "input",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--features", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--factors", type=int, default=DEFAULT_FACTORS)
    parser.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    for path in write_demo_inputs(
        args.output_dir,
        samples=args.samples,
        features=args.features,
        factors=args.factors,
        noise_scale=args.noise_scale,
        seed=args.seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
