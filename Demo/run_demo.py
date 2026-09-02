"""Run the complete factor-analysis workflow on fully synthetic data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from generate_demo_data import DEFAULT_FACTORS, write_demo_inputs


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
ANALYSIS_SCRIPT = REPO_ROOT / "Main_codes" / "Factor_analysis" / "run_fa.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_DIR / "output",
    )
    parser.add_argument("--n-components", type=int, default=DEFAULT_FACTORS)
    args = parser.parse_args()

    started = time.perf_counter()
    input_dir = args.output_dir / "input"
    result_dir = args.output_dir / "results"
    matrix_path, loadings_path, metadata_path = write_demo_inputs(input_dir)

    command = [
        sys.executable,
        str(ANALYSIS_SCRIPT),
        "--input",
        str(matrix_path),
        "--output-dir",
        str(result_dir),
        "--n-components",
        str(args.n_components),
    ]
    subprocess.run(command, check=True)

    result_path = result_dir / "factor_analysis_results.npz"
    with np.load(result_path, allow_pickle=False) as result:
        scores = result["scores"]
        components = result["components"]
        if scores.shape != (240, args.n_components):
            raise RuntimeError(f"unexpected score shape: {scores.shape}")
        if components.shape != (args.n_components, 60):
            raise RuntimeError(f"unexpected component shape: {components.shape}")
        if not np.isfinite(scores).all() or not np.isfinite(components).all():
            raise RuntimeError("demo results contain non-finite values")

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "status": "success",
        "synthetic_only": True,
        "contains_human_data": False,
        "input_shape": [240, 60],
        "n_components": args.n_components,
        "score_shape": list(scores.shape),
        "component_shape": list(components.shape),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "synthetic_matrix": str(matrix_path),
        "synthetic_ground_truth_loadings": str(loadings_path),
        "synthetic_metadata": str(metadata_path),
        "analysis_results": str(result_path),
    }
    summary_path = args.output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Demo completed successfully in {elapsed_seconds:.2f} seconds.")
    print(summary_path)


if __name__ == "__main__":
    main()
