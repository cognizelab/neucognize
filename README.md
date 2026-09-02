# Repository for "Two dominant axes structure high-dimensional object representations in the human ventral temporal cortex"

Authors: Yun Wang, Kaixiang Zhuang, Xinyu Liang, Jianfeng Feng, Martin Hebart, and Deniz Vatansever.

This repository contains the factor-analysis and figure-reproduction code for the manuscript "Two dominant axes structure high-dimensional object representations in the human ventral temporal cortex".

## Keywords

Object vision, ventral temporal cortex, representational geometry, 7T fMRI, dimensionality reduction

## Repository Structure

- `Main_codes/`
  - `Factor_analysis/core_fa.py`: functions for Varimax factor analysis, score standardisation, factor matching, and model comparison.
  - `Factor_analysis/run_fa.py`: command-line script for running the factor-analysis workflow on a user-supplied response matrix.
- `Figure_codes/`
  - `Figure3A/Figure3A.py`: reproduces the two Figure 3A panels.
  - `Figure3B/Figure3B.py`: reproduces the Figure 3B correlation plot.
  - `Figure4C/Figure4C.py`: reproduces the THINGS generalisability panel.
  - `Figure4D/Figure4D.py`: reproduces the CNeuroMod generalisability panel.
  - `Figure_data/`: plot-ready coordinates and summary statistics used by the figure scripts.
- `Demo/`
  - `generate_demo_data.py`: generates a small simulated response matrix.
  - `run_demo.py`: runs the factor-analysis workflow on the simulated data.
- `requirements.txt`: Python package versions used for the analyses.
- `LICENSE`: GNU General Public License v3.0.

The repository does not contain raw fMRI data, behavioural data, stimulus images, semantic embeddings, or participant-level response matrices.

## Instructions for Demo

The demo provides a minimal example of the factor-analysis workflow without using study data. `generate_demo_data.py` creates a 240 x 60 matrix from a Gaussian latent-factor model with a fixed random seed. The simulated rows and columns do not correspond to participants, stimuli, voxels, or brain regions.

From the repository root, run:

```powershell
python Demo/run_demo.py
```

The results are written to `Demo/output/`:

- `input/synthetic_response_matrix.npy`
- `input/synthetic_ground_truth_loadings.npy`
- `input/synthetic_data_metadata.json`
- `results/factor_analysis_results.npz`
- `results/factor_analysis_metadata.json`
- `demo_summary.json`

The demo was tested with Python 3.10.9 on 64-bit Windows and completed in approximately 3 seconds. No specialised hardware is required.

## Prerequisites

The code was developed for Python 3.10. Install the required packages with:

```powershell
python -m pip install -r requirements.txt
```

The required packages are NumPy, pandas, Matplotlib, SciPy, and scikit-learn. Installation normally takes less than 5 minutes under standard network conditions.

## System Requirements

The code has been tested with the following environment:

- Operating system: Windows 11 (64-bit)
- Python: 3.10.9
- Python dependencies: versions listed in `requirements.txt`
- Hardware: standard desktop or laptop; no GPU or other specialised hardware is required

## Figure Reproduction

Run the following commands from the repository root:

```powershell
python Figure_codes/Figure3A/Figure3A.py
python Figure_codes/Figure3B/Figure3B.py
python Figure_codes/Figure4C/Figure4C.py
python Figure_codes/Figure4D/Figure4D.py
```

By default, the figures are saved under `outputs/`. Each script completed in less than 10 seconds in the test environment described above. A different output directory can be specified with `--output-dir`.

The expected files are:

- `outputs/Figure3A/fig3a_factor_space_density.png`
- `outputs/Figure3A/fig3a_factor_space_extreme_points_top9.png`
- `outputs/Figure3B/fig3b_factor_correlation_top10.jpg`
- `outputs/Figure4C/fig4c_generalisability_things.png`
- `outputs/Figure4D/fig4d_generalisability_cneuromod.png`

## Factor Analysis

`run_fa.py` accepts NPY, NPZ, CSV, or TSV matrices with observations in rows and features in columns. For example:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/concept_response_matrix.npy `
  --output-dir path/to/fa_results `
  --n-components 50
```

Repeated measurements can be averaged before fitting by supplying one group label per row:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/repeated_response_matrix.npy `
  --group-labels path/to/concept_ids.npy `
  --output-dir path/to/fa_results
```

Candidate dimensionalities can be compared using repeated cross-validated log-likelihood:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/concept_response_matrix.npy `
  --output-dir path/to/fa_results `
  --evaluate-components 10 20 30 40 50 60 70 80 90 100
```

For cross-dataset factor matching, supply a previously exported result as the reference:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/target_response_matrix.npy `
  --reference path/to/reference/factor_analysis_results.npz `
  --output-dir path/to/matched_fa_results
```

Factors are reordered according to the absolute correlations between component weights using the Hungarian assignment algorithm. Their signs are unchanged by default, as in the manuscript analyses. Use `--align-signs` to apply sign alignment.

## Data Availability

The raw neuroimaging and behavioural data generated in this study cannot be made openly available because of restrictions imposed by institutional ethics approval and participant consent. This repository contains only the plot-ready, non-individual-level values required for the figure panels listed above. Requests for access to restricted data should be directed to the corresponding author.

## Citation

Wang, Y. et al. (2026). *Two dominant axes structure high-dimensional object representations in the human ventral temporal cortex*.

Code repository: https://github.com/cognizelab/neucognize

## License

This repository is licensed under the GNU General Public License v3.0. See `LICENSE` for details.

## Contact

- Prof. Deniz Vatansever
- Institute of Science and Technology for Brain-inspired Intelligence, Fudan University
- Email: deniz@fudan.edu.cn
