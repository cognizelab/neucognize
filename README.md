# Analysis and figure-reproduction code

This repository accompanies the manuscript **"Two dominant axes structure
object representations in the human ventral temporal cortex."** It contains a
compact implementation of the core factor-analysis workflow, scripts and
plot-ready values for selected manuscript figures, and a fully synthetic demo
that can be run without access to the restricted study data.

## Data privacy and repository scope

The repository does **not** contain raw or subject-level data, fMRI
measurements, stimulus images, behavioral records, semantic embeddings,
individual brain maps, or the original factor-analysis input matrices.

The files in `Figure_codes/Figure_data/` are final plot-ready coordinates and
summary statistics for the listed figure panels. The `Demo/` inputs are
generated independently from a documented pseudorandom model and are not
derived from study data.

## Repository structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- Main_codes/
|   `-- Factor_analysis/
|       |-- core_fa.py
|       `-- run_fa.py
|-- Figure_codes/
|   |-- Figure3A/Figure3A.py
|   |-- Figure3B/Figure3B.py
|   |-- Figure4C/Figure4C.py
|   |-- Figure4D/Figure4D.py
|   `-- Figure_data/
|       |-- fig3a_plot_points.csv
|       |-- fig3b_factor_correlations.csv
|       |-- fig4c_things_plot_points.csv
|       `-- fig4d_cneuromod_plot_points.csv
`-- Demo/
    |-- README.md
    |-- generate_demo_data.py
    `-- run_demo.py
```

Generated files are written to `outputs/` or `Demo/output/`; both directories
are excluded from version control.

## System requirements

- Python 3.10
- Package versions listed in `requirements.txt`
- No GPU or non-standard hardware is required for the included synthetic demo
  or figure-reproduction scripts
- The full factor-analysis memory requirement depends on the size of the
  user-supplied matrix

The release environment targets Python 3.10 and the pinned versions in
`requirements.txt`. A complete smoke test was also run on 2 September 2026 on
64-bit Windows build 22621 with Python 3.10.9, NumPy 1.23.1, pandas 1.5.0,
Matplotlib 3.5.3, SciPy 1.11.2, and scikit-learn 1.7.2. No specialized hardware
was used or required for this verification.

## Installation

From the repository root, create and activate a Python 3.10 environment, then
install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Under normal network conditions, installation is expected to take less than
five minutes on a typical desktop or laptop. Actual time depends on network
speed and whether binary packages are already cached.

## Quick synthetic demo

Run the complete public factor-analysis workflow on independently generated
synthetic data:

```powershell
python Demo/run_demo.py
```

The command:

1. generates a 240 x 60 artificial low-rank matrix using the fixed seed
   `20260902`;
2. fits a five-factor model with Varimax rotation and LAPACK SVD;
3. saves factor scores, standardized scores, components, loadings, noise
   variances, and variance summaries;
4. validates the output dimensions and finite values; and
5. writes a machine-readable provenance record confirming that the demo is
   synthetic and contains no human data.

Expected runtime is substantially less than one minute on a typical desktop or
laptop. In the Windows test environment described above, it completed in 2.5
seconds. See `Demo/README.md` for the complete output list and privacy details.

## Figure reproduction

All commands below can be run from the repository root. By default, outputs are
written under `outputs/`.

### Figure 3A

```powershell
python Figure_codes/Figure3A/Figure3A.py
```

Input: `Figure_codes/Figure_data/fig3a_plot_points.csv`

Expected outputs:

- `outputs/Figure3A/fig3a_factor_space_density.png`
- `outputs/Figure3A/fig3a_factor_space_extreme_points_top9.png`

### Figure 3B

```powershell
python Figure_codes/Figure3B/Figure3B.py
```

Input: `Figure_codes/Figure_data/fig3b_factor_correlations.csv`

Expected output:

- `outputs/Figure3B/fig3b_factor_correlation_top10.jpg`

Use `--top-k 15`, for example, to plot a different number of positive and
negative correlations.

### Figure 4C

```powershell
python Figure_codes/Figure4C/Figure4C.py
```

Input: `Figure_codes/Figure_data/fig4c_things_plot_points.csv`

Expected output:

- `outputs/Figure4C/fig4c_generalisability_things.png`

### Figure 4D

```powershell
python Figure_codes/Figure4D/Figure4D.py
```

Input: `Figure_codes/Figure_data/fig4d_cneuromod_plot_points.csv`

Expected output:

- `outputs/Figure4D/fig4d_generalisability_cneuromod.png`

Each figure command is expected to complete in less than one minute on a
typical desktop or laptop. In the Windows test environment described above,
Figures 3A, 3B, 4C, and 4D completed in 5.0, 1.4, 2.5, and 2.5 seconds,
respectively. Use `--output-dir PATH` to select another output directory.

## Core factor-analysis workflow

The reusable analysis code can:

1. optionally average repeated measurements within concept;
2. fit a Varimax-rotated factor-analysis model using LAPACK SVD;
3. evaluate candidate dimensionalities with repeated shuffled K-fold
   log-likelihood;
4. match factors across datasets using absolute loading correlations and the
   Hungarian assignment algorithm; and
5. export scores, standardized scores, components, loadings, noise variances,
   and variance summaries.

The input must be a numeric matrix with observations in rows and features in
columns. For example:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/concept_response_matrix.npy `
  --output-dir path/to/fa_results `
  --n-components 50
```

To average repeated rows before fitting, provide one group identifier per row:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/repeated_response_matrix.npy `
  --group-labels path/to/concept_ids.npy `
  --output-dir path/to/fa_results
```

To run repeated two-fold dimensionality evaluation:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/concept_response_matrix.npy `
  --output-dir path/to/fa_results `
  --evaluate-components 10 20 30 40 50 60 70 80 90 100
```

For cross-dataset component matching, provide a previously exported result:

```powershell
python Main_codes/Factor_analysis/run_fa.py `
  --input path/to/target_response_matrix.npy `
  --reference path/to/reference/factor_analysis_results.npz `
  --output-dir path/to/matched_fa_results
```

By default, component matching reproduces the manuscript workflow by reordering
factors without changing their signs. Add `--align-signs` only when positive
correlations with the reference axes are required.

## Data availability

Only the plot-ready, non-individual-level files required for the listed figure
panels are distributed here. Raw and participant-level neuroimaging data are
not included because their distribution is governed by participant consent,
institutional ethics approval, and the data-access conditions described in the
manuscript. Questions about access should be directed to the corresponding
author.

## Citation

Please cite both the associated manuscript and this code repository:

> Wang, Y. et al. (2026). *Two dominant axes structure object representations
> in the human ventral temporal cortex*. Code and figure-reproduction data.
> https://github.com/cognizelab/neucognize

The citation can be updated with the journal reference and repository DOI after
publication without changing the reproducibility workflow.

## License

This software is distributed under the GNU General Public License version 3.0.
See `LICENSE` for the complete terms.

## Contact

For questions about the manuscript, code, or restricted-data access, contact
Prof. Deniz Vatansever at `deniz@fudan.edu.cn`.
