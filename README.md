# Figure Reproduction Code

This directory contains the plotting code and plot-ready data required to
reproduce the Figure 3a, Figure 3b, Figure 4c, and Figure 4d visualizations.
It also includes a compact implementation of the core factor-analysis
workflow used to generate the factor coordinates.

Only final plotting coordinates and summary statistics are included. The
repository does not contain raw or subject-level data, fMRI measurements,
stimulus images, behavioral records, semantic embeddings, or factor-analysis
inputs.

## Directory structure

```text
code/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- fig3a_plot_points.csv
|   |-- fig3b_factor_correlations.csv
|   |-- fig4c_things_plot_points.csv
|   `-- fig4d_cneuromod_plot_points.csv
|-- fig3a/
|   `-- plot_fig3a.py
|-- fig3b/
|   `-- plot_fig3b.py
|-- fig4c/
|   `-- plot_fig4c.py
|-- fa/
|   |-- core_fa.py
|   `-- run_fa.py
`-- fig4d/
    `-- plot_fig4d.py

figures/
|-- fig3a/
|   |-- fig3a_factor_space_density.png
|   `-- fig3a_factor_space_extreme_points_top9.png
|-- fig3b/
|   `-- fig3b_factor_correlation_top10.jpg
|-- fig4c/
|   `-- fig4c_generalisability_things.png
`-- fig4d/
    `-- fig4d_generalisability_cneuromod.png
```

## Installation

The scripts were tested with Python 3.10 and the package versions listed in
`requirements.txt`.

From the `code` directory, install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Figure 3A

Run:

```powershell
python fig3a\plot_fig3a.py
```

The script reads `data/fig3a_plot_points.csv` and creates two images:

| Output file | Description |
| --- | --- |
| `fig3a_factor_space_density.png` | Category-wise confidence-density regions in the two-dimensional factor space |
| `fig3a_factor_space_extreme_points_top9.png` | Full point cloud with the nine extreme points highlighted for each selection direction |

The input table contains 960 final plotting points and four columns:

| Column | Description |
| --- | --- |
| `point_id` | Stable point identifier used for extreme-point selection |
| `x` | Horizontal plotting coordinate |
| `y` | Vertical plotting coordinate |
| `category` | Category label used for color assignment |

To save the images elsewhere:

```powershell
python fig3a\plot_fig3a.py --output-dir path\to\output
```

## Figure 3B

Run:

```powershell
python fig3b\plot_fig3b.py
```

The script reads `data/fig3b_factor_correlations.csv` and creates:

| Output file | Description |
| --- | --- |
| `fig3b_factor_correlation_top10.jpg` | Strongest positive and negative semantic-dimension correlations for Factors 1 and 2 |

The input table contains the final plot-ready statistics for 66 semantic
dimensions:

| Column | Description |
| --- | --- |
| `dimension_id` | Stable semantic-dimension identifier |
| `dimension` | Semantic-dimension label |
| `factor1_correlation` | Pearson correlation with Factor 1 |
| `factor1_p_value` | Two-sided p value for Factor 1 |
| `factor1_significant` | Whether the Factor 1 correlation has p < 0.05 |
| `factor2_correlation` | Pearson correlation with Factor 2 |
| `factor2_p_value` | Two-sided p value for Factor 2 |
| `factor2_significant` | Whether the Factor 2 correlation has p < 0.05 |

By default, the plot displays the ten strongest positive and ten strongest
negative correlations for each factor. To use a different number:

```powershell
python fig3b\plot_fig3b.py --top-k 15
```

The corresponding output would be named
`fig3b_factor_correlation_top15.jpg`.

## Figure 4C

Run:

```powershell
python fig4c\plot_fig4c.py
```

The script reads `data/fig4c_things_plot_points.csv` and creates:

| Output file | Description |
| --- | --- |
| `fig4c_generalisability_things.png` | Category-wise confidence-density regions showing the generalisability of the two-dimensional factor geometry to THINGS |

The CSV contains 425 final two-dimensional plotting points assigned to 14
categories. It contains only the four columns needed to draw the panel:
`point_id`, `x`, `y`, and `category`.

## Figure 4D

Run:

```powershell
python fig4d\plot_fig4d.py
```

The script reads `data/fig4d_cneuromod_plot_points.csv` and creates:

| Output file | Description |
| --- | --- |
| `fig4d_generalisability_cneuromod.png` | Category-wise confidence-density regions showing the generalisability of the two-dimensional factor geometry to CNeuroMod |

The CSV contains the 425 CNeuroMod plotting points matched to the THINGS
concept set and assigned to the same 14 categories. It contains only
`point_id`, `x`, `y`, and `category`; no source images, embeddings, subject
measurements, or factor-analysis inputs are included.

To save either Figure 4 panel elsewhere, pass `--output-dir`, for example:

```powershell
python fig4d\plot_fig4d.py --output-dir path\to\output
```

## Core factor analysis

The `fa` directory contains the analysis steps shared by the manuscript:

1. optionally average repeated measurements for each concept;
2. fit a 50-component factor-analysis model with Varimax rotation and LAPACK
   SVD;
3. optionally match factors across datasets from the absolute correlations
   between their component weights using the Hungarian assignment algorithm;
4. export factor scores, standardized scores, component weights, loadings,
   noise variances, and variance summaries.

The original response matrices are not distributed in this directory. Supply
an external numeric matrix with observations in rows and features in columns:

```powershell
python fa\run_fa.py `
  --input path\to\concept_response_matrix.npy `
  --output-dir path\to\fa_results
```

If the input contains repeated measurements, provide one group identifier per
row. The script averages the rows within each identifier before fitting FA:

```powershell
python fa\run_fa.py `
  --input path\to\response_matrix.npy `
  --group-labels path\to\concept_ids.npy `
  --output-dir path\to\fa_results
```

To reproduce the repeated two-fold log-likelihood evaluation used to compare
candidate dimensionalities:

```powershell
python fa\run_fa.py `
  --input path\to\concept_response_matrix.npy `
  --output-dir path\to\fa_results `
  --evaluate-components 10 20 30 40 50 60 70 80 90 100
```

For cross-dataset analyses, pass a previously exported
`factor_analysis_results.npz` as the reference:

```powershell
python fa\run_fa.py `
  --input path\to\target_concept_response_matrix.npy `
  --reference path\to\reference\factor_analysis_results.npz `
  --output-dir path\to\matched_fa_results
```

By default, matching reproduces the manuscript code by reordering factors
without changing their signs. Add `--align-signs` only when positive
correlations with the reference axes are required.
