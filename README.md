# CompoSil

Composite Silhouette is a Python package for robust clustering evaluation.
It introduces a composite metric that combines micro-averaged and macro-averaged silhouette using statistical weighting.
This provides a more nuanced assessment of clustering quality, helping identify the optimal number of clusters and compare performance across different clustering scenarios with greater confidence.
The framework is especially useful for data scientists, ML engineers, and researchers who want reliable metrics for centroid-based clustering.

## Overview and Methodology 
In standard clustering evaluation, the **silhouette coefficient** is widely used to measure how well each data point fits into its cluster in terms of intra-cluster cohesion and inter-cluster separation.
It can be aggregated as:
- **Micro-average:** The overall average silhouette score across all data points.
- **Macro-average:** The per-cluster average silhouette score.  

**Composite Silhouette** merges these two perspectives using a statistically-driven weighting strategy.
The method performs repeated subsampled clustering to compute both micro- and macro-averaged silhouette scores.
A Wilcoxon signed-rank test is then applied to their paired differences across subsamples to determine if one consistently and significantly outperforms the other.
The final score is the weighted combination of the sample averages of micro- and macro-averaged scores (*w</sub> · S<sub>micro</sub> + (1-w)</sub> · S<sub>macro</sub>*).
This convex combination keeps the result within the range of the individual scores and ties it meaningfully to both.
When a statistically significant difference is found, the dominant metric receives at least 75% of the total weight, with the exact proportion adjusted based on the mean difference across subsamples. The greater this difference, the more the weighting shifts in its favor, while the other still contributes proportionally—reflecting the relative strength of both perspectives. If no significant difference is found, both sample-average metrics are weighted equally.

> **Note:** The current implementation uses **K-Means** for clustering, which pairs well with silhouette-based evaluation and repeated subsampling. While the method can be adapted to other clustering algorithms, it already offers meaningful, statistically grounded insights for centroid-based clustering tasks.

## Installation 
You can install Composite Silhouette from PyPI: 

```bash
pip install composite-silhouette
```

or directly from the GitHub repository: 

```bash
pip install git+https://github.com/semoglou/composil.git
```

## Quick Start

```python
from composite_silhouette import CompSil
```

### Evaluate a Range of Cluster Counts
```python
from sklearn.datasets import make_blobs

# Generate synthetic 2D data
X, y_true = make_blobs(n_samples=2000, centers=4, cluster_std=1.1, random_state=42)

# Initialize the Composite Silhouette evaluation
cs = CompSil(
    data=X,                        # ndarray or DataFrame
    ground_truth=len(set(y_true))  # (Optional) for visual reference in plots
    k_values=range(2, 11),         # Evaluate cluster counts from 2 to 10
    num_samples=500,               # Number of random subsamples per k
    sample_size=100,               # Number of points in each subsample
    random_state=42,               # Ensures reproducibility
    n_jobs=-1                      # Use all available CPU cores for parallel computation
)

# Run the evaluation for all specified cluster counts
cs.evaluate()

# Retrieve a DataFrame summarizing the results for each k
results_df = cs.get_results_dataframe()

# Get the k with the highest composite silhouette score
best_k = cs.get_optimal_k()

# Plot the silhouette scores and highlight the best k
cs.plot_results()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/semoglou/composite_silhouette/main/results/plot_example.png" alt="Composite Silhouette Plot" width="600"/>
</p>

### Evaluate a Single Cluster Count

```python
cs = CompSil(
    data=X,
    k_values=4
)

cs.evaluate()

# Access the final composite silhouette score directly
score = cs.score_

# Optionally, still access the full results DataFrame
results = cs.get_results_dataframe()
```

## Examples and Notebooks

Additional usage examples and experimental results can be found in the [`results/`](results/) folder:

- [`example.ipynb`](https://github.com/semoglou/composite_silhouette/blob/main/results/example.ipynb)  
  Basic usage of Composite Silhouette on synthetic data.

- [`performance.ipynb`](https://github.com/semoglou/composite_silhouette/blob/main/results/performance.ipynb)  
  Composite silhouette evaluation results on both synthetic and real-world datasets.

These notebooks provide insight into the method's behavior and demonstrate how to apply it in practical settings.

## License

This project is licensed under the [MIT License](https://github.com/semoglou/composite_silhouette/blob/main/LICENSE).

#

<p align="center"><sub>Composite Silhouette · v0.1.0 · Last updated: 04/2025 · MIT License</sub></p>
