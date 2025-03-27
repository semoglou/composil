# Composite Silhouette 

[![PyPI Version](https://img.shields.io/pypi/v/composite-silhouette?logo=pypi)](https://pypi.org/project/composite-silhouette/)

Composite Silhouette is a Python package for robust clustering evaluation.
It introduces a composite metric that combines the traditional silhouette score (micro-average across all samples) with the macro-averaged silhouette (averaged per cluster) using statistical weighting.
This provides a more nuanced assessment of clustering quality, helping identify the optimal number of clusters and compare performance across different clustering scenarios with greater confidence.
The framework is especially useful for data scientists, ML engineers, and researchers who want reliable metrics for centroid-based clustering.

## Overview and Methodology 
In standard clustering evaluation, the **silhouette coefficient** is widely used to measure how well each data point fits into its cluster in terms of intra-cluster cohesion and inter-cluster separation.
It can be aggregated as:
- **Micro-average:** The overall average silhouette score across all data points.
- **Macro-average:** The per-cluster average silhouette score.
**Composite Silhouette** merges these two perspectives using a statistically-driven weighting strategy.
