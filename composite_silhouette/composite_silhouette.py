import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import wilcoxon
from sklearn.utils import resample
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class CompSil:
    """
    A clustering evaluation framework that integrates micro- and macro-averaged silhouette
    scores into a composite metric using statistical weighting. This method provides a
    statistically driven assessment of clustering quality by emphasizing the dominant
    aspect of clustering performance while still accounting for the other.
    It performs subsampled clustering by repeatedly applying K-Means—though other algorithms
    could be employed—to random subsets of the data. For each subsampled
    configuration, it computes the micro-averaged silhouette score (S_micro) for overall
    cohesion and the macro-averaged silhouette score (S_macro) for local compactness, along
    with their differences. These paired differences are then used as input for the
    Wilcoxon Signed-Rank Test to determine whether the discrepancy between S_micro and
    S_macro is statistically significant. If so, the weighting scheme assigns more weight
    to the superior score while proportionally incorporating the other based on their
    average difference. If no significance is found, both are weighted equally. By evaluating
    multiple cluster numbers (k), the method selects the optimal k that maximizes the
    composite silhouette score, providing a refined and statistically sound clustering evaluation.

    Parameters:

    - data: ndarray or DataFrame, shape (n_samples, n_features)
        The input dataset to be clustered. Must be numeric and structured as an array-like object.

    - ground_truth: int, optional, default=None
        The known number of clusters, if available. Used only for reference in visualization
        and does not influence the clustering process.

    - k_values: iterable, default=range(2, 11)
        The range of cluster numbers (k) to evaluate. Each value of k will be tested, and
        clustering quality will be assessed for each. If an integer is provided instead of
        an iterable, the method will perform the evaluation only for that specific k value,
        and the computed silhouette score for this k will be stored in self.score_ attribute.

    - num_samples: int, default=100
        Number of subsampling iterations per k. Each iteration randomly selects a subset of
        the data and applies clustering, ensuring robustness in silhouette score estimation.

    - sample_size: int, default=1000
        Number of data points to sample in each iteration.

    - random_state: int, default=42
        Controls randomization in data subsampling and clustering for reproducibility.

    - n_jobs: int, default=-1
        Number of CPU cores used for parallel execution. A value of -1 utilizes all available
        cores, whereas a positive integer limits the computation to a specified number of threads.
    """
    def __init__(self,
                 data,
                 ground_truth=None,
                 k_values=range(2, 11),
                 num_samples=100,
                 sample_size=1000,
                 random_state=42,
                 n_jobs=-1):
        self.data = data # or data.astype(np.float32) for efficiency
        self.ground_truth = ground_truth
        self.k_values = [k_values] if isinstance(k_values, int) else list(k_values)
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Empty list to store results
        self._results = []

        # Placeholder for the results DataFrame
        self.results_df = pd.DataFrame()

        # Attribute to store composite silhouette score for single k case
        self.score_ = None  # Will be assigned only if one k is evaluated

    def SilhouetteMicro(self, X, labels):
        """
        Computes the micro-averaged silhouette score.

        Parameters:

        - X: array-like of shape (n_samples, n_features)
             Input data points.

        - labels: array-like of shape (n_samples,)
                  Cluster labels for each sample.

        Returns:

        - The micro-averaged silhouette score (float).
        """
        return silhouette_score(X, labels)

    def SilhouetteMacro(self, X, labels):
        """
        Computes the macro-averaged silhouette score.

        Parameters:

        - X: array-like of shape (n_samples, n_features)
             Input data points.

        - labels: array-like of shape (n_samples,)
                  Cluster labels for each sample.

        Returns:

         - The macro-averaged silhouette score (float).
        """
        silhouette_vals = silhouette_samples(X, labels)
        unique_labels = np.unique(labels)
        cluster_means = [
            np.mean(silhouette_vals[labels == lbl]) for lbl in unique_labels
        ]
        return np.mean(cluster_means) if cluster_means else 0

    def evaluate_sample(self, k, i):
        """
        Processes one iteration of subsampled clustering for a specified number of clusters.
        It performs data sampling, applies the clustering algorithm with k clusters, and computes both
        the micro-averaged and macro-averaged Silhouette scores. It then returns these scores along with
        their difference (S_micro - S_macro).

        Parameters:

        - k: int
             Number of clusters.

        - i: int
             Iteration index.

        Returns:

        - (smicro, smacro, difference) (tuple of float), where:
            smicro: micro-averaged silhouette score
            smacro: macro-averaged silhouette score
            difference: smicro - smacro
        """
        # Unique random state for each sample
        seed = self.random_state + i

        # Uniformly sampling without replacement
        sampled_data = resample(
            self.data,
            n_samples=self.sample_size,
            replace=False,
            random_state=seed
        )

        # KMeans clustering (n_init=1 as we have many initializations in the st. test already)
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=1)
        labels = kmeans.fit_predict(sampled_data)

        # Compute S_micro (micro-averaged silhouette score)
        smicro = self.SilhouetteMicro(sampled_data, labels)

        # Compute S_macro (macro-averaged silhouette score)
        smacro = self.SilhouetteMacro(sampled_data, labels)

        # Compute difference (paired differences used in wilcoxon test)
        difference = smicro - smacro

        return smicro, smacro, difference

    def evaluate(self):
        """
        Evaluates clustering performance over a range of k values
        (or a single value) using subsampled clustering.
        Performs clustering on multiple subsamples of the dataset, computes
        both micro- and macro-averaged Silhouette scores for each solution, and applies the
        Wilcoxon Signed-Rank Test to determine the statistical significance of the differences
        between these scores. Based on the test outcomes, it computes a composite Silhouette
        score S_mM (S_micro-macro) as a weighted combination of the average scores across all subsamples
        that aggressively emphasizes the "superior" metric.
        All evaluation metrics are then stored in the results_df DataFrame.
        """
        for k in self.k_values:
            # Parallel execution of samples for current k
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.evaluate_sample)(k, i) for i in range(self.num_samples)
            )

            # Unpack results
            smicro_list, smacro_list, differences = zip(*results)
            smicro_list = np.array(smicro_list)
            smacro_list = np.array(smacro_list)
            differences = np.array(differences)

            # Perform Wilcoxon Signed-Rank Test - at least one non-zero difference
            if np.all(differences == 0):
                p_value = 1.0
            else:
                try:
                    stat, p_value = wilcoxon(differences)
                except Exception as e:
                    print(f"Wilcoxon test failed for k={k}: {e}")
                    p_value = 1.0  # If the test cannot be performed

            if p_value <= 0.05:
                significance = '+'
            elif p_value > 0.05:
                significance = '-'
            else:
                significance = 'N/A'

            # Compute average S_micro and S_macro and their difference across all samples
            mean_diff = np.mean(differences)
            avg_smicro = np.mean(smicro_list)
            avg_smacro = np.mean(smacro_list)

            # Clip mean difference in silhouette scale: [-1,1]
            alpha = np.clip(mean_diff, -1.0, 1.0)

            # Map clipped difference to [0.5, 1]
            epsilon = (1+ np.abs(alpha)) / 2 # Weights will map epsilon to [0.75,1]

            if significance == '+':
                if alpha > 0: # S_micro dominates (more weight to Smicro)
                    w_micro = (1 + epsilon) / 2 # in (0.75, 1]
                    w_macro = 1 - w_micro # in [0, 0.25)
                elif alpha < 0: # S_macro dominates (more weight to Smacro)
                    w_macro = (1 + epsilon) / 2
                    w_micro = 1 - w_macro
                else: # st. significant with zero mean difference (equal weights)
                    w_micro = 0.5
                    w_macro = 0.5
            else: # not st. significant result (equal weights)
                w_micro = 0.5
                w_macro = 0.5

            # Weighted (convex) combination of (sample-)average micro and macro
            Sconvex = w_micro * avg_smicro + w_macro * avg_smacro

            # If only one k, store the composite silhouette score in self.score_
            if len(self.k_values) == 1:
                self.score_ = Sconvex

            result = {
                'number of clusters (k)': k,
                'p-value': p_value,
                'St. significance': significance,
                'Mean difference': mean_diff,
                'avg S_micro': avg_smicro,
                'avg S_macro': avg_smacro,
                'w_micro': w_micro,
                'w_macro': w_macro,
                'S_micro-macro': Sconvex
            }

            self._results.append(result)

        self.results_df = pd.DataFrame(self._results)

    def plot_results(self):
        """
        Plot the combined silhouette scores (S_micro-macro)
        and individual average scores (avg S_micro, avg S_macro)
        against the number of clusters (k).
        """
        if self.results_df.empty:
            raise ValueError("No results available. Run the evaluate() method first.")

        # If only one k was evaluated
        if len(self.results_df) == 1:
            raise ValueError("Cannot generate a plot with only one k value. Evaluate multiple k values.")

        # Maximum scores and their corresponding k
        max_smicro = self.results_df['avg S_micro'].max()
        max_smicro_k = self.results_df.loc[self.results_df['avg S_micro'].idxmax(), 'number of clusters (k)']

        max_smacro = self.results_df['avg S_macro'].max()
        max_smacro_k = self.results_df.loc[self.results_df['avg S_macro'].idxmax(), 'number of clusters (k)']

        max_sconvex = self.results_df['S_micro-macro'].max()
        max_sconvex_k = self.results_df.loc[self.results_df['S_micro-macro'].idxmax(), 'number of clusters (k)']

        # Plot S_micro, S_macro, and S_MM vs. k
        plt.figure(figsize=(10, 6))

        # Micro silhouette scores
        plt.plot(
            self.results_df['number of clusters (k)'], self.results_df['avg S_micro'],
            marker='o', linestyle='-', color='orange', label='avg S_micro'
        )
        plt.plot(
            max_smicro_k, max_smicro, marker='*', color='orange', markersize=12,
            label=f'Max avg S_micro (k={int(max_smicro_k)})'
        )

        # Macro silhouette scores
        plt.plot(
            self.results_df['number of clusters (k)'], self.results_df['avg S_macro'],
            marker='o', linestyle='-', color='blue', label='avg S_macro'
        )
        plt.plot(
            max_smacro_k, max_smacro, marker='*', color='blue', markersize=12,
            label=f'Max avg S_macro (k={int(max_smacro_k)})'
        )

        # Combined silhouette scores - S_mM
        plt.plot(
            self.results_df['number of clusters (k)'], self.results_df['S_micro-macro'],
            marker='o', linestyle='--', color='green', label='S_micro-macro'
        )
        plt.plot(
            max_sconvex_k, max_sconvex, marker='*', color='green', markersize=12,
            label=f'Max S_micro-macro (k={int(max_sconvex_k)})'
        )

        # Vert. red line for ground truth if available
        if self.ground_truth is not None:
            plt.axvline(x=self.ground_truth, color='red', linestyle='--', linewidth=1,
                        label=f'Ground Truth (k={self.ground_truth})')

        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.xticks(self.k_values)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_optimal_k(self):
        """
        Retrieve the optimal number of clusters based on the highest S_micro-macro score.

        Returns:
        - optimal_k: int
            The optimal number of clusters.
        """
        if self.results_df.empty:
            raise ValueError("No results available. Run the evaluate() method first.")

        # Only one k was evaluated
        if len(self.results_df) == 1:
            return int(self.results_df['number of clusters (k)'].iloc[0])

        optimal_row = self.results_df.loc[self.results_df['S_micro-macro'].idxmax()]
        optimal_k = int(optimal_row['number of clusters (k)'])
        return optimal_k

    def get_results_dataframe(self):
        """
        Retrieve the results as a Pandas DataFrame.

        Returns:
        - results_df: DataFrame
            The DataFrame containing evaluation metrics for each k.
        """
        if self.results_df.empty:
            raise ValueError("No results available. Run the evaluate() method first.")
        return self.results_df.set_index('number of clusters (k)', inplace=False)