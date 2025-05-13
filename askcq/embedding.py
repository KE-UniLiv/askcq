"""
CQ Embedding Analysis Module
The module provides functions to analyse and visualise the diversity and
coverage of CQ embeddings from different sets. It includes functions for:
- Internal diversity metrics (avg pairwise cosine similarity, avg distance to centroid).
- Shannon entropy calculation based on k-means clustering.
- Coverage analysis between different sets of embeddings.
- PCA visualisation of embeddings.
"""
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as shannon_entropy_calc # For Shannon entropy
from sklearn.cluster import KMeans # For Shannon entropy via discretization

import warnings

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", module="matplotlib\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn") # For n_init in KMeans


def get_set_data(df, set_id, embed_dim=512):
    """Extracts CQs and their embeddings for a specific set."""
    set_df = df[df['set'] == set_id]
    cqs = set_df['cq'].tolist()
    
    # Ensure embeddings are stacked correctly into a 2D numpy array
    embeddings_list = set_df['embedding'].tolist()
    if not embeddings_list:
        return cqs, np.array([])
    
    # If embeddings are stored as lists, convert them; if already arrays, vstack works
    try:
        embeddings = np.array(embeddings_list)
        if embeddings.ndim == 1: # Potentially a list of lists/arrays that needs vstack
             embeddings = np.vstack(embeddings_list)
    except TypeError: # Handle cases where embeddings might be in a different format
        embeddings = np.array([np.array(e) for e in embeddings_list])

    if embeddings.ndim != 2 or embeddings.shape[1] != embed_dim: # Basic check
        # This might happen if some embeddings are empty or malformed
        # For simplicity, we'll filter out malformed ones here if any, or raise error
        valid_embeddings = [e for e in embeddings_list if hasattr(e, 'shape') and e.shape == (embed_dim,)]
        if not valid_embeddings: return cqs, np.array([])
        embeddings = np.vstack(valid_embeddings)
        # Note: cqs list would need to be filtered accordingly if we drop embeddings here.
        # For now, assume embeddings are mostly clean.

    return cqs, embeddings

def calculate_internal_diversity(embeddings, set_name):
    """
    Calculates internal diversity metrics for a set of embeddings.
    Returns means and standard deviations.
    """
    num_embeddings = embeddings.shape[0]
    results = {
        "num_cqs": num_embeddings,
        "avg_pairwise_cosine_similarity": np.nan, "std_pairwise_cosine_similarity": np.nan,
        "avg_dist_to_centroid": np.nan, "std_dist_to_centroid": np.nan
    }

    if num_embeddings < 2:
        print(f"\n--- Internal Diversity of {set_name} ---")
        print(f"  Not enough embeddings (found {num_embeddings}) to calculate diversity.")
        return results

    # 1. Pairwise Cosine Similarity
    cosine_sim_matrix = cosine_similarity(embeddings)
    upper_triangle_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
    if upper_triangle_indices[0].size > 0:
        pairwise_similarities = cosine_sim_matrix[upper_triangle_indices]
        results["avg_pairwise_cosine_similarity"] = np.mean(pairwise_similarities)
        results["std_pairwise_cosine_similarity"] = np.std(pairwise_similarities)
    else: # Only one unique pair possible, no std dev or only one embedding
        if num_embeddings == 2: # Only one similarity value
             results["avg_pairwise_cosine_similarity"] = cosine_sim_matrix[0,1]
             results["std_pairwise_cosine_similarity"] = 0.0 # Or np.nan if preferred for single value
        # else: avg remains NaN

    # 2. Euclidean Distance to Centroid
    centroid = np.mean(embeddings, axis=0)
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    results["avg_dist_to_centroid"] = np.mean(distances_to_centroid)
    results["std_dist_to_centroid"] = np.std(distances_to_centroid)

    print(f"\n--- Internal Diversity of {set_name} ---")
    print(f"  Number of CQs: {num_embeddings}")
    print(f"  Avg Pairwise Cosine Similarity: {results['avg_pairwise_cosine_similarity']:.4f} (Std: {results['std_pairwise_cosine_similarity']:.4f})")
    print(f"  Avg Euclidean Distance to Centroid: {results['avg_dist_to_centroid']:.4f} (Std: {results['std_dist_to_centroid']:.4f})")
    
    return results

# (calculate_shannon_entropy_for_set remains the same as you provided)
def calculate_shannon_entropy_for_set(embeddings, set_name, n_clusters):
    """Calculates Shannon entropy based on k-means clustering of embeddings."""
    num_embeddings = embeddings.shape[0]
    if num_embeddings < n_clusters:
        print(f"  Cannot calculate Shannon entropy for {set_name}: needs at least {n_clusters} CQs for {n_clusters} clusters (found {num_embeddings}).")
        return np.nan
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        cluster_labels = kmeans.labels_
        counts = np.bincount(cluster_labels, minlength=n_clusters)
        probabilities = counts / num_embeddings
        probabilities = probabilities[probabilities > 0]
        shannon_ent = shannon_entropy_calc(probabilities, base=2)
        print(f"  Shannon Entropy (k={n_clusters}, bits): {shannon_ent:.4f}")
        return shannon_ent
    except Exception as e:
        print(f"  Error calculating Shannon entropy for {set_name}: {e}")
        return np.nan

# (calculate_centroid_similarity remains the same as you provided)
def calculate_centroid_similarity(embeddings1, embeddings2):
    """Calculates cosine similarity between the centroids of two embedding sets."""
    if embeddings1.size == 0 or embeddings2.size == 0:
        return np.nan
    centroid1 = np.mean(embeddings1, axis=0, keepdims=True)
    centroid2 = np.mean(embeddings2, axis=0, keepdims=True)
    return cosine_similarity(centroid1, centroid2)[0][0]


def analyze_set_coverage(
    cqs_covered, embeddings_covered,
    embeddings_covering, threshold,
    set_name_covered, set_name_covering
):
    """
    Analyzes how well `embeddings_covering` cover `embeddings_covered`.
    Returns metrics including novel CQs and std dev of max similarities.
    """
    num_cqs_covered = embeddings_covered.shape[0]
    results = {
        "mean_max_similarity": np.nan, "std_max_similarity": np.nan, "median_max_similarity": np.nan,
        "num_covered": 0, "percentage_covered": 0.0,
        "num_novel": num_cqs_covered, "percentage_novel": 100.0 if num_cqs_covered > 0 else 0.0,
        "novel_cqs_indices": list(range(num_cqs_covered)),
        "novel_cqs_text": list(cqs_covered) # All are novel initially
    }

    if num_cqs_covered == 0:
        print(f"Cannot analyze coverage for {set_name_covered}: no embeddings.")
        # Return default results indicating no coverage possible / no items to cover
        results["percentage_novel"] = 0.0 # Or NaN, depending on preference for empty sets
        results["novel_cqs_indices"] = []
        results["novel_cqs_text"] = []
        return results

    if embeddings_covering.size == 0:
        print(f"Warning: {set_name_covering} has no embeddings. '{set_name_covered}' CQs are all considered novel w.r.t it.")
        # max_sims_per_item already reflects this (all zeros if we were to calculate it)
        # Results initialized above already reflect this state (all novel)
        results["mean_max_similarity"] = 0.0 # Or np.nan
        results["std_max_similarity"] = 0.0  # Or np.nan
        results["median_max_similarity"] = 0.0 # Or np.nan
    else:
        cross_similarities = cosine_similarity(embeddings_covered, embeddings_covering)
        max_sims_per_item = np.max(cross_similarities, axis=1)

        results["mean_max_similarity"] = np.mean(max_sims_per_item)
        results["std_max_similarity"] = np.std(max_sims_per_item)
        results["median_max_similarity"] = np.median(max_sims_per_item)
        
        covered_indices = np.where(max_sims_per_item >= threshold)[0]
        novel_indices = np.where(max_sims_per_item < threshold)[0]
        
        results["num_covered"] = len(covered_indices)
        results["percentage_covered"] = (results["num_covered"] / num_cqs_covered) * 100 if num_cqs_covered > 0 else 0.0
        results["num_novel"] = len(novel_indices)
        results["percentage_novel"] = (results["num_novel"] / num_cqs_covered) * 100 if num_cqs_covered > 0 else 0.0
        
        results["novel_cqs_indices"] = novel_indices.tolist()
        results["novel_cqs_text"] = [cqs_covered[i] for i in novel_indices]

    print(f"  - Coverage of '{set_name_covered}' by '{set_name_covering}':")
    print(f"    Mean Max Similarity: {results['mean_max_similarity']:.4f} (Std: {results['std_max_similarity']:.4f})")
    print(f"    Median Max Similarity: {results['median_max_similarity']:.4f}")
    print(f"    CQs in '{set_name_covered}' covered (sim >= {threshold}): {results['num_covered']} ({results['percentage_covered']:.2f}%)")
    print(f"    Novel CQs in '{set_name_covered}' (sim < {threshold}): {results['num_novel']} ({results['percentage_novel']:.2f}%)")

    return results

# Corrected visualize_all_sets_pca
def visualize_all_sets_pca(df, set_mapping, n_components=2, embedding_col='embedding', set_col='set'):
    """Visualizes embeddings of all sets using PCA."""
    print(f"\n--- Visualizing All Sets (PCA {n_components}D) ---")
    all_embeddings_list = []
    labels = []
    # Use a consistent order for legend items, corresponding to set_mapping iteration
    legend_order = [name for name in set_mapping.values()]


    for set_id, set_name in set_mapping.items(): # Iterate through items to get both id and name
        _, current_embeddings = get_set_data(df, set_id) # Use set_id to fetch data
        if current_embeddings.size > 0:
            all_embeddings_list.append(current_embeddings)
            labels.extend([set_name] * current_embeddings.shape[0])
        else:
            print(f"Warning: No embeddings for set {set_name} (ID: {set_id}) to visualize.")
            if set_name in legend_order: legend_order.remove(set_name)


    if not all_embeddings_list:
        print("No embeddings found for any set. Skipping visualization.")
        return

    all_embeddings_stack = np.vstack(all_embeddings_list)

    if all_embeddings_stack.shape[0] <= n_components:
        print(f"Warning: Number of total samples ({all_embeddings_stack.shape[0]}) is less than or equal to n_components ({n_components}). Skipping PCA.")
        return

    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(all_embeddings_stack)
    explained_var_ratio = pca.explained_variance_ratio_
    print(f"Explained variance by {n_components} components: {np.sum(explained_var_ratio):.4f}")

    plot_df_data = {'Set': labels}
    plot_df_data['PCA1'] = reduced_embeddings[:, 0]
    if n_components >= 2:
        plot_df_data['PCA2'] = reduced_embeddings[:, 1]
    else: # Handle 1D PCA case
        plot_df_data['PCA2'] = np.zeros(len(reduced_embeddings))


    plot_df = pd.DataFrame(plot_df_data)


    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=plot_df, x='PCA1', y='PCA2', hue='Set', hue_order=legend_order, alpha=0.7, s=50)
    plt.title(f'CQ Embeddings for All Sets (PCA)')
    plt.xlabel(f'Principal Component 1 ({explained_var_ratio[0]:.2%} variance)')
    if n_components >= 2:
        plt.ylabel(f'Principal Component 2 ({explained_var_ratio[1]:.2%} variance)')
    else:
        plt.ylabel('') # No y-axis label for 1D PCA if plotted this way
    plt.legend(title='Set Methodology')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()