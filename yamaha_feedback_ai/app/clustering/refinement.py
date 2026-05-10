"""Cluster refinement through semantic similarity merging."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Set
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.logger import logger
from app.utils.config import CLUSTER_SIMILARITY_THRESHOLD, PROCESSED_DATA_DIR


class ClusterRefiner:
    def __init__(self, similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.merge_map = {}

    def compute_cluster_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute centroid for each cluster."""
        centroids = {}
        
        for cluster_id in set(labels) - {-1}:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids[cluster_id] = centroid
        
        return centroids

    def find_similar_clusters(self, centroids: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Find similar clusters using cosine similarity."""
        logger.info(f"Finding similar clusters with threshold {self.similarity_threshold}...")
        
        cluster_ids = sorted(centroids.keys())
        n_clusters = len(cluster_ids)
        
        # Compute pairwise similarities
        centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])
        similarities = cosine_similarity(centroid_matrix)
        
        # Find clusters to merge
        merge_map = {}
        merged_clusters: Set[int] = set()
        
        for i in range(n_clusters):
            if cluster_ids[i] in merged_clusters:
                continue
            
            for j in range(i + 1, n_clusters):
                if cluster_ids[j] in merged_clusters:
                    continue
                
                similarity = similarities[i, j]
                if similarity >= self.similarity_threshold:
                    logger.info(f"Merging cluster {cluster_ids[j]} into {cluster_ids[i]} (similarity: {similarity:.3f})")
                    merge_map[cluster_ids[j]] = cluster_ids[i]
                    merged_clusters.add(cluster_ids[j])
        
        self.merge_map = merge_map
        logger.info(f"Found {len(merge_map)} cluster pairs to merge")
        
        return merge_map

    def apply_merges(self, labels: np.ndarray) -> np.ndarray:
        """Apply cluster merges to labels."""
        new_labels = labels.copy()
        
        for old_cluster_id, new_cluster_id in self.merge_map.items():
            new_labels[labels == old_cluster_id] = new_cluster_id
        
        return new_labels


def refine_clusters(
    reduced_embeddings_path: str,
    clustered_feedback_path: str,
    output_dir: str = None,
) -> str:
    """Refine clusters by merging similar ones."""
    logger.info("Starting cluster refinement...")
    
    # Load data
    embeddings = np.load(reduced_embeddings_path)
    df_clustered = pd.read_csv(clustered_feedback_path, encoding="utf-8")
    
    labels = df_clustered["cluster_id"].values
    original_clusters = len(set(labels) - {-1})
    
    # Perform refinement
    refiner = ClusterRefiner()
    centroids = refiner.compute_cluster_centroids(embeddings, labels)
    refiner.find_similar_clusters(centroids)
    refined_labels = refiner.apply_merges(labels)
    
    # Update dataframe
    df_clustered["cluster_id"] = refined_labels
    refined_clusters = len(set(refined_labels) - {-1})
    
    logger.info(f"Cluster refinement complete: {original_clusters} -> {refined_clusters} clusters")
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_dir = Path(output_dir)
    refined_path = output_dir / "clustered_feedback_refined.csv"
    df_clustered.to_csv(refined_path, index=False, encoding="utf-8")
    logger.info(f"Saved refined clusters to {refined_path}")
    
    return str(refined_path)
