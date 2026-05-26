"""HDBSCAN clustering."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import hdbscan
from app.utils.logger import logger
from app.utils.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_METRIC,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    PROCESSED_DATA_DIR,
)
 

class HDBSCANClustering:
    def __init__(
        self,
        min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
        metric: str = HDBSCAN_METRIC,
        cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
    ):
        logger.info(
            f"Initializing HDBSCAN with params: min_cluster_size={min_cluster_size}, "
            f"metric={metric}, cluster_selection_method={cluster_selection_method}"
        )
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
        )
        self.noise_indices = []
        self.cluster_info = {}

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering."""
        logger.info(f"Clustering {len(embeddings)} embeddings...")
        
        labels = self.clusterer.fit_predict(embeddings)
        
        # Extract cluster information
        self.noise_indices = np.where(labels == -1)[0]
        unique_clusters = set(labels) - {-1}
        
        logger.info(f"Clustering complete:")
        logger.info(f"  - Found {len(unique_clusters)} clusters")
        logger.info(f"  - {len(self.noise_indices)} noise points ({len(self.noise_indices)/len(labels)*100:.1f}%)")
        
        # Compute cluster sizes and confidence
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Get cluster stability/confidence
            stability = self.clusterer.cluster_persistence_.get(cluster_id, 0.0)
            
            self.cluster_info[cluster_id] = {
                "size": cluster_size,
                "stability": stability,
                "percentage": cluster_size / len(labels) * 100,
            }
        
        # Log cluster info
        for cluster_id, info in sorted(self.cluster_info.items()):
            logger.info(f"  Cluster {cluster_id}: {info['size']} points ({info['percentage']:.1f}%), stability={info['stability']:.3f}")
        
        return labels
 
    def get_representative_samples(self, labels: np.ndarray, n_samples: int = 3) -> Dict[int, List[int]]:
        """Get representative sample indices for each cluster."""
        representatives = {}
        
        for cluster_id in set(labels) - {-1}:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            # Select samples spread across the cluster
            if len(cluster_indices) <= n_samples:
                representatives[cluster_id] = cluster_indices.tolist()
            else:
                step = len(cluster_indices) // n_samples
                representatives[cluster_id] = cluster_indices[::step][:n_samples].tolist()
        
        return representatives


def cluster_embeddings(
    reduced_embeddings_path: str,
    embeddings_metadata_path: str,
    feedback_ids_path: str,
    output_dir: str = None,
) -> str:
    """Load reduced embeddings and perform HDBSCAN clustering."""
    logger.info(f"Loading reduced embeddings from {reduced_embeddings_path}")
    
    embeddings = np.load(reduced_embeddings_path)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Load metadata
    metadata = pd.read_csv(embeddings_metadata_path, encoding="utf-8")
    feedback_ids = np.load(feedback_ids_path)
    
    # Perform clustering
    clustering = HDBSCANClustering()
    labels = clustering.cluster(embeddings)
    
    # Get representative samples
    representatives = clustering.get_representative_samples(labels)
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create clustered dataframe
    df_clustered = pd.DataFrame({
        "feedback_id": feedback_ids,
        "cluster_id": labels,
        "cluster_confidence": clustering.clusterer.probabilities_,
        "umap_x": embeddings[:, 0],
        "umap_y": embeddings[:, 1],
        "is_noise": labels == -1,
    })
    
    # Save clustered data
    clustered_path = output_dir / "clustered_feedback.csv"
    df_clustered.to_csv(clustered_path, index=False, encoding="utf-8")
    logger.info(f"Saved clustered feedback to {clustered_path}")
    
    # Save cluster info
    cluster_info_df = pd.DataFrame([
        {
            "cluster_id": cluster_id,
            "size": info["size"],
            "percentage": info["percentage"],
            "stability": info["stability"],
        }
        for cluster_id, info in clustering.cluster_info.items()
    ])
    
    cluster_info_path = output_dir / "cluster_info.csv"
    cluster_info_df.to_csv(cluster_info_path, index=False, encoding="utf-8")
    logger.info(f"Saved cluster info to {cluster_info_path}")
    
    return str(clustered_path)
