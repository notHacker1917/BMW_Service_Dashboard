"""UMAP dimensionality reduction."""
import numpy as np
from pathlib import Path
from typing import Tuple
import umap
from app.utils.logger import logger
from app.utils.config import (
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_METRIC,
    PROCESSED_DATA_DIR,
)


class UMAPReducer:
    def __init__(
        self,
        n_neighbors: int = UMAP_N_NEIGHBORS,
        min_dist: float = UMAP_MIN_DIST,
        n_components: int = UMAP_N_COMPONENTS,
        metric: str = UMAP_METRIC,
    ):
        logger.info(f"Initializing UMAP with params: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}")
        
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
            verbose=True,
        )

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality of embeddings."""
        logger.info(f"Reducing embeddings from {embeddings.shape[1]} to {self.reducer.n_components} dimensions...")
        
        reduced = self.reducer.fit_transform(embeddings)
        logger.info(f"Reduction complete. Output shape: {reduced.shape}")
        
        return reduced


def reduce_embeddings(embeddings_path: str, output_dir: str = None) -> str:
    """Load embeddings and perform UMAP reduction."""
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    reducer = UMAPReducer()
    reduced = reducer.reduce(embeddings)
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reduced embeddings
    reduced_path = output_dir / "embeddings_umap.npy"
    np.save(reduced_path, reduced)
    logger.info(f"Saved UMAP-reduced embeddings to {reduced_path}")
    
    return str(reduced_path)
