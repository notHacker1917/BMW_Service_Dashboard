"""Embedding generation using sentence-transformers."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger
from app.utils.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, DEVICE, PROCESSED_DATA_DIR


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: list) -> np.ndarray:
        """Generate embeddings for texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_from_dataframe(self, df: pd.DataFrame, text_column: str = "formatted_text") -> Tuple[np.ndarray, list]:
        """Generate embeddings from DataFrame."""
        texts = df[text_column].tolist()
        embeddings = self.embed_texts(texts)
        feedback_ids = df["feedback_id"].tolist()
        return embeddings, feedback_ids


def generate_embeddings(input_path: str, output_dir: str = None) -> Tuple[str, str]:
    """Generate and save embeddings from formatted entities."""
    logger.info(f"Loading formatted entities from {input_path}")
    
    df = pd.read_csv(input_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} records")
    
    embedder = Embedder()
    embeddings, feedback_ids = embedder.embed_from_dataframe(df)
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # Save feedback_ids for reference
    ids_path = output_dir / "feedback_ids.npy"
    np.save(ids_path, np.array(feedback_ids))
    logger.info(f"Saved feedback IDs to {ids_path}")
    
    # Save metadata
    metadata = pd.DataFrame({
        "feedback_id": feedback_ids,
        "embedding_index": range(len(feedback_ids)),
        "dimension": embedder.embedding_dim,
    })
    
    metadata_path = output_dir / "embeddings_metadata.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")
    logger.info(f"Saved metadata to {metadata_path}")
    
    return str(embeddings_path), str(metadata_path)
