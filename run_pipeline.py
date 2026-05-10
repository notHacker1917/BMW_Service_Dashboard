"""Main pipeline orchestrator."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "yamaha_feedback_ai"))

import asyncio
import pandas as pd
from app.utils.logger import logger
from app.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR
from app.database import DatabaseManager
from generate_data import generate_synthetic_data, save_raw_data
from app.preprocessing import clean_raw_data
from app.extraction import extract_entities_from_file
from app.embedding import format_from_file, generate_embeddings
from app.clustering import reduce_embeddings, cluster_embeddings, refine_clusters
from app.labeling import label_clusters

# Create data directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


async def run_full_pipeline():
    """Execute complete ML pipeline."""
    logger.info("=" * 80)
    logger.info("YAMAHA FEEDBACK ANALYSIS PIPELINE - FULL EXECUTION")
    logger.info("=" * 80)
    
    # Stage 0: Check for existing data or generate synthetic
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment. Please set it in .env file.")
        logger.info("The pipeline requires an OpenAI API key for entity extraction and labeling.")
        return

    existing_files = list(RAW_DATA_DIR.glob("*.csv"))
    if existing_files:
        raw_data_path = str(max(existing_files, key=os.path.getctime))
        logger.info(f"\n[Stage 0] Using existing raw data: {raw_data_path}")
        
        # Normalize column names for external datasets (like automotive_complaints_300.csv)
        df_raw = pd.read_csv(raw_data_path)
        mapping = {
            'complaint_id': 'feedback_id',
            'complaint_text': 'customer_feedback',
            'mileage_km': 'mileage'
        }
        df_raw = df_raw.rename(columns={k: v for k, v in mapping.items() if k in df_raw.columns})
        df_raw.to_csv(raw_data_path, index=False)
        logger.info(f"✓ Normalized {len(df_raw)} records for pipeline compatibility")
        logger.info("-" * 80)
    else:
        logger.info("\n[Stage 0] Generating Synthetic Data")
        logger.info("-" * 80)
        df = generate_synthetic_data(5000)
        raw_data_path = save_raw_data(df)
        logger.info(f"✓ Generated 5000 synthetic complaint records")
    
    # Initialize database
    logger.info("\n[Setup] Initializing Database")
    logger.info("-" * 80)
    db = DatabaseManager()

    # Populate database with raw feedback for the API and dashboard to function correctly
    df_to_db = pd.read_csv(raw_data_path)
    records = [
        (
            str(row["feedback_id"]),
            str(row["timestamp"]),
            str(row["language"]),
            str(row["vehicle_model"]),
            str(row["domain"]),
            str(row["customer_feedback"]),
            str(row["country"]),
            int(row["mileage"]) if pd.notnull(row.get("mileage")) else 0,
        )
        for _, row in df_to_db.iterrows()
    ]
    
    try:
        db.insert_raw_feedback(records)
        logger.info(f"✓ Database initialized and populated with {len(records)} records")
    except Exception as e:
        logger.warning(f"✓ Database initialization complete (Note: {e})")
    
    # Stage 1: Data Preprocessing
    logger.info("\n[Stage 1] Data Preprocessing & Cleaning")
    logger.info("-" * 80)
    clean_path = str(PROCESSED_DATA_DIR / f"{Path(raw_data_path).stem}_cleaned.csv")
    if not Path(clean_path).exists():
        df_clean, clean_path = clean_raw_data(raw_data_path)
        logger.info(f"✓ Cleaned {len(df_clean)} records")
    else:
        logger.info(f"✓ Skipping Stage 1: Cleaned data already exists at {clean_path}")
    
    # Stage 2: Entity Extraction
    logger.info("\n[Stage 2] GPT-Based Entity Extraction")
    logger.info("-" * 80)
    entities_path = str(Path(clean_path).parent / Path(clean_path).name.replace("_cleaned", "_entities"))
    if not Path(entities_path).exists():
        entities_path = await extract_entities_from_file(clean_path)
        logger.info(f"✓ Extracted entities from cleaned feedback")
    else:
        logger.info(f"✓ Skipping Stage 2: Entities already exist at {entities_path}")
    
    # Stage 3: Entity Formatting
    logger.info("\n[Stage 3] Entity Formatting for Embeddings")
    logger.info("-" * 80)
    formatted_path = str(Path(entities_path).parent / Path(entities_path).name.replace("_entities", "_formatted"))
    if not Path(formatted_path).exists():
        formatted_path = format_from_file(entities_path)
        logger.info(f"✓ Formatted entities for semantic processing")
    else:
        logger.info(f"✓ Skipping Stage 3: Formatted data already exists")
    
    # Stage 4: Embedding Generation
    logger.info("\n[Stage 4] Semantic Embedding Generation")
    logger.info("-" * 80)
    embeddings_path = str(PROCESSED_DATA_DIR / "embeddings.npy")
    metadata_path = str(PROCESSED_DATA_DIR / "embeddings_metadata.csv")
    if not Path(embeddings_path).exists():
        embeddings_path, metadata_path = generate_embeddings(formatted_path)
        logger.info(f"✓ Generated embeddings with BGE model")
    else:
        logger.info(f"✓ Skipping Stage 4: Embeddings already exist")
    
    # Stage 5: Dimensionality Reduction
    logger.info("\n[Stage 5] UMAP Dimensionality Reduction")
    logger.info("-" * 80)
    reduced_path = str(PROCESSED_DATA_DIR / "embeddings_umap.npy")
    if not Path(reduced_path).exists():
        reduced_path = reduce_embeddings(embeddings_path)
        logger.info(f"✓ Reduced embeddings to 15 dimensions")
    else:
        logger.info(f"✓ Skipping Stage 5: UMAP reduction already exists")
    
    # Stage 6: HDBSCAN Clustering
    logger.info("\n[Stage 6] HDBSCAN Clustering")
    logger.info("-" * 80)
    feedback_ids_path = str(Path(embeddings_path).parent / "feedback_ids.npy")
    clustered_path = cluster_embeddings(
        reduced_path,
        metadata_path,
        feedback_ids_path,
    )
    logger.info(f"✓ Clustered feedback into semantic groups")
    
    # Stage 7: Cluster Refinement
    logger.info("\n[Stage 7] Cluster Refinement & Merging")
    logger.info("-" * 80)
    refined_path = refine_clusters(reduced_path, clustered_path)
    logger.info(f"✓ Refined clusters by merging similar patterns")
    
    # Stage 8: Cluster Labeling
    logger.info("\n[Stage 8] Cluster Label Generation")
    logger.info("-" * 80)
    labels_path = await label_clusters(refined_path, entities_path, clean_path)
    logger.info(f"✓ Generated cluster labels with TF-IDF + GPT")
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    # Terminal Visual Result
    try:
        df_report = pd.read_csv(labels_path)
        print("\n" + " " * 25 + "📋 ANALYSIS SUMMARY REPORT")
        print("=" * 80)
        print(df_report[['cluster_id', 'label', 'failure_frequency', 'root_component']].to_string(index=False))
        print("=" * 80 + "\n")
    except Exception as e:
        logger.error(f"Could not generate terminal report: {e}")

    logger.info(f"\nOutput files:")
    logger.info(f"  - Raw data: {raw_data_path}")
    logger.info(f"  - Cleaned data: {clean_path}")
    logger.info(f"  - Entities: {entities_path}")
    logger.info(f"  - Embeddings: {embeddings_path}")
    logger.info(f"  - Clusters: {refined_path}")
    logger.info(f"  - Labels: {labels_path}")
    logger.info(f"\nStart Streamlit dashboard:")
    logger.info(f"  streamlit run yamaha_feedback_ai/app/dashboard/app.py")
    logger.info(f"\nStart FastAPI server:")
    logger.info(f"  uvicorn yamaha_feedback_ai.app.api.main:app --reload")


if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
