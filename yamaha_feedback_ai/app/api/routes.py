"""FastAPI routes for the Yamaha feedback analysis system."""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import os
from pathlib import Path
from app.utils.logger import logger
from app.utils.config import RAW_DATA_DIR, OUTPUT_DATA_DIR
from app.database import DatabaseManager
from app.preprocessing import clean_raw_data
from app.extraction import extract_entities_from_file
from app.embedding import format_from_file, generate_embeddings
from app.clustering import reduce_embeddings, cluster_embeddings, refine_clusters
from app.labeling import label_clusters

router = APIRouter(prefix="/api", tags=["analysis"])
db = DatabaseManager()


@router.post("/upload-feedback")
async def upload_feedback(file: UploadFile = File(...)):
    """Upload raw motorcycle complaints."""
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        # Save to raw directory
        output_path = RAW_DATA_DIR / file.filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        # Insert into database
        records = [
            (
                row["feedback_id"],
                row["timestamp"],
                row["language"],
                row["vehicle_model"],
                row["domain"],
                row["customer_feedback"],
                row["country"],
                row.get("mileage", 0),
            )
            for _, row in df.iterrows()
        ]
        
        inserted = db.insert_raw_feedback(records)
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows_uploaded": inserted,
            "path": str(output_path),
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def _run_pipeline_task():
    """Internal helper to run the heavy ML pipeline tasks."""
    try:
        raw_files = list(RAW_DATA_DIR.glob("*.csv"))
        if not raw_files:
            logger.error("Background Pipeline: No raw files found.")
            return

        latest_file = max(raw_files, key=os.path.getctime)
        
        # Execute Stages
        df_clean, clean_path = clean_raw_data(str(latest_file))
        entities_path = await extract_entities_from_file(clean_path)
        formatted_path = format_from_file(entities_path)
        embeddings_path, metadata_path = generate_embeddings(formatted_path)
        reduced_path = reduce_embeddings(embeddings_path)
        
        feedback_ids_path = str(Path(embeddings_path).parent / "feedback_ids.npy")
        clustered_path = cluster_embeddings(reduced_path, metadata_path, feedback_ids_path)
        refined_path = refine_clusters(reduced_path, clustered_path)
        
        await label_clusters(refined_path, entities_path, str(latest_file))
        logger.info("Background Pipeline: Completed successfully.")
    except Exception as e:
        logger.error(f"Background Pipeline Failed: {e}")


@router.post("/run-pipeline")
async def run_pipeline(background_tasks: BackgroundTasks):
    """Start the full analysis pipeline."""
    try:
        raw_files = list(RAW_DATA_DIR.glob("*.csv"))
        if not raw_files:
            raise HTTPException(status_code=400, detail="No raw data files found")
        
        # Schedule the heavy work
        background_tasks.add_task(_run_pipeline_task)
        
        return {
            "status": "success",
            "message": "Pipeline execution started in background",
        }
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-clusters")
async def get_clusters():
    """Get cluster data and statistics."""
    try:
        clustered_files = list(OUTPUT_DATA_DIR.glob("*refined.csv"))
        if not clustered_files:
            raise HTTPException(status_code=404, detail="No clustered data found")
        
        latest = max(clustered_files, key=os.path.getctime)
        df = pd.read_csv(latest)
        
        # Compute statistics
        n_clusters = len(set(df["cluster_id"]) - {-1})
        n_noise = len(df[df["cluster_id"] == -1])
        
        return {
            "clusters": n_clusters,
            "noise_points": n_noise,
            "total_points": len(df),
            "file": str(latest),
            "preview": df.head(10).to_dict("records"),
        }
    
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-analytics")
async def get_analytics():
    """Get analytics and insights."""
    try:
        labels_files = list(OUTPUT_DATA_DIR.glob("*labels.csv"))
        if not labels_files:
            raise HTTPException(status_code=404, detail="No labels found")
        
        latest = max(labels_files, key=os.path.getctime)
        df_labels = pd.read_csv(latest)
        
        # Top failure patterns
        top_failures = (
            df_labels.nlargest(10, "failure_frequency")[
                ["label", "failure_frequency", "root_component", "recurring_symptom"]
            ]
            .to_dict("records")
        )
        
        # Component breakdown
        components = df_labels["root_component"].value_counts().head(5).to_dict()
        
        return {
            "top_failures": top_failures,
            "components": components,
            "total_clusters": len(df_labels),
            "total_failures": df_labels["failure_frequency"].sum(),
        }
    
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-feedback")
async def search_feedback(query: str = None, cluster_id: int = None):
    """Search feedback records with filtering."""
    try:
        clustered_files = list(OUTPUT_DATA_DIR.glob("*refined.csv"))
        if not clustered_files:
            raise HTTPException(status_code=404, detail="No data found")
        
        latest = max(clustered_files, key=os.path.getctime)
        df_clusters = pd.read_csv(latest)
        
        # Filter by cluster if provided
        if cluster_id is not None:
            df_clusters = df_clusters[df_clusters["cluster_id"] == cluster_id]

        feedback_ids = df_clusters["feedback_id"].tolist()
        
        # Fetch details from DB
        target_ids = feedback_ids[:50]
        placeholder = ', '.join(['?'] * len(target_ids))
        sql = f"SELECT * FROM raw_feedback WHERE feedback_id IN ({placeholder})"
        
        params = list(target_ids)
        if query:
            sql += " AND customer_feedback LIKE ?"
            params.append(f"%{query}%")
            
        raw_records = db.execute_query(sql, tuple(params))
        results = []
        for rec in raw_records:
            results.append(dict(rec))
        
        return {"count": len(results), "results": results}
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-report")
async def export_report():
    """Export analysis report as CSV."""
    try:
        labels_files = list(OUTPUT_DATA_DIR.glob("*labels.csv"))
        if not labels_files:
            raise HTTPException(status_code=404, detail="No report available")
        
        latest = max(labels_files, key=os.path.getctime)
        
        return FileResponse(
            path=latest,
            media_type="text/csv",
            filename="yamaha_analysis_report.csv"
        )
    
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Yamaha Feedback Analysis"}
