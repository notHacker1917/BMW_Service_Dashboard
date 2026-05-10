"""
ARCHITECTURE OVERVIEW - Yamaha Feedback Analysis System

This document explains the end-to-end architecture and data flow.

═══════════════════════════════════════════════════════════════════════════════

1. DATA INGESTION LAYER
─────────────────────────────────────────────────────────────────────────────

INPUT: Multilingual motorcycle complaints
├── Format: CSV with columns
│   ├── feedback_id: Unique identifier
│   ├── timestamp: When reported
│   ├── language: ISO 639-1 code (en, de, da, pl, fr, it, es, nl)
│   ├── vehicle_model: Yamaha model (R15, MT-07, etc.)
│   ├── domain: Issue category (Interior, PowerTrain, Display)
│   ├── customer_feedback: Raw complaint text
│   ├── country: Geographic region
│   └── mileage: Vehicle mileage
│
└── Storage: SQLite database (raw_feedback table)


2. PREPROCESSING LAYER (Stage 1)
─────────────────────────────────────────────────────────────────────────────

GOAL: Clean and normalize raw text

INPUT: raw_feedback.csv (5000 records)

PROCESSING:
├── Remove duplicates (2-3%)
├── Remove null rows (0.5-1%)
├── Normalize whitespace
├── Remove VINs (17-char patterns)
├── Remove emails and phone numbers
├── Remove leading/trailing spaces
└── Ensure UTF-8 encoding (99.9%)

OUTPUT: cleaned_feedback.csv (~4800 records)
└── Stored in: database.cleaned_feedback table


3. ENTITY EXTRACTION LAYER (Stage 2)
─────────────────────────────────────────────────────────────────────────────

GOAL: Extract structured information from unstructured text

INPUT: cleaned_feedback.csv + original language

PROCESSING (GPT-4o-mini):
├── Send text to LLM with system prompt
├── Request JSON extraction:
│   ├── component: affected part (display, engine, clutch)
│   ├── failure_mode: type of failure (freezing, overheating)
│   ├── symptom: observable behavior (navigation crash)
│   ├── severity: critical|high|medium|low
│   └── driving_condition: rainy, highway, city traffic, etc.
│
├── Handle malformed JSON (repair extraction)
├── Track confidence scores
└── Implement rate limiting (5 req/sec) and retries

OUTPUT: structured_entities.csv
├── Each row: feedback_id + 5 structured fields
├── Confidence score: 0.85-0.95
└── Stored in: database.structured_entities table


4. SEMANTIC REPRESENTATION LAYER (Stage 3)
─────────────────────────────────────────────────────────────────────────────

GOAL: Convert entities into semantic text for embeddings

INPUT: structured_entities.csv

PROCESSING:
├── Format as: "Component: X | Failure: Y | Symptom: Z | ..."
├── Reduces multilingual noise
└── Creates consistent input for embedding model

OUTPUT: formatted_entities.csv
└── One formatted_text column per row


5. EMBEDDING GENERATION LAYER (Stage 4)
─────────────────────────────────────────────────────────────────────────────

GOAL: Convert text to semantic vectors

MODEL: BAAI/bge-large-en-v1.5
├── 1024-dimensional embeddings
├── Pre-trained on 430M sentence pairs
├── Optimized for semantic search
└── GPU-compatible

INPUT: formatted_entities.csv (~4800 rows)

PROCESSING:
├── Batch encoding (32 texts at a time)
├── GPU acceleration if available
├── Progress tracking
└── Cache embeddings locally

OUTPUT:
├── embeddings.npy (4800 × 1024 matrix)
├── embeddings_metadata.csv (index, dimension, timestamp)
└── feedback_ids.npy (4800 IDs for reference)


6. DIMENSIONALITY REDUCTION LAYER (Stage 5)
─────────────────────────────────────────────────────────────────────────────

GOAL: Reduce 1024D to 15D while preserving semantic structure

ALGORITHM: UMAP
├── n_neighbors: 15 (local neighborhood size)
├── min_dist: 0.1 (minimum separation)
├── n_components: 15 (output dimensions)
├── metric: cosine (similarity measure)
└── random_state: 42 (reproducibility)

WHY 15D?
├── HDBSCAN clustering works better in moderate dimensions
├── 2D/3D insufficient for complex failure patterns
├── 15D is sweet spot: enough info, fast computation
└── Can visualize first 2-3 dimensions

INPUT: embeddings.npy (4800 × 1024)

OUTPUT: embeddings_umap.npy (4800 × 15)


7. CLUSTERING LAYER (Stage 6)
─────────────────────────────────────────────────────────────────────────────

GOAL: Find groups of similar failure patterns

ALGORITHM: HDBSCAN
├── min_cluster_size: 10 (minimum group size)
├── metric: euclidean (distance measure in 15D space)
├── cluster_selection_method: eom (excess of mass)
└── prediction_data: True (allow new predictions)

WHY HDBSCAN?
├── Density-based (finds natural clusters)
├── Handles noise points (outliers = -1 label)
├── Adaptive clustering (variable cluster sizes)
├── No need to specify number of clusters
└── Good for hierarchical structure

INPUT: embeddings_umap.npy (4800 × 15)

OUTPUT:
├── cluster_id: -1 (noise) or 0,1,2,... (clusters)
├── cluster_confidence: confidence in assignment
├── Typical results: 10-15 clusters, 5-10% noise

STORAGE:
├── clustered_feedback.csv
│   ├── feedback_id
│   ├── cluster_id
│   ├── cluster_confidence
│   ├── umap_x, umap_y (2D projection for visualization)
│   └── is_noise (boolean)
└── cluster_info.csv (statistics per cluster)


8. CLUSTER REFINEMENT LAYER (Stage 7)
─────────────────────────────────────────────────────────────────────────────

GOAL: Merge semantically similar clusters

INPUT: HDBSCAN clusters (e.g., 12 clusters)

PROCESSING:
├── Compute centroid for each cluster
├── Calculate pairwise cosine similarities
├── Find clusters with similarity > 0.90
├── Merge redundant clusters
└── Reassign labels to merged cluster IDs

EXAMPLE:
├── Before: Cluster 3 "Display Lag" vs Cluster 7 "Navigation Freeze"
├── Similarity: 0.92 > threshold 0.90
├── Merge: Cluster 7 → Cluster 3
└── Result: Consolidated as single failure pattern

OUTPUT: clustered_feedback_refined.csv
└── Reduced from 12 to 10 clusters (2 merges)


9. LABELING LAYER (Stage 8)
─────────────────────────────────────────────────────────────────────────────

GOAL: Generate human-readable cluster labels

TWO-STEP PROCESS:

STEP 1: TF-IDF Keyword Extraction
├── Extract top 5 keywords per cluster
├── Keywords: "display", "freezing", "navigation", "crash", "system"
└── Reflects cluster content

STEP 2: GPT Label Generation
├── Send keywords + sample complaints to GPT
├── Request label format: "failure_mode + condition"
├── Example outputs:
│   ├── "Display Freezing During Navigation"
│   ├── "Engine Overheating on Highways"
│   ├── "Bluetooth Disconnect After Rain"
│   └── "Clutch Slipping in City Traffic"
├── Also extract:
│   ├── root_component: "display", "engine", "battery"
│   ├── recurring_symptom: system behavior
│   ├── failure_frequency: number of complaints
│   └── representative_complaint: sample text

OUTPUT: cluster_labels.csv
├── Each row: One cluster with human-readable label
├── Confidence: 0.85-0.95
└── Stored in: database.cluster_labels table


10. VISUALIZATION LAYER (Stage 9)
─────────────────────────────────────────────────────────────────────────────

FRONTEND: Streamlit Dashboard

PAGES:

1. OVERVIEW
   ├── Key Metrics (total issues, clusters, noise %)
   ├── Top 10 failure clusters (bar chart)
   ├── Component distribution (pie chart)
   └── UMAP landscape (scatter plot of all complaints)

2. CLUSTER EXPLORER
   ├── Select failure pattern from dropdown
   ├── Show cluster statistics
   ├── Display symptom description
   ├── List representative complaints
   └── Browse all member complaints

3. FAILURE ANALYTICS
   ├── Severity distribution (critical/high/medium/low)
   ├── Geographic breakdown (by country)
   ├── Language distribution
   └── Vehicle model breakdown

4. SEARCH
   ├── Filter by vehicle, language, domain, country
   ├── Full-text search on complaint text
   └── Download filtered dataset

5. EXPORT
   ├── Download full cluster analysis as CSV
   ├── Preview table with all clusters
   └── Engineering-ready format


11. API LAYER
─────────────────────────────────────────────────────────────────────────────

FastAPI Server with REST endpoints:

POST /api/upload-feedback
├── Accept CSV file upload
├── Insert into database
└── Return row count

POST /api/run-pipeline
├── Trigger full pipeline execution
├── Return file paths and status
└── Support background processing

GET /api/get-clusters
├── Return cluster statistics
├── Include cluster sizes and confidence
└── Preview clustered data

GET /api/get-analytics
├── Top failures by frequency
├── Component breakdown
├── System-wide statistics

GET /api/search-feedback
├── Query complaints
├── Filter by parameters
└── Full-text search

POST /api/export-report
├── Download analysis as CSV
└── Include all cluster labels

GET /api/health
└── System status check


12. STORAGE LAYER
─────────────────────────────────────────────────────────────────────────────

DATABASE: SQLite (feedback.db)

TABLES:
├── raw_feedback: Original complaints with metadata
├── cleaned_feedback: After preprocessing
├── structured_entities: Extracted entities (component, failure, etc.)
├── embeddings: Metadata for 1024D embeddings
├── clusters: HDBSCAN cluster assignments
└── cluster_labels: Generated labels and summaries

FILE STORAGE:

Processed Data (PROCESSED_DATA_DIR):
├── embeddings.npy: Full 1024D vectors
├── embeddings_umap.npy: 15D vectors
├── embeddings_metadata.csv: Index and dimension info
├── feedback_ids.npy: Reference mapping
├── clustered_feedback.csv: Cluster assignments + coordinates
├── clustered_feedback_refined.csv: After merging
└── cluster_info.csv: Cluster statistics

Output/Reports (OUTPUT_DATA_DIR):
└── All CSV files ready for consumption

Logs (LOGS_DIR):
└── yamaha_ai.log: Complete execution trace


13. DATA FLOW SUMMARY
─────────────────────────────────────────────────────────────────────────────

Raw CSV (5000 rows)
    ↓
Preprocessing (remove PII, duplicates) → 4800 rows
    ↓
GPT Entity Extraction (component, failure, severity)
    ↓
Format for Embeddings (semantic text)
    ↓
BAAI Embeddings (1024D vectors)
    ↓
UMAP Reduction (15D)
    ↓
HDBSCAN Clustering (assign to groups)
    ↓
Cluster Refinement (merge similar clusters)
    ↓
TF-IDF + GPT Labeling (human-readable names)
    ↓
Streamlit Dashboard (interactive exploration)
    ↓
FastAPI Endpoints (programmatic access)


14. QUALITY ASSURANCE
─────────────────────────────────────────────────────────────────────────────

VALIDATION CHECKS:

Stage 1 (Preprocessing):
├── UTF-8 encoding valid
├── No null values in critical fields
└── Duplicates removed

Stage 2 (Extraction):
├── JSON is valid (repair if needed)
├── All required fields present
└── Confidence > 0 for valid extractions

Stage 4 (Embeddings):
├── Shape matches expectations (N × 1024)
├── No NaN values
└── Metadata consistency

Stage 6 (Clustering):
├── Cluster IDs contiguous
├── Confidence scores in [0, 1]
├── Coordinates numeric and finite

Stage 8 (Labeling):
├── Labels not empty
├── Components identified
└── Frequency counts positive

═══════════════════════════════════════════════════════════════════════════════
"""

print(__doc__)
