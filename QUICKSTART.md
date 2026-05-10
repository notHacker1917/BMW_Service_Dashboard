# 🚀 Quick Start Guide - Yamaha Dashboard

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key
Edit `.env`:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 3: Verify Environment
```bash
python verify_env.py
```

Expected output:
```
✓ Python 3.11+
✓ All dependencies
✓ Directory structure
✓ Environment ready!
```

### Step 4: Run Full Pipeline
```bash
python run_pipeline.py
```

This will:
1. Generate 5000 synthetic multilingual complaints
2. Process through all 9 ML pipeline stages
3. Create 10-15 semantic clusters
4. Generate engineering-readable labels

**Expected duration:** 8-12 minutes

### Step 5: View Results

**Option A - Dashboard (Recommended)**
```bash
streamlit run yamaha_feedback_ai/app/dashboard/app.py
```
Opens at: http://localhost:8501

**Option B - API Server**
```bash
uvicorn yamaha_feedback_ai.app.api.main:app --reload
```
API at: http://localhost:8000

---

## What Gets Generated?

```
yamaha_feedback_ai/
  data/
    raw/
      └── yamaha_feedback.csv (5000 complaints)
    processed/
      ├── yamaha_feedback_cleaned.csv
      ├── yamaha_feedback_entities.csv
      ├── yamaha_feedback_formatted.csv
      ├── embeddings.npy (5000 × 1024)
      ├── embeddings_umap.npy (5000 × 15)
      └── feedback_ids.npy
    outputs/
      ├── clustered_feedback_refined.csv (with cluster IDs)
      ├── cluster_info.csv (cluster statistics)
      └── yamaha_feedback_cluster_labels.csv (labeled clusters)
  database/
    └── feedback.db (SQLite)
  logs/
    └── yamaha_ai.log
```

---

## Dashboard Pages

### 1. 📊 Overview
- Key metrics (total issues, clusters, noise%)
- Top 10 failure clusters by frequency
- Component distribution (pie chart)
- Semantic landscape visualization (UMAP scatter)

### 2. 🔍 Cluster Explorer
- Select failure pattern
- View symptom description
- See representative complaints
- Browse all complaints in cluster

### 3. 📈 Failure Analytics
- Severity distribution
- Geographic breakdown by country
- Language distribution
- Vehicle model breakdown

### 4. 🔎 Search
- Filter by vehicle, language, domain, country
- View raw complaint text
- Download filtered dataset

### 5. 📥 Export
- Download full cluster analysis as CSV
- Preview all clusters with failure frequency

---

## API Endpoints (Quick Reference)

```bash
# Upload complaints
curl -X POST "http://localhost:8000/api/upload-feedback" \
  -F "file=@complaints.csv"

# Get clusters
curl "http://localhost:8000/api/get-clusters"

# Get analytics
curl "http://localhost:8000/api/get-analytics"

# Search complaints
curl "http://localhost:8000/api/search-feedback?query=display"

# Export report
curl "http://localhost:8000/api/export-report" -o report.csv

# Health check
curl "http://localhost:8000/api/health"
```

---

## Troubleshooting

**Q: "OPENAI_API_KEY not set"**
A: Add real API key to `.env` file

**Q: "No module named 'torch'"**
A: Run `pip install -r requirements.txt` again

**Q: "Streamlit not found"**
A: Ensure virtual environment is activated

**Q: "Slow embedding generation"**
A: Use GPU - set `DEVICE=cuda` in config.py

**Q: "Port 8501 already in use"**
A: Use different port: `streamlit run ... --server.port 8502`

---

## Performance Tips

1. **GPU Acceleration** (10x faster embeddings)
   - Set `DEVICE=cuda` in `yamaha_feedback_ai/app/utils/config.py`
   - Requires: CUDA 11.8+, cuDNN

2. **Smaller Dataset** (for testing)
   - In `run_pipeline.py`: `generate_synthetic_data(1000)` instead of 5000

3. **Parallel Processing** (future enhancement)
   - Currently single-threaded
   - Can be parallelized per stage

---

## Expected Results

After running the full pipeline, you should see:

**Typical Cluster Examples:**
- "Engine Overheating at Highway Speeds" (45 complaints)
- "Display Freezing During Navigation" (38 complaints)  
- "Bluetooth Connection Loss" (32 complaints)
- "Clutch Slipping in City Traffic" (28 complaints)
- "Fuel Sensor Malfunction" (24 complaints)

**Dashboard Metrics:**
- Total Issues: ~4,800 (after cleaning)
- Clusters Found: 10-15
- Noise Points: 250-350 (5-10%)
- Avg Cluster Size: 300-400

**Languages Detected:**
- English: ~15%
- German: ~15%
- Danish: ~12%
- Polish: ~12%
- French: ~12%
- Italian: ~12%
- Spanish: ~12%
- Dutch: ~12%

---

## Next Steps

After successfully running the pipeline:

1. **Analyze Clusters** - Use dashboard to explore failure patterns
2. **Export Reports** - Download cluster analysis for engineering teams
3. **Refine Labels** - Manually adjust cluster labels if needed
4. **Set Alerts** - Configure thresholds for critical failures
5. **Integrate** - Connect to production feedback intake systems

---

## Support Files

- `README.md` - Full documentation
- `verify_env.py` - Environment checker
- `QUICKSTART.md` - This file
- `.env` - Configuration template

---

**Status:** ✅ Ready to deploy | **Version:** 1.0.0 | **Python:** 3.11+
