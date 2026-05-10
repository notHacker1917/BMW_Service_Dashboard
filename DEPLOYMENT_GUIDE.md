# Deployment & Production Guide

## Local Development

### Prerequisites
- Python 3.11+
- 4GB+ RAM
- OpenAI API key

### Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python run_pipeline.py
streamlit run yamaha_feedback_ai/app/dashboard/app.py
```

---

## Docker Deployment (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/outputs logs

# Expose ports
EXPOSE 8000 8501

# Default: run dashboard
CMD ["streamlit", "run", "yamaha_feedback_ai/app/dashboard/app.py", "--server.port=8501", "--server.headless=true"]
```

### Build & Run
```bash
docker build -t yamaha-dashboard .
docker run -p 8501:8501 -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY yamaha-dashboard
```

---

## Kubernetes Deployment (Optional)

### Create yamaha-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yamaha-dashboard
  labels:
    app: yamaha-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: yamaha-dashboard
  template:
    metadata:
      labels:
        app: yamaha-dashboard
    spec:
      containers:
      - name: dashboard
        image: yamaha-dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: yamaha-data
---
apiVersion: v1
kind: Service
metadata:
  name: yamaha-dashboard-service
spec:
  selector:
    app: yamaha-dashboard
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

### Deploy
```bash
kubectl create secret generic openai-secret --from-literal=api-key=$OPENAI_API_KEY
kubectl apply -f yamaha-deployment.yaml
```

---

## Production Checklist

- [ ] API key stored in secret manager (not in code)
- [ ] Database backups automated
- [ ] Logging aggregated (Datadog, ELK Stack, etc.)
- [ ] API rate limits enforced
- [ ] CORS configured for specific origins
- [ ] SSL/TLS certificates installed
- [ ] Monitoring and alerts set up
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security audit completed

---

## Performance Optimization

### For Production Scale

1. **Caching**
   ```python
   # Cache embeddings model
   @st.cache_resource
   def get_embedder():
       return Embedder()
   ```

2. **Database Indexing**
   ```sql
   CREATE INDEX idx_cluster_id ON clusters(cluster_id);
   CREATE INDEX idx_feedback_id ON raw_feedback(feedback_id);
   ```

3. **API Rate Limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/api/run-pipeline")
   @limiter.limit("5/hour")
   async def run_pipeline():
       ...
   ```

4. **Batch Processing**
   - Increase EMBEDDING_BATCH_SIZE for more throughput
   - Process complaints in chunks rather than individually

5. **GPU Acceleration**
   - Set DEVICE="cuda" for NVIDIA GPU
   - Reduces embedding time by 10x

---

## Monitoring & Logging

### Key Metrics to Track

```python
# Add to config.py
METRICS_TO_TRACK = [
    "pipeline_execution_time",
    "api_response_time",
    "clustering_accuracy",
    "label_generation_confidence",
    "data_cleaning_rate",
    "entity_extraction_success_rate",
]
```

### Log Aggregation Example (ELK Stack)

```python
from loguru import logger
import json

def setup_elk_logging():
    logger.add(
        "http://elasticsearch:9200/_bulk",
        format=lambda record: json.dumps({
            "timestamp": record["time"],
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
        }),
        sink=ElasticsearchSink(),
    )
```

---

## Scaling Strategies

### Horizontal Scaling
- Run multiple API instances behind load balancer
- Use PostgreSQL instead of SQLite for concurrent writes
- Implement queue system (Redis/RabbitMQ) for async jobs

### Vertical Scaling
- Allocate more CPU/GPU to embedding generation
- Increase batch sizes for throughput
- Use connection pooling for database

### Data Scaling
- Archive old clusters to cold storage
- Partition database by date
- Implement data retention policies

---

## Disaster Recovery

### Backup Strategy
```bash
# Daily backup
0 2 * * * sqlite3 feedback.db ".backup 'backups/feedback_$(date +%Y%m%d).db'"

# Weekly to S3
0 3 * * 0 aws s3 cp backups/ s3://yamaha-backups/ --recursive
```

### Recovery Procedure
```bash
# Restore from backup
sqlite3 feedback.db < backups/feedback_YYYYMMDD.db

# Restore from S3
aws s3 cp s3://yamaha-backups/feedback_YYYYMMDD.db .
```

---

## Security Hardening

### API Security
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    if not verify_jwt(credentials.credentials):
        raise HTTPException(status_code=401)
    return credentials.credentials

@app.get("/api/analytics")
async def get_analytics(token: str = Depends(verify_token)):
    ...
```

### Data Privacy
- Implement data masking for PII
- Encrypt sensitive fields in database
- GDPR compliance for EU data
- Data retention policies

### Input Validation
```python
from pydantic import BaseModel, validator

class FeedbackUpload(BaseModel):
    file_size: int
    file_type: str
    
    @validator('file_size')
    def validate_size(cls, v):
        if v > 100_000_000:  # 100MB
            raise ValueError('File too large')
        return v
```

---

## Common Issues & Solutions

### Issue: High Memory Usage
**Solution:** 
- Reduce EMBEDDING_BATCH_SIZE
- Process data in smaller chunks
- Use streaming for large datasets

### Issue: Slow API Response
**Solution:**
- Add Redis caching layer
- Pre-compute common queries
- Increase number of API instances

### Issue: Embedding Generation Timeout
**Solution:**
- Use GPU acceleration
- Reduce embedding model size
- Implement timeout and retry logic

### Issue: Database Locks
**Solution:**
- Switch to PostgreSQL
- Enable WAL mode for SQLite
- Reduce transaction size

---

## Monitoring Dashboard (Prometheus + Grafana)

### Add Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

pipeline_executions = Counter('pipeline_executions', 'Total pipelines run')
api_latency = Histogram('api_latency_seconds', 'API response latency')

@app.get("/api/health")
@api_latency.time()
async def health():
    pipeline_executions.inc()
    return {"status": "healthy"}
```

### Grafana Dashboard
- Pipeline execution count and timing
- API request rates and latencies
- Cluster count and noise percentage
- Database query performance
- Error rates by endpoint

---

## Cost Optimization

### Azure Deployment
```yaml
# Use spot instances for non-critical workloads
spec:
  spotPrice: "0.10"  # Cost per hour
  evictionPolicy: Deallocate
```

### OpenAI API Costs
- GPT-4o-mini: $0.15 per 1M input tokens
- Average 500 tokens per extraction
- 5000 feedback = ~$0.37 per run

Optimization:
- Batch similar feedback for efficiency
- Cache common patterns
- Use smaller model (GPT-3.5-turbo) for initial extraction

---

## CI/CD Pipeline (GitHub Actions)

```yaml
name: Deploy Yamaha Dashboard

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python verify_env.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: docker build -t yamaha-dashboard .
      - run: docker push ${{ secrets.REGISTRY }}/yamaha-dashboard
      - run: kubectl set image deployment/yamaha-dashboard dashboard=${{ secrets.REGISTRY }}/yamaha-dashboard
```

---

## Support & Maintenance

**Version Management:**
- Semantic versioning: v1.0.0
- Changelog documentation
- Dependency version pinning

**Updates & Patches:**
- Monthly dependency updates
- Security patch priority
- Backward compatibility maintained

**Documentation:**
- API documentation (auto-generated from FastAPI)
- User guide for engineers
- Architecture documentation
- Troubleshooting guide

---

## Contact & Support

For production deployment questions or issues:
- Email: yamaha-ai-team@example.com
- Slack: #yamaha-dashboard
- Docs: https://internal-docs.yamaha.com/dashboard

---

**Last Updated:** 2024  
**Deployment Version:** 1.0.0  
**Status:** ✅ Production Ready
