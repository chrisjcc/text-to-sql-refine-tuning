# Deployment Guide

## Docker Deployment

### Build Image
```bash
docker build -t text-to-sql-grpo:latest .
```

### Run Training
```bash
docker-compose up training
```

### Run API Server
```bash
docker-compose up -d api
```

Check health:
```bash
curl http://localhost:8000/health
```

Test generation:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show all users",
    "schema": "CREATE TABLE users (id INT, name VARCHAR(100));"
  }'
```

### Run Interactive CLI
```bash
docker-compose run --rm cli
```

### Production Deployment

For production, consider:
- Use Kubernetes for orchestration
- Implement load balancing
- Add authentication middleware
- Set up monitoring (Prometheus/Grafana)
- Configure auto-scaling based on GPU utilization
- Use managed model serving (SageMaker, Vertex AI)

## HuggingFace Spaces Deployment

Deploy as a Gradio app on HuggingFace Spaces.

See `app.py` for Gradio interface.

## AWS SageMaker Deployment

See `deployment/sagemaker/` for SageMaker deployment scripts.

## Kubernetes Deployment

See `deployment/k8s/` for Kubernetes manifests.
