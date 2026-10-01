# AI-Driven Citizen Grievance & Sentiment Analysis System

An end-to-end NLP pipeline that ingests NYC 311 service requests, classifies complaints into municipal departments, scores sentiment, and prioritises tickets by urgency — served through a FastAPI backend and Streamlit frontend.


## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Priority System](#priority-system)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [Contributors](#contributors)
- [License](#license)

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│   Frontend       │     │   Backend API    │     │   ML Models           │
│   (Streamlit)    │◄───►│   (FastAPI)      │◄───►│   Transformers        │
│   localhost:8501 │     │   localhost:8000 │     │   + Scikit-learn      │
└──────────────────┘     └──────────────────┘     └───────────────────────┘
         │                        │                          │
         ▼                        ▼                          ▼
  Single / Batch          POST /predict             Sentiment Analysis
  Grievance Input         POST /batch_predict       Department Routing
  CSV Upload              GET  /health               Urgency Scoring
                          GET  /metrics
```

## Features

- **Sentiment Analysis** — 4-class transformer model (positive / neutral / negative / critical) fine-tuned on legacy and India-oriented grievance examples
- **Department Routing** — India-oriented routing for water, roads, electricity, health, environment, police, welfare, education, municipal, revenue, agriculture, food distribution, and non-complaints
- **Urgency Scoring** — escalation rules for life safety, electrical hazards, violence, medical emergencies, pollution exposure, vulnerable residents, and essential-service outages
- **FastAPI Backend** — production-ready REST API with full endpoint coverage
- **Streamlit Frontend** — web interface for single and batch grievance submission with analytics dashboard
- **Batch Processing** — bulk analysis via CSV upload through `/batch_predict`

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Baviyas/citizen-grievance-nlp.git
cd citizen-grievance-nlp

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r api/requirements.txt
pip install -r frontend/requirements-frontend.txt

# 4. Provision and verify the local model artifacts
python scripts/train_india_department_model.py
python scripts/finetune_sentiment_model.py
python scripts/verify_model_artifacts.py --strict
# Optional: record exact artifact hashes for a deployment
python scripts/verify_model_artifacts.py \
  --strict \
  --write-checksums evaluation/model_artifact_checksums.json

# 5. Train models (the routing model excludes target/post-resolution leakage)
python scripts/train_department_model.py
# Run the sentiment notebooks separately when sentiment data is available.

# Evaluate on manually authored grievances kept outside training data
python scripts/evaluate_manual_grievances.py

# Prepare real NYC 311 records for human review. Taxonomy suggestions are not
# ground truth; see docs/REAL_DATA_REVIEW.md.
python scripts/prepare_real_nyc311_labels.py
python scripts/validate_reviewed_labels.py
python scripts/train_department_model.py
python scripts/evaluate_real_nyc311.py

# Train a separate provisional three-class model on real 311 records.
# This does not replace the four-class API model.
python scripts/assistant_review_real_nyc311.py
python scripts/train_real_3class_model.py

# The API uses the India-oriented department model and India-finetuned sentiment
# model by default.
python api/app.py

# Train India-oriented department routing.
python scripts/train_india_department_model.py

# Fine-tune sentiment and evaluate all four sentiment classes.
python scripts/finetune_sentiment_model.py

# Retrain the real-data model after regenerating assistant-reviewed labels:
python scripts/assistant_review_real_nyc311.py
python scripts/train_real_5class_model.py

# Use the NYC five-class artifact only for comparison:
$env:DEPARTMENT_MODEL_VARIANT = "real_5class"
python api/app.py

# Use the legacy four-class artifact only for comparison:
$env:DEPARTMENT_MODEL_VARIANT = "legacy_4class"
python api/app.py

The India department dataset currently contains 140 curated authored examples
(10 per department). The trainer combines word and character TF-IDF features to
improve tolerance for spelling variation and Hinglish wording. The data is
suitable for application prototyping, but production deployment should replace
it with substantially larger, independently labelled Indian grievance records.

# 6. Start the backend API
cd api && python app.py
# → http://localhost:8000  |  Swagger UI: http://localhost:8000/docs

# 7. Start the frontend (new terminal)
cd frontend && streamlit run app.py
# → http://localhost:8501
```

### Model artifacts

Model binaries are intentionally not committed to Git because the sentiment
weights are hundreds of megabytes. The required default artifacts and their
reproducible provisioning commands are recorded in
`config/model_artifacts.json`.

Run the verifier before starting a prediction-ready deployment:

```bash
python scripts/verify_model_artifacts.py --strict
```

The API can still start in degraded mode when artifacts are absent; `/health`
reports the missing model state and prediction requests return `503`.
Routing keywords and precedence terms are maintained in
`api/routing_rules.py`, and request bodies reject empty or oversized complaint
text before model inference.

## Configuration

### Environment Variables

```bash
API_BASE_URL=http://localhost:8000
USE_GPU=1          # 0 for CPU, 1 for GPU
```

### Streamlit Secrets

Create `frontend/.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
USE_GPU = 0
```

## Priority System

| Priority | Score | SLA | Description |
|----------|-------|-----|-------------|
| P1 — Critical | 80 – 100 | 2 hours | Life-threatening / immediate danger |
| P2 — High | 60 – 79 | 24 hours | Urgent infrastructure issues |
| P3 — Medium | 40 – 59 | 3 days | Standard maintenance / repair |
| P4 — Low | 0 – 39 | 7 days | Routine requests |

## Testing

```bash
# Run API test suite
cd api && python -m pytest test_api.py -v

# Manual curl test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Water pipe broken, flooding street"}'

# Frontend smoke test
cd frontend && streamlit run app.py
# Visit http://localhost:8501
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Models not found | Run all notebooks in order first |
| Port already in use | `python -m uvicorn api.app:app --port 8001` |
| CUDA out of memory | `export CUDA_VISIBLE_DEVICES=""` to force CPU |
| Import errors | `pip install --upgrade -r requirements.txt` |
| Streamlit secrets error | Create `frontend/.streamlit/secrets.toml` with `API_BASE_URL` |

## Deployment

### Docker

```bash
docker build -t grievance-api .
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" grievance-api
```

The Docker build deliberately excludes local model binaries from the build
context. Mount a provisioned `models` directory as shown above, or copy the
same artifacts into `/app/models` through your deployment's artifact store.
The build runs a non-strict manifest check so CI can build the degraded image;
production deployment should run `python scripts/verify_model_artifacts.py
--strict` before serving traffic.

## Contributors

| Name | GitHub |
|------|--------|
| Vasi Khan | [@vasi2904k](https://github.com/vasi2904k) |
| Bhumi Shah | [@code-with-bhumi](https://github.com/code-with-bhumi) |
| Baviya | [@Baviyas](https://github.com/Baviyas) |

## License

This project is licensed under the [MIT License](LICENSE).