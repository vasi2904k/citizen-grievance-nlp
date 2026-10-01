# API Reference Guide

Complete documentation of all available API endpoints for the Citizen Grievance Management System.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint

**Endpoint:** `GET /`

**Description:** API information and available endpoints

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "Citizen Grievance Analysis API",
  "version": "1.0.0",
  "documentation": "/docs",
  "endpoints": {
    "health": "GET /health",
    "stats": "GET /stats",
    "predict": "POST /predict",
    "batch_predict": "POST /batch_predict",
    "metrics": "GET /metrics",
  }
}
```
---

### 2. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health and model status

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "sentiment_model_loaded": true,
  "department_model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-04-01T12:00:00.000000"
}
```

The API reports `status: "unhealthy"` when either model is unavailable. Model
artifacts are local/generated files and are not committed to Git; deployers must
mount or provision the configured sentiment and department model directories.

**Status Codes:**
- `200` - API is healthy and operational
- `503` - Service unavailable

---

### 3. Get Statistics

**Endpoint:** `GET /stats`

**Description:** Get system configuration and available options

**Request:**
```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "departments": [
    "Environment",
    "Non-Complaint",
    "Social & Health Services",
    "Transport"
  ],
  "priority_tiers": [
    "P1",
    "P2",
    "P3",
    "P4"
  ],
  "sentiment_types": [
    "critical",
    "negative",
    "neutral",
    "positive"
  ],
  "models": {
    "routing_model": "Logistic Regression",
    "sentiment_model": "DistilBERT"
  },
  "timestamp": "2026-04-05T12:00:00.000000"
}
```

**Status Codes:**
- `200` - Success

---

### 4. Single Prediction

**Endpoint:** `POST /predict`

**Description:** Simplified single prediction endpoint (legacy compatibility)

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "Road has huge pothole. URGENT!"}'
```

**Response:**
```json
{
  "complaint_text": "Road has huge pothole. URGENT!",
  "predicted_department": "Transport",
  "supporting_departments": [],
  "department_confidence": 0.9542,
  "sentiment": "critical",
  "sentiment_confidence": 0.9123,
  "urgency_score": 9.25,
  "priority": "CRITICAL",
  "recommended_action": "Dispatch emergency team immediately.",
  "timestamp": "2026-04-05T12:30:45"
}
```

---

### 5. Batch Prediction

**Endpoint:** `POST /batch_predict`

**Description:** Process multiple complaints with simplified response format (legacy compatibility)

**Request:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      "Road has huge pothole. URGENT!",
      "Water pipe is broken",
      "Electricity is cut off"
    ]
  }'
```

**Response:**
```json
{
  "total_complaints": 3,
  "predictions": [
    {
      "complaint_text": "Road has huge pothole. URGENT!",
      "predicted_department": "Transport",
      "supporting_departments": [],
      "department_confidence": 0.9542,
      "sentiment": "critical",
      "sentiment_confidence": 0.9123,
      "urgency_score": 9.25,
      "priority": "CRITICAL",
      "recommended_action": "Dispatch emergency team immediately.",
      "timestamp": "2026-04-05T12:30:45"
    }
    // ... more results
  ],
  "processing_time": 0.123
}
```

For multi-agency emergencies, `predicted_department` is the primary responder and
`supporting_departments` lists departments that should coordinate. For example,
an accident with severe bleeding is routed primarily to `public_health`, with
`roads_transport` and `police_public_safety` included for scene control and
traffic/public-safety response.

---

### 6. Get Model Metrics

**Endpoint:** `GET /metrics`

**Description:** Get detailed model performance metrics

**Request:**
```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "sentiment_metrics": {
    "accuracy": 0.875,
    "f1_score": 0.8642,
    "precision": 0.8758,
    "recall": 0.875
  },
  "department_metrics": {
    "accuracy": 0.8333,
    "f1_score": 0.8301,
    "precision": 0.839,
    "recall": 0.8333
  },
  "total_predictions": 1250,
  "timestamp": "2026-04-05T12:00:00.000000"
}
```

## Interactive API Documentation

When the API is running, visit:

**Swagger UI:** http://localhost:8000/docs

These provide interactive endpoints for testing the API directly from your browser.

## Department model variants

The API uses the India-oriented department model by default:

```text
Water Supply & Sewerage
Roads & Transport
Electricity & Power
Public Health
Environment & Pollution
Police & Public Safety
Women & Child Welfare
Social Welfare
Education
Municipal Services
Revenue & Land Records
Agriculture & Rural Development
Public Distribution System
Non-Complaint
```

To explicitly test the NYC five-class artifact, set:

```powershell
$env:DEPARTMENT_MODEL_VARIANT = "real_5class"
python api/app.py
```

To explicitly test the legacy four-class artifact, set:

```powershell
$env:DEPARTMENT_MODEL_VARIANT = "legacy_4class"
python api/app.py
```

The India department and sentiment models use curated authored examples until
independently labelled Indian grievance records are available. The NYC
`Non-Complaint` class likewise uses curated neutral/service-information
examples. These data limitations should be considered before production use.
The India department holdout is currently only a prototype evaluation
(approximately 57% accuracy and 53% macro-F1 after expanding to 140 curated
examples and adding character features). Expand and independently review the
Indian grievance dataset before relying on standalone classifier confidence.
