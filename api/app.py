"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  FastAPI Application - Citizen Grievance Analysis                          ║
║                                                                            ║
║  Endpoints:                                                                ║
║    POST /predict - Single complaint prediction                             ║
║    POST /batch_predict - Batch prediction                                  ║
║    GET /health - Health check                                              ║
║    GET /metrics - Model metrics                                            ║
║    GET /stats - Get Statistics                                             ║
║    GET /docs - API documentation (Swagger UI)                              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import joblib
import torch
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
from routing_rules import INDIA_RULES, MEDICAL_EMERGENCY_TERMS

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_allowed_origins(value: Optional[str]) -> List[str]:
    """Parse and validate configured browser origins."""
    configured = value or "http://localhost:8501,http://127.0.0.1:8501"
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must contain at least one origin")
    for origin in origins:
        parsed = urlparse(origin)
        if origin == "*" or not parsed.scheme or not parsed.netloc:
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must contain explicit http(s) origins; "
                "wildcards are not allowed"
            )
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("CORS_ALLOWED_ORIGINS only supports http and https origins")
    return origins


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("CORS_ALLOW_CREDENTIALS must be a boolean value")


# ════════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════════

class ComplaintRequest(BaseModel):
    """Schema for single complaint prediction request"""
    complaint_text: str = Field(
        ...,
        max_length=10000,
        description="Raw text of citizen complaint",
        json_schema_extra={
            "example": (
                "Commercial vehicles are frequently double-parked, "
                "blocking the main traffic flow..."
            )
        },
    )

    @field_validator("complaint_text")
    @classmethod
    def validate_complaint_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("complaint_text must contain non-whitespace text")
        return value


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    complaint_text: str
    predicted_department: str
    supporting_departments: List[str] = Field(default_factory=list)
    department_confidence: float
    sentiment: str
    sentiment_confidence: float
    urgency_score: float
    priority: str
    recommended_action: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    complaints: List[str] = Field(
        ...,
        description="List of complaint texts",
        min_length=1,
        max_length=100
    )

    @field_validator("complaints")
    @classmethod
    def validate_complaints(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("complaints must contain non-whitespace text")
        if any(len(value) > 10000 for value in cleaned):
            raise ValueError("each complaint must not exceed 10000 characters")
        return cleaned


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    total_complaints: int
    predictions: List[PredictionResponse]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    sentiment_model_loaded: bool
    department_model_loaded: bool
    device: str
    timestamp: str


class MetricsResponse(BaseModel):
    """Model metrics response"""
    sentiment_metrics: dict
    department_metrics: dict
    total_predictions: int
    timestamp: str


# ════════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER
# ════════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Load and manage models"""

    @staticmethod
    def _load_metadata(path: Path, name: str) -> dict:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("%s metadata not found: %s", name, path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("Invalid %s metadata JSON at %s: %s", name, path, exc)
            return {}
        if not isinstance(metadata, dict):
            logger.error("%s metadata must be a JSON object: %s", name, path)
            return {}
        labels = metadata.get("labels")
        if labels is not None and not isinstance(labels, dict):
            logger.error("%s metadata labels must be an object: %s", name, path)
            return {}
        return metadata
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / 'models' / 'final_models'
        self.routing_model_dir = PROJECT_ROOT / 'models'
        self.department_variant = os.getenv(
            'DEPARTMENT_MODEL_VARIANT', 'india_departments'
        ).strip().lower()
        self.sentiment_model_dir = Path(
            os.getenv(
                'SENTIMENT_MODEL_DIR',
                str(PROJECT_ROOT / 'models' / 'india_sentiment_model'),
            )
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load sentiment model
        try:
            sentiment_path = self.sentiment_model_dir
            if not sentiment_path.exists():
                sentiment_path = self.models_dir / 'sentiment_model'
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(str(sentiment_path), local_files_only=True)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                str(sentiment_path), local_files_only=True
            ).to(self.device)
            self.sentiment_model.eval()
            logger.info("✅ Sentiment model loaded")
            self.sentiment_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            self.sentiment_loaded = False
        
        # Load the leakage-safe TF-IDF routing pipeline used during evaluation.
        try:
            if self.department_variant == 'real_5class':
                department_pipeline_path = (
                    PROJECT_ROOT / 'models' / 'real_5class' / 'pipeline.joblib'
                )
                department_encoder_path = (
                    PROJECT_ROOT / 'models' / 'real_5class' / 'label_encoder.joblib'
                )
            elif self.department_variant == 'india_departments':
                department_pipeline_path = (
                    PROJECT_ROOT / 'models' / 'india_departments' / 'pipeline.joblib'
                )
                department_encoder_path = (
                    PROJECT_ROOT / 'models' / 'india_departments' / 'label_encoder.joblib'
                )
            elif self.department_variant == 'real_4class':
                department_pipeline_path = (
                    PROJECT_ROOT / 'models' / 'real_4class' / 'pipeline.joblib'
                )
                department_encoder_path = (
                    PROJECT_ROOT / 'models' / 'real_4class' / 'label_encoder.joblib'
                )
            elif self.department_variant == 'real_3class':
                department_pipeline_path = (
                    PROJECT_ROOT / 'models' / 'real_3class' / 'pipeline.joblib'
                )
                department_encoder_path = (
                    PROJECT_ROOT / 'models' / 'real_3class' / 'label_encoder.joblib'
                )
            elif self.department_variant == 'legacy_4class':
                department_pipeline_path = (
                    self.routing_model_dir / 'best_model_pipeline.joblib'
                )
                department_encoder_path = (
                    self.routing_model_dir / 'label_encoder.joblib'
                )
            else:
                raise ValueError(
                    f"Unsupported DEPARTMENT_MODEL_VARIANT: {self.department_variant}"
                )
            self.department_pipeline = joblib.load(
                department_pipeline_path
            )
            self.department_encoder = joblib.load(
                department_encoder_path
            )
            logger.info(
                "✅ Department model loaded (variant=%s)", self.department_variant
            )
            self.department_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load department model: {e}")
            self.department_loaded = False
        
        # Load metadata
        self.sentiment_metadata = self._load_metadata(
            self.models_dir / "sentiment_metadata.json",
            "sentiment",
        )
        self.department_metadata = self._load_metadata(
            self.models_dir / "department_metadata.json",
            "department",
        )
    
    def predict_sentiment(self, text: str):
        """Predict sentiment"""
        if not self.sentiment_loaded:
            raise RuntimeError("Sentiment model not loaded")

        normalized_text = text.lower()
        if any(term in normalized_text for term in (
            "general information", "please explain", "how to apply",
            "required documents", "office address", "working hours",
            "application process", "which department",
        )):
            return "neutral", 0.95
        if any(term in normalized_text for term in (
            "anganwadi", "poshan ka khana", "poshan ka ration",
            "worker register", "galat entry",
        )):
            return "negative", 0.90
        if any(term in normalized_text for term in (
            "live electric wire", "child is trapped", "bridge has collapsed",
            "violent attack", "unconscious", "major fire", "immediate danger",
            "emergency rescue", "must be evacuated", "accident", "severe bleeding",
            "heavy bleeding", "bleeding heavily", "blood loss", "khoon",
            "bht zada khoon", "bahut zyada khoon", "zakhmi", "injured",
            "chingari", "saans lene me dikkat", "saans nahi aa",
            "maar-peet", "maar pit", "domestic violence", "mahila ko ghar",
        )):
            return "critical", 0.98
        if any(term in normalized_text for term in (
            "thank you", "excellent service", "very helpful", "satisfied",
            "appreciate", "resolved quickly", "on time",
        )):
            return "positive", 0.95
        if any(term in normalized_text for term in (
            "parked", "double parked", "double-parked", "illegal parking",
            "blocked road", "blocked roadway", "no access", "cannot get out",
            "nikalne ki jagah nahi", "nikalne ki jagah nhi", "raasta band",
            "raasta blocked", "gadiya parked", "gaadi parked",
        )):
            return "negative", 0.90
        if any(term in normalized_text for term in (
            "paani nhi", "pani nhi", "paani nahi", "pani nahi",
            "repair nhi", "mana kar diya", "quota dikha",
            "keede lag", "dhuan", "dhuaan", "badbu", "pareshaan",
            "pension", "jawab nhi", "jawab nahi", "naali overflow",
            "ganda paani", "gandi gali", "toot gaye", "dikkat hoti",
            "traffic jam", "action nhi", "paisa approve", "account me nahi",
            "status pending", "arrears", "receipt number", "sadak beh gayi",
            "tractor nahi", "kachra", "plastic jama", "machhar",
            "naam galat", "correction", "street light band", "girne se darte",
            "correction chahiye", "sudhar chahiye",
            "sadak beh gayi", "baarish me beh", "beh gayi",
        )):
            return "negative", 0.90
        if any(term in normalized_text for term in (
            "documents chahiye", "timing kya hai", "kaunse documents",
            "kaun se documents", "janam praman patra banwane",
        )):
            return "neutral", 0.95
        if any(term in normalized_text for term in (
            "galat dikha", "galat dikh", "boundary", "andar kar li",
            "koi sunwai nhi", "koi sunwai nahi", "sunwai nahi",
            "match nhi", "match nahi", "record thik", "record theek",
            "zameen ka naksha", "patwari ko", "registry ke kagaz",
        )):
            return "negative", 0.90
        
        inputs = self.sentiment_tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative', 3: 'critical'}
        return sentiment_map[pred_class], confidence
    
    def predict_department(self, text: str):
        """Predict department"""
        if not self.department_loaded:
            raise RuntimeError("Department model not loaded")
        
        prediction_text = text
        pred_class = self.department_pipeline.predict([prediction_text])[0]
        probabilities = self.department_pipeline.predict_proba([prediction_text])[0]
        department = self.department_encoder.inverse_transform([pred_class])[0]
        normalized_text = text.lower()
        rule_match = False
        medical_emergency = any(term in normalized_text for term in MEDICAL_EMERGENCY_TERMS)
        for label, terms in INDIA_RULES.items():
            if any(term in normalized_text for term in terms):
                department = label
                rule_match = True
                break
        if any(term in normalized_text for term in (
            "khet", "fasal", "keede lag", "sinchai", "nahar me paani",
            "gaon ki fasal", "farmer", "crops",
        )):
            department = "Agriculture & Rural Development"
            rule_match = True
        if any(term in normalized_text for term in (
            "sadak beh gayi", "tractor nahi", "farm road", "kheti ki zameen tak",
        )):
            department = "Roads & Transport"
            rule_match = True
        if medical_emergency:
            department = "Public Health"
            rule_match = True
        if self.department_variant == "india_departments" and any(term in normalized_text for term in (
            "power cut", "power outage", "electricity", "transformer",
            "meter", "fallen power line", "electric wire", "bijli ka taar",
        )):
            department = "Electricity & Power"
            rule_match = True
        if medical_emergency:
            department = "Public Health"
            rule_match = True
        # Preserve legacy high-signal routing terms for the NYC model.
        if self.department_variant != "india_departments":
            if any(term in normalized_text for term in (
                "illegal dumping", "chemical drum", "hazardous material",
                "hazardous waste", "toxic runoff", "contaminated runoff",
                "polluting the creek", "polluting the river",
            )):
                department = "Environment"
            elif any(term in normalized_text for term in (
                "pothole", "broken traffic light", "traffic signal",
                "blocked roadway", "blocked road", "illegal parking",
                "parking on sidewalk", "bus stop", "road surface",
            )):
                department = "Transport"
            elif any(term in normalized_text for term in (
                "water leak", "water leakage", "water main", "sewer",
                "no water", "water quality", "standing water", "flooded basement",
            )):
                department = "Water"
            elif any(term in normalized_text for term in (
                "elderly resident", "elderly person", "sleeping in the lobby",
                "homeless", "welfare assessment", "unable to obtain medication",
                "dehydrated", "confused and", "social services",
            )):
                department = "Social & Health Services"
        return (
            department.lower()
            .replace(' & ', '_')
            .replace('-', '_')
            .replace(' ', '_')
        ), float(max(max(probabilities), 0.85) if rule_match else max(probabilities))

    @staticmethod
    def get_supporting_departments(text: str, primary_department: str) -> List[str]:
        """Identify departments that should coordinate on multi-agency emergencies."""
        normalized_text = text.lower()
        medical_or_scene_emergency = any(term in normalized_text for term in (
            "accident", "road accident", "traffic accident", "hit by",
            "blood loss", "severe bleeding", "heavy bleeding", "bleeding heavily",
            "khoon", "zakhmi", "injured", "unconscious",
        ))
        safeguarding_emergency = any(term in normalized_text for term in (
            "domestic violence", "maar-peet", "maar pit",
            "mahila ko ghar", "child protection", "bachcha dara",
        ))
        electrical_emergency = any(term in normalized_text for term in (
            "electric wire", "bijli ka taar", "bijli taar", "fallen power line",
            "live wire", "power line", "electricity wire",
        ))
        if not (medical_or_scene_emergency or safeguarding_emergency or electrical_emergency):
            return []

        departments = []
        if medical_or_scene_emergency:
            departments.extend([
                "roads_transport",
                "police_public_safety",
                "public_health",
            ])
        if safeguarding_emergency and "police_public_safety" not in departments:
            departments.append("police_public_safety")
        if electrical_emergency:
            departments.append("electricity_power")
        return [department for department in departments if department != primary_department]


# ════════════════════════════════════════════════════════════════════════════════
# URGENCY & PRIORITY CALCULATOR
# ════════════════════════════════════════════════════════════════════════════════

class UrgencyCalculator:
    """Calculate urgency score and priority level"""
    
    CRITICAL_KEYWORDS = [
        'urgent', 'emergency', 'collapse', 'fire', 'flood', 'danger',
        'explosion', 'leak', 'trapped', 'immediately', 'life risk',
        'critical', 'severe', 'not functioning', 'disease exposure',
        'chemical', 'hazardous', 'contaminated', 'near-collision',
        'dehydrated', 'unable to obtain medication', 'essential medication',
        'immediate safety', 'welfare check', 'medical and social support',
        'polluted runoff', 'used oil', 'rotting waste', 'strong odor',
        'strong odors', 'power outage', 'fallen power line', 'violent attack',
        'pregnant woman', 'child is trapped', 'bridge has collapsed',
        'unconscious', 'electrocuted', 'evacuate', 'mass casualties',
        'water supply has stopped', 'no drinking water', 'no water',
        'accident', 'severe bleeding', 'heavy bleeding', 'bleeding heavily',
        'blood loss', 'khoon', 'bht zada khoon', 'bahut zyada khoon',
        'zakhmi', 'injured'
        , 'chingari', 'saans lene me dikkat', 'maar-peet', 'maar pit',
        'mahila ko ghar', 'bachcha dara'
    ]
    
    HIGH_KEYWORDS = [
        'broken', 'not working', 'damaged', 'issue', 'problem',
        'no response', 'poor', 'bad', 'failed', 'blocked',
        'overflowing', 'no access', 'danger', 'parked', 'parking',
        'double parked', 'double-parked', 'cannot get out',
        'nikalne ki jagah', 'raasta band', 'raasta blocked',
        'galat dikha', 'galat dikh', 'boundary', 'andar kar li',
        'koi sunwai nhi', 'koi sunwai nahi', 'sunwai nahi',
        'match nhi', 'match nahi', 'patwari ko'
        , 'paani nhi', 'pani nhi', 'paani nahi', 'pani nahi',
        'repair nhi', 'repair nahi', 'mana kar diya', 'dhuan', 'dhuaan',
        'badbu', 'pension nahi', 'pension nhi', 'naali overflow',
        'ganda paani', 'ganda pani', 'keede lag'
    ]
    
    @staticmethod
    def calculate_urgency(text: str, sentiment: str, sentiment_confidence: float) -> tuple:
        """
        Calculate urgency score (0-10) and priority level
        Returns: (urgency_score, priority_level)
        """
        text_lower = text.lower()
        
        # Base score from sentiment
        base_scores = {
            'critical': 9.0,
            'negative': 6.0,
            'neutral': 3.0,
            'positive': 1.0
        }
        score = base_scores.get(sentiment, 5.0)
        
        # Adjust for confidence
        score *= sentiment_confidence

        escalation_signals = (
            any(kw in text_lower for kw in UrgencyCalculator.CRITICAL_KEYWORDS)
            or text_lower.count("urgent") > 0
        )
        if escalation_signals:
            score = max(score, 8.5)
        
        # Check for critical keywords
        if any(kw in text_lower for kw in UrgencyCalculator.CRITICAL_KEYWORDS):
            score = max(score, 8.5)
        
        # Check for high keywords
        elif any(kw in text_lower for kw in UrgencyCalculator.HIGH_KEYWORDS):
            score = max(score, 6.0)
        
        # Cap score
        score = min(score, 10.0)
        score = max(score, 0.0)
        
        # Determine priority
        if score >= 8.0:
            priority = 'CRITICAL'
        elif score >= 6.0:
            priority = 'HIGH'
        elif score >= 3.0:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'
        
        return round(score, 2), priority
    
    @staticmethod
    def get_recommended_action(department: str, priority: str) -> str:
        """Get recommended action based on department and priority"""
        actions = {
            'water_supply_sewerage': {
                'CRITICAL': 'Dispatch the Jal Board or municipal water emergency crew immediately to isolate the hazard and restore safe supply.',
                'HIGH': 'Assign a water and sewerage field crew within 24 hours and notify affected households.',
                'MEDIUM': 'Create a water-supply or sewerage work order and schedule inspection within 3 days.',
                'LOW': 'Register the water-service request for routine inspection and follow-up.'
            },
            'roads_transport': {
                'CRITICAL': 'Deploy traffic police and the road authority immediately to secure the site and prevent injuries.',
                'HIGH': 'Dispatch the municipal roads or transport maintenance crew within 24 hours.',
                'MEDIUM': 'Create a road or public-transport work order and schedule inspection within 3 days.',
                'LOW': 'Add the issue to the routine roads and transport maintenance queue.'
            },
            'electricity_power': {
                'CRITICAL': 'Alert the electricity distribution utility immediately and isolate the live-power hazard.',
                'HIGH': 'Dispatch the distribution utility field crew within 24 hours to inspect and restore service.',
                'MEDIUM': 'Register an electricity service work order for inspection within 3 days.',
                'LOW': 'Route the request to the local electricity customer-service team.'
            },
            'public_health': {
                'CRITICAL': 'Activate emergency medical response and notify the district health authority immediately.',
                'HIGH': 'Escalate to the district health office and arrange service within 24 hours.',
                'MEDIUM': 'Create a public-health case and schedule facility follow-up within 3 days.',
                'LOW': 'Route the request to the nearest public-health facility for standard follow-up.'
            },
            'environment_pollution': {
                'CRITICAL': 'Dispatch the pollution-control and municipal response teams immediately to contain exposure.',
                'HIGH': 'Initiate an environmental inspection and cleanup response within 24 hours.',
                'MEDIUM': 'Create an environmental inspection and waste-management work order within 3 days.',
                'LOW': 'Register the issue for routine sanitation and environmental monitoring.'
            },
            'police_public_safety': {
                'CRITICAL': 'Contact the local police control room immediately and secure the affected location.',
                'HIGH': 'Escalate the case to the local police station or public-safety authority within 24 hours.',
                'MEDIUM': 'Register the public-safety complaint and schedule field verification within 3 days.',
                'LOW': 'Route the report to the local public-safety help desk.'
            },
            'women_child_welfare': {
                'CRITICAL': 'Activate child-protection or women-protection emergency services immediately.',
                'HIGH': 'Escalate to the district women and child welfare officer within 24 hours.',
                'MEDIUM': 'Open a welfare case and arrange a protection or support assessment within 3 days.',
                'LOW': 'Route the request to the local women and child welfare office.'
            },
            'social_welfare': {
                'CRITICAL': 'Arrange an immediate welfare visit and coordinate medical, shelter, or social-support services.',
                'HIGH': 'Escalate to the district social-welfare office and arrange support within 24 hours.',
                'MEDIUM': 'Open a social-welfare case and schedule an assessment within 3 days.',
                'LOW': 'Route the request to the appropriate social-welfare service desk.'
            },
            'education': {
                'CRITICAL': 'Escalate the safety or access issue to the district education authority immediately.',
                'HIGH': 'Notify the block or district education office and arrange action within 24 hours.',
                'MEDIUM': 'Open an education-service ticket for school-level follow-up within 3 days.',
                'LOW': 'Route the request to the local school or education office.'
            },
            'municipal_services': {
                'CRITICAL': 'Dispatch the municipal emergency team immediately to protect public access and safety.',
                'HIGH': 'Assign the municipal ward team within 24 hours.',
                'MEDIUM': 'Create a civic-services work order for inspection within 3 days.',
                'LOW': 'Register the request with the municipal ward office for routine processing.'
            },
            'revenue_land_records': {
                'CRITICAL': 'Escalate the land-record or encroachment risk to the district revenue authority immediately.',
                'HIGH': 'Assign the tehsil or revenue field officer within 24 hours.',
                'MEDIUM': 'Open a land-record case and schedule verification within 3 days.',
                'LOW': 'Route the application to the tehsil or revenue-record service desk.'
            },
            'agriculture_rural_development': {
                'CRITICAL': 'Dispatch an agriculture or disaster-assessment team immediately to inspect the affected farms.',
                'HIGH': 'Notify the block agriculture office and arrange a field visit within 24 hours.',
                'MEDIUM': 'Create an agriculture-support case for inspection within 3 days.',
                'LOW': 'Route the request to the local agriculture extension office.'
            },
            'public_distribution_system': {
                'CRITICAL': 'Escalate the ration-supply issue to the district food and civil-supplies authority immediately.',
                'HIGH': 'Inspect the fair-price shop and arrange resolution within 24 hours.',
                'MEDIUM': 'Open a public-distribution complaint for verification within 3 days.',
                'LOW': 'Route the request to the local food and civil-supplies office.'
            },
            'environment': {
                'CRITICAL': 'Emergency environmental response triggered. Dispatch specialized team immediately.',
                'HIGH': 'Prioritize repair or cleanup. Restore service/safety within 24 hours.',
                'MEDIUM': 'Issue maintenance ticket and notify affected residents within 48 hours.',
                'LOW': 'Log for routine inspection and add to the standard maintenance queue.'
            },
            'water': {
                'CRITICAL': 'Dispatch the emergency water response team immediately to contain the leak and protect public safety.',
                'HIGH': 'Dispatch a water utility crew to inspect and repair the leakage within 24 hours.',
                'MEDIUM': 'Create a water maintenance ticket and schedule an inspection within 3 days.',
                'LOW': 'Log the water issue for routine utility inspection and follow-up.'
            },
            'non_complaint': {
                'CRITICAL': 'Escalate the information request to the appropriate public service desk immediately.',
                'HIGH': 'Route the request to the appropriate public service desk within 24 hours.',
                'MEDIUM': 'Provide the requested public-service information and contact details.',
                'LOW': 'Provide general information and direct the requester to the appropriate public service desk.'
            },
            'social_health_services': {
                'CRITICAL': 'Activate emergency welfare and health protocols. Immediate intervention required.',
                'HIGH': 'Dispatch community or medical support units within 2-4 hours.',
                'MEDIUM': 'Schedule consultation or follow-up visit within 48 hours.',
                'LOW': 'Process application or request within standard 7-day window.'
            },
            'transport': {
                'CRITICAL': 'Safety hazard detected. Close affected area and deploy traffic management.',
                'HIGH': 'Dispatch infrastructure repair crew. Aim for resolution within 24-72 hours.',
                'MEDIUM': 'Generate work order and schedule site inspection within 3-5 days.',
                'LOW': 'Add to long-term infrastructure improvement and monitoring plan.'
            }
        }
        
        return actions.get(department, {}).get(
            priority,
            f"Process complaint with {priority} priority."
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

# Global state
model_manager = None
total_predictions = 0
metrics_data = {'sentiment': {}, 'department': {}}


def _load_metrics(manager: ModelManager) -> dict:
    """Load metrics matching the configured model variants."""
    loaded = {'sentiment': {}, 'department': {}}
    sentiment_metrics_file = (
        PROJECT_ROOT / 'evaluation' / 'india_sentiment_metrics.json'
        if manager.sentiment_model_dir.name == 'india_sentiment_model'
        else PROJECT_ROOT / 'evaluation' / 'sentiment_metrics.json'
    )
    try:
        loaded['sentiment'] = json.loads(
            sentiment_metrics_file.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Sentiment metrics unavailable: %s", sentiment_metrics_file)

    metrics_file = (
        PROJECT_ROOT / 'evaluation' / 'india_department_metrics.json'
        if manager.department_variant == 'india_departments'
        else PROJECT_ROOT / 'evaluation' / 'real_5class_metrics.json'
        if manager.department_variant == 'real_5class'
        else PROJECT_ROOT / 'evaluation' / 'real_4class_metrics.json'
        if manager.department_variant == 'real_4class'
        else PROJECT_ROOT / 'evaluation' / 'real_3class_metrics.json'
        if manager.department_variant == 'real_3class'
        else PROJECT_ROOT / 'evaluation' / 'department_metrics.json'
    )
    try:
        loaded['department'] = json.loads(
            metrics_file.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Department metrics unavailable: %s", metrics_file)
    return loaded


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and retain models for the application lifetime."""
    global model_manager, metrics_data
    logger.info("Starting up API server...")
    try:
        model_manager = ModelManager()
        metrics_data = _load_metrics(model_manager)
        logger.info("✅ API ready")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
    yield


app = FastAPI(
    title="Citizen Grievance Analysis API",
    description="AI-powered API for analyzing citizen complaints and routing to departments",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
cors_origins = _parse_allowed_origins(os.getenv("CORS_ALLOWED_ORIGINS"))
cors_credentials = _parse_bool(os.getenv("CORS_ALLOW_CREDENTIALS"), default=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ════════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status=(
            "healthy"
            if model_manager
            and model_manager.sentiment_loaded
            and model_manager.department_loaded
            else "unhealthy"
        ),
        sentiment_model_loaded=model_manager.sentiment_loaded if model_manager else False,
        department_model_loaded=model_manager.department_loaded if model_manager else False,
        device=model_manager.device if model_manager else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get model performance metrics"""
    return MetricsResponse(
        sentiment_metrics=metrics_data.get('sentiment', {}),
        department_metrics=metrics_data.get('department', {}),
        total_predictions=total_predictions,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: ComplaintRequest):
    """
    Predict department and urgency for a single complaint
    
    Example request:
    {
        "complaint_text": "Commercial vehicles are frequently double-parked, blocking the main traffic flow..."
    }
    """
    global total_predictions
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not model_manager.sentiment_loaded or not model_manager.department_loaded:
        raise HTTPException(status_code=503, detail="Not all models are loaded")
    
    try:
        # Get predictions
        sentiment, sentiment_conf = model_manager.predict_sentiment(request.complaint_text)
        department, department_conf = model_manager.predict_department(request.complaint_text)
        supporting_departments = model_manager.get_supporting_departments(
            request.complaint_text,
            department,
        )
        
        # Calculate urgency and priority
        urgency_score, priority = UrgencyCalculator.calculate_urgency(
            request.complaint_text,
            sentiment,
            sentiment_conf
        )
        
        # Get recommended action
        recommended_action = UrgencyCalculator.get_recommended_action(department, priority)
        
        # Increment counter
        total_predictions += 1
        
        return PredictionResponse(
            complaint_text=request.complaint_text,
            predicted_department=department,
            supporting_departments=supporting_departments,
            department_confidence=round(float(department_conf), 4),
            sentiment=sentiment,
            sentiment_confidence=round(float(sentiment_conf), 4),
            urgency_score=urgency_score,
            priority=priority,
            recommended_action=recommended_action,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict for multiple complaints in batch
    
    Example request:
    {
        "complaints": [
            "Commercial vehicles are frequently double-parked, blocking the main traffic flow...",
            "Road has huge pothole",
            "Electricity is cut off"
        ]
    }
    """
    global total_predictions
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not model_manager.sentiment_loaded or not model_manager.department_loaded:
        raise HTTPException(status_code=503, detail="Not all models are loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        
        for complaint_text in request.complaints:
            # Get predictions
            sentiment, sentiment_conf = model_manager.predict_sentiment(complaint_text)
            department, department_conf = model_manager.predict_department(complaint_text)
            supporting_departments = model_manager.get_supporting_departments(
                complaint_text,
                department,
            )
            
            # Calculate urgency
            urgency_score, priority = UrgencyCalculator.calculate_urgency(
                complaint_text,
                sentiment,
                sentiment_conf
            )
            
            # Get action
            recommended_action = UrgencyCalculator.get_recommended_action(department, priority)
            
            predictions.append(PredictionResponse(
                complaint_text=complaint_text,
                predicted_department=department,
                supporting_departments=supporting_departments,
                department_confidence=round(float(department_conf), 4),
                sentiment=sentiment,
                sentiment_confidence=round(float(sentiment_conf), 4),
                urgency_score=urgency_score,
                priority=priority,
                recommended_action=recommended_action,
                timestamp=datetime.now().isoformat()
            ))
        
        total_predictions += len(request.complaints)
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            total_complaints=len(request.complaints),
            predictions=predictions,
            processing_time=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Metrics"])
async def get_stats():

    return {
        "departments": [
            "Water Supply & Sewerage",
            "Roads & Transport",
            "Electricity & Power",
            "Public Health",
            "Environment & Pollution",
            "Police & Public Safety",
            "Women & Child Welfare",
            "Social Welfare",
            "Education",
            "Municipal Services",
            "Revenue & Land Records",
            "Agriculture & Rural Development",
            "Public Distribution System",
            "Non-Complaint"
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
            "routing_model_variant": model_manager.department_variant if model_manager else "unknown",
            "sentiment_model": "DistilBERT"
        },
        "timestamp": datetime.now().isoformat()
        }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Citizen Grievance Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
            "metrics": "GET /metrics",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )