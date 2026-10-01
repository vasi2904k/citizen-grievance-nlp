"""
Comprehensive test suite for Citizen Grievance Analysis API
"""
import pytest
import json
import warnings
from fastapi.testclient import TestClient
from app import app, UrgencyCalculator
warnings.filterwarnings("ignore")

client = TestClient(app, raise_server_exceptions=False)

class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "device" in data
        assert "timestamp" in data

    def test_health_check_has_model_status(self):
        response = client.get("/health")
        data = response.json()
        assert "sentiment_model_loaded" in data
        assert "department_model_loaded" in data

class TestMetricsEndpoint:
    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "sentiment_metrics" in data
        assert "department_metrics" in data
        assert "total_predictions" in data
        assert "timestamp" in data

class TestPredictEndpoint:
    def test_predict_valid_request(self):
        payload = {"complaint_text": "Water pipe is broken near my house."}
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predicted_department" in data
            assert "sentiment" in data
            assert "urgency_score" in data
            assert "priority" in data

    def test_predict_various_complaints(self):
        test_cases = [
            "Road has a large pothole blocking traffic",
            "Hospital is not providing proper treatment",
            "Electricity is cut off in my area",
        ]
        for complaint in test_cases:
            response = client.post("/predict", json={"complaint_text": complaint})
            assert response.status_code in [200, 503]

    def test_hinglish_blocked_parking_routes_to_roads(self):
        text = (
            "mere ghr ke bahar bht saari gadiya parked h nikalne ki jagah nhi h, "
            "jaldi aa kr inhe hatao or sahi karo raasta"
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json={"complaint_text": text})
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            return
        data = response.json()
        assert data["predicted_department"] == "roads_transport"
        assert data["department_confidence"] >= 0.85
        assert data["sentiment"] == "negative"
        assert data["priority"] == "HIGH"

    def test_hinglish_accident_bleeding_routes_to_health(self):
        text = "mere ghr ke bahar accident ho gaya h, or ek insaan ka khoon bht zada nikl rha h"
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json={"complaint_text": text})
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            return
        data = response.json()
        assert data["predicted_department"] == "public_health"
        assert data["department_confidence"] >= 0.85
        assert data["sentiment"] == "critical"
        assert data["priority"] == "CRITICAL"
        assert data["urgency_score"] >= 8.0
        assert data["supporting_departments"] == [
            "roads_transport",
            "police_public_safety",
        ]

    def test_complex_hinglish_emergency_coordinates_electricity(self):
        text = (
            "mere mohalle ke main road par kal raat se ek bada ped girra hua h, "
            "uske niche ek aadmi dab gaya tha aur abhi bhi usko bht zyada chot lagi h, "
            "khoon ruk nhi raha. road puri tarah block h aur paas me bijli ka taar "
            "bhi toot ke latak rha h. ambulance ko aane me dikkat ho rhi h"
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json={"complaint_text": text})
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            return
        data = response.json()
        assert data["predicted_department"] == "public_health"
        assert data["supporting_departments"] == [
            "roads_transport",
            "police_public_safety",
            "electricity_power",
        ]
        assert data["priority"] == "CRITICAL"

    def test_hinglish_land_record_dispute_is_negative_and_high(self):
        text = (
            "hamare gaon ki zameen ka naksha online galat dikha rha h, "
            "padosi ne boundary thodi si andar kar li h aur patwari ko 3 baar "
            "bolne ke baad bhi koi sunwai nhi hui. registry ke kagaz me survey "
            "number bhi match nhi kar raha, kripya jaldi se site naap kar record "
            "thik karwao."
        )
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json={"complaint_text": text})
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            return
        data = response.json()
        assert data["predicted_department"] == "revenue_land_records"
        assert data["sentiment"] == "negative"
        assert data["priority"] == "HIGH"

    def test_predict_empty_text(self):
        payload = {"complaint_text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 422, 503]

    def test_predict_long_text(self):
        long_text = "complaint text " * 100
        payload = {"complaint_text": long_text}
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]

class TestBatchPredictEndpoint:
    def test_batch_predict_valid_request(self):
        payload = {"complaints": ["Water pipe is broken", "Road has pothole"]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["total_complaints"] == 2

    def test_batch_predict_single_complaint(self):
        payload = {"complaints": ["Water issue"]}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code in [200, 503]

    def test_batch_predict_max_complaints(self):
        complaints = ["complaint " + str(i) for i in range(50)]
        payload = {"complaints": complaints}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code in [200, 503]

    def test_batch_predict_exceeds_max(self):
        complaints = ["complaint " + str(i) for i in range(101)]
        payload = {"complaints": complaints}
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 422

class TestUrgencyCalculator:
    def test_critical_priority_assignment(self):
        text = "URGENT! Water pipe collapsed - emergency!"
        urgency, priority = UrgencyCalculator.calculate_urgency(text, "critical", 0.95)
        assert priority == "CRITICAL"
        assert urgency >= 8.0

    def test_low_priority_assignment(self):
        text = "All good, thank you"
        urgency, priority = UrgencyCalculator.calculate_urgency(text, "positive", 0.90)
        assert priority == "LOW"
        assert urgency <= 5.0

    def test_india_emergency_escalation(self):
        text = "A live electric wire has fallen across the road and people could be electrocuted."
        urgency, priority = UrgencyCalculator.calculate_urgency(text, "neutral", 0.95)
        assert priority == "CRITICAL"
        assert urgency >= 8.0

    def test_urgency_bounds(self):
        for text, sentiment, conf in [
            ("EMERGENCY!!!", "critical", 0.95),
            ("No issues", "positive", 0.90),
            ("Minor problem", "neutral", 0.70)
        ]:
            urgency, _ = UrgencyCalculator.calculate_urgency(text, sentiment, conf)
            assert 0 <= urgency <= 10

class TestRootEndpoint:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

class TestRequestValidation:
    def test_missing_complaint_text(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_invalid_json(self):
        response = client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

class TestErrorHandling:
    def test_404_endpoint(self):
        response = client.get("/nonexistent")
        assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])