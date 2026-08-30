import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "BanglaEduAI API is running. Use POST /api/v1/analyze"}

def test_analyze_valid():
    payload = {"question": "আলোক সংশ্লেষণে ক্লোরোফিলের ভূমিকা ব্যাখ্যা কর।"}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Check that all expected tasks are present
    expected_tasks = ["subject", "topic", "question_type", "difficulty", "cognitive_level"]
    for task in expected_tasks:
        assert task in data["data"]
        assert "label" in data["data"][task]
        assert "confidence" in data["data"][task]

def test_analyze_empty():
    response = client.post("/api/v1/analyze", json={"question": ""})
    # Empty question will still be processed; should not crash
    assert response.status_code == 200
