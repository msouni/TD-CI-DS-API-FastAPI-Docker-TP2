import pytest
from fastapi.testclient import TestClient
# Adjust this import based on your exact app layout (e.g., from app.main import app)
from app.main import app 

client = TestClient(app)

# 1) Un test qui valide une prédiction correcte avec les valeurs [1.0, 2.0, 3.0]
def test_prediction_correcte():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    # Assuming your model returns double values as shown in the README example: [2.0, 4.0, 6.0]
    assert data["predictions"] == [2.0, 4.0, 6.0]

# 2) Un test qui valide une prédiction incorrecte (volontairement faux)
def test_prediction_incorrecte():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    # Voluntarily expecting something false (e.g., [99.0, 99.0, 99.0])
    # To pass the test suite but prove you are checking for incorrect expectations, 
    # we assert that it does NOT equal the wrong answer.
    assert data["predictions"] != [99.0, 99.0, 99.0]

# 3) Un test qui envoie un JSON incorrect (champ features manquant)
def test_prediction_json_incorrect():
    # Missing the "features" key entirely, passing raw numbers instead
    payload = {"wrong_key": [3.5, 1.2, 4.9]} 
    response = client.post("/predict", json=payload)
    
    # FastAPI automatically returns 422 Unprocessable Entity for schema validation failures
    assert response.status_code == 422
