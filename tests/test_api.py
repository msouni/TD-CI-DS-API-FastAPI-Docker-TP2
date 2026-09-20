import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_prediction_correcte():
    """Vérifier les prédictions pour les valeurs [1.0, 2.0, 3.0]."""

    # Créer un client qui appelle notre application FastAPI.
    with TestClient(app) as client:

        # Envoyer les valeurs à la route /predict.
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    # Vérifier que la requête a réussi.
    assert response.status_code == 200

    # Vérifier que les prédictions correspondent à y = 2x.
    assert response.json()["predictions"] == pytest.approx(
        [2.0, 4.0, 6.0]
    )