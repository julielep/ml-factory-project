# """Tests des endpoints FastAPI du module src.api.main."""

# from unittest.mock import MagicMock, patch

# from fastapi.testclient import TestClient

# from src.api.main import app

# client = TestClient(app)


# def test_root():
#     """Vérifie que l'endpoint racine répond correctement."""
#     response = client.get("/")

#     assert response.status_code == 200
#     assert response.json() == {"message": "API MLflow is running"}


# def test_health():
#     """Vérifie que l'endpoint /health retourne un statut OK."""
#     response = client.get("/health")

#     assert response.status_code == 200
#     assert response.json() == {"status": "ok"}


# @patch("src.api.main.load_production_model")
# def test_model_info_success(mock_load_production_model):
#     """Vérifie que /model-info retourne un succès si le modèle est chargé."""
#     mock_load_production_model.return_value = MagicMock()

#     response = client.get("/model-info")

#     assert response.status_code == 200
#     assert response.json() == {"message": "Modele Production charge avec succes"}


# @patch("src.api.main.load_production_model")
# def test_model_info_failure(mock_load_production_model):
#     """Vérifie que /model-info retourne une erreur si le modèle est absent."""
#     mock_load_production_model.return_value = None

#     response = client.get("/model-info")

#     assert response.status_code == 500
#     assert response.json() == {"detail": "Modele non charge"}


# @patch("src.api.main.load_production_model")
# def test_predict_success(mock_load_production_model):
#     """Vérifie que /predict retourne une prédiction et une version."""
#     mock_model = MagicMock()
#     mock_model.predict.return_value = [1]
#     mock_load_production_model.return_value = (mock_model, 3)

#     payload = {
#         "sepal_length": 5.1,
#         "sepal_width": 3.5,
#         "petal_length": 1.4,
#         "petal_width": 0.2,
#     }

#     response = client.post("/predict", json=payload)

#     assert response.status_code == 200
#     assert response.json() == {
#         "prediction": 1,
#         "version": 3,
#     }

#     mock_model.predict.assert_called_once_with([[5.1, 3.5, 1.4, 0.2]])
