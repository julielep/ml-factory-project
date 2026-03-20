"""src/api/main.py

API FastAPI pour servir le modèle MLflow en production.
Fournit des endpoints pour la santé de l'API, les informations du modèle
et la prédiction sur les données Iris.
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.load_minio import prepare_minio
from src.api.load_model import load_production_model

app = FastAPI(title="MLflow Model API")

# Initialisation MinIO dès l'import
prepare_minio()


class IrisFeatures(BaseModel):
    """Classe représentant les caractéristiques de l'Iris pour la prédiction.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    """Endpoint racine pour vérifier que l'API est en ligne.

    Returns:
        dict: Message indiquant que l'API fonctionne.
    """
    return {"message": "API MLflow is running"}


@app.get("/health")
def health():
    """Endpoint de vérification de l'état de santé de l'API.

    Returns:
        dict: Statut de l'API ("ok" si opérationnelle).
    """
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Endpoint pour récupérer les informations sur le modèle en production.

    Returns:
        dict: Message confirmant que le modèle Production est chargé avec succès.

    Raises:
        HTTPException: Si le modèle n'a pas pu être chargé.
    """
    model = load_production_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modele non charge")
    return {"message": "Modele Production charge avec succes"}


@app.post("/predict")
def predict(features: IrisFeatures):
    """Endpoint pour effectuer une prédiction à partir des caractéristiques de l'Iris.

    Args:
        features (IrisFeatures): Données d'entrée pour la prédiction.

    Returns:
        dict: Résultat de la prédiction et version du modèle utilisée.
              {"prediction": <classe prédite>, "version": <version du modèle>}
    """
    model, version = load_production_model()

    # Préparer les données dans le format attendu par MLflow / scikit-learn
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0]),
        "version": version
    }


if __name__ == "__main__":
    """Démarrage de l'API avec Uvicorn sur le host et port configurés."""
    prepare_minio()
    port_env = os.getenv("FASTAPI_PORT", "9090")
    host_url = "0.0.0.0"
    try:
        port = int(port_env)
    except (ValueError, TypeError):
        port = 9090
    uvicorn.run(app, host=host_url, port=port, log_level="debug")