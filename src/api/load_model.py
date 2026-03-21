"""src/api/load_model.py"""

import os

import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
from fastapi import HTTPException
from mlflow import MlflowClient

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# IMPORTANT : configure MLflow globalement
mlflow.set_tracking_uri(MLFLOW_URI)

# Initialisation du client MLflow
client = MlflowClient(tracking_uri=MLFLOW_URI)

# Cache pour éviter de recharger le modèle si la version n'a pas changé
state = {"model": None, "version": None}

MODEL_NAME = "iris_model"
MODEL_ALIAS = "production"


def load_production_model():
    """Vérifie la version en production et recharge si nécessaire."""
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prod_version = alias_info.version

        if state["model"] is None or prod_version != state["version"]:
            print(f"[INFO] Chargement de la version {prod_version} depuis MinIO...")
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            state["model"] = mlflow.pyfunc.load_model(model_uri)
            state["version"] = prod_version
            print(f"[OK] Modele {MODEL_NAME} v{prod_version} charge")

        return state["model"], prod_version

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur MLflow: {str(e)}")


if __name__ == "__main__":
    model, version = load_production_model()
    print(model)
    print(f"Version chargee : {version}")
