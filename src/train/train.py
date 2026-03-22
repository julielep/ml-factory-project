"""Module d'entraînement et d'enregistrement du modèle dans MLflow."""

import os

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.train.services.prep_data import prepare_data

load_dotenv()


def configure_mlflow():
    """Configure MLflow uniquement hors mode test."""
    if os.getenv("TESTING") == "True":
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("My_Experiment")


def train_and_register(model, params, X_train, X_test, y_train, y_test):
    """Entraîne un modèle, log les métriques et l'enregistre dans MLflow."""
    configure_mlflow()

    model_name = "iris_model"

    with mlflow.start_run():
        # 1. Entraînement
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # 2. Métriques
        accuracy = accuracy_score(y_test, preds)

        # 3. Log des paramètres et métriques
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # 4. Log + enregistrement du modèle dans MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="LogisticRegression",
            registered_model_name=model_name,
        )

    print(f"✅ Modèle enregistré dans MLflow Registry : {model_name}")


def assign_production_alias(model_name="iris_model"):
    """Assigne l'alias 'production' à la dernière version du modèle."""
    if os.getenv("TESTING") == "True":
        return

    client = MlflowClient()

    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.set_registered_model_alias(model_name, "production", latest_version)


def main():
    """Pipeline complet d'entraînement."""
    # Chargement des données
    X_train, X_test, y_train, y_test = prepare_data()

    # Modèles
    rf_model = RandomForestClassifier()
    lr_model = LogisticRegression(max_iter=200)

    # Paramètres à logger (on peut logger ceux du modèle réellement entraîné aussi)
    params = rf_model.get_params()

    # Entraîner + enregistrer
    train_and_register(lr_model, params, X_train, X_test, y_train, y_test)

    # Gestion de l'alias 'Production'
    assign_production_alias("iris_model")


if __name__ == "__main__":
    main()