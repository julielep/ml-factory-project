import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.train.services.prep_data import prepare_data

load_dotenv()



def train_and_register(model, params, X_train, X_test, y_train, y_test):
    # Configuration MLflow AVANT le run
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("My_Experiment")

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
            registered_model_name=model_name
        )

    print(f"✅ Modèle enregistré dans MLflow Registry : {model_name}")


# Chargement des données
X_train, X_test, y_train, y_test = prepare_data()

# Modèle
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()

# Paramètres à logger
params = rf_model.get_params()

# Entraîner + enregistrer
train_and_register(lr_model, params, X_train, X_test, y_train, y_test)

# 3. Gestion de l'Alias 'Production' via MlflowClient 
client = MlflowClient()

# On récupère la toute dernière version créée 
latest_version = client.get_latest_versions("iris_model", stages=["None"])[0].version

# On lui attribue l'alias 'Production'
# client.set_registered_model_alias("iris_model", "Production", latest_version)

# print(f"✅ Alias 'Production' assigné à la version {latest_version} de iris_model")