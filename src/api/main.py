# import os

# import boto3

# os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
# os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
# os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin"

# def prepare_minio():
#     """Vérifie si le bucket 'mlflow' existe, sinon le crée"""


#     s3 = boto3.client('s3', endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'])


#     buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]


#     if 'mlflow' not in buckets:
#         s3.create_bucket(Bucket='mlflow')
#         print("Bucket 'mlflow' créé avec succès.")


"""src/api/main.py"""
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.load_minio import prepare_minio
from src.api.load_model import load_production_model

app = FastAPI(title="MLflow Model API")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "API MLflow is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    model = load_production_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Modele non charge")
    return {"message": "Modele Production charge avec succes"}


@app.post("/predict")
def predict(features: IrisFeatures):
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
    prepare_minio()
    port_env = os.getenv("FASTAPI_PORT", "9090")
    host_url = "0.0.0.0"
    try:
        port = int(port_env)
    except (ValueError, TypeError):
        port = 9090
    uvicorn.run(app, host=host_url, port=port, log_level="debug")