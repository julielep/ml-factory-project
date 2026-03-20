import os

import boto3


def prepare_minio():
    """Vérifie si le bucket 'mlflow' existe, sinon le crée"""


    s3 = boto3.client('s3', endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'])


    buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]


    if 'mlflow' not in buckets:
        s3.create_bucket(Bucket='mlflow')
        print("Bucket 'mlflow' créé avec succès.")