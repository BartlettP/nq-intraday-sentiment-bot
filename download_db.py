import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='us-east-2'
)

s3.download_file(
    os.environ['S3_BUCKET'],
    'predictions.db',
    'predictions.db'
)
print("Downloaded predictions.db")