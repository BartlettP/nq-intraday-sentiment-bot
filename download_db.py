"""
Download the production predictions database from S3.

Writes to the project's canonical database path (src/config.py -> DB_PATH),
so the backtest and health scripts read exactly what you just downloaded.
Previously this dropped the file in the repo root while everything else read
from data/, which is how the two copies drifted apart.
"""
import os
import shutil
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config  # noqa: E402


def main():
    if not (config.S3_BUCKET and config.AWS_ACCESS_KEY and config.AWS_SECRET_KEY):
        sys.exit("S3 is not configured. Set S3_BUCKET, AWS_ACCESS_KEY_ID and "
                 "AWS_SECRET_ACCESS_KEY in .env")

    s3 = boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY,
        aws_secret_access_key=config.AWS_SECRET_KEY,
        region_name=config.S3_REGION,
    )

    dest = config.DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    # Keep a backup — this overwrites whatever local history you had.
    if os.path.exists(dest):
        backup = dest + '.bak'
        shutil.copy2(dest, backup)
        print(f"Backed up existing database to {backup}")

    tmp = dest + '.download'
    s3.download_file(config.S3_BUCKET, config.S3_KEY, tmp)
    os.replace(tmp, dest)  # atomic swap, so a failed download can't truncate

    size = os.path.getsize(dest)
    print(f"Downloaded s3://{config.S3_BUCKET}/{config.S3_KEY} -> {dest} ({size:,} bytes)")


if __name__ == '__main__':
    main()
