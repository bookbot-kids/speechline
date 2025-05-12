import boto3
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime


def setup_logging(log_dir="logs"):
    """Set up logging with file and console output."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    log_filename = datetime.now().strftime("%Y_%m_%d_%H:%M:%S-prompts_download.log")
    log_path = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_s3_client():
    """Initialize and return an S3 client."""
    return boto3.client("s3")


def list_subfolders(bucket_name, prefix):
    """List subfolders in the given S3 bucket under the specified prefix."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    subfolders = []

    if "CommonPrefixes" in response:
        for prefix in response["CommonPrefixes"]:
            subfolders.append(prefix["Prefix"])

    return subfolders


def download_audio_files(bucket_name, folder_path, output_dir):
    """Download audio files from S3 and save them to the specified local directory."""
    s3 = get_s3_client()
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    if "Contents" in objects:
        for obj in objects["Contents"]:
            file_key = obj["Key"]
            if file_key.endswith(".aac"):
                # Create a local path that mirrors the S3 structure
                subfolder_name = folder_path.split("/")[-2]  # Get the subfolder name
                local_subfolder = os.path.join(output_dir, subfolder_name)
                Path(local_subfolder).mkdir(parents=True, exist_ok=True)

                local_path = os.path.join(local_subfolder, os.path.basename(file_key))
                s3.download_file(bucket_name, file_key, local_path)
                logger.info(f"Downloaded {file_key} to {local_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio files from S3 prompts")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save downloaded files",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save log files"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting download process with args: {args}")

    try:
        subfolders = list_subfolders(args.bucket, "dropbox/prompts/")
        for subfolder in subfolders:
            logger.info(f"Processing subfolder: {subfolder}")
            download_audio_files(args.bucket, subfolder, args.output_dir)

        logger.info("Download process completed successfully")

    except Exception as e:
        logger.error(f"Error during download process: {str(e)}", exc_info=True)
        raise
