"""Download bookbot user audio and book transcript files from Amazon S3."""
import boto3
import os
from pathlib import Path
import subprocess
import json
from p_tqdm import p_map
from functools import partial
from typing import List, Tuple
from datetime import datetime, timedelta
import argparse
from datasets import Dataset, load_dataset, Audio
from datasets import concatenate_datasets
import logging
from datasets import IterableDataset


def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_filename = datetime.now().strftime('%Y_%m_%d_%H:%M:%S-bb_s3_download.log')
    log_path = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)

def get_s3_client():
    """Initialize and return an S3 client."""
    return boto3.client('s3')

def list_subfolders(bucket_name, languages):
    """List subfolders in the given S3 bucket that start with any of the specified languages,
    including those in both dropbox/ and dropbox/practiceError/ paths.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        languages (List[str]): List of language prefixes to filter folders by (e.g. ['en', 'es'])
    """
    language_folders = []
    s3 = get_s3_client()
    
    # Check dropbox root path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="dropbox/", Delimiter='/')
    
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            if any(folder_name.startswith(lang) for lang in languages):
                language_folders.append(prefix['Prefix'])
    
    # Check dropbox/practiceError path
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="dropbox/practiceError/", Delimiter='/')
    
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            if any(folder_name.startswith(lang) for lang in languages):
                language_folders.append(prefix['Prefix'])
    
    return language_folders
    
def download_and_convert_to_wav(bucket_name: str, file_key: str, output_wav_path: str, 
                              num_channels: int = 1, sampling_rate: int = 16000) -> None:
    """Download AAC from S3 and convert to WAV using a temporary file."""
    s3 = get_s3_client()
    
    # Create a temporary file for the AAC content
    temp_aac = output_wav_path + '.temp.aac'
    try:
        # Download to temporary file
        s3.download_file(bucket_name, file_key, temp_aac)
        
        # Convert using proven ffmpeg command
        cmd = [
            "ffmpeg",
            "-loglevel", "error",  # Show errors only
            "-y",
            "-i", temp_aac,
            "-acodec", "pcm_s16le",
            "-ac", str(num_channels),
            "-ar", str(sampling_rate),
            str(output_wav_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_aac):
            os.remove(temp_aac)

def list_objects_paginated(s3, bucket: str, prefix: str) -> List[dict]:
    """Get all objects from an S3 bucket/prefix, handling pagination"""
    objects = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            objects.extend(page['Contents'])
    
    return objects

def parse_date(date_str):
    """Convert date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None

def download_audio_and_transcripts(bucket_name: str, folder_path: str, output_dir: str, 
                                 after_date: datetime = None, convert_to_wav: bool = True):
    """
    Download audio and transcripts from an S3 bucket, convert audio to WAV format, and save them to the specified output directory.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_path (str): The path to the folder in the S3 bucket containing the audio and transcript files.
        output_dir (str): The directory where the downloaded and converted files will be saved.
        after_date (datetime, optional): Only download files modified after this date. Defaults to None.
        convert_to_wav (bool, optional): Whether to convert audio files to WAV format. Defaults to True.

    Returns:
        None
    """
    s3 = get_s3_client()
    
    # Modify how we extract the output path to preserve directory structure
    path_parts = folder_path.rstrip('/').split('/')
    if len(path_parts) > 2:  # If it's in a subdirectory like practiceError
        subfolder_path = os.path.join(*path_parts[1:])  # Skip the first 'dropbox' part
    else:
        subfolder_path = path_parts[-1]
    
    output_path = os.path.join(output_dir, subfolder_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    objects = list_objects_paginated(s3, bucket_name, folder_path)
    
    if objects:
        # Filter objects by date if after_date is specified
        if after_date:
            logger.info(f"After date: {after_date}")
            objects = [obj for obj in objects if obj['LastModified'].replace(tzinfo=None) > after_date]
        
        # First, find all transcript files (both .txt and .json)
        transcript_files = set()
        for obj in objects:
            if obj['Key'].endswith(('.txt', '.json')):
                base_name = os.path.splitext(os.path.basename(obj['Key']))[0]
                transcript_files.add(base_name)
        
        logger.info(f"\nProcessing files from {folder_path} to {output_path}")
        
        new_rows = []
        
        for obj in objects:
            file_key = obj['Key']
            base_name = os.path.splitext(os.path.basename(file_key))[0]
            
            if file_key.endswith('.aac') and base_name in transcript_files:
                wav_path = os.path.join(output_path, f"{base_name}.wav")
                txt_path = os.path.join(output_path, f"{base_name}.txt")
                
                # Skip only if both WAV and TXT files exist
                if os.path.exists(wav_path) and os.path.exists(txt_path):
                    logger.info(f"Skipping {base_name} - WAV and TXT files already exist")
                    continue
                
                # Download and process transcript
                transcript_text = ""
                if any(obj['Key'].endswith(f"{base_name}.json") for obj in objects): # Handle JSON transcript
                    json_key = os.path.splitext(file_key)[0] + '.json'
                    logger.info(f"Downloading JSON transcript: {os.path.basename(json_key)}")
                    
                    temp_json = os.path.join(output_path, "temp.json")
                    s3.download_file(bucket_name, json_key, temp_json)
                    
                    try:
                        with open(temp_json, 'r') as f:
                            transcript_data = json.load(f)
                        # Extract text from JSON - adjust this based on your JSON structure
                        transcript_text = transcript_data.get('text', '')
                        
                        # Optionally save as txt file for reference
                        with open(txt_path, 'w') as f:
                            f.write(transcript_text)
                    finally:
                        if os.path.exists(temp_json):
                            os.remove(temp_json)
                else: # Handle regular .txt transcript
                    txt_key = os.path.splitext(file_key)[0] + '.txt'
                    logger.info(f"Downloading transcript: {os.path.basename(txt_key)}")
                    s3.download_file(bucket_name, txt_key, txt_path)
                    with open(txt_path, 'r') as f:
                        transcript_text = f.read()
                
                if transcript_text:
                    # Download and convert audio
                    try:
                        download_and_convert_to_wav(bucket_name, file_key, wav_path)
                        new_rows.append({
                            "id": base_name,
                            "audio": wav_path,
                            "text": transcript_text,
                            "speaker": base_name.split('_')[0],
                            "accent": folder_path.split('/')[-1],
                            "language": folder_path.split('/')[-1].split('_')[0]
                        })
                    except Exception as e:
                        logger.info(f"Error processing {file_key}: {str(e)}")
                        continue
        
        return new_rows

# Last run on 2024-11-13
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Download and convert BookBot audio files from S3')
    parser.add_argument('--languages', type=str, nargs='+', required=True,
                       help='List of language codes to filter folders (e.g. "en es")')
    parser.add_argument('--bucket', type=str, required=True,
                       help='S3 bucket name')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Local directory to save downloaded files')
    parser.add_argument('--after-date', type=str,
                       help='Only process files created after this date (YYYY-MM-DD)')
    parser.add_argument('--hf-dataset', type=str, required=True,
                       help='Hugging Face dataset ID (e.g. "username/dataset-name")')
    parser.add_argument('--append-to-dataset', action='store_true',
                        help='Append new rows to existing dataset if set, otherwise create a new dataset')

    args = parser.parse_args()
    
    logger.info(f"Starting download process with args: {args}")
    
    # Convert after_date string to datetime if provided
    after_date = None
    if args.after_date:
        try:
            after_date = datetime.strptime(args.after_date, '%Y-%m-%d')
            logger.info(f"Filtering files after date: {after_date}")
        except ValueError:
            logger.error(f"Invalid date format: {args.after_date}")
            exit(1)

    if args.append_to_dataset:
        # Load existing dataset from Hub
        hf_dataset = load_dataset(args.hf_dataset, split="train", num_proc = os.cpu_count())
    else:
        hf_dataset = None
    
    languages = args.languages
    subfolders = list_subfolders(args.bucket, languages)    
    
    all_new_rows = []
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    
    try:
        s3 = get_s3_client()
        logger.info("Successfully connected to S3")
        
        # Open the manifest file in append mode once
        for subfolder in subfolders:
            if "practiceError" in subfolder:
                logger.debug(f"Skipping practice error folder: {subfolder}")
                continue
            else:
                logger.info(f"Processing subfolder: {subfolder}")
                new_rows = download_audio_and_transcripts(
                    args.bucket,
                    subfolder,
                    args.output_dir,
                    after_date=after_date,
                    convert_to_wav=True
                )
                all_new_rows.extend(new_rows)
        
        if all_new_rows:
            logger.info(f"\nAppending {len(all_new_rows)} total new entries to dataset...")
            
            with open(manifest_path, "w") as manifest_file:
                json.dump(all_new_rows, manifest_file)

            # Create a new dataset with all new rows
            new_dataset = Dataset.from_dict({
                "id": [row["id"] for row in all_new_rows],
                "audio": [row["audio"] for row in all_new_rows],
                "text": [row["text"] for row in all_new_rows],
                "speaker": [row["speaker"] for row in all_new_rows],
                "language": [row["language"] for row in all_new_rows]
            })
            
            # Cast audio column to Audio() feature
            new_dataset = new_dataset.cast_column("audio", Audio())
            # Concatenate with existing dataset
            
            if args.append_to_dataset:  
                updated_dataset = concatenate_datasets([hf_dataset, new_dataset])
                logger.info(f"Length of updated dataset: {len(updated_dataset)}")
                
                # Push to the Hub once with all updates
                updated_dataset.push_to_hub(args.hf_dataset, private=True, max_shard_size="500MB")
                logger.info(f"Successfully pushed updated dataset to {args.hf_dataset}")
            else:
                new_dataset.push_to_hub(args.hf_dataset, private=True, max_shard_size="500MB")
                logger.info(f"Successfully pushed new dataset to {args.hf_dataset}")
        
        logger.info("Download process completed successfully")
        
    except Exception as e:
        logger.error(f"Error during download process: {str(e)}", exc_info=True)
        raise
