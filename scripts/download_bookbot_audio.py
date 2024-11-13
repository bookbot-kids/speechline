# %%
import boto3
import os
from pathlib import Path
import subprocess
from p_tqdm import p_map
from functools import partial


def get_s3_client():
    """Initialize and return an S3 client."""
    return boto3.client('s3')

def list_subfolders(bucket_name, language):
    """List subfolders in the given S3 bucket and folder path that start with the specified language.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        folder_path (str): Prefix/folder path in the bucket
        language (str): Language prefix to filter folders by (e.g. 'en', 'es')
    """
    folder_path = "dropbox/"
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path, Delimiter='/')
    
    if 'CommonPrefixes' in response:
        # Filter folders that start with the language prefix
        language_folders = []
        for prefix in response['CommonPrefixes']:
            folder_name = prefix['Prefix'].rstrip('/').split('/')[-1]
            if folder_name.startswith(language):
                language_folders.append(prefix['Prefix'])
        return language_folders
    else:
        return []

def convert_audio_to_wav(input_audio_path: str, num_channels: int = 1, sampling_rate: int = 16000) -> None:
    """Convert audio file to WAV using ffmpeg."""
    output_path = os.path.splitext(input_audio_path)[0] + '.wav'
    cmd = [
        'ffmpeg', '-y',
        '-i', input_audio_path,
        '-acodec', 'pcm_s16le',
        '-ar', str(sampling_rate),
        '-ac', str(num_channels),
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
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

def download_audio_and_transcripts(bucket_name, folder_path, output_dir, convert_to_wav=True):
    """Download audio files and their corresponding transcripts."""
    s3 = get_s3_client()
    
    subfolder_name = folder_path.rstrip('/').split('/')[-1]
    output_path = os.path.join(output_dir, subfolder_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    folder_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    
    if 'Contents' in folder_response:
        # First, find all transcript files
        transcript_files = set()
        for obj in folder_response['Contents']:
            if obj['Key'].endswith('.txt'):
                base_name = os.path.splitext(os.path.basename(obj['Key']))[0]
                transcript_files.add(base_name)
        
        print(f"\nProcessing files from {folder_path} to {output_path}")
        
        # Process audio files that have matching transcripts
        audio_conversion_tasks = []
        for obj in folder_response['Contents']:
            file_key = obj['Key']
            base_name = os.path.splitext(os.path.basename(file_key))[0]
            
            if file_key.endswith('.aac') and base_name in transcript_files:
                # Download transcript
                txt_key = os.path.splitext(file_key)[0] + '.txt'
                txt_path = os.path.join(output_path, os.path.basename(txt_key))
                print(f"Downloading transcript: {os.path.basename(txt_key)}")
                s3.download_file(bucket_name, txt_key, txt_path)
                
                # Prepare WAV conversion task
                wav_path = os.path.join(output_path, f"{base_name}.wav")
                audio_conversion_tasks.append((bucket_name, file_key, wav_path))
        
        # Convert all audio files in parallel
        if audio_conversion_tasks:
            print("\nConverting audio files to WAV format...")
            fn = partial(lambda x: download_and_convert_to_wav(*x))
            _ = p_map(fn, audio_conversion_tasks)



if __name__ == "__main__":
    language = "en"
    bucket = "bookbot-speech"
    subfolders = list_subfolders(bucket, language)
    
    output_dir = "bookbot_en"
    download_audio_and_transcripts(bucket, "dropbox/en-ar/", output_dir, convert_to_wav=True)
# %%
