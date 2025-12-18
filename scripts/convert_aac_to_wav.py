#!/usr/bin/env python3
"""
Convert AAC audio files to WAV format for speechline processing.
Recursively converts all .aac files in a directory while preserving structure.
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import subprocess
import sys


def convert_aac_to_wav(input_file: str, output_file: str) -> bool:
    """
    Convert a single AAC file to WAV using ffmpeg.
    
    Args:
        input_file: Path to input AAC file
        output_file: Path to output WAV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use ffmpeg to convert AAC to WAV (16kHz, mono)
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono
            '-y',            # overwrite output file
            output_file
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert AAC files to WAV format for speechline processing'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing AAC files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save WAV files'
    )
    parser.add_argument(
        '--preserve_structure',
        action='store_true',
        default=True,
        help='Preserve directory structure in output'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        sys.exit(1)
    
    # Find all AAC files
    aac_files = list(input_path.rglob("*.aac"))
    
    if not aac_files:
        print(f"No AAC files found in {input_path}")
        sys.exit(1)
    
    print(f"Found {len(aac_files)} AAC files to convert")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for aac_file in tqdm(aac_files, desc="Converting AAC to WAV"):
        # Determine output file path
        if args.preserve_structure:
            # Preserve relative path structure
            rel_path = aac_file.relative_to(input_path)
            wav_file = output_path / rel_path.with_suffix('.wav')
        else:
            # Flat structure in output directory
            wav_file = output_path / aac_file.with_suffix('.wav').name
        
        # Create parent directory if needed
        wav_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert the file
        if convert_aac_to_wav(str(aac_file), str(wav_file)):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_path}")


if __name__ == '__main__':
    main()