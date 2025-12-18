import argparse
import json
import os
from pathlib import Path

def process_tsv_file(tsv_path):
    """Process a single TSV file and extract required information."""
    transcript = []
    duration = 0
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Skip empty files
        if not lines:
            return None
            
        # Process each line
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            if len(parts) >= 3:  # Ensure we have start_time, end_time, and word
                word = parts[2]
                transcript.append(word)
                
                # Update duration from the end_time of the last valid line
                duration = float(parts[1])
    
    # Get audio filepath from TSV filename
    audio_path = str(tsv_path).replace('.tsv', '.wav')
    
    return {
        "audio_filepath": audio_path,
        "duration": duration,
        "text": " ".join(transcript)
    }

def create_manifest(input_path, output_path):
    """Create manifest file from all TSV files in input directory."""
    input_dir = Path(input_path)
    manifest_entries = []
    
    # Process all TSV files in the input directory
    for tsv_file in input_dir.glob('**/*.tsv'):
        entry = process_tsv_file(tsv_file)
        if entry:
            manifest_entries.append(entry)
    
    # Write manifest file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')

def main():
    parser = argparse.ArgumentParser(
        description='Create manifest file for NVIDIA Speech Data Explorer from TSV files'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to directory containing TSV files'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output manifest file'
    )
    
    args = parser.parse_args()
    create_manifest(args.input_path, args.output_path)

if __name__ == '__main__':
    main()