#!/usr/bin/env python3
"""
Create Ground Truth Files for Segmented Audio

This script creates ground truth .txt files for each segmented audio file by copying
the original ground truth text from the non-segmented source files.

File Structure:
- Segmented audio files: {uuid}_{timestamp}_Speaker_{speaker_id}_{start_time}-{end_time}.{ext}
- Ground truth files: {uuid}_{timestamp}_Speaker_{speaker_id}_{start_time}-{end_time}.txt
  - Contains: Single line with original ground truth text only

- Source files: {uuid}_{timestamp}.txt or .json (ground truth from original recording)
  - .txt: plain text
  - .json: JSON with "text" field containing ground truth
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def extract_base_name(filename: str) -> str:
    """
    Extract the base name (uuid_timestamp) from a segmented filename.
    
    Handles multiple patterns:
    - Standard: {uuid}_{timestamp}_Speaker_{speaker_id}_{times}.txt
    - With device: {uuid}_{timestamp}_{device}_Speaker_{speaker_id}_{times}.txt
    - Guest: guest_{uuid}_{timestamp}_Speaker_{speaker_id}_{times}.txt
    - Guest with device: guest_{uuid}_{timestamp}_{device}_Speaker_{speaker_id}_{times}.txt
    
    Example:
        Input: "000ad660-7fa5-4965-a4d6-9c5df02a4196_1726097791618_Speaker_6E652_0002.022-0004.300.txt"
        Output: "000ad660-7fa5-4965-a4d6-9c5df02a4196_1726097791618"
        
        Input: "guest_123bef41-3789-4cdd-829f-260bc1603904_1747388595912_iPhone13,4_Speaker_38D36_0002.562-0006.764.txt"
        Output: "guest_123bef41-3789-4cdd-829f-260bc1603904_1747388595912"
    
    Args:
        filename: The segmented file name
        
    Returns:
        Base name containing UUID and timestamp (with optional guest_ prefix)
    """
    # Try pattern with guest_ prefix first: guest_{uuid}_{timestamp}_...
    match = re.match(r'^(guest_[0-9a-f\-]+_\d+)(?:_[^_]+)?_Speaker_', filename)
    if match:
        return match.group(1)
    
    # Try standard pattern: {uuid}_{timestamp}_...
    match = re.match(r'^([0-9a-f\-]+_\d+)(?:_[^_]+)?_Speaker_', filename)
    if match:
        return match.group(1)
    
    return ""


def find_ground_truth(base_name: str, original_dir: str) -> Optional[str]:
    """
    Find and extract ground truth text from original .txt or .json file.
    
    Checks for files in this order:
    1. {base_name}.txt - plain text ground truth
    2. {base_name}.json - JSON with "text" field
    
    Args:
        base_name: Base filename (uuid_timestamp)
        original_dir: Directory containing original ground truth files
        
    Returns:
        Ground truth text if found, None otherwise
    """
    # Check for .txt file first
    txt_path = os.path.join(original_dir, f"{base_name}.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"  Warning: Error reading {txt_path}: {e}")
            return None
    
    # Check for .json file
    json_path = os.path.join(original_dir, f"{base_name}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract "text" field from JSON
                text = data.get("text", "")
                if text:
                    return text.strip()
                else:
                    print(f"  Warning: No 'text' field in {json_path}")
                    return None
        except json.JSONDecodeError as e:
            print(f"  Warning: Invalid JSON in {json_path}: {e}")
            return None
        except Exception as e:
            print(f"  Warning: Error reading {json_path}: {e}")
            return None
    
    return None


def create_ground_truth_file(segment_filename: str, segment_dir: str, ground_truth: str) -> bool:
    """
    Create a ground truth .txt file for a segment.
    
    Creates a new .txt file with the same name as the segment file,
    containing only the original ground truth text.
    
    Args:
        segment_filename: Name of the segment file (e.g., uuid_timestamp_Speaker_id_times.aac)
        segment_dir: Directory containing segment files
        ground_truth: Ground truth text to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create .txt filename by replacing extension
        base_name = os.path.splitext(segment_filename)[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(segment_dir, txt_filename)
        
        # Write ground truth as single line
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(ground_truth)
        
        return True
        
    except Exception as e:
        print(f"  Error creating {txt_path}: {e}")
        return False


def process_single_file(filename: str, segment_dir: str, original_dir: str) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single segment file by creating its ground truth .txt file.
    
    Args:
        filename: Name of the segment file (audio file with .aac, .mp3, etc.)
        segment_dir: Directory containing segment files
        original_dir: Directory containing original ground truth files
        
    Returns:
        Tuple of (filename, success, error_message)
    """
    # Extract base name
    base_name = extract_base_name(filename)
    if not base_name:
        return (filename, False, "Could not extract base name")
    
    # Find ground truth
    ground_truth = find_ground_truth(base_name, original_dir)
    if ground_truth is None:
        return (filename, False, "No ground truth found")
    
    # Create ground truth .txt file
    if create_ground_truth_file(filename, segment_dir, ground_truth):
        return (filename, True, None)
    else:
        return (filename, False, "Failed to create ground truth file")


def process_directory(segment_dir: str, original_dir: str, max_workers: int = 10) -> Dict[str, int]:
    """
    Process all segment files in a directory.
    
    Args:
        segment_dir: Directory containing segment files
        original_dir: Directory containing original ground truth files
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'processed': 0,
        'updated': 0,
        'no_ground_truth': 0,
        'errors': 0
    }
    
    # Verify directories exist
    if not os.path.exists(segment_dir):
        print(f"  ⚠️  Segment directory not found: {segment_dir}")
        return stats
    
    if not os.path.exists(original_dir):
        print(f"  ⚠️  Original directory not found: {original_dir}")
        return stats
    
    # Get all audio segment files (not .txt files, as those are the ground truth we're creating)
    segment_files = [f for f in os.listdir(segment_dir)
                    if f.endswith(('.aac', '.mp3', '.wav', '.m4a', '.flac'))]
    
    if not segment_files:
        return stats
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, filename, segment_dir, original_dir): filename
            for filename in segment_files
        }
        
        for future in as_completed(futures):
            stats['processed'] += 1
            filename, success, error = future.result()
            
            if success:
                stats['updated'] += 1
            elif error == "No ground truth found":
                stats['no_ground_truth'] += 1
            else:
                stats['errors'] += 1
    
    return stats


def main():
    """
    Main function to process all language directories in /mnt/Store07/Bookbot.
    """
    # Configuration
    base_segment_dir = "/mnt/Store07/Bookbot"
    base_original_dir = "/mnt/Store06/Datasets"
    
    # Language to path mapping
    lang_mappings = {
        'en': 'English/S3',
        'id': 'Indonesian/S3',
        'sw': 'Swahili/S3'
    }
    
    print("=" * 80)
    print("Creating Ground Truth Files for Segmented Audio - Batch Processing")
    print("=" * 80)
    print(f"Base segment directory: {base_segment_dir}")
    print(f"Base original directory: {base_original_dir}")
    print()
    
    # Get all language directories
    try:
        all_dirs = [d for d in os.listdir(base_segment_dir) 
                   if os.path.isdir(os.path.join(base_segment_dir, d))]
    except Exception as e:
        print(f"Error reading directory {base_segment_dir}: {e}")
        return
    
    print(f"Found {len(all_dirs)} language directories to process\n")
    
    # Group directories by base language
    lang_dirs = defaultdict(list)
    for dir_name in all_dirs:
        # Extract base language (e.g., 'en' from 'en-au')
        base_lang = dir_name.split('-')[0]
        if base_lang in lang_mappings:
            lang_dirs[base_lang].append(dir_name)
    
    # Overall statistics
    total_stats = {
        'processed': 0,
        'updated': 0,
        'no_ground_truth': 0,
        'errors': 0,
        'dirs_processed': 0,
        'dirs_skipped': 0
    }
    
    # Process each language group
    for base_lang, dirs in sorted(lang_dirs.items()):
        print(f"\n{'=' * 80}")
        print(f"Processing {base_lang.upper()} directories ({len(dirs)} total)")
        print(f"{'=' * 80}")
        
        for dir_name in sorted(dirs):
            segment_dir = os.path.join(base_segment_dir, dir_name)
            original_dir = os.path.join(base_original_dir, lang_mappings[base_lang], dir_name)
            
            print(f"\n📁 {dir_name}:")
            print(f"   Segment: {segment_dir}")
            print(f"   Original: {original_dir}")
            
            # Process directory
            stats = process_directory(segment_dir, original_dir)
            
            if stats['processed'] > 0:
                total_stats['dirs_processed'] += 1
                total_stats['processed'] += stats['processed']
                total_stats['updated'] += stats['updated']
                total_stats['no_ground_truth'] += stats['no_ground_truth']
                total_stats['errors'] += stats['errors']
                
                print(f"   ✓ Processed: {stats['processed']} segment files")
                print(f"   ✓ Created ground truth: {stats['updated']} files")
                if stats['no_ground_truth'] > 0:
                    print(f"   ⚠ No ground truth: {stats['no_ground_truth']} files")
                if stats['errors'] > 0:
                    print(f"   ✗ Errors: {stats['errors']} files")
            else:
                total_stats['dirs_skipped'] += 1
                print(f"   ⊘ No files to process")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Directories processed:          {total_stats['dirs_processed']}")
    print(f"Directories skipped:            {total_stats['dirs_skipped']}")
    print(f"Total segment files processed:  {total_stats['processed']}")
    print(f"Ground truth files created:     {total_stats['updated']}")
    print(f"No ground truth found:          {total_stats['no_ground_truth']}")
    print(f"Errors:                         {total_stats['errors']}")
    print("=" * 80)


if __name__ == '__main__':
    main()