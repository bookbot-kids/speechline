#!/usr/bin/env python3
"""
Analyze Ground Truth Coverage in Original Datasets

This script analyzes the original dataset directories to determine:
1. How many audio files exist
2. How many have corresponding .txt files
3. How many have corresponding .json files
4. Coverage percentages
"""

import os
from pathlib import Path
from collections import defaultdict

def analyze_directory(original_dir: str) -> dict:
    """
    Analyze a single directory for ground truth coverage.
    
    Args:
        original_dir: Directory containing original audio and ground truth files
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'audio_files': 0,
        'txt_files': 0,
        'json_files': 0,
        'audio_with_txt': 0,
        'audio_with_json': 0,
        'audio_with_either': 0,
        'audio_with_both': 0,
        'audio_with_neither': 0
    }
    
    if not os.path.exists(original_dir):
        return stats
    
    # Get all files
    try:
        all_files = os.listdir(original_dir)
    except Exception as e:
        print(f"  Error reading {original_dir}: {e}")
        return stats
    
    # Categorize files by extension
    audio_extensions = {'.aac', '.mp3', '.wav', '.m4a', '.flac'}
    audio_base_names = set()
    txt_base_names = set()
    json_base_names = set()
    
    for filename in all_files:
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext in audio_extensions:
            audio_base_names.add(name)
        elif ext == '.txt':
            txt_base_names.add(name)
        elif ext == '.json':
            json_base_names.add(name)
    
    # Calculate statistics
    stats['audio_files'] = len(audio_base_names)
    stats['txt_files'] = len(txt_base_names)
    stats['json_files'] = len(json_base_names)
    
    # Check coverage for each audio file
    for base_name in audio_base_names:
        has_txt = base_name in txt_base_names
        has_json = base_name in json_base_names
        
        if has_txt:
            stats['audio_with_txt'] += 1
        if has_json:
            stats['audio_with_json'] += 1
        if has_txt or has_json:
            stats['audio_with_either'] += 1
        if has_txt and has_json:
            stats['audio_with_both'] += 1
        if not has_txt and not has_json:
            stats['audio_with_neither'] += 1
    
    return stats


def main():
    """
    Main function to analyze all language directories.
    """
    # Configuration
    base_original_dir = "/mnt/Store06/Datasets"
    
    # Language to path mapping
    lang_mappings = {
        'en': 'English/S3',
        'id': 'Indonesian/S3',
        'sw': 'Swahili/S3'
    }
    
    # Get directories to analyze (same as segment directories)
    base_segment_dir = "/mnt/Store07/Bookbot"
    
    print("=" * 80)
    print("Ground Truth Coverage Analysis")
    print("=" * 80)
    print()
    
    try:
        all_dirs = [d for d in os.listdir(base_segment_dir) 
                   if os.path.isdir(os.path.join(base_segment_dir, d))]
    except Exception as e:
        print(f"Error reading directory {base_segment_dir}: {e}")
        return
    
    # Group directories by base language
    lang_dirs = defaultdict(list)
    for dir_name in all_dirs:
        base_lang = dir_name.split('-')[0]
        if base_lang in lang_mappings:
            lang_dirs[base_lang].append(dir_name)
    
    # Overall statistics
    total_stats = {
        'audio_files': 0,
        'txt_files': 0,
        'json_files': 0,
        'audio_with_txt': 0,
        'audio_with_json': 0,
        'audio_with_either': 0,
        'audio_with_both': 0,
        'audio_with_neither': 0,
        'dirs_analyzed': 0
    }
    
    # Process each language group
    for base_lang, dirs in sorted(lang_dirs.items()):
        print(f"\n{'=' * 80}")
        print(f"Analyzing {base_lang.upper()} directories ({len(dirs)} total)")
        print(f"{'=' * 80}")
        
        lang_stats = {
            'audio_files': 0,
            'txt_files': 0,
            'json_files': 0,
            'audio_with_txt': 0,
            'audio_with_json': 0,
            'audio_with_either': 0,
            'audio_with_both': 0,
            'audio_with_neither': 0
        }
        
        for dir_name in sorted(dirs):
            original_dir = os.path.join(base_original_dir, lang_mappings[base_lang], dir_name)
            
            stats = analyze_directory(original_dir)
            
            if stats['audio_files'] > 0:
                total_stats['dirs_analyzed'] += 1
                
                # Add to language totals
                for key in lang_stats.keys():
                    lang_stats[key] += stats[key]
                
                # Print directory details if significant
                if stats['audio_files'] > 100:  # Only show dirs with >100 audio files
                    txt_pct = (stats['audio_with_txt'] / stats['audio_files'] * 100) if stats['audio_files'] > 0 else 0
                    json_pct = (stats['audio_with_json'] / stats['audio_files'] * 100) if stats['audio_files'] > 0 else 0
                    either_pct = (stats['audio_with_either'] / stats['audio_files'] * 100) if stats['audio_files'] > 0 else 0
                    
                    print(f"\n  {dir_name}:")
                    print(f"    Audio files: {stats['audio_files']:,}")
                    print(f"    With .txt: {stats['audio_with_txt']:,} ({txt_pct:.1f}%)")
                    print(f"    With .json: {stats['audio_with_json']:,} ({json_pct:.1f}%)")
                    print(f"    With either: {stats['audio_with_either']:,} ({either_pct:.1f}%)")
                    print(f"    With neither: {stats['audio_with_neither']:,} ({100-either_pct:.1f}%)")
        
        # Print language summary
        if lang_stats['audio_files'] > 0:
            print(f"\n  {base_lang.upper()} TOTALS:")
            print(f"    Total audio files: {lang_stats['audio_files']:,}")
            print(f"    With .txt: {lang_stats['audio_with_txt']:,} ({lang_stats['audio_with_txt']/lang_stats['audio_files']*100:.1f}%)")
            print(f"    With .json: {lang_stats['audio_with_json']:,} ({lang_stats['audio_with_json']/lang_stats['audio_files']*100:.1f}%)")
            print(f"    With either: {lang_stats['audio_with_either']:,} ({lang_stats['audio_with_either']/lang_stats['audio_files']*100:.1f}%)")
            print(f"    With both: {lang_stats['audio_with_both']:,} ({lang_stats['audio_with_both']/lang_stats['audio_files']*100:.1f}%)")
            print(f"    With neither: {lang_stats['audio_with_neither']:,} ({lang_stats['audio_with_neither']/lang_stats['audio_files']*100:.1f}%)")
            
            # Add to total
            for key in total_stats.keys():
                if key != 'dirs_analyzed':
                    total_stats[key] += lang_stats[key]
    
    # Print final summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    if total_stats['audio_files'] > 0:
        print(f"Directories analyzed:           {total_stats['dirs_analyzed']}")
        print(f"Total audio files:              {total_stats['audio_files']:,}")
        print(f"Audio with .txt ground truth:   {total_stats['audio_with_txt']:,} ({total_stats['audio_with_txt']/total_stats['audio_files']*100:.1f}%)")
        print(f"Audio with .json ground truth:  {total_stats['audio_with_json']:,} ({total_stats['audio_with_json']/total_stats['audio_files']*100:.1f}%)")
        print(f"Audio with either format:       {total_stats['audio_with_either']:,} ({total_stats['audio_with_either']/total_stats['audio_files']*100:.1f}%)")
        print(f"Audio with both formats:        {total_stats['audio_with_both']:,} ({total_stats['audio_with_both']/total_stats['audio_files']*100:.1f}%)")
        print(f"Audio with NO ground truth:     {total_stats['audio_with_neither']:,} ({total_stats['audio_with_neither']/total_stats['audio_files']*100:.1f}%)")
    else:
        print("No audio files found in original directories")
    
    print("=" * 80)


if __name__ == '__main__':
    main()