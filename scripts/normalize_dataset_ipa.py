#!/usr/bin/env python3
"""
Dataset IPA Normalization Script

This script normalizes IPA transcriptions in datasets to canonical form.
It can process various dataset formats and update them with normalized IPA.

This ensures compatibility with the lexicon that has been processed with
canonical normalization.

Usage:
    python scripts/normalize_dataset_ipa.py <input_file> <output_file> [--format <format>]
    
Formats:
    - jsonl: JSON Lines format (default)
    - csv: CSV format with 'ipa' or 'transcript' column
    - tsv: TSV format with 'ipa' or 'transcript' column
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from ipa_to_class_mapping import normalize_ipa, get_normalization_stats


def normalize_jsonl(input_file: Path, output_file: Path) -> Dict[str, int]:
    """
    Normalize IPA in JSONL format.
    
    Expected format: Each line is a JSON object with 'phonemes' field
    containing space-separated IPA phonemes.
    """
    stats = {
        'total_lines': 0,
        'normalized_count': 0,
        'errors': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            stats['total_lines'] += 1
            
            try:
                data = json.loads(line.strip())
                
                # Check for phonemes field (LibriPhone format)
                if 'phonemes' in data:
                    original_ipa = data['phonemes']
                    # LibriPhone uses space-separated phonemes
                    normalized_ipa = normalize_ipa(original_ipa, merge_spaces=True)
                    
                    if normalized_ipa != original_ipa:
                        stats['normalized_count'] += 1
                    
                    # Update with normalized version
                    data['phonemes_normalized'] = normalized_ipa
                    data['phonemes_original'] = original_ipa
                    
                # Check for transcript field
                elif 'transcript' in data:
                    original_ipa = data['transcript']
                    normalized_ipa = normalize_ipa(original_ipa, merge_spaces=True)
                    
                    if normalized_ipa != original_ipa:
                        stats['normalized_count'] += 1
                    
                    data['transcript_normalized'] = normalized_ipa
                    data['transcript_original'] = original_ipa
                
                # Write updated line
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                stats['errors'] += 1
                # Write original line on error
                fout.write(line)
            
            # Progress indicator
            if stats['total_lines'] % 1000 == 0:
                print(f"Processed {stats['total_lines']:,} lines...")
    
    return stats


def normalize_csv_tsv(input_file: Path, output_file: Path, delimiter: str = ',') -> Dict[str, int]:
    """
    Normalize IPA in CSV/TSV format.
    
    Expected format: CSV/TSV with 'ipa' or 'transcript' column.
    """
    stats = {
        'total_rows': 0,
        'normalized_count': 0,
        'errors': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter=delimiter)
        fieldnames = reader.fieldnames
        
        # Determine which column to normalize
        ipa_column = None
        if 'ipa' in fieldnames:
            ipa_column = 'ipa'
        elif 'transcript' in fieldnames:
            ipa_column = 'transcript'
        elif 'phonemes' in fieldnames:
            ipa_column = 'phonemes'
        else:
            raise ValueError(f"No IPA column found. Available columns: {fieldnames}")
        
        # Add normalized column
        new_fieldnames = list(fieldnames) + [f'{ipa_column}_normalized', f'{ipa_column}_original']
        
        with open(output_file, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=new_fieldnames, delimiter=delimiter)
            writer.writeheader()
            
            for row in reader:
                stats['total_rows'] += 1
                
                try:
                    original_ipa = row[ipa_column]
                    normalized_ipa = normalize_ipa(original_ipa, merge_spaces=True)
                    
                    if normalized_ipa != original_ipa:
                        stats['normalized_count'] += 1
                    
                    # Add normalized columns
                    row[f'{ipa_column}_normalized'] = normalized_ipa
                    row[f'{ipa_column}_original'] = original_ipa
                    
                    writer.writerow(row)
                    
                except Exception as e:
                    print(f"Error processing row {stats['total_rows']}: {e}")
                    stats['errors'] += 1
                
                # Progress indicator
                if stats['total_rows'] % 1000 == 0:
                    print(f"Processed {stats['total_rows']:,} rows...")
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Normalize IPA transcriptions in datasets to canonical form',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Normalize LibriPhone JSONL dataset
    python scripts/normalize_dataset_ipa.py data/libriphone.jsonl data/libriphone_normalized.jsonl
    
    # Normalize CSV dataset
    python scripts/normalize_dataset_ipa.py data/dataset.csv data/dataset_normalized.csv --format csv
    
    # Normalize TSV dataset
    python scripts/normalize_dataset_ipa.py data/dataset.tsv data/dataset_normalized.tsv --format tsv
        """
    )
    
    parser.add_argument('input_file', type=Path, help='Input dataset file')
    parser.add_argument('output_file', type=Path, help='Output normalized dataset file')
    parser.add_argument('--format', choices=['jsonl', 'csv', 'tsv'], default='jsonl',
                        help='Dataset format (default: jsonl)')
    
    args = parser.parse_args()
    
    # Check input exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    print("=" * 80)
    print("DATASET IPA NORMALIZATION")
    print("=" * 80)
    print(f"\nInput:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Format: {args.format}")
    print()
    
    # Process based on format
    try:
        if args.format == 'jsonl':
            stats = normalize_jsonl(args.input_file, args.output_file)
            total_key = 'total_lines'
        else:
            delimiter = '\t' if args.format == 'tsv' else ','
            stats = normalize_csv_tsv(args.input_file, args.output_file, delimiter)
            total_key = 'total_rows'
        
        # Print summary
        print("\n" + "=" * 80)
        print("NORMALIZATION SUMMARY")
        print("=" * 80)
        print(f"Total records:     {stats[total_key]:,}")
        print(f"Normalized:        {stats['normalized_count']:,} ({stats['normalized_count']/stats[total_key]*100:.1f}%)")
        print(f"Unchanged:         {stats[total_key] - stats['normalized_count']:,}")
        print(f"Errors:            {stats['errors']:,}")
        print()
        print(f"✓ Output saved to: {args.output_file}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nError during normalization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())