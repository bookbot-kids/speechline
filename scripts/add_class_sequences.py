#!/usr/bin/env python3
"""
Process IPA CSV files and add phonetic class sequences.

This script reads CSV files with word/IPA pairs and adds a class_sequence column
by converting each IPA transcription to its phonetic class representation.

Outputs:
1. Full CSV with all word-IPA pairs + class sequences
2. Unique IPA CSV with deduplicated IPA transcriptions

Usage:
    python scripts/add_class_sequences.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ipa_to_class_mapping import parse_ipa_sequence, validate_ipa_string


def process_csv(input_path, output_full_path, output_unique_path, output_first_occurrence_path):
    """
    Process IPA CSV and generate three output files.
    
    Args:
        input_path: Path to input CSV (word, ipa columns)
        output_full_path: Path for full output CSV (word, ipa, class_sequence)
        output_unique_path: Path for unique IPA output CSV (ipa, class_sequence, word_examples)
        output_first_occurrence_path: Path for cumulative IPA symbols CSV (word, ipa, class_sequence) - includes rows with new IPA symbols
    
    Returns:
        dict: Statistics including 'seen_ipa_symbols' set
    """
    stats = {
        'total_rows': 0,
        'parsed_ok': 0,
        'parse_errors': 0,
        'unique_ipa': 0,
        'error_examples': []
    }
    
    # Store all rows for full output
    full_rows = []
    
    # Store first occurrence rows (includes row if it has any new IPA symbols)
    first_occurrence_rows = []
    seen_ipa_symbols = set()  # Track individual IPA symbols (characters), not full strings
    
    # Store unique IPA with example words
    unique_ipa = {}  # ipa -> (class_sequence, [example_words])
    
    print("=" * 70)
    print("Processing IPA CSV")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_full_path} (full - all rows)")
    print(f"        {output_first_occurrence_path} (cumulative IPA symbols - rows with new symbols)")
    print(f"        {output_unique_path} (unique - aggregated with examples)")
    print()
    
    # Read and process input CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total_rows'] += 1
            
            # Progress indicator
            if stats['total_rows'] % 10000 == 0:
                print(f"  Processed {stats['total_rows']:,} rows... "
                      f"{stats['parsed_ok']:,} OK, {stats['parse_errors']:,} errors")
            
            word = row['word']
            ipa = row['ipa']
            
            # Parse IPA to class sequence
            try:
                class_seq = parse_ipa_sequence(ipa, strict=False)
                stats['parsed_ok'] += 1
                
                row_data = {
                    'word': word,
                    'ipa': ipa,
                    'class_sequence': class_seq
                }
                
                # Add to full rows
                full_rows.append(row_data)
                
                # Add to first occurrence if this row contains any new IPA symbols
                # Extract individual IPA symbols (characters) from the IPA string
                ipa_symbols = set(ipa)
                
                # Check if there are any new symbols not seen before
                new_symbols = ipa_symbols - seen_ipa_symbols
                
                if new_symbols:
                    # This row has at least one new IPA symbol, include it
                    first_occurrence_rows.append(row_data)
                    seen_ipa_symbols.update(new_symbols)
                    stats['unique_ipa'] += 1
                
                # Track unique IPA with examples
                if ipa not in unique_ipa:
                    unique_ipa[ipa] = (class_seq, [])
                
                # Add example word (limit to 5 examples per IPA)
                if len(unique_ipa[ipa][1]) < 5:
                    unique_ipa[ipa][1].append(word)
                
            except Exception as e:
                stats['parse_errors'] += 1
                
                # Store error example
                if len(stats['error_examples']) < 10:
                    stats['error_examples'].append((word, ipa, str(e)[:100]))
                
                # Add with empty class sequence
                full_rows.append({
                    'word': word,
                    'ipa': ipa,
                    'class_sequence': ''
                })
    
    print(f"\nFinished processing {stats['total_rows']:,} rows")
    print()
    
    # Write full output CSV
    print("Writing full CSV (all rows)...")
    with open(output_full_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'ipa', 'class_sequence'])
        writer.writeheader()
        writer.writerows(full_rows)
    print(f"  Wrote {len(full_rows):,} rows to {output_full_path}")
    
    # Write first occurrence CSV
    print("\nWriting cumulative IPA symbols CSV (rows with new symbols)...")
    with open(output_first_occurrence_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'ipa', 'class_sequence'])
        writer.writeheader()
        writer.writerows(first_occurrence_rows)
    print(f"  Wrote {len(first_occurrence_rows):,} rows to {output_first_occurrence_path}")
    
    # Write unique IPA CSV with examples
    print("\nWriting aggregated unique IPA CSV...")
    unique_rows = []
    for ipa, (class_seq, examples) in sorted(unique_ipa.items()):
        unique_rows.append({
            'ipa': ipa,
            'class_sequence': class_seq,
            'word_examples': ', '.join(examples)
        })
    
    with open(output_unique_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ipa', 'class_sequence', 'word_examples'])
        writer.writeheader()
        writer.writerows(unique_rows)
    print(f"  Wrote {len(unique_rows):,} rows to {output_unique_path}")
    
    # Print statistics
    print()
    print("=" * 70)
    print("PROCESSING STATISTICS")
    print("=" * 70)
    print(f"Total rows:              {stats['total_rows']:,}")
    print(f"Successfully parsed:     {stats['parsed_ok']:,} ({stats['parsed_ok']/stats['total_rows']*100:.1f}%)")
    print(f"Parse errors:            {stats['parse_errors']:,} ({stats['parse_errors']/stats['total_rows']*100:.1f}%)")
    print(f"Rows with new symbols:   {stats['unique_ipa']:,}")
    print(f"Unique IPA symbols seen: {len(seen_ipa_symbols):,}")
    print(f"Compression ratio:       {stats['total_rows']/stats['unique_ipa']:.1f}x")
    
    if stats['error_examples']:
        print(f"\nFirst {len(stats['error_examples'])} parse errors:")
        for word, ipa, error in stats['error_examples']:
            print(f"  {word} ({ipa}): {error}")
    
    print()
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    
    # Add seen symbols to stats for later reporting
    stats['seen_ipa_symbols'] = seen_ipa_symbols
    
    return stats


def analyze_class_sequences(output_full_path):
    """
    Analyze the generated class sequences.
    """
    print()
    print("=" * 70)
    print("CLASS SEQUENCE ANALYSIS")
    print("=" * 70)
    
    class_counts = defaultdict(int)
    sequence_lengths = []
    
    with open(output_full_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row['class_sequence']
            if seq:
                sequence_lengths.append(len(seq))
                for char in seq:
                    class_counts[char] += 1
    
    # Print class frequency
    print("\nPhonetic class frequency:")
    total_classes = sum(class_counts.values())
    for class_sym in sorted(class_counts.keys()):
        count = class_counts[class_sym]
        pct = count / total_classes * 100
        print(f"  {class_sym}: {count:,} ({pct:.1f}%)")
    
    # Print sequence length statistics
    if sequence_lengths:
        avg_len = sum(sequence_lengths) / len(sequence_lengths)
        min_len = min(sequence_lengths)
        max_len = max(sequence_lengths)
        print(f"\nSequence length statistics:")
        print(f"  Average: {avg_len:.1f}")
        print(f"  Min: {min_len}")
        print(f"  Max: {max_len}")
    
    print()


def main():
    """Main entry point."""
    # Paths
    input_path = Path('data/english_words_ipa.csv')
    output_full_path = Path('data/english_words_with_classes.csv')
    output_first_occurrence_path = Path('data/english_unique_first_occurrence.csv')
    output_unique_path = Path('data/english_unique_ipa_classes.csv')
    
    # Check input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Process CSV
    try:
        stats = process_csv(input_path, output_full_path, output_unique_path, output_first_occurrence_path)
        
        # Analyze results
        analyze_class_sequences(output_full_path)
        
        print(f"Success! Generated three files:")
        print(f"  1. {output_full_path} - Full dataset ({stats['total_rows']:,} rows)")
        print(f"  2. {output_first_occurrence_path} - Cumulative IPA symbols ({stats['unique_ipa']:,} rows, covering {len(stats['seen_ipa_symbols']):,} unique symbols)")
        print(f"  3. {output_unique_path} - Aggregated with examples")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()