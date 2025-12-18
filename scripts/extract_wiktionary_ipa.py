#!/usr/bin/env python3
"""
Extract words with IPA phonemes from Wiktionary JSONL data.

This script processes a Wiktionary extract file in JSONL format and extracts
all words with IPA pronunciations for specified languages (English, Spanish,
Indonesian, and Swahili).

Usage:
    python extract_wiktionary_ipa.py <input_file.jsonl>
    
Output:
    extracted_words_ipa.csv - CSV file with word and ipa columns
"""

import json
import csv
import sys
from collections import defaultdict
from pathlib import Path


# Target language codes
TARGET_LANGUAGES = {'en', 'es', 'id', 'sw'}

# Language names for reporting
LANG_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'id': 'Indonesian',
    'sw': 'Swahili'
}


def extract_ipa_from_entry(entry):
    """
    Extract all IPA pronunciations from a single entry.
    
    Args:
        entry: Dictionary containing the parsed JSON entry
        
    Returns:
        List of IPA strings, or empty list if none found
    """
    ipa_list = []
    
    # Check if sounds field exists and is a list
    if 'sounds' not in entry or not isinstance(entry['sounds'], list):
        return ipa_list
    
    # Extract IPA from each sound object
    for sound in entry['sounds']:
        if isinstance(sound, dict) and 'ipa' in sound:
            ipa = sound['ipa']
            # Only add non-empty IPA values
            if ipa and isinstance(ipa, str) and ipa.strip():
                ipa_list.append(ipa.strip())
    
    return ipa_list


def process_jsonl_file(input_path):
    """
    Process the JSONL file and extract words with IPA.
    
    Args:
        input_path: Path to input JSONL file
    """
    # Use separate sets for each language to store unique (word, ipa) pairs
    word_ipa_by_lang = {
        'en': set(),
        'es': set(),
        'id': set(),
        'sw': set()
    }
    
    # Statistics
    stats = {
        'total_lines': 0,
        'valid_entries': 0,
        'entries_with_ipa': 0,
        'by_language': defaultdict(int),
        'errors': 0
    }
    
    print(f"Processing file: {input_path}")
    print(f"Target languages: {', '.join([LANG_NAMES[code] for code in TARGET_LANGUAGES])}")
    print("-" * 60)
    
    # Process file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1
            
            # Progress indicator every 10,000 lines
            if line_num % 10000 == 0:
                total_pairs = sum(len(pairs) for pairs in word_ipa_by_lang.values())
                print(f"Processed {line_num:,} lines... "
                      f"Found {total_pairs:,} unique word-IPA pairs")
            
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON
                entry = json.loads(line)
                
                # Check if this is a target language
                lang_code = entry.get('lang_code')
                if lang_code not in TARGET_LANGUAGES:
                    continue
                
                stats['valid_entries'] += 1
                
                # Get the word and convert to lowercase
                word = entry.get('word')
                if not word:
                    continue
                
                word = word.lower().strip()
                
                # Extract all IPA pronunciations
                ipa_list = extract_ipa_from_entry(entry)
                
                if ipa_list:
                    stats['entries_with_ipa'] += 1
                    stats['by_language'][lang_code] += 1
                    
                    # Add each (word, ipa) pair to the language-specific set
                    # The set automatically handles deduplication
                    for ipa in ipa_list:
                        word_ipa_by_lang[lang_code].add((word, ipa))
                
            except json.JSONDecodeError as e:
                stats['errors'] += 1
                if stats['errors'] <= 10:  # Only print first 10 errors
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 10:
                    print(f"Warning: Error processing line {line_num}: {e}")
    
    print(f"\nFinished processing {stats['total_lines']:,} lines")
    print("-" * 60)
    
    # Write results to separate CSV files for each language
    print(f"\nWriting results to separate CSV files...")
    
    output_files = {
        'en': 'data/english_words_ipa.csv',
        'es': 'data/spanish_words_ipa.csv',
        'id': 'data/indonesian_words_ipa.csv',
        'sw': 'data/swahili_words_ipa.csv'
    }
    
    for lang_code, output_path in output_files.items():
        pairs = word_ipa_by_lang[lang_code]
        if pairs:
            # Sort by word for better readability
            sorted_pairs = sorted(pairs, key=lambda x: (x[0].lower(), x[1]))
            
            with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['word', 'ipa'])
                writer.writerows(sorted_pairs)
            
            print(f"  {LANG_NAMES[lang_code]}: {len(pairs):,} pairs → {output_path}")
        else:
            print(f"  {LANG_NAMES[lang_code]}: 0 pairs (no file created)")
    
    # Calculate total unique pairs
    total_pairs = sum(len(pairs) for pairs in word_ipa_by_lang.values())
    
    # Print statistics
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total lines processed: {stats['total_lines']:,}")
    print(f"Entries in target languages: {stats['valid_entries']:,}")
    print(f"Entries with IPA data: {stats['entries_with_ipa']:,}")
    print(f"Total unique word-IPA pairs: {total_pairs:,}")
    print(f"Parse errors: {stats['errors']:,}")
    print("\nBreakdown by language:")
    for lang_code in sorted(TARGET_LANGUAGES):
        count = stats['by_language'][lang_code]
        pairs_count = len(word_ipa_by_lang[lang_code])
        print(f"  {LANG_NAMES[lang_code]} ({lang_code}): {count:,} entries → {pairs_count:,} unique pairs")
    print("\nOutput files created:")
    for lang_code, output_path in output_files.items():
        if word_ipa_by_lang[lang_code]:
            print(f"  - {output_path}")
    print("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python extract_wiktionary_ipa.py <input_file.jsonl>")
        print("\nExample:")
        print("  python extract_wiktionary_ipa.py data/raw-wiktextract-data.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Check if file is readable
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            pass
    except Exception as e:
        print(f"Error: Cannot read file: {e}")
        sys.exit(1)
    
    # Process the file
    try:
        process_jsonl_file(input_path)
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