#!/usr/bin/env python3
"""
Consolidated IPA Processing Pipeline

This script processes word lists with IPA transcriptions and:
1. Normalizes IPA to canonical form for consistency
2. Applies accent rules to generate phonetic variations
3. Adds phonetic class sequences using updated IPA_TO_CLASS mapping
4. Automatically deduplicates (word, ipa) combinations
5. Outputs a single consolidated CSV file

Usage:
    python scripts/process_words.py
"""

import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from ipa_to_class_mapping import parse_ipa_sequence, normalize_ipa, IPA_TO_CLASS


def get_vowels_and_consonants():
    """
    Extract all vowels and consonants from IPA_TO_CLASS mapping.
    Returns: (vowel_chars, consonant_chars)
    """
    vowels = set()
    consonants = set()
    
    for ipa_char, class_char in IPA_TO_CLASS.items():
        # Skip multi-character sequences for regex character class
        if len(ipa_char) > 1:
            continue
            
        if class_char == 'ə':  # Vowel class
            vowels.add(ipa_char)
        else:  # All other classes are consonants
            consonants.add(ipa_char)
    
    return vowels, consonants


def expand_regex_pattern(pattern):
    """
    Expand phonetic class placeholders in regex patterns.
    
    V = vowels
    C = consonants
    # = word boundary
    """
    vowels, consonants = get_vowels_and_consonants()
    
    # Create regex character classes
    vowel_pattern = '[' + ''.join(sorted(vowels)) + ']'
    consonant_pattern = '[' + ''.join(sorted(consonants)) + ']'
    
    # Word boundary (start/end or space)
    boundary = r'(?:^|$|\s)'
    
    # Replace placeholders
    expanded = pattern.replace('V', vowel_pattern)
    expanded = expanded.replace('C', consonant_pattern)
    expanded = expanded.replace('#', boundary)
    
    return expanded


def apply_rule(ipa, match_regex, op, out_phone):
    """
    Apply a single accent rule to an IPA string.
    
    Returns: (modified_ipa, was_modified)
    """
    try:
        # Expand regex pattern
        pattern = expand_regex_pattern(match_regex)
        
        # Check if pattern matches
        if not re.search(pattern, ipa):
            return ipa, False
        
        # Apply operation based on op type
        if op == 'S':  # Substitute
            modified = re.sub(pattern, out_phone, ipa)
        elif op == 'D':  # Delete
            modified = re.sub(pattern, '', ipa)
        elif op == 'A':  # Add
            modified = re.sub(pattern, lambda m: m.group(0) + out_phone, ipa)
        else:
            return ipa, False
        
        # Check if actually modified
        if modified != ipa:
            return modified, True
        else:
            return ipa, False
            
    except Exception:
        # If regex fails, skip this rule
        return ipa, False


def load_accent_rules(rules_file):
    """Load accent rules from CSV file."""
    rules = []
    with open(rules_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules.append({
                'name': row['name'],
                'match_regex': row['match_regex'],
                'op': row['op'],
                'out_phone': row['out_phone']
            })
    return rules


def process_word(word, ipa, rules, seen_combinations, ipa_to_process, rule_usage):
    """
    Process a single word: collect original and variations.
    
    Args:
        word: Word text
        ipa: IPA transcription
        rules: List of accent rules
        seen_combinations: Set tracking (word, ipa) pairs
        ipa_to_process: List to collect (word, ipa) pairs for class sequence calculation
        rule_usage: Dict tracking rule application counts
    """
    # Add original if not seen
    combo = (word, ipa)
    if combo not in seen_combinations:
        ipa_to_process.append(combo)
        seen_combinations.add(combo)
    
    # Try applying each rule to generate variations
    for rule in rules:
        modified_ipa, was_modified = apply_rule(
            ipa,
            rule['match_regex'],
            rule['op'],
            rule['out_phone']
        )
        
        if was_modified:
            # Check if this combination already exists
            combo = (word, modified_ipa)
            if combo not in seen_combinations:
                ipa_to_process.append(combo)
                seen_combinations.add(combo)
                rule_usage[rule['name']] += 1


def main():
    """Main processing pipeline."""
    # File paths
    input_file = 'data/english_words_ipa.csv'
    output_file = 'data/english_words_processed.csv'
    rules_file = 'data/accent_rules_intra.csv'
    
    print("=" * 80)
    print("CONSOLIDATED IPA PROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nInput:  {input_file}")
    print(f"Rules:  {rules_file}")
    print(f"Output: {output_file}")
    print("\nProcessing steps:")
    print("  1. Normalize IPA to canonical form")
    print("  2. Apply accent rules to generate variations")
    print("  3. Calculate phonetic class sequences")
    print("  4. Deduplicate (word, ipa) combinations")
    print()
    
    # Check input files exist
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    if not Path(rules_file).exists():
        print(f"Error: Rules file not found: {rules_file}")
        sys.exit(1)
    
    # Load accent rules
    print("Loading accent rules...")
    rules = load_accent_rules(rules_file)
    print(f"Loaded {len(rules)} accent rules\n")
    
    # Statistics
    stats = {
        'total_input_rows': 0,
        'normalized_count': 0,
        'variations_added': 0,
        'parse_errors': 0
    }
    rule_usage = {rule['name']: 0 for rule in rules}
    
    # Storage
    ipa_to_process = []  # List of (word, ipa) tuples
    seen_combinations = set()  # Track (word, ipa) pairs
    
    # Step 1: Normalize IPA and apply accent rules
    print("Step 1: Normalizing IPA and applying accent rules...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total_input_rows'] += 1
            
            word = row['word']
            ipa = row['ipa']
            
            # Normalize IPA to canonical form
            normalized_ipa = normalize_ipa(ipa)
            if normalized_ipa != ipa:
                stats['normalized_count'] += 1
            
            # Count before processing
            count_before = len(ipa_to_process)
            
            # Process word with normalized IPA and generate variations
            process_word(word, normalized_ipa, rules, seen_combinations, ipa_to_process, rule_usage)
            
            # Track variations
            count_after = len(ipa_to_process)
            if count_after > count_before:
                # First one added is original, rest are variations
                stats['variations_added'] += (count_after - count_before - 1)
            
            # Progress indicator
            if stats['total_input_rows'] % 10000 == 0:
                print(f"  Processed {stats['total_input_rows']:,} input rows, "
                      f"collected {len(ipa_to_process):,} unique (word, ipa) pairs...")
    
    print(f"✓ Collected {len(ipa_to_process):,} unique (word, ipa) combinations")
    
    # Step 2: Calculate class sequences for all collected pairs
    print(f"\nStep 2: Calculating class sequences for {len(ipa_to_process):,} entries...")
    all_rows = []
    
    for idx, (word, ipa) in enumerate(ipa_to_process):
        # Calculate class sequence
        try:
            class_seq = parse_ipa_sequence(ipa, strict=False)
        except Exception:
            class_seq = ''
            stats['parse_errors'] += 1
        
        all_rows.append({
            'word': word,
            'ipa': ipa,
            'class_sequence': class_seq
        })
        
        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"  Calculated {idx + 1:,} class sequences...")
    
    print(f"✓ Calculated all class sequences")
    
    # Step 3: Write output
    print(f"\nStep 3: Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['word', 'ipa', 'class_sequence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"✓ Wrote {len(all_rows):,} rows")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Input rows:             {stats['total_input_rows']:,}")
    print(f"IPA normalized:         {stats['normalized_count']:,} ({stats['normalized_count']/stats['total_input_rows']*100:.1f}%)")
    print(f"Variations generated:   {stats['variations_added']:,}")
    print(f"Total output rows:      {len(all_rows):,}")
    print(f"Growth factor:          {len(all_rows) / stats['total_input_rows']:.2f}x")
    print(f"Unique combinations:    {len(seen_combinations):,}")
    print(f"Parse errors:           {stats['parse_errors']:,}")
    
    # Rule usage statistics
    print("\n" + "=" * 80)
    print("RULE USAGE STATISTICS")
    print("=" * 80)
    
    sorted_rules = sorted(rule_usage.items(), key=lambda x: x[1], reverse=True)
    active_rules = [(name, count) for name, count in sorted_rules if count > 0]
    
    if active_rules:
        for rule_name, count in active_rules:
            pct = (count / stats['total_input_rows']) * 100
            print(f"  {rule_name[:50]:50s} {count:6,} ({pct:5.1f}%)")
    
    unused_rules = [name for name, count in sorted_rules if count == 0]
    if unused_rules:
        print(f"\nUnused rules ({len(unused_rules)}):")
        for rule_name in unused_rules:
            print(f"  - {rule_name}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Output saved to: {output_file}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)