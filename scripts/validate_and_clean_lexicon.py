#!/usr/bin/env python3
"""
Validate and clean the IPA lexicon by detecting and removing problematic entries.

This script identifies several types of bad lexicon entries:
1. Incomplete IPA (fragments, partial transcriptions)
2. Severely mismatched length (word >> class sequence)
3. Invalid IPA symbols
4. Malformed entries
"""

import csv
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.phonetics import normalize_ipa, ipa_to_artemes, validate_ipa

def analyze_entry(word: str, ipa: str) -> Dict[str, any]:
    """Analyze a lexicon entry for quality issues."""
    issues = []
    severity = 'ok'
    
    # Clean IPA
    ipa_clean = ipa.strip('/')
    
    # 1. Check for empty or very short IPA
    if len(ipa_clean) == 0:
        issues.append('empty_ipa')
        severity = 'critical'
        return {'issues': issues, 'severity': severity, 'classes': '', 'keep': False}
    
    if len(ipa_clean) == 1 and len(word) > 2:
        issues.append('ipa_too_short')
        severity = 'critical'
    
    # 2. Check for IPA fragments (ends with -, incomplete)
    if ipa_clean.endswith('-') or ipa_clean.startswith('-'):
        issues.append('ipa_fragment')
        severity = 'critical'
    
    # 3. Try to parse and get classes
    try:
        normalized = normalize_ipa(ipa_clean)
        classes = ipa_to_artemes(normalized)
    except Exception as e:
        issues.append(f'parse_error: {e}')
        severity = 'critical'
        return {'issues': issues, 'severity': severity, 'classes': '', 'keep': False}
    
    # 4. Check class sequence length vs word length
    word_len = len(word)
    class_len = len(classes)
    
    if class_len == 0:
        issues.append('no_classes_generated')
        severity = 'critical'
    elif class_len == 1 and word_len > 4:
        # Long word but only 1 phonetic class - likely corrupt
        issues.append('severe_length_mismatch')
        severity = 'critical'
    elif class_len < word_len / 3:
        # Class sequence suspiciously shorter than word
        issues.append('length_ratio_suspicious')
        severity = 'warning'
    
    # 5. Check for invalid IPA patterns
    if re.search(r'[0-9]', ipa_clean):
        issues.append('contains_numbers')
        severity = 'critical'
    
    # 6. Validate IPA string
    is_valid, unknown_symbols, excluded_symbols = validate_ipa(normalized)
    if not is_valid:
        if unknown_symbols:
            issues.append(f"unknown_symbols: {unknown_symbols}")
            severity = 'warning'
    
    # Determine if we should keep this entry
    keep = severity != 'critical'
    
    return {
        'issues': issues,
        'severity': severity,
        'classes': classes,
        'normalized_ipa': normalized,
        'keep': keep
    }


def clean_lexicon(input_file: str, output_file: str, report_file: str):
    """Clean lexicon by removing problematic entries."""
    
    print("=" * 80)
    print("LEXICON VALIDATION AND CLEANING")
    print("=" * 80)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Report: {report_file}")
    
    total_entries = 0
    removed_entries = []
    kept_entries = []
    warning_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            total_entries += 1
            word, ipa = row[0], row[1]
            
            analysis = analyze_entry(word, ipa)
            
            if analysis['keep']:
                kept_entries.append((word, ipa, analysis))
                if analysis['severity'] == 'warning':
                    warning_entries.append((word, ipa, analysis))
            else:
                removed_entries.append((word, ipa, analysis))
    
    # Write cleaned lexicon
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for word, ipa, analysis in kept_entries:
            writer.writerow([word, ipa])
    
    # Write report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LEXICON CLEANING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total entries processed: {total_entries}\n")
        f.write(f"Entries kept:            {len(kept_entries)} ({100*len(kept_entries)/total_entries:.1f}%)\n")
        f.write(f"Entries removed:         {len(removed_entries)} ({100*len(removed_entries)/total_entries:.1f}%)\n")
        f.write(f"Entries with warnings:   {len(warning_entries)} ({100*len(warning_entries)/total_entries:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("REMOVED ENTRIES (Critical Issues)\n")
        f.write("=" * 80 + "\n\n")
        
        for word, ipa, analysis in removed_entries[:100]:
            f.write(f"{word:30} {ipa:20} -> Issues: {', '.join(analysis['issues'])}\n")
        
        if len(removed_entries) > 100:
            f.write(f"\n... and {len(removed_entries) - 100} more removed entries\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("ENTRIES WITH WARNINGS (Kept but suspicious)\n")
        f.write("=" * 80 + "\n\n")
        
        for word, ipa, analysis in warning_entries[:50]:
            f.write(f"{word:30} {ipa:20} -> Issues: {', '.join(analysis['issues'])}\n")
        
        if len(warning_entries) > 50:
            f.write(f"\n... and {len(warning_entries) - 50} more warning entries\n")
    
    print(f"\n✓ Cleaned lexicon written to: {output_file}")
    print(f"✓ Detailed report written to: {report_file}")
    print(f"\nSummary:")
    print(f"  Kept:    {len(kept_entries):6,} entries")
    print(f"  Removed: {len(removed_entries):6,} entries")
    print(f"  Warnings: {len(warning_entries):6,} entries")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and clean IPA lexicon')
    parser.add_argument('--input', default='data/english_words_ipa.csv',
                       help='Input lexicon file')
    parser.add_argument('--output', default='data/english_words_ipa_cleaned.csv',
                       help='Output cleaned lexicon file')
    parser.add_argument('--report', default='data/lexicon_cleaning_report.txt',
                       help='Cleaning report file')
    
    args = parser.parse_args()
    
    clean_lexicon(args.input, args.output, args.report)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the cleaning report:")
    print(f"   less {args.report}")
    print("\n2. If satisfied, replace original lexicon:")
    print(f"   mv {args.output} {args.input}")
    print("\n3. Reprocess the lexicon:")
    print("   python3 scripts/process_words.py")
    print("\n4. Re-run phoneme matching tests")
    print("   python3 -m pytest tests/test_libriphone.py -v")


if __name__ == '__main__':
    main()