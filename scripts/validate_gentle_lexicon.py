#!/usr/bin/env python3
"""
Validate Gentle Lexicon and Find Missing Words

This script:
1. Checks if lexicons are in sync across different Gentle locations
2. Finds words from Common Voice error transcripts that are missing
3. Validates that the speechline lexicon has better coverage
"""

import sys
import hashlib
from pathlib import Path
from typing import Set, List, Tuple
import re

# Gentle locations
GENTLE_DIR = Path("/mnt/Projects/Projects/AudioProcessing/gentle")
SPEECHLINE_LEXICON = Path("data/align_lexicon.txt")

GENTLE_LOCATIONS = [
    GENTLE_DIR / "exp/langdir/phones/align_lexicon.txt",
    GENTLE_DIR / "exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt"
]

ERRORS_DIR = Path("errors/common_voice")


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file"""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_lexicon_words(lexicon_path: Path) -> Set[str]:
    """Load all words from a lexicon file"""
    words = set()
    if not lexicon_path.exists():
        return words
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                word = parts[0].lower()
                words.add(word)
    
    return words


def extract_words_from_transcript(text: str) -> List[str]:
    """Extract words from transcript text"""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s\'-]', ' ', text.lower())
    words = text.split()
    return [w for w in words if w]


def find_missing_words_from_errors() -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Find missing words from error transcript files"""
    missing_words = set()
    examples = []
    
    if not ERRORS_DIR.exists():
        print(f"Warning: Errors directory not found: {ERRORS_DIR}")
        return missing_words, examples
    
    error_files = list(ERRORS_DIR.glob("*.txt"))
    
    for error_file in error_files[:20]:  # Limit to 20 files for analysis
        try:
            with open(error_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    transcript = lines[0].strip()
                    not_found_line = lines[2].strip()
                    
                    if "NOT_FOUND_IN_AUDIO:" in not_found_line:
                        # Extract the list of words
                        match = re.search(r'\[(.*?)\]', not_found_line)
                        if match:
                            words_str = match.group(1)
                            words = [w.strip().strip("'\"").lower() 
                                   for w in words_str.split(',')]
                            for word in words:
                                if word:
                                    missing_words.add(word)
                                    if len(examples) < 10:
                                        examples.append((word, transcript))
        except Exception as e:
            print(f"Error reading {error_file}: {e}")
    
    return missing_words, examples


def check_lexicon_status():
    """Check the status of all lexicon files"""
    print("=" * 70)
    print("GENTLE LEXICON VALIDATION")
    print("=" * 70)
    print()
    
    # Check MD5 hashes
    print("1. CHECKING LEXICON FILE SYNC")
    print("-" * 70)
    
    speechline_md5 = calculate_md5(SPEECHLINE_LEXICON)
    print(f"Speechline lexicon: {SPEECHLINE_LEXICON}")
    print(f"  MD5: {speechline_md5}")
    print(f"  Size: {SPEECHLINE_LEXICON.stat().st_size / 1024 / 1024:.1f} MB" 
          if SPEECHLINE_LEXICON.exists() else "  NOT FOUND")
    print()
    
    all_match = True
    for gentle_lex in GENTLE_LOCATIONS:
        gentle_md5 = calculate_md5(gentle_lex)
        matches = "✓ MATCH" if gentle_md5 == speechline_md5 else "✗ DIFFERENT"
        print(f"Gentle lexicon: {gentle_lex}")
        print(f"  MD5: {gentle_md5}")
        print(f"  Status: {matches}")
        if gentle_md5 != speechline_md5:
            all_match = False
        print()
    
    # Check vocabulary coverage
    print("2. VOCABULARY COVERAGE ANALYSIS")
    print("-" * 70)
    
    print("Loading lexicons...")
    speechline_words = load_lexicon_words(SPEECHLINE_LEXICON)
    print(f"Speechline lexicon: {len(speechline_words):,} unique words")
    
    gentle_words = {}
    for gentle_lex in GENTLE_LOCATIONS:
        words = load_lexicon_words(gentle_lex)
        gentle_words[gentle_lex.name] = words
        print(f"Gentle {gentle_lex.name}: {len(words):,} unique words")
    print()
    
    # Find missing words from error files
    print("3. ANALYZING ERROR TRANSCRIPTS")
    print("-" * 70)
    
    missing_words, examples = find_missing_words_from_errors()
    print(f"Found {len(missing_words)} unique words marked as NOT_FOUND_IN_AUDIO")
    print()
    
    if missing_words:
        # Check which words are in speechline lexicon
        found_in_speechline = missing_words & speechline_words
        not_in_speechline = missing_words - speechline_words
        
        print(f"Words in Speechline lexicon: {len(found_in_speechline)} "
              f"({len(found_in_speechline)/len(missing_words)*100:.1f}%)")
        
        # Check coverage in each Gentle location
        for name, words in gentle_words.items():
            found_in_gentle = missing_words & words
            print(f"Words in Gentle {name}: {len(found_in_gentle)} "
                  f"({len(found_in_gentle)/len(missing_words)*100:.1f}%)")
        
        print()
        print("Sample missing words that ARE in Speechline lexicon:")
        print("-" * 70)
        for i, (word, transcript) in enumerate(examples[:10], 1):
            in_speechline = "✓" if word in speechline_words else "✗"
            print(f"{i}. {word} {in_speechline}")
            print(f"   From: {transcript[:60]}...")
        print()
        
        if not_in_speechline:
            print(f"\nWords NOT in Speechline lexicon: {len(not_in_speechline)}")
            sample_not_in = list(not_in_speechline)[:10]
            print(f"Sample: {', '.join(sample_not_in)}")
            print("(These may be proper nouns or OCR errors)")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_match:
        print("✓ All lexicon files are IN SYNC")
        print("✓ Speechline lexicon is deployed to all Gentle locations")
    else:
        print("✗ Lexicon files are OUT OF SYNC")
        print("✗ Speechline lexicon needs to be deployed to Gentle")
    
    print()
    print("NEXT STEPS:")
    if not all_match:
        print("1. Run: python scripts/update_gentle_lexicon.py")
        print("   This will update both Gentle locations and rebuild FST files")
    print("2. Run validation tests to verify improvements")
    print("3. Re-process error files to check if words are now found")
    print()


def main():
    check_lexicon_status()


if __name__ == '__main__':
    main()