#!/usr/bin/env python3
"""
Generate IPA phoneme to phonetic class mapping.

This script:
1. Reads the phonetic classification table from data/manner_of_articulation.csv
2. Extracts all unique IPA symbols from data/english_words_ipa.csv
3. Maps each IPA symbol to its phonetic class
4. Uses inference rules for unmapped symbols
5. Outputs a Python dictionary in scripts/ipa_to_class_mapping.py

Usage:
    python scripts/generate_ipa_class_mapping.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path


# IPA symbols to exclude (modifiers, boundaries, brackets, etc.)
EXCLUDE_SYMBOLS = {
    # Brackets and delimiters
    '(', ')', '[', ']', '/', '|',
    # Stress and prosody
    'ˈ',   # primary stress
    'ˌ',   # secondary stress
    'ː',   # length mark
    'ˑ',   # half-long
    '.',   # syllable boundary
    'ʔ',   # glottal stop
    # Tone marks
    '˥', '˦', '˧', '˨', '˩', '↓', '↗',
    # Typography and punctuation
    '–', '…', '‿', '²', '³', '¹', '⁴', '⁵', '⁻', 'ⁿ', '︎', "'",
    # Common diacritics (combining characters)
    '̃',   # nasalization
    '̚',   # unreleased
    'ʰ',   # aspiration
    'ʷ',   # labialization
    'ʲ',   # palatalization
    '̩',   # syllabic (handled separately)
    'ʱ',   # breathy voice
    '̀', '́', '̂', '̄', '̆', '̈', '̊', '̍',
    '̙', '̝', '̞', '̟', '̠', '̥', '̪', '̬', '̯', '̰', '̹', '̺', '̽',
    '͆', '͇', '͜', '͡',
    # Superscripts (various)
    'ʳ', 'ʴ', 'ʼ', 'ˀ', '˔', '˞', 'ˠ', 'ˡ', 'ˤ', '˭',
    'ᵈ', 'ᵊ', 'ᵐ', 'ᵒ', 'ᶴ', '᷈',
    # Spaces and separators
    ' ', '-', '~',
    '',    # empty string
}

# Latin letters with diacritics (used in various orthographies)
# Map to their base IPA equivalent
LATIN_DIACRITIC_MAP = {
    # Vowels - a variants
    'á': 'a', 'ã': 'a', 'ä': 'a', 'ā': 'a', 'ă': 'a',
    # Vowels - e variants
    'é': 'e', 'ẽ': 'e', 'ĕ': 'e',
    # Vowels - i variants
    'í': 'i', 'ï': 'i', 'ĩ': 'i', 'ĭ': 'i', 'ǐ': 'i',
    # Vowels - o variants
    'ó': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ŏ': 'o',
    # Vowels - u variants
    'ú': 'u', 'ü': 'u', 'ũ': 'u', 'ŭ': 'u',
    # Consonants
    'ç': 's',  # cedilla c represents /s/ sound
    'ḿ': 'm',  # m with acute
    'y': 'j',  # Latin y represents palatal approximant /j/
}

# Multi-character IPA sequences (must be checked before single characters)
MULTI_CHAR_SEQUENCES = [
    'tʃ', 'dʒ',           # affricates
    'aɪ', 'eɪ', 'oʊ', 'aʊ', 'ɔɪ',  # diphthongs
    'l̩',                  # syllabic lateral (from table)
]

# Base phoneme categories for inference
# Expanded to include more IPA symbols from various languages
VOWELS = set('ieaouəɪʊɛæɑɒʌ' +
             'ɐɔɘɚɝɜɞɤɨɯɵɶʉʏœ')  # Additional vowels
STOPS_LABIAL = set('pbɓ')  # Including implosive
STOPS_CORONAL = set('tdɾɖʈɽ')  # Including retroflex
STOPS_DORSAL = set('kgqcɟɡɢ')  # Including palatal and uvular (ɡ with tail)
FRICATIVES_LABIODENTAL = set('fvɸβ')  # Including bilabial
FRICATIVES_DENTAL = set('θð')
FRICATIVES_ALVEOLAR_SIB = set('sz')
FRICATIVES_POSTALVEOLAR = set('ʃʒɕʑ')  # Including alveolo-palatal
FRICATIVES_GLOTTAL = set('hɦɣχʕx')  # Other back/glottal fricatives
NASALS = set('mnŋɱɴɲɳ')  # Including palatal and retroflex
LATERALS = set('lɫɬɭɺʎʟ')  # Including lateral fricative and retroflex
RHOTICS = set('ɹrɻʁʀ')
GLIDES = set('jwɥʋʍ')  # Including labiodental and labial-velar
CLICKS = set('ʘǀǁǃʙ')  # Click consonants


def load_classification_table(filepath='data/manner_of_articulation.csv'):
    """Load the phonetic classification table and create base mapping."""
    ipa_to_class = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_symbol = row['Class symbol']
            phonemes_str = row['Representative phonemes']
            
            # Parse phonemes (comma-separated, may have spaces)
            phonemes = [p.strip() for p in phonemes_str.split(',')]
            
            for phoneme in phonemes:
                if phoneme:
                    ipa_to_class[phoneme] = class_symbol
    
    return ipa_to_class


def strip_diacritics(ipa_string):
    """
    Remove diacritics and modifiers from IPA string.
    Handles syllabic consonants and Latin diacritics specially.
    """
    # Handle syllabic consonants before stripping
    syllabic_map = {
        'l̩': 'l',
        'r̩': 'r',
        'ɹ̩': 'ɹ',
        'm̩': 'm',
        'n̩': 'n',
        'ŋ̩': 'ŋ',
    }
    
    for syllabic, base in syllabic_map.items():
        ipa_string = ipa_string.replace(syllabic, base)
    
    # Convert Latin letters with diacritics to base IPA
    for latin, base in LATIN_DIACRITIC_MAP.items():
        ipa_string = ipa_string.replace(latin, base)
    
    # Remove all excluded symbols
    for symbol in EXCLUDE_SYMBOLS:
        ipa_string = ipa_string.replace(symbol, '')
    
    return ipa_string


def extract_ipa_symbols(csv_path='data/english_words_ipa.csv', max_lines=None):
    """
    Extract all unique IPA symbols from the CSV file.
    Handles multi-character sequences properly.
    """
    symbols = set()
    lines_processed = 0
    
    print(f"Extracting IPA symbols from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            lines_processed += 1
            
            if max_lines and lines_processed > max_lines:
                print(f"Stopped after {max_lines} lines (for testing)")
                break
            
            if lines_processed % 50000 == 0:
                print(f"  Processed {lines_processed:,} lines, found {len(symbols)} unique symbols")
            
            ipa = row.get('ipa', '')
            if not ipa:
                continue
            
            # Clean the IPA string
            ipa = strip_diacritics(ipa)
            
            # Extract symbols (multi-char first, then single chars)
            remaining = ipa
            while remaining:
                matched = False
                
                # Try multi-character sequences first
                for seq in MULTI_CHAR_SEQUENCES:
                    if remaining.startswith(seq):
                        symbols.add(seq)
                        remaining = remaining[len(seq):]
                        matched = True
                        break
                
                # If no multi-char match, take single character
                if not matched and remaining:
                    char = remaining[0]
                    if char not in EXCLUDE_SYMBOLS:
                        symbols.add(char)
                    remaining = remaining[1:]
    
    print(f"Extraction complete: {lines_processed:,} lines processed")
    print(f"Found {len(symbols)} unique IPA symbols\n")
    
    return symbols


def infer_class(symbol, base_mapping):
    """
    Infer phonetic class for unmapped symbols using phonetic properties.
    Handles extended IPA symbols from various languages.
    """
    # Already mapped
    if symbol in base_mapping:
        return base_mapping[symbol]
    
    # Vowels and diphthongs → V (ə)
    if any(v in symbol for v in VOWELS):
        return 'ə'
    
    # Stops
    if symbol in STOPS_LABIAL:
        return 'p'
    if symbol in STOPS_CORONAL:
        return 't'
    if symbol in STOPS_DORSAL:
        return 'k'
    
    # Fricatives
    if symbol in FRICATIVES_LABIODENTAL:
        return 'f'
    if symbol in FRICATIVES_DENTAL:
        return 'θ'
    if symbol in FRICATIVES_ALVEOLAR_SIB:
        return 's'
    if symbol in FRICATIVES_POSTALVEOLAR:
        return 'ʃ'
    if symbol in FRICATIVES_GLOTTAL:
        return 'h'
    
    # Sonorants
    if symbol in NASALS:
        return 'n'
    if symbol in LATERALS:
        return 'l'
    if symbol in RHOTICS:
        return 'ɹ'
    if symbol in GLIDES:
        return 'j'
    
    # Clicks → map to labial stops (arbitrary but consistent)
    if symbol in CLICKS:
        return 'p'
    
    # Unknown - return None for manual review
    return None


def generate_mapping(base_mapping, all_symbols):
    """
    Generate complete IPA to class mapping.
    Returns dict and stats about mapping coverage.
    """
    complete_mapping = {}
    stats = {
        'total': len(all_symbols),
        'base_mapped': 0,
        'inferred': 0,
        'unmapped': 0,
        'unmapped_symbols': []
    }
    
    for symbol in sorted(all_symbols):
        if symbol in base_mapping:
            complete_mapping[symbol] = base_mapping[symbol]
            stats['base_mapped'] += 1
        else:
            inferred = infer_class(symbol, base_mapping)
            if inferred:
                complete_mapping[symbol] = inferred
                stats['inferred'] += 1
            else:
                stats['unmapped'] += 1
                stats['unmapped_symbols'].append(symbol)
    
    return complete_mapping, stats


def write_mapping_file(mapping, stats, output_path='scripts/ipa_to_class_mapping.py'):
    """Write the mapping dictionary to a Python file."""
    
    # Group symbols by class for organized output
    class_to_symbols = defaultdict(list)
    for symbol, class_sym in sorted(mapping.items()):
        class_to_symbols[class_sym].append(symbol)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""IPA Phoneme to Phonetic Class Mapping\n\n')
        f.write('Auto-generated mapping from IPA symbols to phonetic class symbols.\n')
        f.write('Based on the robust phonetic classification system.\n\n')
        f.write(f'Statistics:\n')
        f.write(f'  Total symbols: {stats["total"]}\n')
        f.write(f'  Base mapped: {stats["base_mapped"]}\n')
        f.write(f'  Inferred: {stats["inferred"]}\n')
        f.write(f'  Unmapped: {stats["unmapped"]}\n')
        if stats['unmapped_symbols']:
            f.write(f'  Unmapped symbols: {", ".join(stats["unmapped_symbols"])}\n')
        f.write('"""\n\n')
        
        f.write('# IPA symbol to phonetic class mapping\n')
        f.write('IPA_TO_CLASS = {\n')
        
        # Write by class for better organization
        class_names = {
            'ə': 'V - Vowels/Diphthongs',
            's': 'SIB-ALV - Alveolar Sibilants',
            'ʃ': 'SIB-POST - Postalveolar Sibilants',
            'f': 'FR-LAB - Labiodental Fricatives',
            'θ': 'FR-DENT - Dental Fricatives',
            'h': 'FR-GLOT - Glottal Fricatives',
            'p': 'ST-LAB - Labial Stops',
            't': 'ST-COR - Coronal Stops',
            'k': 'ST-DOR - Dorsal Stops',
            'n': 'NAS - Nasals',
            'l': 'LAT - Laterals',
            'ɹ': 'RHO - Rhotics',
            'j': 'GLD - Glides',
        }
        
        for class_sym in ['ə', 's', 'ʃ', 'f', 'θ', 'h', 'p', 't', 'k', 'n', 'l', 'ɹ', 'j']:
            if class_sym in class_to_symbols:
                f.write(f'\n    # {class_names[class_sym]}\n')
                for symbol in sorted(class_to_symbols[class_sym]):
                    f.write(f"    '{symbol}': '{class_sym}',\n")
        
        f.write('}\n')
    
    print(f"Mapping written to {output_path}")


def main():
    """Main execution."""
    print("="*70)
    print("IPA to Phonetic Class Mapping Generator")
    print("="*70 + "\n")
    
    # Step 1: Load base mapping
    print("Step 1: Loading phonetic classification table...")
    base_mapping = load_classification_table()
    print(f"  Loaded {len(base_mapping)} base phoneme mappings\n")
    
    # Step 2: Extract all IPA symbols from CSV
    print("Step 2: Extracting IPA symbols from CSV...")
    all_symbols = extract_ipa_symbols()
    
    # Step 3: Generate complete mapping
    print("Step 3: Generating complete mapping...")
    complete_mapping, stats = generate_mapping(base_mapping, all_symbols)
    print(f"  Base mapped: {stats['base_mapped']}")
    print(f"  Inferred: {stats['inferred']}")
    print(f"  Unmapped: {stats['unmapped']}")
    if stats['unmapped_symbols']:
        print(f"  Unmapped symbols: {', '.join(stats['unmapped_symbols'])}")
    print()
    
    # Step 4: Write output file
    print("Step 4: Writing mapping to Python file...")
    write_mapping_file(complete_mapping, stats)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Mapping file created: scripts/ipa_to_class_mapping.py")
    print(f"Total mappings: {len(complete_mapping)}")


if __name__ == '__main__':
    main()