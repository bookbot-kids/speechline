#!/usr/bin/env python3
"""
Tests for the unified phonetics library
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.phonetics import (
    normalize_ipa,
    ipa_to_artemes,
    ipa_to_arpa,
    arpa_to_ipa,
    arpa_to_artemes,
    validate_ipa
)

def test_normalize_ipa():
    """Test IPA normalization"""
    print("Testing IPA normalization...")
    
    tests = [
        ("ˈhɛˌloʊ", "hɛloʊ"),  # Remove stress
        ("ð ə", "ðə"),          # Merge spaces
        ("t͡ʃ", "tʃ"),          # Normalize affricate
        ("kʰæt", "kæt"),        # Remove aspiration
        ("ɡ", "g"),             # Normalize g
        ("r", "ɹ"),             # r to approximant
    ]
    
    for input_ipa, expected in tests:
        result = normalize_ipa(input_ipa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} normalize_ipa('{input_ipa}') = '{result}' (expected '{expected}')")
    
    print()

def test_ipa_to_artemes():
    """Test IPA to artemes conversion"""
    print("Testing IPA to artemes...")
    
    tests = [
        ("hɛloʊ", "hələə"),   # hello (oʊ is diphthong → əə)
        ("kæt", "kət"),       # cat
        ("tʃɛɹ", "ʃəɹ"),      # chair
        ("ðə", "θə"),         # the
    ]
    
    for input_ipa, expected in tests:
        result = ipa_to_artemes(input_ipa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} ipa_to_artemes('{input_ipa}') = '{result}' (expected '{expected}')")
    
    print()

def test_arpa_to_ipa():
    """Test ARPA to IPA conversion"""
    print("Testing ARPA to IPA...")
    
    tests = [
        ("K AE1 T", "kæt"),           # cat
        ("HH EH L OW", "hɛloʊ"),      # hello
        ("DH AH0", "ðʌ"),             # the (unstressed)
        ("CH EH R", "tʃɛɹ"),          # chair
    ]
    
    for input_arpa, expected in tests:
        result = arpa_to_ipa(input_arpa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} arpa_to_ipa('{input_arpa}') = '{result}' (expected '{expected}')")
    
    print()

def test_ipa_to_arpa():
    """Test IPA to ARPA conversion"""
    print("Testing IPA to ARPA...")
    
    tests = [
        ("kæt", "K AE T"),         # cat
        ("hɛloʊ", "HH EH L OW"),   # hello
        ("tʃɛɹ", "CH EH R"),       # chair
    ]
    
    for input_ipa, expected in tests:
        result = ipa_to_arpa(input_ipa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} ipa_to_arpa('{input_ipa}') = '{result}' (expected '{expected}')")
    
    print()

def test_arpa_to_artemes():
    """Test ARPA to artemes conversion"""
    print("Testing ARPA to artemes...")
    
    tests = [
        ("K AE1 T", "kət"),       # cat
        ("HH EH L OW", "hələə"),  # hello (OW diphthong → əə)
    ]
    
    for input_arpa, expected in tests:
        result = arpa_to_artemes(input_arpa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} arpa_to_artemes('{input_arpa}') = '{result}' (expected '{expected}')")
    
    print()

def test_validate_ipa():
    """Test IPA validation"""
    print("Testing IPA validation...")
    
    result = validate_ipa("kæt")
    print(f"  validate_ipa('kæt'): valid={result['valid']}, unknown={result['unknown_symbols']}")
    
    result = validate_ipa("ˈkæt")
    print(f"  validate_ipa('ˈkæt'): valid={result['valid']}, excluded={result['excluded_symbols']}")
    
    print()

def test_round_trip():
    """Test round-trip conversions"""
    print("Testing round-trip conversions...")
    
    # IPA → ARPA → IPA
    ipa1 = "kæt"
    arpa = ipa_to_arpa(ipa1)
    ipa2 = arpa_to_ipa(arpa)
    print(f"  IPA→ARPA→IPA: '{ipa1}' → '{arpa}' → '{ipa2}'")
    
    # ARPA → IPA → Artemes
    arpa1 = "K AE T"
    ipa = arpa_to_ipa(arpa1)
    artemes = ipa_to_artemes(ipa)
    print(f"  ARPA→IPA→Artemes: '{arpa1}' → '{ipa}' → '{artemes}'")
    
    print()

def main():
    """Run all tests"""
    print("=" * 70)
    print("PHONETICS LIBRARY TESTS")
    print("=" * 70)
    print()
    
    test_normalize_ipa()
    test_ipa_to_artemes()
    test_arpa_to_ipa()
    test_ipa_to_arpa()
    test_arpa_to_artemes()
    test_validate_ipa()
    test_round_trip()
    
    print("=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()