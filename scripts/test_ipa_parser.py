#!/usr/bin/env python3
"""
Test script for IPA to phonetic class parser.

Tests various IPA sequences to ensure correct parsing and error handling.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ipa_to_class_mapping import (
    parse_ipa_sequence,
    validate_ipa_string,
    get_class_name,
    IPA_TO_CLASS,
)


def test_basic_parsing():
    """Test basic IPA parsing."""
    print("=" * 70)
    print("TEST 1: Basic Parsing")
    print("=" * 70)
    
    test_cases = [
        ("hɛloʊ", "hələ", "hello"),
        ("kæt", "kət", "cat"),
        ("dɒg", "tək", "dog"),
        ("bʊk", "pək", "book"),
        ("sɪŋ", "sən", "sing"),
    ]
    
    for ipa, expected, word in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{word}' ({ipa}) → {result} (expected: {expected})")
    print()


def test_affricates():
    """Test affricate parsing."""
    print("=" * 70)
    print("TEST 2: Affricates")
    print("=" * 70)
    
    test_cases = [
        ("tʃɛɹ", "ʃəɹ", "chair (simple)"),  # ɛ→ə
        ("t͡ʃɛɹ", "ʃəɹ", "chair (tie-bar ͡)"),  # ɛ→ə
        ("dʒʌmp", "ʃənp", "jump (simple)"),
        ("d͡ʒʌmp", "ʃənp", "jump (tie-bar ͡)"),
        ("ts", "s", "alveolar affricate ts"),
        ("t͡s", "s", "alveolar affricate t͡s"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result} (expected: {expected})")
    print()


def test_stress_removal():
    """Test stress mark removal."""
    print("=" * 70)
    print("TEST 3: Stress Mark Removal")
    print("=" * 70)
    
    test_cases = [
        ("ˈkæt", "kət", "primary stress"),
        ("ˌsɛkənd", "səkənt", "secondary stress"),
        ("ˈhɛˌloʊ", "hələ", "both stresses"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result} (expected: {expected})")
    print()


def test_diphthongs():
    """Test diphthong parsing."""
    print("=" * 70)
    print("TEST 4: Diphthongs")
    print("=" * 70)
    
    test_cases = [
        ("haɪ", "hə", "high (aɪ)"),
        ("naʊ", "nə", "now (aʊ)"),
        ("deɪ", "tə", "day (eɪ)"),
        ("goʊ", "kə", "go (oʊ)"),
        ("bɔɪ", "pə", "boy (ɔɪ)"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result} (expected: {expected})")
    print()


def test_syllabic_consonants():
    """Test syllabic consonant parsing."""
    print("=" * 70)
    print("TEST 5: Syllabic Consonants")
    print("=" * 70)
    
    test_cases = [
        ("bɑtl̩", "pətl", "bottle (syllabic l)"),
        ("bʌtn̩", "pətn", "button (syllabic n)"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result} (expected: {expected})")
    print()


def test_fricative_merge():
    """Test non-glottal fricative coarse merge."""
    print("=" * 70)
    print("TEST 6: Fricative Coarse Merge (to h)")
    print("=" * 70)
    
    test_cases = [
        ("x", "h", "velar x"),
        ("ɣ", "h", "velar ɣ"),
        ("χ", "h", "uvular χ"),
        ("ʕ", "h", "pharyngeal ʕ"),
        ("ħ", "h", "pharyngeal ħ"),
        ("ç", "h", "palatal ç"),
        ("ʝ", "h", "palatal ʝ"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result}")
    print()


def test_clicks_and_special():
    """Test click and special consonant mappings."""
    print("=" * 70)
    print("TEST 7: Clicks and Special Consonants")
    print("=" * 70)
    
    test_cases = [
        ("ʘ", "p", "bilabial click → p"),
        ("ǀ", "t", "dental click → t"),
        ("ǁ", "l", "lateral click → l"),
        ("ǃ", "t", "alveolar click → t"),
        ("ɓ", "p", "bilabial implosive → p"),
        ("ʙ", "p", "bilabial trill → p"),
    ]
    
    for ipa, expected, desc in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {ipa} → {result}")
    print()


def test_error_handling():
    """Test error handling for unknown symbols."""
    print("=" * 70)
    print("TEST 8: Error Handling")
    print("=" * 70)
    
    # Test strict mode (should raise error)
    print("Testing strict mode with unknown symbol...")
    try:
        result = parse_ipa_sequence("hello@world", strict=True)
        print("✗ Should have raised ValueError for '@'")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:80]}...")
    
    # Test non-strict mode (should skip)
    print("\nTesting non-strict mode with unknown symbol...")
    result = parse_ipa_sequence("hɛlo@wɜld", strict=False)
    print(f"✓ Non-strict mode: 'hɛlo@wɜld' → '{result}' (@ skipped)")
    print()


def test_validation():
    """Test IPA string validation."""
    print("=" * 70)
    print("TEST 9: Validation Function")
    print("=" * 70)
    
    test_cases = [
        "hɛloʊ",  # valid
        "ˈkæt",   # has stress
        "ʘaka",   # has click (now mapped)
        "test@",  # has unknown
    ]
    
    for ipa in test_cases:
        validation = validate_ipa_string(ipa)
        print(f"\nIPA: '{ipa}'")
        print(f"  Valid: {validation['valid']}")
        if validation['unknown_symbols']:
            print(f"  Unknown: {validation['unknown_symbols']}")
        if validation['excluded_symbols']:
            print(f"  Excluded: {validation['excluded_symbols']}")
    print()


def test_real_words():
    """Test with real English words."""
    print("=" * 70)
    print("TEST 10: Real English Words")
    print("=" * 70)
    
    test_cases = [
        ("ˈæpl̩", "əpl", "apple"),
        ("ˈwɔtɚ", "jətə", "water"),  # ɚ is vowel→ə, not rhotic
        ("tʃɪldɹən", "ʃəltɹən", "children"),
        ("ˈsɪmpəl", "sənpəl", "simple"),  # m→n (all nasals→n)
        ("kəmˈpjutɚ", "kənpjətə", "computer"),  # m→n, ɚ→ə
    ]
    
    for ipa, expected, word in test_cases:
        result = parse_ipa_sequence(ipa)
        status = "✓" if result == expected else "✗"
        print(f"{status} {word}: {ipa} → {result} (expected: {expected})")
    print()


def test_class_statistics():
    """Print mapping statistics."""
    print("=" * 70)
    print("MAPPING STATISTICS")
    print("=" * 70)
    
    # Count by class
    class_counts = {}
    for symbol, class_sym in IPA_TO_CLASS.items():
        class_counts[class_sym] = class_counts.get(class_sym, 0) + 1
    
    print(f"\nTotal mapped symbols: {len(IPA_TO_CLASS)}")
    print(f"Total classes: {len(class_counts)}")
    
    print("\nSymbols per class:")
    for class_sym in sorted(class_counts.keys()):
        count = class_counts[class_sym]
        name = get_class_name(class_sym)
        print(f"  {class_sym}: {count:3d} symbols - {name}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "IPA PARSER TEST SUITE" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    test_basic_parsing()
    test_affricates()
    test_stress_removal()
    test_diphthongs()
    test_syllabic_consonants()
    test_fricative_merge()
    test_clicks_and_special()
    test_error_handling()
    test_validation()
    test_real_words()
    test_class_statistics()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()