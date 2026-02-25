#!/usr/bin/env python3
"""
Comprehensive test coverage for the unified phonetics library.
Tests all phonetic instances across IPA, ARPA, and Artemes.
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


class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def record(self, test_name, passed, expected=None, actual=None):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append({
                'test': test_name,
                'expected': expected,
                'actual': actual
            })
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests:  {self.total}")
        print(f"Passed:       {self.passed} ({100*self.passed/self.total:.1f}%)")
        print(f"Failed:       {self.failed}")
        
        if self.failures:
            print("\nFailed tests:")
            for failure in self.failures:
                print(f"  ✗ {failure['test']}")
                print(f"    Expected: {failure['expected']}")
                print(f"    Got:      {failure['actual']}")
        
        print("=" * 70)


results = TestResults()


def test_arpa_vowels():
    """Test all 15 ARPA vowels"""
    print("\n" + "=" * 70)
    print("Testing ARPA Vowels (15 phonemes)")
    print("=" * 70)
    
    vowel_tests = [
        # (ARPA, IPA, Arteme, Word)
        ("AA", "ɑ", "ə", "father"),
        ("AE", "æ", "ə", "cat"),
        ("AH", "ʌ", "ə", "cut"),
        ("AO", "ɔ", "ə", "caught"),
        ("AW", "aʊ", "əə", "how"),
        ("AY", "aɪ", "əə", "hide"),
        ("EH", "ɛ", "ə", "red"),
        ("ER", "ɝ", "əɹ", "her"),
        ("EY", "eɪ", "əə", "say"),
        ("IH", "ɪ", "ə", "it"),
        ("IY", "i", "ə", "eat"),
        ("OW", "oʊ", "əə", "go"),
        ("OY", "ɔɪ", "əə", "boy"),
        ("UH", "ʊ", "ə", "book"),
        ("UW", "u", "ə", "too"),
    ]
    
    for arpa, ipa, arteme, word in vowel_tests:
        # Test ARPA to IPA
        result_ipa = arpa_to_ipa(arpa)
        passed = result_ipa == ipa
        status = "✓" if passed else "✗"
        print(f"  {status} {arpa:3} → {ipa:3} (got: {result_ipa:3}) - {word}")
        results.record(f"ARPA vowel {arpa} to IPA", passed, ipa, result_ipa)
        
        # Test IPA to ARPA
        result_arpa = ipa_to_arpa(ipa)
        passed = result_arpa == arpa
        results.record(f"IPA vowel {ipa} to ARPA", passed, arpa, result_arpa)
        
        # Test IPA to Arteme
        result_arteme = ipa_to_artemes(ipa)
        passed = result_arteme == arteme
        results.record(f"IPA vowel {ipa} to Arteme", passed, arteme, result_arteme)


def test_arpa_consonants():
    """Test all 24 ARPA consonants"""
    print("\n" + "=" * 70)
    print("Testing ARPA Consonants (24 phonemes)")
    print("=" * 70)
    
    consonant_tests = [
        # (ARPA, IPA, Arteme, Word)
        ("B", "b", "p", "bee"),
        ("CH", "tʃ", "ʃ", "cheese"),
        ("D", "d", "t", "day"),
        ("DH", "ð", "θ", "this"),
        ("F", "f", "f", "fee"),
        ("G", "g", "k", "green"),
        ("HH", "h", "h", "he"),
        ("JH", "dʒ", "ʃ", "gee"),
        ("K", "k", "k", "key"),
        ("L", "l", "l", "lee"),
        ("M", "m", "n", "me"),
        ("N", "n", "n", "knee"),
        ("NG", "ŋ", "n", "sing"),
        ("P", "p", "p", "pea"),
        ("R", "ɹ", "ɹ", "read"),
        ("S", "s", "s", "sea"),
        ("SH", "ʃ", "ʃ", "she"),
        ("T", "t", "t", "tea"),
        ("TH", "θ", "θ", "theta"),
        ("V", "v", "f", "vee"),
        ("W", "w", "ə", "we"),
        ("Y", "j", "ə", "yield"),
        ("Z", "z", "s", "zee"),
        ("ZH", "ʒ", "ʃ", "seizure"),
    ]
    
    for arpa, ipa, arteme, word in consonant_tests:
        # Test ARPA to IPA
        result_ipa = arpa_to_ipa(arpa)
        passed = result_ipa == ipa
        status = "✓" if passed else "✗"
        print(f"  {status} {arpa:3} → {ipa:3} (got: {result_ipa:3}) - {word}")
        results.record(f"ARPA consonant {arpa} to IPA", passed, ipa, result_ipa)
        
        # Test IPA to ARPA
        result_arpa = ipa_to_arpa(ipa)
        passed = result_arpa == arpa
        results.record(f"IPA consonant {ipa} to ARPA", passed, arpa, result_arpa)
        
        # Test IPA to Arteme
        result_arteme = ipa_to_artemes(ipa)
        passed = result_arteme == arteme
        results.record(f"IPA consonant {ipa} to Arteme", passed, arteme, result_arteme)


def test_stress_markers():
    """Test ARPA stress marker handling"""
    print("\n" + "=" * 70)
    print("Testing ARPA Stress Markers")
    print("=" * 70)
    
    stress_tests = [
        ("AE0", "æ", "unstressed"),
        ("AE1", "æ", "primary stress"),
        ("AE2", "æ", "secondary stress"),
        ("K AE1 T", "kæt", "cat with stress"),
        ("HH AH0 L OW1", "hʌloʊ", "hello with stress"),
    ]
    
    for arpa, expected_ipa, description in stress_tests:
        result = arpa_to_ipa(arpa)
        passed = result == expected_ipa
        status = "✓" if passed else "✗"
        print(f"  {status} {arpa:15} → {expected_ipa:6} - {description}")
        results.record(f"Stress marker: {description}", passed, expected_ipa, result)


def test_diphthongs():
    """Test all diphthongs"""
    print("\n" + "=" * 70)
    print("Testing Diphthongs")
    print("=" * 70)
    
    diphthong_tests = [
        ("aɪ", "AY", "əə", "hide"),
        ("aʊ", "AW", "əə", "how"),
        ("eɪ", "EY", "əə", "say"),
        ("oʊ", "OW", "əə", "go"),
        ("ɔɪ", "OY", "əə", "boy"),
    ]
    
    for ipa, arpa, arteme, word in diphthong_tests:
        # IPA to ARPA
        result_arpa = ipa_to_arpa(ipa)
        passed = result_arpa == arpa
        status = "✓" if passed else "✗"
        print(f"  {status} IPA {ipa:3} → ARPA {arpa:3} (got: {result_arpa:3}) - {word}")
        results.record(f"Diphthong {ipa} to ARPA", passed, arpa, result_arpa)
        
        # IPA to Arteme (should be two vowels)
        result_arteme = ipa_to_artemes(ipa)
        passed = result_arteme == arteme
        status = "✓" if passed else "✗"
        print(f"  {status} IPA {ipa:3} → Arteme {arteme:3} (got: {result_arteme:3}) - {word}")
        results.record(f"Diphthong {ipa} to Arteme", passed, arteme, result_arteme)


def test_arteme_classes():
    """Test all 12 arteme classes"""
    print("\n" + "=" * 70)
    print("Testing Arteme Classes (12 classes)")
    print("=" * 70)
    
    arteme_tests = [
        # (Arteme, IPA examples, Description)
        ("ə", ["a", "e", "i", "o", "u", "æ", "ɛ", "ɪ"], "Vowels/Glides"),
        ("s", ["s", "z"], "Alveolar Sibilants"),
        ("ʃ", ["ʃ", "ʒ", "tʃ", "dʒ"], "Postalveolar Sibilants"),
        ("f", ["f", "v"], "Labiodental Fricatives"),
        ("θ", ["θ", "ð"], "Dental Fricatives"),
        ("h", ["h"], "Glottal Fricatives"),
        ("p", ["p", "b"], "Labial Stops"),
        ("t", ["t", "d"], "Coronal Stops"),
        ("k", ["k", "g"], "Dorsal Stops"),
        ("n", ["m", "n", "ŋ"], "Nasals"),
        ("l", ["l"], "Laterals"),
        ("ɹ", ["ɹ"], "Rhotics"),
    ]
    
    for arteme, ipa_examples, description in arteme_tests:
        print(f"\n  Class '{arteme}' - {description}:")
        for ipa in ipa_examples:
            result = ipa_to_artemes(ipa)
            passed = result == arteme
            status = "✓" if passed else "✗"
            print(f"    {status} {ipa:3} → {arteme:3} (got: {result:3})")
            results.record(f"Arteme class {arteme} from {ipa}", passed, arteme, result)


def test_multi_character_sequences():
    """Test multi-character IPA sequences"""
    print("\n" + "=" * 70)
    print("Testing Multi-Character Sequences")
    print("=" * 70)
    
    sequence_tests = [
        # (IPA, ARPA, Arteme, Description)
        ("tʃ", "CH", "ʃ", "voiceless postalveolar affricate"),
        ("dʒ", "JH", "ʃ", "voiced postalveolar affricate"),
        ("t͡ʃ", "CH", "ʃ", "affricate with tie-bar (top)"),
        ("d͡ʒ", "JH", "ʃ", "affricate with tie-bar (top)"),
        ("t͜ʃ", "CH", "ʃ", "affricate with tie-bar (bottom)"),
        ("d͜ʒ", "JH", "ʃ", "affricate with tie-bar (bottom)"),
    ]
    
    for ipa, arpa, arteme, description in sequence_tests:
        # Normalize first
        normalized = normalize_ipa(ipa)
        
        # IPA to ARPA
        result_arpa = ipa_to_arpa(normalized)
        passed = result_arpa == arpa
        status = "✓" if passed else "✗"
        print(f"  {status} {ipa:5} → {arpa:3} (got: {result_arpa:3}) - {description}")
        results.record(f"Multi-char {ipa} to ARPA", passed, arpa, result_arpa)
        
        # IPA to Arteme
        result_arteme = ipa_to_artemes(normalized)
        passed = result_arteme == arteme
        results.record(f"Multi-char {ipa} to Arteme", passed, arteme, result_arteme)


def test_normalization_variants():
    """Test IPA normalization variants"""
    print("\n" + "=" * 70)
    print("Testing IPA Normalization Variants")
    print("=" * 70)
    
    normalization_tests = [
        # (Input, Expected, Description)
        ("ˈhɛloʊ", "hɛloʊ", "remove primary stress"),
        ("hɛˌloʊ", "hɛloʊ", "remove secondary stress"),
        ("ˈhɛˌloʊ", "hɛloʊ", "remove both stress markers"),
        ("ð ə k æ t", "ðəkæt", "merge spaces"),
        ("t͡ʃ", "tʃ", "normalize tie-bar (top)"),
        ("d͜ʒ", "dʒ", "normalize tie-bar (bottom)"),
        ("kʰæt", "kæt", "remove aspiration"),
        ("siːŋ", "siŋ", "remove length marker"),
        ("ɡ", "g", "normalize g with tail"),
        ("r", "ɹ", "trill to approximant"),
        ("ɨ", "ɪ", "close central to front"),
        ("ɐ", "ə", "near-open to mid"),
        ("ʔ", "", "remove glottal stop"),
        ("y", "j", "latin y to IPA j"),
    ]
    
    for input_ipa, expected, description in normalization_tests:
        result = normalize_ipa(input_ipa)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} '{input_ipa:10}' → '{expected:10}' (got: '{result:10}') - {description}")
        results.record(f"Normalize: {description}", passed, expected, result)


def test_common_words():
    """Test conversion of common words"""
    print("\n" + "=" * 70)
    print("Testing Common Words")
    print("=" * 70)
    
    word_tests = [
        # (Word, ARPA, IPA, Arteme)
        ("cat", "K AE1 T", "kæt", "kət"),
        ("dog", "D AO1 G", "dɔg", "tək"),
        ("hello", "HH EH L OW", "hɛloʊ", "hələə"),
        ("world", "W ER L D", "wɝld", "əəɹlt"),  # ɝ is r-colored schwa → əɹ
        ("the", "DH AH0", "ðʌ", "θə"),
        ("chair", "CH EH R", "tʃɛɹ", "ʃəɹ"),
        ("sing", "S IH NG", "sɪŋ", "sən"),
        ("read", "R IY D", "ɹid", "ɹət"),
        ("house", "HH AW S", "haʊs", "həəs"),
        ("boy", "B OY", "bɔɪ", "pəə"),
    ]
    
    for word, arpa, ipa, arteme in word_tests:
        print(f"\n  Word: '{word}'")
        
        # ARPA to IPA
        result_ipa = arpa_to_ipa(arpa)
        passed = result_ipa == ipa
        status = "✓" if passed else "✗"
        print(f"    {status} ARPA '{arpa}' → IPA '{ipa}' (got: '{result_ipa}')")
        results.record(f"Word '{word}' ARPA to IPA", passed, ipa, result_ipa)
        
        # IPA to ARPA (without stress)
        arpa_base = arpa.replace("0", "").replace("1", "").replace("2", "")
        result_arpa = ipa_to_arpa(ipa)
        passed = result_arpa == arpa_base
        status = "✓" if passed else "✗"
        print(f"    {status} IPA '{ipa}' → ARPA '{arpa_base}' (got: '{result_arpa}')")
        results.record(f"Word '{word}' IPA to ARPA", passed, arpa_base, result_arpa)
        
        # IPA to Arteme
        result_arteme = ipa_to_artemes(ipa)
        passed = result_arteme == arteme
        status = "✓" if passed else "✗"
        print(f"    {status} IPA '{ipa}' → Arteme '{arteme}' (got: '{result_arteme}')")
        results.record(f"Word '{word}' IPA to Arteme", passed, arteme, result_arteme)


def test_validation():
    """Test IPA validation"""
    print("\n" + "=" * 70)
    print("Testing IPA Validation")
    print("=" * 70)
    
    validation_tests = [
        ("kæt", True, [], [], "valid word"),
        ("ˈkæt", True, [], ["ˈ"], "stress marker excluded"),
        ("ð ə", True, [], [" "], "space excluded"),
        ("123", False, ["1", "2", "3"], [], "invalid symbols"),  # x,y,z have mappings
    ]
    
    for ipa, expected_valid, expected_unknown, expected_excluded, description in validation_tests:
        result = validate_ipa(ipa)
        passed = (
            result['valid'] == expected_valid and
            set(result['unknown_symbols']) == set(expected_unknown) and
            set(result['excluded_symbols']) == set(expected_excluded)
        )
        status = "✓" if passed else "✗"
        print(f"  {status} '{ipa:10}' - {description}")
        print(f"      valid={result['valid']}, unknown={result['unknown_symbols']}, excluded={result['excluded_symbols']}")
        results.record(f"Validation: {description}", passed, 
                      f"valid={expected_valid}", f"valid={result['valid']}")


def test_round_trips():
    """Test round-trip conversions"""
    print("\n" + "=" * 70)
    print("Testing Round-Trip Conversions")
    print("=" * 70)
    
    round_trip_tests = [
        ("kæt", "K AE T"),
        ("hɛloʊ", "HH EH L OW"),
        ("tʃɛɹ", "CH EH R"),
        ("dʒim", "JH IY M"),
    ]
    
    for original_ipa, expected_arpa in round_trip_tests:
        # IPA → ARPA → IPA
        arpa = ipa_to_arpa(original_ipa)
        restored_ipa = arpa_to_ipa(arpa)
        passed = restored_ipa == original_ipa
        status = "✓" if passed else "✗"
        print(f"  {status} IPA→ARPA→IPA: '{original_ipa}' → '{arpa}' → '{restored_ipa}'")
        results.record(f"Round-trip IPA '{original_ipa}'", passed, original_ipa, restored_ipa)


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    # Empty strings
    result = normalize_ipa("")
    passed = result == ""
    status = "✓" if passed else "✗"
    print(f"  {status} Empty string normalization")
    results.record("Empty string normalization", passed, "", result)
    
    result = ipa_to_arpa("")
    passed = result == ""
    status = "✓" if passed else "✗"
    print(f"  {status} Empty string IPA to ARPA")
    results.record("Empty string IPA to ARPA", passed, "", result)
    
    result = arpa_to_ipa("")
    passed = result == ""
    status = "✓" if passed else "✗"
    print(f"  {status} Empty string ARPA to IPA")
    results.record("Empty string ARPA to IPA", passed, "", result)
    
    # Unknown symbols (non-strict mode) - numbers have no mapping
    result = ipa_to_artemes("123", strict=False)
    passed = result == ""
    status = "✓" if passed else "✗"
    print(f"  {status} Unknown symbols (non-strict): got '{result}'")
    results.record("Unknown symbols non-strict", passed, "", result)
    
    # Mixed valid and invalid - letters with mappings still convert
    result = ipa_to_artemes("kæt123", strict=False)
    passed = result == "kət"
    status = "✓" if passed else "✗"
    print(f"  {status} Mixed valid/invalid: got '{result}'")
    results.record("Mixed valid/invalid", passed, "kət", result)


def main():
    """Run all comprehensive tests"""
    print("=" * 70)
    print("COMPREHENSIVE PHONETICS LIBRARY TESTS")
    print("=" * 70)
    
    test_arpa_vowels()
    test_arpa_consonants()
    test_stress_markers()
    test_diphthongs()
    test_arteme_classes()
    test_multi_character_sequences()
    test_normalization_variants()
    test_common_words()
    test_validation()
    test_round_trips()
    test_edge_cases()
    
    results.print_summary()


if __name__ == '__main__':
    main()