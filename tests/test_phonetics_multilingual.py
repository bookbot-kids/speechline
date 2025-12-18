"""
Multilingual Phonetics Tests

Tests for Spanish, Bahasa Indonesia, and Swahili language support
in the unified phonetics library.

Coverage:
- Spanish: Nasalized vowels, flaps, palatal sounds
- Indonesian: Glottal stops, schwa, diverse vowels
- Swahili: Pre-nasalized consonants, implosives, clicks
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.phonetics import (
    normalize_ipa,
    ipa_to_artemes,
    validate_ipa,
)


class TestSpanish:
    """Test Spanish phonetic conversions."""
    
    def test_spanish_flaps(self):
        """Test Spanish flap/tap consonants."""
        # pero (but) - /ˈpeɾo/
        assert normalize_ipa("/ˈpeɾo/") == "peɹo"
        assert ipa_to_artemes("/ˈpeɾo/") == "pəɹə"
        
        # caro (expensive) - /ˈkaɾo/
        assert normalize_ipa("/ˈkaɾo/") == "kaɹo"
        assert ipa_to_artemes("/ˈkaɾo/") == "kəɹə"
    
    def test_spanish_theta(self):
        """Test Spanish theta (interdental fricative)."""
        # corazón (heart) - /koɾaˈθon/
        assert normalize_ipa("/koɾaˈθon/") == "koɹaθon"
        assert ipa_to_artemes("/koɾaˈθon/") == "kəɹəθən"
        
        # zapato (shoe) - /θaˈpato/
        assert normalize_ipa("/θaˈpato/") == "θapato"
        assert ipa_to_artemes("/θaˈpato/") == "θəpətə"
    
    def test_spanish_palatal_nasal(self):
        """Test Spanish palatal nasal ñ."""
        # año (year) - /ˈaɲo/
        assert normalize_ipa("/ˈaɲo/") == "aɲo"
        assert ipa_to_artemes("/ˈaɲo/") == "ənə"
        
        # niño (child) - /ˈniɲo/
        assert normalize_ipa("/ˈniɲo/") == "niɲo"
        assert ipa_to_artemes("/ˈniɲo/") == "nənə"
    
    def test_spanish_lateral_palatal(self):
        """Test Spanish palatal lateral ll."""
        # calle (street) - /ˈkaʎe/
        assert normalize_ipa("/ˈkaʎe/") == "kaʎe"
        assert ipa_to_artemes("/ˈkaʎe/") == "kələ"
        
        # lluvia (rain) - /ˈʎuβja/
        assert normalize_ipa("/ˈʎuβja/") == "ʎuβja"
        assert ipa_to_artemes("/ˈʎuβja/") == "ləfəə"  # j → ə (glide)
    
    def test_spanish_nasalized_vowels(self):
        """Test Spanish nasalized vowels (rare, but in loanwords)."""
        # Some dialects nasalize vowels near nasal consonants
        assert normalize_ipa("/kãsa/") == "kasa"
        assert ipa_to_artemes("/kãsa/") == "kəsə"
        
        assert normalize_ipa("/põte/") == "pote"
        assert ipa_to_artemes("/põte/") == "pətə"
    
    def test_spanish_common_words(self):
        """Test common Spanish words."""
        # hola (hello) - /ˈola/
        assert normalize_ipa("/ˈola/") == "ola"
        assert ipa_to_artemes("/ˈola/") == "ələ"
        
        # gracias (thanks) - /ˈɡɾaθjas/
        assert normalize_ipa("/ˈɡɾaθjas/") == "gɹaθjas"
        assert ipa_to_artemes("/ˈɡɾaθjas/") == "kɹəθəəs"  # j → ə (glide)
        
        # español (Spanish) - /espaˈɲol/
        assert normalize_ipa("/espaˈɲol/") == "espaɲol"
        assert ipa_to_artemes("/espaˈɲol/") == "əspənəl"


class TestIndonesian:
    """Test Indonesian (Bahasa Indonesia) phonetic conversions."""
    
    def test_indonesian_glottal_stop(self):
        """Test Indonesian glottal stops."""
        # tidak (no) - /ˈtidaʔ/
        assert normalize_ipa("/ˈtidaʔ/") == "tida"
        assert ipa_to_artemes("/ˈtidaʔ/") == "tətə"
        
        # rakyat (people) - /ˈrakjat/  # sometimes with glottal stop
        assert normalize_ipa("/ˈrakjaʔ/") == "ɹakja"  # r → ɹ normalization
        assert ipa_to_artemes("/ˈrakjaʔ/") == "ɹəkəə"  # j → ə (glide)
    
    def test_indonesian_schwa(self):
        """Test Indonesian schwa."""
        # selamat (greetings) - /səˈlamat/
        assert normalize_ipa("/səˈlamat/") == "səlamat"
        assert ipa_to_artemes("/səˈlamat/") == "sələnət"
        
        # terima (thank) - /təˈɾima/
        assert normalize_ipa("/təˈɾima/") == "təɹima"
        assert ipa_to_artemes("/təˈɾima/") == "təɹənə"
    
    def test_indonesian_ng_sound(self):
        """Test Indonesian ng sound."""
        # dengan (with) - /dəŋan/
        assert normalize_ipa("/dəŋan/") == "dəŋan"
        assert ipa_to_artemes("/dəŋan/") == "tənən"
        
        # jangan (don't) - /d͡ʒaŋan/
        assert normalize_ipa("/d͡ʒaŋan/") == "dʒaŋan"
        assert ipa_to_artemes("/d͡ʒaŋan/") == "ʃənən"
    
    def test_indonesian_vowel_variants(self):
        """Test Indonesian vowel variations."""
        # Words with various vowels
        assert normalize_ipa("/ˈmakan/") == "makan"  # eat
        assert ipa_to_artemes("/ˈmakan/") == "nəkən"
        
        assert normalize_ipa("/ˈminum/") == "minum"  # drink
        assert ipa_to_artemes("/ˈminum/") == "nənən"
        
        assert normalize_ipa("/ˈbuku/") == "buku"  # book
        assert ipa_to_artemes("/ˈbuku/") == "pəkə"
    
    def test_indonesian_common_words(self):
        """Test common Indonesian words."""
        # apa (what) - /ˈapa/
        assert normalize_ipa("/ˈapa/") == "apa"
        assert ipa_to_artemes("/ˈapa/") == "əpə"
        
        # siapa (who) - /siˈapa/
        assert normalize_ipa("/siˈapa/") == "siapa"
        assert ipa_to_artemes("/siˈapa/") == "səəpə"
        
        # bagaimana (how) - /baɡajˈmana/
        assert normalize_ipa("/baɡajˈmana/") == "bagajmana"
        assert ipa_to_artemes("/baɡajˈmana/") == "pəkəənənə"  # j → ə (glide)


class TestSwahili:
    """Test Swahili phonetic conversions."""
    
    def test_swahili_prenasalized_consonants(self):
        """Test Swahili pre-nasalized consonants."""
        # simba (lion) - /ˈsi.ᵐbɑ/
        assert normalize_ipa("/ˈsi.ᵐbɑ/") == "siᵐbɑ"
        assert ipa_to_artemes("/ˈsi.ᵐbɑ/") == "sənpə"
        
        # jambo (hello) - /ˈʄɑ.ᵐbɔ/
        assert normalize_ipa("/ˈʄɑ.ᵐbɔ/") == "ʄɑᵐbɔ"
        assert ipa_to_artemes("/ˈʄɑ.ᵐbɔ/") == "tənpə"
        
        # ndizi (banana) - /ˈⁿdi.zi/
        assert normalize_ipa("/ˈⁿdi.zi/") == "ⁿdizi"
        assert ipa_to_artemes("/ˈⁿdi.zi/") == "ntəsə"
        
        # ngoma (drum) - /ˈŋɡɔma/
        assert normalize_ipa("/ˈŋɡɔma/") == "ŋgɔma"
        assert ipa_to_artemes("/ˈŋɡɔma/") == "nkənə"
    
    def test_swahili_implosives(self):
        """Test Swahili implosive consonants."""
        # habari (news) - /hɑˈɓɑ.ɾi/
        assert normalize_ipa("/hɑˈɓɑ.ɾi/") == "hɑɓɑɹi"
        assert ipa_to_artemes("/hɑˈɓɑ.ɾi/") == "həpəɹə"
        
        # ndege (bird) - /ˈⁿdɛ.ɠɛ/
        assert normalize_ipa("/ˈⁿdɛ.ɠɛ/") == "ⁿdɛɠɛ"
        assert ipa_to_artemes("/ˈⁿdɛ.ɠɛ/") == "ntəkə"
        
        # adui (enemy) - /ɑˈɗui/
        assert normalize_ipa("/ɑˈɗui/") == "ɑɗui"  # ɑ is preserved (IPA open back vowel)
        assert ipa_to_artemes("/ɑˈɗui/") == "ətəə"
    
    def test_swahili_aspirated_consonants(self):
        """Test Swahili aspirated consonants."""
        # kuku (chicken) - /ˈku.ku/ or /ˈkʰu.ku/
        assert normalize_ipa("/ˈkʰu.ku/") == "kuku"
        assert ipa_to_artemes("/ˈkʰu.ku/") == "kəkə"
        
        # tatu (three) - /ˈtɑ.tu/ or /ˈtʰɑ.tu/
        assert normalize_ipa("/ˈtʰɑ.tu/") == "tɑtu"
        assert ipa_to_artemes("/ˈtʰɑ.tu/") == "tətə"
    
    def test_swahili_dental_vs_alveolar(self):
        """Test Swahili dental and alveolar distinctions."""
        # taka (want) - /ˈtɑ.kɑ/ or /ˈt̪ɑ.kɑ/
        assert normalize_ipa("/ˈt̪ɑ.kɑ/") == "tɑkɑ"  # ɑ preserved
        assert ipa_to_artemes("/ˈt̪ɑ.kɑ/") == "təkə"
        
        # dhakari (doctor) - /ðɑˈkɑ.ri/
        assert normalize_ipa("/ðɑˈkɑ.ri/") == "ðɑkɑɹi"  # ɑ preserved
        assert ipa_to_artemes("/ðɑˈkɑ.ri/") == "θəkəɹə"
    
    def test_swahili_common_words(self):
        """Test common Swahili words."""
        # maji (water) - /ˈmɑ.d͡ʒi/
        assert normalize_ipa("/ˈmɑ.d͡ʒi/") == "mɑdʒi"  # ɑ preserved
        assert ipa_to_artemes("/ˈmɑ.d͡ʒi/") == "nəʃə"
        
        # asante (thank you) - /ɑˈsɑn.te/
        assert normalize_ipa("/ɑˈsɑn.te/") == "ɑsɑnte"  # ɑ preserved
        assert ipa_to_artemes("/ɑˈsɑn.te/") == "əsəntə"
        
        # karibu (welcome) - /kɑˈɾi.bu/
        assert normalize_ipa("/kɑˈɾi.bu/") == "kɑɹibu"  # ɑ preserved
        assert ipa_to_artemes("/kɑˈɾi.bu/") == "kəɹəpə"
        
        # safari (journey) - /sɑˈfɑ.ɾi/
        assert normalize_ipa("/sɑˈfɑ.ɾi/") == "sɑfɑɹi"  # ɑ preserved
        assert ipa_to_artemes("/sɑˈfɑ.ɾi/") == "səfəɹə"


class TestMultilingualEdgeCases:
    """Test edge cases across all three languages."""
    
    def test_complex_consonant_clusters(self):
        """Test complex consonant clusters."""
        # Spanish: transcripción
        assert normalize_ipa("/tɾanskɾipˈθjon/") == "tɹanskɹipθjon"
        assert ipa_to_artemes("/tɾanskɾipˈθjon/") == "tɹənskɹəpθəən"  # j→ə (palatal approximant)
        
        # Swahili: mchungaji (shepherd) - note: ɡ → g normalization
        assert normalize_ipa("/m̩.tʃuˈᵑɡɑ.d͡ʒi/") == "mtʃuᵑgɑdʒi"
        assert ipa_to_artemes("/m̩.tʃuˈᵑɡɑ.d͡ʒi/") == "nʃənkəʃə"
    
    def test_diacritic_stripping(self):
        """Test that all diacritics are properly stripped."""
        # Various diacritics should be removed
        assert normalize_ipa("/kʰaʲ/") == "ka"  # Aspiration + palatalization
        assert normalize_ipa("/tʷaˤ/") == "ta"  # Labialization + pharyngealization
        assert normalize_ipa("/s̪a̹/") == "sa"   # Dental + rounding diacritics
    
    def test_validation_multilingual(self):
        """Test validation with multilingual symbols."""
        # Spanish word should validate
        result = validate_ipa("/koɾaˈθon/")
        assert result['valid'] is True
        
        # Indonesian word should validate
        result = validate_ipa("/təˈɾima/")
        assert result['valid'] is True
        
        # Swahili word should validate
        result = validate_ipa("/ˈʄɑ.ᵐbɔ/")
        assert result['valid'] is True
    
    def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid inputs."""
        assert normalize_ipa("") == ""
        assert ipa_to_artemes("") == ""
        
        # Numeric characters are kept but unmapped (ignored in arteme conversion)
        assert normalize_ipa("/k1a2t3/") == "k1a2t3"
        assert ipa_to_artemes("/k1a2t3/") == "kət"  # numbers ignored in arteme


class TestRealWorldExamples:
    """Test real-world examples from the language datasets."""
    
    def test_spanish_dataset_samples(self):
        """Test actual examples from spanish_words_ipa.csv."""
        # Common Spanish phonemes
        examples = [
            ("/ˈpeɾo/", "peɹo", "pəɹə"),      # pero (but)
            ("/ˈkaɾne/", "kaɹne", "kəɹnə"),   # carne (meat)
            ("/aˈmiɡo/", "amigo", "ənəkə"),   # amigo (friend)
        ]
        
        for ipa, expected_norm, expected_art in examples:
            assert normalize_ipa(ipa) == expected_norm
            assert ipa_to_artemes(ipa) == expected_art
    
    def test_indonesian_dataset_samples(self):
        """Test actual examples from indonesian_words_ipa.csv."""
        examples = [
            ("/ˈbaik/", "baik", "pəək"),        # baik (good) - both vowels → ə
            ("/ˈsaja/", "saja", "səəə"),        # saya (I/me)
            ("/iˈtu/", "itu", "ətə"),           # itu (that)
        ]
        
        for ipa, expected_norm, expected_art in examples:
            assert normalize_ipa(ipa) == expected_norm
            assert ipa_to_artemes(ipa) == expected_art
    
    def test_swahili_dataset_samples(self):
        """Test actual examples from swahili_words_ipa.csv."""
        examples = [
            ("/ˈwɑ.tu/", "wɑtu", "əətə"),       # watu (people) - w → ə
            ("/ˈmɛzɑ/", "mɛzɑ", "nəsə"),        # meza (table)
            ("/ˈki.tu/", "kitu", "kətə"),       # kitu (thing)
        ]
        
        for ipa, expected_norm, expected_art in examples:
            assert normalize_ipa(ipa) == expected_norm
            assert ipa_to_artemes(ipa) == expected_art


def run_tests():
    """Run all test classes."""
    test_classes = [
        TestSpanish,
        TestIndonesian,
        TestSwahili,
        TestMultilingualEdgeCases,
        TestRealWorldExamples,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running {test_class.__name__}")
        print('='*70)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(test_instance, method_name)
            
            try:
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())