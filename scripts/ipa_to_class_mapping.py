"""IPA Phoneme to Phonetic Class Mapping and Parser

This module provides:
1. Canonical IPA normalization to standardize different transcription styles
2. Mapping from IPA symbols to robust phonetic classes
3. Parser function to convert IPA sequences to class sequences

Based on the robust phonetic classification system from manner_of_articulation.csv

Key Features:
- Normalizes variant IPA representations to canonical forms
- Handles space-separated vs continuous phoneme sequences
- Converts IPA to 12-class phonetic system
- Supports multi-character sequences (affricates, diphthongs)

Statistics (corrected mapping):
  Total base symbols: ~50+
  Excluded symbols: ~90+ (diacritics, stress, tone, etc.)
  Click/trill/implosive mappings: Added to main mapping
"""

import unicodedata
from typing import Tuple, Dict, List

# =============================================================================
# EXCLUSION TABLES
# =============================================================================

# Symbols to exclude completely (non-phonemic annotations)
EXCLUDE_SYMBOLS = {
    # Brackets and delimiters
    '(', ')', '[', ']', '/', '|',
    
    # Stress and prosody markers
    'ˈ',   # primary stress
    'ˌ',   # secondary stress
    'ː',   # length mark
    'ˑ',   # half-long
    '.',   # syllable boundary
    'ʔ',   # glottal stop (often transcription artifact)
    
    # Tone marks and intonation
    '˥', '˦', '˧', '˨', '˩',  # tone levels
    '↓', '↗',  # tone movement
    
    # Typography and punctuation
    '–', '…', '‿', '²', '³', '¹', '⁴', '⁵', '⁻', 'ⁿ', '︎', "'",
    
    # Common diacritics (combining characters)
    '̃',   # nasalization
    '̚',   # unreleased
    'ʰ',   # aspiration
    'ʷ',   # labialization
    'ʲ',   # palatalization
    '̩',   # syllabic (handled via syllabic consonant mapping)
    'ʱ',   # breathy voice
    '̀', '́', '̂', '̄', '̆', '̈', '̊', '̍',
    '̙', '̝', '̞', '̟', '̠', '̥', '̪', '̬', '̯', '̰', '̹', '̺', '̽',
    '̧',   # cedilla (from NFD decomposition of ç)
    '͆', '͇', '͜', '͡',  # tie bars (except when part of affricate)
    
    # Superscripts and other modifiers
    'ʳ', 'ʴ', 'ʼ', 'ˀ', '˔', '˞', 'ˠ', 'ˡ', 'ˤ', '˭',
    'ᵈ', 'ᵊ', 'ᵐ', 'ᵒ', 'ᶴ', '᷈',
    
    # Spaces and separators
    ' ', '-', '~', '',
}

# =============================================================================
# CANONICAL IPA NORMALIZATION MAP
# =============================================================================

# Map alternative IPA representations to canonical forms
# This ensures compatibility between different IPA sources (lexicons, datasets, etc.)
IPA_NORMALIZATION_MAP = {
    # Vowel normalizations
    'ɨ': 'ɪ',      # Close central → close front (near-equivalent)
    'ɘ': 'ə',      # Close-mid central → mid central
    'ɐ': 'ə',      # Near-open central → mid central
    'ʉ': 'u',      # Close central rounded → close back rounded
    
    # Consonant normalizations
    'ɡ': 'g',      # G with tail → standard g
    'r': 'ɹ',      # Trill/tap → approximant (English uses approximant)
    'ɾ': 'ɹ',      # Tap/flap → approximant (American English)
    'ʔ': '',       # Glottal stop → remove (not phonemic in English)
    
    # Common transcription variants
    'y': 'j',      # Latin y → IPA j for palatal approximant
}

# Affricate normalization: all tie-bar forms to simple sequences
AFFRICATE_NORMALIZATION = {
    # With tie bars
    't͡ʃ': 'tʃ',
    'd͡ʒ': 'dʒ',
    't͜ʃ': 'tʃ',
    'd͜ʒ': 'dʒ',
    't͡s': 'ts',
    'd͡z': 'dz',
    't͜s': 'ts',
    'd͜z': 'dz',
}

# =============================================================================
# MULTI-CHARACTER SEQUENCES
# =============================================================================

# Must be checked in this order (longest first) for greedy matching
MULTI_CHAR_SEQUENCES = [
    # Tie-bar affricates (3 characters including combining tie-bar ͡ or ͜)
    't͡ʃ', 'd͡ʒ', 't͡s', 'd͡z',
    't͜ʃ', 'd͜ʒ', 't͜s', 'd͜z',  # alternate tie-bar
    
    # Simple affricates (2 characters)
    'tʃ', 'dʒ', 'ts', 'dz',
    
    # Diphthongs (2 characters)
    'aɪ', 'eɪ', 'oʊ', 'aʊ', 'ɔɪ',
    
    # Syllabic consonants (base + syllabic diacritic)
    'l̩', 'r̩', 'ɹ̩', 'm̩', 'n̩', 'ŋ̩',
]

# =============================================================================
# IPA TO CLASS MAPPING
# =============================================================================

IPA_TO_CLASS = {
    # -------------------------------------------------------------------------
    # V - Vowels/Diphthongs → ə
    # All vowels and diphthongs collapsed to single class
    # Examples: s(i)t, b(e)d, c(a)t, h(o)t, b(oo)k, (a)bout, pr(i)ce, f(a)ce
    # -------------------------------------------------------------------------
    'a': 'ə',
    'e': 'ə',
    'i': 'ə',
    'o': 'ə',
    'u': 'ə',
    'æ': 'ə',
    'œ': 'ə',
    'ɐ': 'ə',
    'ɑ': 'ə',
    'ɒ': 'ə',
    'ɔ': 'ə',
    'ɘ': 'ə',
    'ə': 'ə',
    'ɚ': 'əɹ',  # Rhotic schwa (American English) = schwa + rhotic
    'ɛ': 'ə',
    'ɜ': 'ə',
    'ɝ': 'əɹ',  # R-colored schwa = schwa + rhotic (must map to 2 classes like ər)
    'ɞ': 'ə',
    'ɤ': 'ə',
    'ɨ': 'ə',
    'ɪ': 'ə',
    'ɯ': 'ə',
    'ɵ': 'ə',
    'ɶ': 'ə',
    'ʉ': 'ə',
    'ʊ': 'ə',
    'ʌ': 'ə',
    'ʏ': 'ə',
    
    # Diphthongs (multi-char sequences)
    'aɪ': 'əə',
    'aʊ': 'əə',
    'eɪ': 'əə',
    'oʊ': 'əə',
    'ɔɪ': 'əə',
    
    # -------------------------------------------------------------------------
    # SIB-ALV - Alveolar Sibilants → s
    # Includes alveolar sibilant fricatives and affricates
    # Examples: (s)it, (z)oo, ea(s)y, pi(zz)a, ca(ts)
    # -------------------------------------------------------------------------
    's': 's',
    'z': 's',
    
    # Alveolar affricates (added)
    'ts': 's',
    'dz': 's',
    't͡s': 's',
    'd͡z': 's',
    't͜s': 's',
    'd͜z': 's',
    
    # -------------------------------------------------------------------------
    # SIB-POST - Postalveolar Sibilants → ʃ
    # Includes postalveolar/alveolo-palatal sibilants and affricates
    # Examples: (sh)e, vi(si)on, (ch)air, (j)udge
    # -------------------------------------------------------------------------
    'ʃ': 'ʃ',
    'ʒ': 'ʃ',
    'ɕ': 'ʃ',
    'ʑ': 'ʃ',
    
    # Postalveolar affricates (tie-bar forms added)
    'tʃ': 'ʃ',
    'dʒ': 'ʃ',
    't͡ʃ': 'ʃ',
    'd͡ʒ': 'ʃ',
    't͜ʃ': 'ʃ',
    'd͜ʒ': 'ʃ',
    
    # -------------------------------------------------------------------------
    # FR-LAB - Labiodental Fricatives → f
    # Examples: (f)ine, (v)oice, (ph)one, de(v)elop [labiodental approximant]
    # -------------------------------------------------------------------------
    'f': 'f',
    'v': 'f',
    'ɸ': 'f',  # bilabial fricative
    'β': 'f',  # bilabial fricative
    'ʋ': 'f',  # labiodental approximant (treated as fricative)
    
    # -------------------------------------------------------------------------
    # FR-DENT - Dental Fricatives → θ
    # Examples: (th)in, (th)is, bo(th)
    # -------------------------------------------------------------------------
    'θ': 'θ',
    'ð': 'θ',
    
    # -------------------------------------------------------------------------
    # FR-GLOT - Glottal & Other Fricatives → h
    # NOTE: Deliberate coarse merge of non-sibilant fricatives for robustness
    # Includes glottal, velar, uvular, pharyngeal, and palatal fricatives
    # Examples: (h)at, lo(ch) [Scottish], Ba(ch) [German], A(h)med [Arabic]
    # -------------------------------------------------------------------------
    'h': 'h',
    'ɦ': 'h',  # breathy-voiced glottal
    
    # Velar/uvular fricatives (coarse merge)
    'x': 'h',  # voiceless velar
    'ɣ': 'h',  # voiced velar
    'χ': 'h',  # voiceless uvular
    
    # Pharyngeal fricatives (coarse merge)
    'ʕ': 'h',  # voiced pharyngeal
    'ħ': 'h',  # voiceless pharyngeal
    
    # Palatal fricatives (coarse merge)
    'ç': 'h',  # voiceless palatal (handled before NFD in parser)
    'ʝ': 'h',  # voiced palatal
    
    # -------------------------------------------------------------------------
    # ST-LAB - Labial Stops → p
    # Includes bilabial stops, clicks, trills, and implosives (mapped to nearest)
    # Examples: (p)en, (b)at, a(pp)le, (mwah) [kiss sound]
    # -------------------------------------------------------------------------
    'p': 'p',
    'b': 'p',
    
    # Click consonants - mapped to nearest place equivalent
    'ʘ': 'p',  # bilabial click → labial stop
    'ɓ': 'p',  # bilabial implosive → labial stop
    'ʙ': 'p',  # bilabial trill → labial stop (uncommon, but labial articulation)
    
    # -------------------------------------------------------------------------
    # ST-COR - Coronal Stops → t
    # Includes alveolar, dental, retroflex stops, and coronal clicks
    # Examples: (t)wo, (d)ay, wa(t)er, bu(tt)er [flap], !(K)ung [click]
    # -------------------------------------------------------------------------
    't': 't',
    'd': 't',
    'ɾ': 't',  # flap
    'ɖ': 't',  # retroflex
    'ʈ': 't',  # retroflex
    'ɽ': 't',  # retroflex flap
    
    # Click consonants - coronal articulation
    'ǀ': 't',  # dental click → coronal stop
    'ǃ': 't',  # alveolar click → coronal stop
    'ǂ': 't',  # palatal click → coronal stop (between t and k)
    'ɗ': 't',  # alveolar implosive → coronal stop
    'ʄ': 't',  # palatal implosive → coronal stop
    
    # -------------------------------------------------------------------------
    # ST-DOR - Dorsal Stops → k
    # Includes velar, uvular, and palatal stops
    # Examples: (c)at, (g)o, bac(k), s(qu)are
    # -------------------------------------------------------------------------
    'k': 'k',
    'g': 'k',
    'ɡ': 'k',  # g with tail
    'q': 'k',  # uvular
    'ɢ': 'k',  # uvular
    'c': 'k',  # palatal
    'ɟ': 'k',  # palatal
    
    # -------------------------------------------------------------------------
    # NAS - Nasals → n
    # All nasals collapsed to single class
    # Examples: (m)ap, (n)ot, si(ng), ca(ny)on
    # -------------------------------------------------------------------------
    'm': 'n',
    'n': 'n',
    'ŋ': 'n',
    'ɱ': 'n',  # labiodental
    'ɲ': 'n',  # palatal
    'ɳ': 'n',  # retroflex
    'ɴ': 'n',  # uvular
    
    # -------------------------------------------------------------------------
    # LAT - Laterals → l
    # Includes lateral approximants, fricatives, and lateral clicks
    # Examples: (l)ight, bott(le), mi(ll)ion, (Ll)anelli [Welsh]
    # -------------------------------------------------------------------------
    'l': 'l',
    'ɫ': 'l',  # velarized/dark l
    'ɬ': 'l',  # lateral fricative
    'ɭ': 'l',  # retroflex
    'ɺ': 'l',  # lateral flap
    'ʎ': 'l',  # palatal
    'ʟ': 'l',  # velar
    
    # Lateral click
    'ǁ': 'l',  # lateral click → lateral
    
    # Syllabic lateral (multi-char)
    'l̩': 'l',
    
    # -------------------------------------------------------------------------
    # RHO - Rhotics → ɹ
    # All r-sounds collapsed to single class
    # Examples: (r)ed, a(rr)ive, ca(r), Pa(r)is [French uvular]
    # -------------------------------------------------------------------------
    'ɹ': 'ɹ',
    'r': 'ɹ',
    'ɻ': 'ɹ',  # retroflex
    'ʀ': 'ɹ',  # uvular trill
    'ʁ': 'ɹ',  # uvular fricative
    
    # Syllabic rhotics (multi-char)
    'r̩': 'ɹ',
    'ɹ̩': 'ɹ',
    
    # -------------------------------------------------------------------------
    # GLD - Glides → ə (mapped to vowels)
    # Includes palatal and labial-velar approximants
    # Examples: (y)es, (w)e, h(u)ge, (wh)ich
    # Note: Glides treated as vowel-like for phonetic class purposes
    # -------------------------------------------------------------------------
    'j': 'ə',
    'y': 'ə',  # Latin y used for palatal approximant (alternate notation)
    'w': 'ə',
    'ɥ': 'ə',  # labial-palatal
    'ʍ': 'ə',  # voiceless labial-velar
}

# =============================================================================
# CANONICAL NORMALIZATION FUNCTION
# =============================================================================

def normalize_ipa(ipa_string: str,
                  remove_stress: bool = True,
                  merge_spaces: bool = True) -> str:
    """
    Normalize IPA string to canonical form before parsing.
    
    This function standardizes different IPA transcription styles to ensure
    compatibility between lexicons and datasets.
    
    Normalization steps:
    1. Unicode NFD decomposition
    2. Normalize affricates (tie-bar forms to simple)
    3. Remove stress markers and non-phonemic annotations
    4. Merge space-separated phonemes (e.g., "ð ə" → "ðə")
    5. Apply character-level normalization map
    6. Remove remaining combining marks
    
    Args:
        ipa_string: Input IPA transcription
        remove_stress: If True, remove stress markers (ˈˌ)
        merge_spaces: If True, convert space-separated phonemes to continuous
        
    Returns:
        Normalized IPA string ready for class sequence conversion
        
    Examples:
        >>> normalize_ipa("ˈhɛˌloʊ")
        'hɛloʊ'
        
        >>> normalize_ipa("ð ə")
        'ðə'
        
        >>> normalize_ipa("t͡ʃ")
        'tʃ'
        
        >>> normalize_ipa("kʰæt")
        'kæt'
    """
    if not ipa_string:
        return ""
    
    # Step 1: Handle special cases before NFD (ç would decompose incorrectly)
    result = ipa_string.replace('ç', '\x00PALATAL_FRIC\x00')
    
    # Step 2: Unicode normalization (NFD decomposition)
    result = unicodedata.normalize('NFD', result)
    
    # Step 3: Restore special cases
    result = result.replace('\x00PALATAL_FRIC\x00', 'ç')
    
    # Step 4: Normalize affricates (longest first for greedy matching)
    for affricate, canonical in sorted(AFFRICATE_NORMALIZATION.items(),
                                       key=lambda x: len(x[0]),
                                       reverse=True):
        result = result.replace(affricate, canonical)
    
    # Step 5: Remove excluded symbols
    if remove_stress:
        for symbol in EXCLUDE_SYMBOLS:
            result = result.replace(symbol, '')
    
    # Step 6: Merge spaces if requested (some datasets use space-separated phonemes)
    if merge_spaces:
        result = result.replace(' ', '')
    
    # Step 7: Apply normalization map
    normalized_chars = []
    i = 0
    while i < len(result):
        # Try multi-character sequences first (affricates, diphthongs)
        matched = False
        for length in [3, 2]:
            if i + length <= len(result):
                seq = result[i:i+length]
                # Check if it's a valid multi-char sequence we want to keep
                if seq in MULTI_CHAR_SEQUENCES or seq in IPA_TO_CLASS:
                    normalized_chars.append(seq)
                    i += length
                    matched = True
                    break
        
        if not matched:
            # Single character - apply normalization
            char = result[i]
            if char in IPA_NORMALIZATION_MAP:
                replacement = IPA_NORMALIZATION_MAP[char]
                if replacement:  # Empty string means remove
                    normalized_chars.append(replacement)
            else:
                normalized_chars.append(char)
            i += 1
    
    result = ''.join(normalized_chars)
    
    # Step 8: Final cleanup - remove any remaining combining marks
    result = ''.join(char for char in result
                     if unicodedata.category(char) != 'Mn')
    
    return result


# =============================================================================
# PARSER FUNCTION
# =============================================================================

def parse_ipa_sequence(ipa_string, strict=True, remove_excluded=True, normalize=True):
    """
    Parse IPA sequence and convert to phonetic class sequence.
    
    This function handles multi-character IPA sequences (affricates, diphthongs)
    and converts them to their corresponding phonetic class symbols.
    
    Args:
        ipa_string (str): IPA transcription to parse
        strict (bool): If True, raise error on unknown symbols. If False, skip them.
        remove_excluded (bool): If True, remove symbols in EXCLUDE_SYMBOLS first
        normalize (bool): If True, apply canonical normalization before parsing (default: True)
        
    Returns:
        str: Sequence of phonetic class symbols (e.g., "ənsə" for "ansa")
        
    Raises:
        ValueError: If unknown character encountered in strict mode
        
    Examples:
        >>> parse_ipa_sequence("hɛˈloʊ")
        'hələ'
        
        >>> parse_ipa_sequence("tʃɛɹ")
        'ʃəɹ'
        
        >>> parse_ipa_sequence("ˈkæt")
        'kət'
        
        >>> parse_ipa_sequence("ð ə")  # space-separated
        'θə'
    """
    if not ipa_string:
        return ""
    
    # Step 0: Apply canonical normalization if requested
    if normalize:
        ipa_string = normalize_ipa(ipa_string,
                                   remove_stress=remove_excluded,
                                   merge_spaces=True)
    else:
        # Legacy path: manual normalization
        # Step 0a: Handle special cases that would be broken by NFD normalization
        ipa_string = ipa_string.replace('ç', '\x00PALATAL_FRIC\x00')
        
        # Step 0b: Normalize Unicode to decomposed form (NFD)
        ipa_string = unicodedata.normalize('NFD', ipa_string)
        
        # Step 0c: Restore special cases
        ipa_string = ipa_string.replace('\x00PALATAL_FRIC\x00', 'ç')
        
        # Step 1: Remove excluded symbols if requested
        if remove_excluded:
            for symbol in EXCLUDE_SYMBOLS:
                ipa_string = ipa_string.replace(symbol, '')
    
    # Step 2: Parse sequence with greedy multi-char matching
    result = []
    pos = 0
    
    while pos < len(ipa_string):
        matched = False
        
        # Try multi-character sequences first (longest match)
        for seq in MULTI_CHAR_SEQUENCES:
            if ipa_string[pos:pos+len(seq)] == seq:
                # Check if in mapping
                if seq in IPA_TO_CLASS:
                    result.append(IPA_TO_CLASS[seq])
                    pos += len(seq)
                    matched = True
                    break
        
        # If no multi-char match, try single character
        if not matched:
            char = ipa_string[pos]
            
            if char in IPA_TO_CLASS:
                result.append(IPA_TO_CLASS[char])
                pos += 1
            elif char in EXCLUDE_SYMBOLS:
                # Excluded symbol not yet removed - skip
                pos += 1
            else:
                # Unknown symbol
                if strict:
                    raise ValueError(
                        f"Unknown IPA symbol at position {pos}: '{char}' (U+{ord(char):04X})\n"
                        f"Context: ...{ipa_string[max(0,pos-3):pos]}[{char}]{ipa_string[pos+1:min(len(ipa_string),pos+4)]}..."
                    )
                else:
                    # Skip unknown symbol in non-strict mode
                    pos += 1
    
    return ''.join(result)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_class_name(class_symbol):
    """Get human-readable name for class symbol."""
    class_names = {
        'ə': 'V (Vowels/Diphthongs/Glides)',
        's': 'SIB-ALV (Alveolar Sibilants)',
        'ʃ': 'SIB-POST (Postalveolar Sibilants)',
        'f': 'FR-LAB (Labiodental Fricatives)',
        'θ': 'FR-DENT (Dental Fricatives)',
        'h': 'FR-GLOT (Glottal & Other Fricatives)',
        'p': 'ST-LAB (Labial Stops)',
        't': 'ST-COR (Coronal Stops)',
        'k': 'ST-DOR (Dorsal Stops)',
        'n': 'NAS (Nasals)',
        'l': 'LAT (Laterals)',
        'ɹ': 'RHO (Rhotics)',
    }
    return class_names.get(class_symbol, f'Unknown ({class_symbol})')


def validate_ipa_string(ipa_string):
    """
    Validate IPA string and return analysis.
    
    Returns:
        dict with keys: valid, unknown_symbols, excluded_symbols
    """
    # Normalize first
    ipa_string = unicodedata.normalize('NFD', ipa_string)
    
    unknown = []
    excluded = []
    
    pos = 0
    while pos < len(ipa_string):
        matched = False
        
        # Check multi-char sequences
        for seq in MULTI_CHAR_SEQUENCES:
            if ipa_string[pos:pos+len(seq)] == seq:
                pos += len(seq)
                matched = True
                break
        
        if not matched:
            char = ipa_string[pos]
            if char in IPA_TO_CLASS:
                pos += 1
            elif char in EXCLUDE_SYMBOLS:
                excluded.append(char)
                pos += 1
            else:
                unknown.append(char)
                pos += 1
    
    return {
        'valid': len(unknown) == 0,
        'unknown_symbols': list(set(unknown)),
        'excluded_symbols': list(set(excluded))
    }


# =============================================================================
# UTILITY FUNCTIONS FOR NORMALIZATION
# =============================================================================

def get_normalization_stats(original: str, normalized: str) -> Dict[str, any]:
    """
    Get statistics about what changed during normalization.
    
    Returns dict with:
        - changed: bool
        - original_length: int
        - normalized_length: int
        - diff: str
    """
    return {
        'changed': original != normalized,
        'original_length': len(original),
        'normalized_length': len(normalized),
        'diff': f"{original} → {normalized}" if original != normalized else "unchanged"
    }


def normalize_ipa_word_list(word_ipa_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Normalize a list of (word, ipa) tuples.
    
    Args:
        word_ipa_pairs: List of (word, ipa) tuples
        
    Returns:
        List of (word, normalized_ipa) tuples with duplicates removed
    """
    normalized = []
    seen = set()
    
    for word, ipa in word_ipa_pairs:
        norm_ipa = normalize_ipa(ipa)
        pair = (word, norm_ipa)
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    
    return normalized


# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "3.0.0"
__all__ = [
    'IPA_TO_CLASS',
    'EXCLUDE_SYMBOLS',
    'MULTI_CHAR_SEQUENCES',
    'IPA_NORMALIZATION_MAP',
    'AFFRICATE_NORMALIZATION',
    'normalize_ipa',
    'parse_ipa_sequence',
    'get_class_name',
    'validate_ipa_string',
    'get_normalization_stats',
    'normalize_ipa_word_list',
]
