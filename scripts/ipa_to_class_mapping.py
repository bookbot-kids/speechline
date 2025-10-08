"""IPA Phoneme to Phonetic Class Mapping and Parser

This module provides mapping from IPA symbols to robust phonetic classes
and a parser function to convert IPA sequences to class sequences.

Based on the robust phonetic classification system from manner_of_articulation.csv

Statistics (corrected mapping):
  Total base symbols: ~50+
  Excluded symbols: ~90+ (diacritics, stress, tone, etc.)
  Click/trill/implosive mappings: Added to main mapping
"""

import unicodedata

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

# No longer using FUTURE_REMAP - all symbols now mapped to nearest English equivalents

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
    'ɚ': 'ə',
    'ɛ': 'ə',
    'ɜ': 'ə',
    'ɝ': 'ə',
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
    # Examples: (f)ine, (v)oice, (ph)one
    # -------------------------------------------------------------------------
    'f': 'f',
    'v': 'f',
    'ɸ': 'f',  # bilabial fricative
    'β': 'f',  # bilabial fricative
    
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
    # GLD - Glides → j
    # Includes palatal and labial-velar approximants
    # Examples: (y)es, (w)e, h(u)ge, (wh)ich
    # -------------------------------------------------------------------------
    'j': 'j',
    'y': 'j',  # Latin y used for palatal approximant (alternate notation)
    'w': 'j',
    'ɥ': 'j',  # labial-palatal
    'ʋ': 'j',  # labiodental
    'ʍ': 'j',  # voiceless labial-velar
}

# =============================================================================
# PARSER FUNCTION
# =============================================================================

def parse_ipa_sequence(ipa_string, strict=True, remove_excluded=True):
    """
    Parse IPA sequence and convert to phonetic class sequence.
    
    This function handles multi-character IPA sequences (affricates, diphthongs)
    and converts them to their corresponding phonetic class symbols.
    
    Args:
        ipa_string (str): IPA transcription to parse
        strict (bool): If True, raise error on unknown symbols. If False, skip them.
        remove_excluded (bool): If True, remove symbols in EXCLUDE_SYMBOLS first
        
    Returns:
        str: Sequence of phonetic class symbols (e.g., "ənsə" for "ansa")
        
    Raises:
        ValueError: If unknown character encountered in strict mode
        
    Examples:
        >>> parse_ipa_sequence("hɛˈloʊ")
        'hələ'
        
        >>> parse_ipa_sequence("tʃɛɹ")
        'ʃɛɹ'
        
        >>> parse_ipa_sequence("ˈkæt")  # stress mark removed
        'kæt' -> 'kət'
    """
    if not ipa_string:
        return ""
    
    # Step 0a: Handle special cases that would be broken by NFD normalization
    # ç (U+00E7) would decompose to c+̧, but c→k (stop) not h (fricative)
    # So we need to replace it before NFD normalization
    ipa_string = ipa_string.replace('ç', '\x00PALATAL_FRIC\x00')  # temporary placeholder
    
    # Step 0b: Normalize Unicode to decomposed form (NFD)
    # This handles precomposed characters like ĩ (U+0129) → i + combining tilde
    ipa_string = unicodedata.normalize('NFD', ipa_string)
    
    # Step 0c: Restore special cases and map them
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
        'ə': 'V (Vowels/Diphthongs)',
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
        'j': 'GLD (Glides)',
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
# MODULE METADATA
# =============================================================================

__version__ = "2.1.0"
__all__ = [
    'IPA_TO_CLASS',
    'EXCLUDE_SYMBOLS',
    'MULTI_CHAR_SEQUENCES',
    'parse_ipa_sequence',
    'get_class_name',
    'validate_ipa_string',
]
