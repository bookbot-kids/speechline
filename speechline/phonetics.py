"""
Unified Phonetics Library

This library provides normalization and conversion between three phonetic representation systems:

1. **IPA (International Phonetic Alphabet)**
   - Standard: Unicode IPA symbols
   - Example: "hɛloʊ" (hello), "ðə" (the), "kæt" (cat)
   - Variants supported: Space-separated ("ð ə"), tie-bar affricates ("t͡ʃ"), stress marks ("ˈhɛloʊ")
   
2. **ARPA (ARPABET - American English phoneset)**
   - Standard: ASCII phoneme codes with stress markers (0=unstressed, 1=primary, 2=secondary)
   - Example: "HH AH0 L OW1" (hello), "DH AH0" (the), "K AE1 T" (cat)
   - 39 phonemes (15 vowels + 24 consonants) + stress variants
   
3. **Artemes (Articulatory phonetic classes)**
   - Standard: 12-class phonetic system based on manner and place of articulation
   - Example: "hələə" (hello), "θə" (the), "kət" (cat)
   - Classes: ə(vowels), s(alv-sib), ʃ(post-sib), f(labio), θ(dental), h(glottal),
             p(labial-stop), t(coronal-stop), k(dorsal-stop), n(nasal), l(lateral), ɹ(rhotic)
   - One-way conversion only (IPA/ARPA → Artemes, not reversible)

## Complete Character Sets

### ARPA (39 Phonemes + Stress)

**Vowels (15):**
    AA  - father   (ɑ)      AE  - cat      (æ)      AH  - cut      (ʌ)
    AO  - caught   (ɔ)      AW  - how      (aʊ)     AY  - hide     (aɪ)
    EH  - red      (ɛ)      ER  - her      (ɝ)      EY  - say      (eɪ)
    IH  - it       (ɪ)      IY  - eat      (i)      OW  - go       (oʊ)
    OY  - boy      (ɔɪ)     UH  - book     (ʊ)      UW  - too      (u)

**Consonants (24):**
    B   - bee      (b)      CH  - cheese   (tʃ)     D   - day      (d)
    DH  - this     (ð)      F   - fee      (f)      G   - green    (g)
    HH  - he       (h)      JH  - gee      (dʒ)     K   - key      (k)
    L   - lee      (l)      M   - me       (m)      N   - knee     (n)
    NG  - ping     (ŋ)      P   - pea      (p)      R   - read     (ɹ)
    S   - sea      (s)      SH  - she      (ʃ)      T   - tea      (t)
    TH  - theta    (θ)      V   - vee      (v)      W   - we       (w)
    Y   - yield    (j)      Z   - zee      (z)      ZH  - seizure  (ʒ)

**Stress Markers:** 0 (unstressed), 1 (primary), 2 (secondary)

### IPA (100+ Symbols)

**Core Vowels:**
    a, e, i, o, u          - Basic vowels
    æ (ash)                - cat
    ɑ (script a)           - father
    ɔ (open o)             - caught
    ə (schwa)              - about
    ɛ (epsilon)            - red
    ɪ (small cap i)        - it
    ʊ (upsilon)            - book
    ʌ (turned v)           - cut
    ɝ (r-colored schwa)    - her

**Diphthongs:** aɪ (hide), aʊ (how), eɪ (say), oʊ (go), ɔɪ (boy)

**Common Consonants:**
    p, b, t, d, k, g       - Stops
    f, v, θ, ð, s, z       - Fricatives
    ʃ (esh), ʒ (ezh)       - Postalveolar fricatives
    tʃ, dʒ                 - Affricates
    m, n, ŋ (eng)          - Nasals
    l, ɹ (turned r)        - Liquids
    j, w, h                - Glides

**Stress Markers (excluded):** ˈ (primary), ˌ (secondary), ː (length)

### Artemes (12 Classes)

    ə - Vowels/Glides       (all vowel sounds, w, j, y)
    s - Alveolar Sibilants  (s, z, ts, dz)
    ʃ - Postalv. Sibilants  (sh, zh, ch, j → ʃ, ʒ, tʃ, dʒ)
    f - Labiodental Fric.   (f, v)
    θ - Dental Fricatives   (th → θ, ð)
    h - Glottal Fricatives  (h and other fricatives)
    p - Labial Stops        (p, b)
    t - Coronal Stops       (t, d)
    k - Dorsal Stops        (k, g)
    n - Nasals              (m, n, ng → ŋ)
    l - Laterals            (l variants)
    ɹ - Rhotics             (r sounds)

## Comparison Table

| Word   | ARPA              | IPA      | Artemes |
|--------|-------------------|----------|---------|
| cat    | K AE1 T           | kæt      | kət     |
| hello  | HH EH L OW        | hɛloʊ    | hələə   |
| the    | DH AH0            | ðə       | θə      |
| chair  | CH EH R           | tʃɛɹ     | ʃəɹ     |
| read   | R IY D            | ɹid      | ɹət     |
| sing   | S IH NG           | sɪŋ      | sən     |

## Conversion Flow

```
Input (IPA/ARPA)
    ↓
normalize_ipa() ←── arpa_to_ipa()
    ↓
Canonical IPA (central hub)
    ↓
├→ ARPA (ipa_to_arpa)
└→ Artemes (ipa_to_artemes)
```

## Usage Examples

```python
from speechline.phonetics import (
    normalize_ipa,
    ipa_to_arpa,
    arpa_to_ipa,
    ipa_to_artemes,
    arpa_to_artemes
)

# Normalize IPA variants
canonical = normalize_ipa("ˈhɛˌloʊ")  # → "hɛloʊ"
canonical = normalize_ipa("ð ə")       # → "ðə"
canonical = normalize_ipa("t͡ʃ")       # → "tʃ"

# Convert between IPA and ARPA
arpa = ipa_to_arpa("hɛloʊ")           # → "HH EH L OW"
ipa = arpa_to_ipa("K AE T")            # → "kæt"

# Convert to artemes
artemes = ipa_to_artemes("hɛloʊ")     # → "hələə"
artemes = arpa_to_artemes("K AE T")   # → "kət"
```

Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)
Licensed under the Apache License, Version 2.0
"""

import unicodedata
from typing import Dict, List, Tuple, Set

__version__ = "1.0.0"

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
    '–', '…', '‿', '²', '³', '¹', '⁴', '⁵', '⁻', '︎', "'",
    
    # Common diacritics (combining characters)
    '̃',   # nasalization
    '̚',   # unreleased
    'ʰ',   # aspiration
    'ʷ',   # labialization
    'ʲ',   # palatalization
    '̩',   # syllabic
    'ʱ',   # breathy voice
    '̀', '́', '̂', '̄', '̆', '̈', '̊', '̍',
    '̙', '̝', '̞', '̟', '̠', '̥', '̪', '̬', '̯', '̰', '̹', '̺', '̽',
    '̧',   # cedilla
    '͆', '͇', '͜', '͡',  # tie bars
    
    # Superscripts and other modifiers (excluding pre-nasalized markers)
    'ʳ', 'ʴ', 'ʼ', 'ˀ', '˔', '˞', 'ˠ', 'ˡ', 'ˤ', '˭',
    'ᵈ', 'ᵊ', 'ᵒ', 'ᶴ', '᷈',
    
    # Spaces and separators
    ' ', '-', '~', '',
}

# =============================================================================
# CANONICAL IPA NORMALIZATION MAP
# =============================================================================

# Map alternative IPA representations to canonical forms
IPA_NORMALIZATION_MAP = {
    # Vowel normalizations
    'ɨ': 'ɪ',      # Close central → close front
    'ɘ': 'ə',      # Close-mid central → mid central
    'ɐ': 'ə',      # Near-open central → mid central
    'ʉ': 'u',      # Close central rounded → close back rounded
    
    # Nasalized vowels → base vowels (nasalization stripped)
    'ã': 'a',      # Nasalized a
    'ẽ': 'e',      # Nasalized e
    'ĩ': 'i',      # Nasalized i
    'õ': 'o',      # Nasalized o
    'ũ': 'u',      # Nasalized u
    
    # Vowels with diacritics → base vowels
    'ä': 'a',      # a with diaeresis
    'é': 'e',      # e with acute
    'ĕ': 'e',      # e with breve
    'ū': 'u',      # u with macron
    
    # Consonant normalizations
    'ɡ': 'g',      # G with tail → standard g
    'r': 'ɹ',      # Trill/tap → approximant (English uses approximant)
    'ɾ': 'ɹ',      # Tap/flap → approximant
    
    # Common transcription variants
    'y': 'j',      # Latin y → IPA j for palatal approximant
}

# Affricate normalization: all tie-bar forms to simple sequences
AFFRICATE_NORMALIZATION = {
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
    # Tie-bar affricates (3 characters)
    't͡ʃ', 'd͡ʒ', 't͡s', 'd͡z',
    't͜ʃ', 'd͜ʒ', 't͜s', 'd͜z',
    
    # Simple affricates (2 characters)
    'tʃ', 'dʒ', 'ts', 'dz',
    
    # Diphthongs (2 characters)
    'aɪ', 'eɪ', 'oʊ', 'aʊ', 'ɔɪ',
    
    # Syllabic consonants
    'l̩', 'r̩', 'ɹ̩', 'm̩', 'n̩', 'ŋ̩',
]

# =============================================================================
# IPA TO ARTEME MAPPING
# =============================================================================

IPA_TO_ARTEME = {
    # Vowels/Diphthongs/Glides → ə
    'a': 'ə', 'e': 'ə', 'i': 'ə', 'o': 'ə', 'u': 'ə',
    'æ': 'ə', 'œ': 'ə', 'ɐ': 'ə', 'ɑ': 'ə', 'ɒ': 'ə', 'ɔ': 'ə',
    'ɘ': 'ə', 'ə': 'ə', 'ɚ': 'əɹ', 'ɛ': 'ə', 'ɜ': 'ə', 'ɝ': 'əɹ',
    'ɞ': 'ə', 'ɤ': 'ə', 'ɨ': 'ə', 'ɪ': 'ə', 'ɯ': 'ə', 'ɵ': 'ə',
    'ɶ': 'ə', 'ʉ': 'ə', 'ʊ': 'ə', 'ʌ': 'ə', 'ʏ': 'ə',
    
    # Nasalized vowels (Spanish, Portuguese) → ə
    'ã': 'ə', 'ẽ': 'ə', 'ĩ': 'ə', 'õ': 'ə', 'ũ': 'ə',
    
    # Vowels with diacritics (Indonesian, etc.) → ə
    'ä': 'ə', 'é': 'ə', 'ĕ': 'ə', 'ū': 'ə',
    
    # Diphthongs
    'aɪ': 'əə', 'aʊ': 'əə', 'eɪ': 'əə', 'oʊ': 'əə', 'ɔɪ': 'əə',
    
    # Glides
    'j': 'ə', 'y': 'ə', 'w': 'ə', 'ɥ': 'ə', 'ʍ': 'ə',
    
    # Alveolar Sibilants → s
    's': 's', 'z': 's',
    'ts': 's', 'dz': 's', 't͡s': 's', 'd͡z': 's', 't͜s': 's', 'd͜z': 's',
    
    # Postalveolar Sibilants → ʃ
    'ʃ': 'ʃ', 'ʒ': 'ʃ', 'ɕ': 'ʃ', 'ʑ': 'ʃ',
    'tʃ': 'ʃ', 'dʒ': 'ʃ', 't͡ʃ': 'ʃ', 'd͡ʒ': 'ʃ', 't͜ʃ': 'ʃ', 'd͜ʒ': 'ʃ',
    
    # Labiodental Fricatives → f
    'f': 'f', 'v': 'f', 'ɸ': 'f', 'β': 'f', 'ʋ': 'f',
    
    # Dental Fricatives → θ
    'θ': 'θ', 'ð': 'θ',
    
    # Glottal & Other Fricatives → h
    'h': 'h', 'ɦ': 'h', 'x': 'h', 'ɣ': 'h', 'χ': 'h',
    'ʕ': 'h', 'ħ': 'h', 'ç': 'h', 'ʝ': 'h',
    'ʔ': 'h',  # Glottal stop (Indonesian, etc.) → h
    
    # Labial Stops → p
    'p': 'p', 'b': 'p', 'ʘ': 'p', 'ɓ': 'p', 'ʙ': 'p',
    
    # Coronal Stops → t
    't': 't', 'd': 't', 'ɾ': 't', 'ɖ': 't', 'ʈ': 't', 'ɽ': 't',
    'ǀ': 't', 'ǃ': 't', 'ǂ': 't', 'ɗ': 't', 'ʄ': 't',
    
    # Dorsal Stops → k
    'k': 'k', 'g': 'k', 'ɡ': 'k', 'q': 'k', 'ɢ': 'k', 'c': 'k', 'ɟ': 'k',
    'ɠ': 'k',  # Voiced velar implosive (Swahili) → k
    
    # Nasals → n
    'm': 'n', 'n': 'n', 'ŋ': 'n', 'ɱ': 'n', 'ɲ': 'n', 'ɳ': 'n', 'ɴ': 'n',
    
    # Pre-nasalized consonant markers (Swahili, etc.) → n
    'ᵐ': 'n',  # Superscript m (prenasalized bilabial)
    'ⁿ': 'n',  # Superscript n (prenasalized alveolar)
    'ᵑ': 'n',  # Superscript ŋ (prenasalized velar)
    
    # Laterals → l
    'l': 'l', 'ɫ': 'l', 'ɬ': 'l', 'ɭ': 'l', 'ɺ': 'l', 'ʎ': 'l', 'ʟ': 'l', 'ǁ': 'l', 'l̩': 'l',
    
    # Rhotics → ɹ
    'ɹ': 'ɹ', 'r': 'ɹ', 'ɻ': 'ɹ', 'ʀ': 'ɹ', 'ʁ': 'ɹ', 'r̩': 'ɹ', 'ɹ̩': 'ɹ',
}

# =============================================================================
# ARPA TO IPA MAPPING (Canonical)
# =============================================================================

# Simplified ARPA to IPA mapping (base phonemes, ignoring stress)
ARPA_TO_IPA = {
    # Vowels
    'AA': 'ɑ',    # father
    'AE': 'æ',    # cat
    'AH': 'ʌ',    # cut
    'AO': 'ɔ',    # caught
    'AW': 'aʊ',   # how
    'AY': 'aɪ',   # hide
    'EH': 'ɛ',    # red
    'ER': 'ɝ',    # her
    'EY': 'eɪ',   # say
    'IH': 'ɪ',    # it
    'IY': 'i',    # eat
    'OW': 'oʊ',   # go
    'OY': 'ɔɪ',   # boy
    'UH': 'ʊ',    # book
    'UW': 'u',    # too
    
    # Consonants
    'B': 'b',     # bee
    'CH': 'tʃ',   # cheese
    'D': 'd',     # day
    'DH': 'ð',    # this
    'F': 'f',     # fee
    'G': 'g',     # green
    'HH': 'h',    # he
    'JH': 'dʒ',   # gee
    'K': 'k',     # key
    'L': 'l',     # lee
    'M': 'm',     # me
    'N': 'n',     # knee
    'NG': 'ŋ',    # ping
    'P': 'p',     # pea
    'R': 'ɹ',     # read
    'S': 's',     # sea
    'SH': 'ʃ',    # she
    'T': 't',     # tea
    'TH': 'θ',    # theta
    'V': 'v',     # vee
    'W': 'w',     # we
    'Y': 'j',     # yield
    'Z': 'z',     # zee
    'ZH': 'ʒ',    # seizure
    
    # Special
    'spn': '',    # spoken noise
}

# Generate reverse mapping (IPA to ARPA)
IPA_TO_ARPA = {v: k for k, v in ARPA_TO_IPA.items() if v}

# =============================================================================
# NORMALIZATION FUNCTION
# =============================================================================

def normalize_ipa(ipa_string: str,
                  remove_stress: bool = True,
                  merge_spaces: bool = True) -> str:
    """
    Normalize IPA string to canonical form.
    
    Handles variant IPA transcription styles to ensure compatibility
    between different lexicons and datasets.
    
    Normalization steps:
    1. Unicode NFD decomposition
    2. Normalize affricates (tie-bar forms to simple)
    3. Remove stress markers and non-phonemic annotations
    4. Merge space-separated phonemes
    5. Apply character-level normalization map
    6. Remove remaining combining marks
    
    Args:
        ipa_string: Input IPA transcription
        remove_stress: If True, remove stress markers (ˈˌ)
        merge_spaces: If True, convert space-separated to continuous
        
    Returns:
        Normalized IPA string ready for conversion
        
    Examples:
        >>> normalize_ipa("ˈhɛˌloʊ")
        'hɛloʊ'
        >>> normalize_ipa("ð ə")
        'ðə'
        >>> normalize_ipa("t͡ʃ")
        'tʃ'
    """
    if not ipa_string:
        return ""
    
    # Handle special cases before NFD (ç would decompose incorrectly)
    result = ipa_string.replace('ç', '\x00PALATAL_FRIC\x00')
    
    # Unicode normalization (NFD decomposition)
    result = unicodedata.normalize('NFD', result)
    
    # Restore special cases
    result = result.replace('\x00PALATAL_FRIC\x00', 'ç')
    
    # Normalize affricates (longest first)
    for affricate, canonical in sorted(AFFRICATE_NORMALIZATION.items(),
                                       key=lambda x: len(x[0]),
                                       reverse=True):
        result = result.replace(affricate, canonical)
    
    # Remove excluded symbols
    if remove_stress:
        for symbol in EXCLUDE_SYMBOLS:
            result = result.replace(symbol, '')
    
    # Merge spaces if requested
    if merge_spaces:
        result = result.replace(' ', '')
    
    # Apply normalization map
    normalized_chars = []
    i = 0
    while i < len(result):
        # Try multi-character sequences first
        matched = False
        for length in [3, 2]:
            if i + length <= len(result):
                seq = result[i:i+length]
                if seq in MULTI_CHAR_SEQUENCES or seq in IPA_TO_ARTEME:
                    normalized_chars.append(seq)
                    i += length
                    matched = True
                    break
        
        if not matched:
            # Single character - apply normalization
            char = result[i]
            if char in IPA_NORMALIZATION_MAP:
                replacement = IPA_NORMALIZATION_MAP[char]
                if replacement:
                    normalized_chars.append(replacement)
            else:
                normalized_chars.append(char)
            i += 1
    
    result = ''.join(normalized_chars)
    
    # Final cleanup - remove remaining combining marks
    result = ''.join(char for char in result
                     if unicodedata.category(char) != 'Mn')
    
    return result

# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def ipa_to_artemes(ipa_string: str, strict: bool = False) -> str:
    """
    Convert IPA to arteme (phonetic class) sequence.
    
    Args:
        ipa_string: IPA transcription (will be normalized first)
        strict: If True, raise error on unknown symbols
        
    Returns:
        Arteme sequence (e.g., "hələ" for "hello")
        
    Examples:
        >>> ipa_to_artemes("hɛloʊ")
        'hələ'
        >>> ipa_to_artemes("tʃɛɹ")
        'ʃəɹ'
    """
    # Normalize first
    ipa_string = normalize_ipa(ipa_string)
    
    if not ipa_string:
        return ""
    
    result = []
    pos = 0
    
    while pos < len(ipa_string):
        matched = False
        
        # Try multi-character sequences first
        for seq in MULTI_CHAR_SEQUENCES:
            if ipa_string[pos:pos+len(seq)] == seq:
                if seq in IPA_TO_ARTEME:
                    result.append(IPA_TO_ARTEME[seq])
                    pos += len(seq)
                    matched = True
                    break
        
        if not matched:
            char = ipa_string[pos]
            if char in IPA_TO_ARTEME:
                result.append(IPA_TO_ARTEME[char])
                pos += 1
            elif char in EXCLUDE_SYMBOLS:
                pos += 1
            else:
                if strict:
                    raise ValueError(f"Unknown IPA symbol: '{char}' at position {pos}")
                pos += 1
    
    return ''.join(result)

def arpa_to_ipa(arpa_string: str) -> str:
    """
    Convert ARPA phoneme sequence to canonical IPA.
    
    Args:
        arpa_string: Space-separated ARPA phonemes (e.g., "K AE1 T")
        
    Returns:
        IPA transcription (e.g., "kæt")
        
    Examples:
        >>> arpa_to_ipa("K AE1 T")
        'kæt'
        >>> arpa_to_ipa("HH EH L OW")
        'hɛloʊ'
    """
    if not arpa_string:
        return ""
    
    # Split into phonemes
    phonemes = arpa_string.split()
    
    result = []
    for phoneme in phonemes:
        # Strip stress markers (0, 1, 2)
        base_phoneme = phoneme.rstrip('012')
        
        if base_phoneme in ARPA_TO_IPA:
            result.append(ARPA_TO_IPA[base_phoneme])
        elif phoneme == '':
            continue
        else:
            # Unknown ARPA phoneme - keep as is
            result.append(phoneme)
    
    return ''.join(result)

def ipa_to_arpa(ipa_string: str) -> str:
    """
    Convert canonical IPA to ARPA phoneme sequence.
    
    Note: This is a best-effort conversion. Some IPA sequences may not
    have exact ARPA equivalents. Stress information is lost.
    
    Args:
        ipa_string: IPA transcription (will be normalized first)
        
    Returns:
        Space-separated ARPA phonemes (e.g., "K AE T")
        
    Examples:
        >>> ipa_to_arpa("kæt")
        'K AE T'
        >>> ipa_to_arpa("hɛloʊ")
        'HH EH L OW'
    """
    # Normalize first
    ipa_string = normalize_ipa(ipa_string)
    
    if not ipa_string:
        return ""
    
    result = []
    pos = 0
    
    while pos < len(ipa_string):
        matched = False
        
        # Try multi-character sequences first
        for length in [3, 2]:
            if pos + length <= len(ipa_string):
                seq = ipa_string[pos:pos+length]
                if seq in IPA_TO_ARPA:
                    result.append(IPA_TO_ARPA[seq])
                    pos += length
                    matched = True
                    break
        
        if not matched:
            char = ipa_string[pos]
            if char in IPA_TO_ARPA:
                result.append(IPA_TO_ARPA[char])
            pos += 1
    
    return ' '.join(result)

def arpa_to_artemes(arpa_string: str) -> str:
    """
    Convert ARPA to artemes via IPA.
    
    Args:
        arpa_string: Space-separated ARPA phonemes
        
    Returns:
        Arteme sequence
        
    Examples:
        >>> arpa_to_artemes("K AE T")
        'kət'
    """
    ipa = arpa_to_ipa(arpa_string)
    return ipa_to_artemes(ipa)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_ipa(ipa_string: str) -> Dict[str, any]:
    """
    Validate IPA string and return analysis.
    
    Args:
        ipa_string: IPA transcription to validate
        
    Returns:
        Dict with keys: valid, unknown_symbols, excluded_symbols
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
            if char in IPA_TO_ARTEME:
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
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Normalization
    'normalize_ipa',
    
    # Conversions
    'ipa_to_artemes',
    'ipa_to_arpa',
    'arpa_to_ipa',
    'arpa_to_artemes',
    
    # Validation
    'validate_ipa',
    
    # Constants
    'IPA_TO_ARTEME',
    'ARPA_TO_IPA',
    'IPA_TO_ARPA',
    'IPA_NORMALIZATION_MAP',
    'AFFRICATE_NORMALIZATION',
    'EXCLUDE_SYMBOLS',
    'MULTI_CHAR_SEQUENCES',
]