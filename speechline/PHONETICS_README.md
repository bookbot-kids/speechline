# Phonetics Library

Unified phonetics library for normalization and conversion between IPA, ARPA, and Arteme phonetic representation systems.

## Overview

This library provides a comprehensive toolkit for working with phonetic transcriptions across three different representation systems:

- **IPA (International Phonetic Alphabet)**: Unicode-based international standard
- **ARPA (ARPABET)**: ASCII-based American English phoneset
- **Artemes**: 12-class articulatory phonetic system

All conversions flow through canonical IPA as the central hub, ensuring consistency and accuracy.

## Installation

```python
from speechline.phonetics import (
    normalize_ipa,
    ipa_to_arpa,
    arpa_to_ipa,
    ipa_to_artemes,
    arpa_to_artemes,
    validate_ipa
)
```

## Quick Start

### Normalize IPA

```python
# Remove stress markers
canonical = normalize_ipa("ˈhɛˌloʊ")  # → "hɛloʊ"

# Merge space-separated phonemes
canonical = normalize_ipa("ð ə")       # → "ðə"

# Normalize affricate tie-bars
canonical = normalize_ipa("t͡ʃ")       # → "tʃ"

# Normalize aspiration and diacritics
canonical = normalize_ipa("kʰæt")     # → "kæt"

# Normalize character variants
canonical = normalize_ipa("ɡ")        # → "g"
canonical = normalize_ipa("r")        # → "ɹ"
```

### Convert Between IPA and ARPA

```python
# IPA to ARPA
arpa = ipa_to_arpa("hɛloʊ")           # → "HH EH L OW"
arpa = ipa_to_arpa("kæt")             # → "K AE T"
arpa = ipa_to_arpa("tʃɛɹ")            # → "CH EH R"

# ARPA to IPA (strips stress markers)
ipa = arpa_to_ipa("K AE1 T")          # → "kæt"
ipa = arpa_to_ipa("HH EH L OW")       # → "hɛloʊ"
ipa = arpa_to_ipa("DH AH0")           # → "ðə"
```

### Convert to Artemes

```python
# IPA to Artemes
artemes = ipa_to_artemes("hɛloʊ")     # → "hələə"
artemes = ipa_to_artemes("kæt")       # → "kət"
artemes = ipa_to_artemes("sɪŋ")       # → "sən"

# ARPA to Artemes (via IPA)
artemes = arpa_to_artemes("K AE1 T")  # → "kət"
artemes = arpa_to_artemes("S IH NG")  # → "sən"
```

### Validate IPA

```python
result = validate_ipa("kæt")
# {'valid': True, 'unknown_symbols': [], 'excluded_symbols': []}

result = validate_ipa("ˈkæt")
# {'valid': True, 'unknown_symbols': [], 'excluded_symbols': ['ˈ']}

result = validate_ipa("xyz123")
# {'valid': False, 'unknown_symbols': ['x', 'y', 'z', '1', '2', '3'], 'excluded_symbols': []}
```

## Character Sets

### ARPA (39 Phonemes)

#### Vowels (15)
| ARPA | IPA | Example | Word |
|------|-----|---------|------|
| AA | ɑ | AA | f**a**ther |
| AE | æ | AE | c**a**t |
| AH | ʌ | AH | c**u**t |
| AO | ɔ | AO | c**au**ght |
| AW | aʊ | AW | h**ow** |
| AY | aɪ | AY | h**i**de |
| EH | ɛ | EH | r**e**d |
| ER | ɝ | ER | h**er** |
| EY | eɪ | EY | s**ay** |
| IH | ɪ | IH | **i**t |
| IY | i | IY | **ea**t |
| OW | oʊ | OW | g**o** |
| OY | ɔɪ | OY | b**oy** |
| UH | ʊ | UH | b**oo**k |
| UW | u | UW | t**oo** |

#### Consonants (24)
| ARPA | IPA | Example | Word |
|------|-----|---------|------|
| B | b | B | **b**ee |
| CH | tʃ | CH | **ch**eese |
| D | d | D | **d**ay |
| DH | ð | DH | **th**is |
| F | f | F | **f**ee |
| G | g | G | **g**reen |
| HH | h | HH | **h**e |
| JH | dʒ | JH | **g**ee |
| K | k | K | **k**ey |
| L | l | L | **l**ee |
| M | m | M | **m**e |
| N | n | N | k**n**ee |
| NG | ŋ | NG | pi**ng** |
| P | p | P | **p**ea |
| R | ɹ | R | **r**ead |
| S | s | S | **s**ea |
| SH | ʃ | SH | **sh**e |
| T | t | T | **t**ea |
| TH | θ | TH | **th**eta |
| V | v | V | **v**ee |
| W | w | W | **w**e |
| Y | j | Y | **y**ield |
| Z | z | Z | **z**ee |
| ZH | ʒ | ZH | sei**z**ure |

#### Stress Markers
- **0**: Unstressed (e.g., AH0)
- **1**: Primary stress (e.g., AE1)
- **2**: Secondary stress (e.g., AH2)

### IPA (100+ Symbols)

#### Basic Vowels
```
a, e, i, o, u          Simple vowels
æ (ash)                cat
ɑ (script a)           father
ɔ (open o)             caught
ə (schwa)              about
ɛ (epsilon)            red
ɪ (small cap i)        it
ʊ (upsilon)            book
ʌ (turned v)           cut
ɝ (r-colored schwa)    her
```

#### Diphthongs
```
aɪ    hide
aʊ    how
eɪ    say
oʊ    go
ɔɪ    boy
```

#### Consonants by Category

**Stops**
```
p, b    Labial (pea, bee)
t, d    Coronal (tea, day)
k, g    Dorsal (key, go)
```

**Fricatives**
```
f, v      Labiodental (fee, vee)
θ, ð      Dental (theta, this)
s, z      Alveolar (sea, zee)
ʃ, ʒ      Postalveolar (she, seizure)
h         Glottal (he)
```

**Affricates**
```
tʃ, dʒ    Postalveolar (cheese, gee)
```

**Nasals**
```
m         Labial (me)
n         Alveolar (knee)
ŋ         Velar (sing)
```

**Liquids**
```
l         Lateral (lee)
ɹ         Rhotic (read)
```

**Glides**
```
j         Palatal (yield)
w         Labiovelar (we)
```

### Artemes (12 Classes)

| Symbol | Class | IPA Symbols | Examples |
|--------|-------|-------------|----------|
| ə | Vowels/Glides | All vowels, w, j, y | a, e, i, o, u, æ, ɛ, ɪ, ʊ, ʌ, ə, w, j |
| s | Alveolar Sibilants | s, z, ts, dz | **s**ea, **z**ee |
| ʃ | Postalveolar Sibilants | ʃ, ʒ, tʃ, dʒ | **sh**e, sei**z**ure, **ch**eese |
| f | Labiodental Fricatives | f, v | **f**ee, **v**ee |
| θ | Dental Fricatives | θ, ð | **th**eta, **th**is |
| h | Glottal Fricatives | h, x, ɣ, χ, ʕ, ħ, ç | **h**e |
| p | Labial Stops | p, b | **p**ea, **b**ee |
| t | Coronal Stops | t, d | **t**ea, **d**ay |
| k | Dorsal Stops | k, g | **k**ey, **g**o |
| n | Nasals | m, n, ŋ | **m**e, k**n**ee, si**ng** |
| l | Laterals | l, ɫ, ɬ, ɭ | **l**ee |
| ɹ | Rhotics | r, ɹ, ɻ, ʀ, ʁ | **r**ead |

## Conversion Examples

### Common Words

| Word | ARPA | IPA | Artemes |
|------|------|-----|---------|
| cat | `K AE1 T` | `kæt` | `kət` |
| dog | `D AO1 G` | `dɔg` | `tək` |
| hello | `HH EH L OW` | `hɛloʊ` | `hələə` |
| world | `W ER L D` | `wɝld` | `əɹlt` |
| the | `DH AH0` | `ðə` | `θə` |
| chair | `CH EH R` | `tʃɛɹ` | `ʃəɹ` |
| sing | `S IH NG` | `sɪŋ` | `sən` |
| read | `R IY D` | `ɹid` | `ɹət` |
| house | `HH AW S` | `haʊs` | `həəs` |
| boy | `B OY` | `bɔɪ` | `pəə` |

### Diphthongs

| Diphthong | ARPA | IPA | Artemes | Example |
|-----------|------|-----|---------|---------|
| Long A | `EY` | `eɪ` | `əə` | s**ay** |
| Long I | `AY` | `aɪ` | `əə` | h**i**de |
| Long O | `OW` | `oʊ` | `əə` | g**o** |
| OW sound | `AW` | `aʊ` | `əə` | h**ow** |
| OY sound | `OY` | `ɔɪ` | `əə` | b**oy** |

### Multi-Character Sequences

| Sequence | Type | ARPA | IPA | Artemes |
|----------|------|------|-----|---------|
| CH | Affricate | `CH` | `tʃ` | `ʃ` |
| JH | Affricate | `JH` | `dʒ` | `ʃ` |
| SH | Fricative | `SH` | `ʃ` | `ʃ` |
| TH (voiceless) | Fricative | `TH` | `θ` | `θ` |
| TH (voiced) | Fricative | `DH` | `ð` | `θ` |
| NG | Nasal | `NG` | `ŋ` | `n` |
| ZH | Fricative | `ZH` | `ʒ` | `ʃ` |

## Multilingual Support

The phonetics library has been extended to support Spanish, Indonesian (Bahasa Indonesia), and Swahili phonetic systems, in addition to American English. The library handles language-specific phonemes, diacritics, and phonetic features unique to each language.

### Coverage Statistics

| Language | Unique Symbols | Mapped | Coverage |
|----------|----------------|--------|----------|
| **Spanish** | 18 | 15 | 83.3% |
| **Indonesian** | 11 | 9 | 81.8% |
| **Swahili** | 66 | 57 | 86.3% |
| **Overall** | 95 | 76 | 80.0% |

*Remaining unmapped symbols are primarily diacritics that are properly handled through normalization.*

### Supported Language-Specific Features

#### Spanish
- **Flaps**: ɾ (normalized to ɹ)
- **Dental fricative**: θ (theta, as in "gracias")
- **Palatal nasal**: ɲ (as in "niño")
- **Palatal lateral**: ʎ (as in "calle")
- **Nasalized vowels**: ã, õ, ẽ
- **Trilled r**: r (normalized to ɹ)

```python
# Spanish examples
normalize_ipa("/ˈpeɾo/")           # → "peɹo" (pero - but)
normalize_ipa("/ɡɾaˈθjas/")        # → "gɹaθjas" (gracias - thank you)
normalize_ipa("/ˈniɲo/")           # → "niɲo" (niño - child)
ipa_to_artemes("/ɡɾaˈθjas/")       # → "kɹəθəs"
```

#### Indonesian (Bahasa Indonesia)
- **Glottal stops**: ʔ (often transcription artifact, excluded)
- **Schwa**: ə (central vowel)
- **Velar nasal**: ŋ (ng sound)
- **Diverse vowels**: a, e, i, o, u with variants
- **Consonant clusters**: Complex onset and coda clusters

```python
# Indonesian examples
normalize_ipa("/ˈbaɡus/")          # → "bagus" (bagus - good)
normalize_ipa("/sə.ˈla.mat/")      # → "səlamat" (selamat - greetings)
normalize_ipa("/ˈtɛɹima/")         # → "tɛɹima" (terima - receive)
ipa_to_artemes("/ˈbaɡus/")         # → "pəkəs"
```

#### Swahili
- **Pre-nasalized consonants**: ᵐb, ⁿd, ᵑg (as in "simba", "ndizi", "ngoma")
- **Implosives**: ɓ, ɗ, ɠ (as in "habari", "adui")
- **Aspirated consonants**: kʰ, tʰ, pʰ (diacritics stripped)
- **Dental vs alveolar**: t̪ (dental diacritic handled)
- **Open back vowel**: ɑ (preserved, not normalized to 'a')
- **Flaps**: ɾ (normalized to ɹ)

```python
# Swahili examples
normalize_ipa("/ˈsi.ᵐbɑ/")         # → "siᵐbɑ" (simba - lion)
normalize_ipa("/ˈⁿdi.zi/")         # → "ⁿdizi" (ndizi - banana)
normalize_ipa("/hɑˈɓɑ.ɾi/")        # → "hɑɓɑɹi" (habari - news)
ipa_to_artemes("/ˈsi.ᵐbɑ/")        # → "sənpə"
ipa_to_artemes("/hɑˈɓɑ.ɾi/")       # → "həpəɹə"
```

### Phonetic Mappings

#### Nasalized Vowels
All nasalized vowels map to their base form in normalization and to schwa (ə) in artemes:

| IPA | Normalized | Arteme | Languages |
|-----|------------|--------|-----------|
| ã | a | ə | Spanish, Portuguese |
| õ | o | ə | Spanish, Portuguese |
| ẽ | e | ə | Spanish, Portuguese |
| ĩ | i | ə | Spanish, Portuguese |
| ũ | u | ə | Spanish, Portuguese |

#### Pre-nasalized Consonants
Pre-nasalized markers are preserved in normalization and map to nasal (n) in artemes:

| IPA | Preserved | Arteme | Example |
|-----|-----------|--------|---------|
| ᵐ | ᵐ | n | si**mb**a (lion) |
| ⁿ | ⁿ | n | **nd**izi (banana) |
| ᵑ | ᵑ | n | **ng**oma (drum) |

#### Implosives
Implosive consonants map to their closest plosive equivalents:

| IPA | Arteme | Example |
|-----|--------|---------|
| ɓ | p | ha**b**ari (news) |
| ɗ | t | a**d**ui (enemy) |
| ɠ | k | nde**g**e (bird) |

### Common Words Examples

#### Spanish

| Word | IPA | Normalized | Artemes | Translation |
|------|-----|------------|---------|-------------|
| hola | /ˈola/ | ola | ələ | hello |
| gracias | /ɡɾaˈθjas/ | gɹaθjas | kɹəθəs | thank you |
| agua | /ˈaɡwa/ | agwa | əkəə | water |
| niño | /ˈniɲo/ | niɲo | nənə | child |

#### Indonesian

| Word | IPA | Normalized | Artemes | Translation |
|------|-----|------------|---------|-------------|
| halo | /ˈhalo/ | halo | hələ | hello |
| terima kasih | /ˈtɛɹima ˈkasih/ | tɛɹima kasih | tɛɹənə kəsəh | thank you |
| air | /ˈair/ | aiɹ | əəɹ | water |
| baik | /ˈbaik/ | baik | pəək | good |

#### Swahili

| Word | IPA | Normalized | Artemes | Translation |
|------|-----|------------|---------|-------------|
| jambo | /ˈʄɑ.ᵐbɔ/ | ʄɑᵐbɔ | tənpə | hello |
| asante | /ɑˈsɑn.te/ | ɑsɑnte | əsəntə | thank you |
| maji | /ˈmɑ.d͡ʒi/ | mɑdʒi | nəʃə | water |
| simba | /ˈsi.ᵐbɑ/ | siᵐbɑ | sənpə | lion |

### Dataset Compatibility

The library has been tested with real-world datasets:

- **Spanish**: 9M+ tokens from `data/spanish_words_ipa.csv`
- **Indonesian**: 805K+ tokens from `data/indonesian_words_ipa.csv`
- **Swahili**: 310 words from `data/swahili_words_ipa.csv`

All conversions maintain phonetic accuracy while handling language-specific features appropriately.

### Testing

Comprehensive multilingual test suite with 23 tests covering:
- Spanish flaps, theta, palatals, and nasalized vowels
- Indonesian glottal stops, schwa, and ng sounds
- Swahili pre-nasalized consonants, implosives, and aspirated consonants
- Edge cases with complex consonant clusters and diacritics
- Real-world dataset samples from all three languages

Run tests: `python tests/test_phonetics_multilingual.py`

## Advanced Features

### IPA Normalization

The normalization function handles various IPA transcription styles:

```python
# Stress markers
normalize_ipa("ˈhɛˌloʊ")      # → "hɛloʊ"

# Space-separated phonemes
normalize_ipa("ð ə k æ t")    # → "ðəkæt"

# Tie-bar affricates
normalize_ipa("t͡ʃɛɹ")        # → "tʃɛɹ"
normalize_ipa("d͜ʒ")          # → "dʒ"

# Diacritics (aspiration, length, etc.)
normalize_ipa("kʰæt")         # → "kæt"
normalize_ipa("siːŋ")         # → "siŋ"

# Character variants
normalize_ipa("ɡ")            # → "g"  (g with tail)
normalize_ipa("r")            # → "ɹ"  (trill to approximant)
normalize_ipa("ɨ")            # → "ɪ"  (close central to front)
```

### Round-Trip Conversions

```python
# IPA → ARPA → IPA
original = "kæt"
arpa = ipa_to_arpa(original)      # "K AE T"
restored = arpa_to_ipa(arpa)      # "kæt"
assert original == restored

# ARPA → IPA → Artemes
arpa = "K AE1 T"
ipa = arpa_to_ipa(arpa)           # "kæt"
artemes = ipa_to_artemes(ipa)     # "kət"
```

### Error Handling

```python
# Unknown symbols in strict mode
try:
    ipa_to_artemes("xyz123", strict=True)
except ValueError as e:
    print(f"Error: {e}")  # "Unknown IPA symbol: 'x' at position 0"

# Non-strict mode (default) - unknown symbols are skipped
result = ipa_to_artemes("xyz123", strict=False)  # ""
```

## API Reference

### Functions

#### `normalize_ipa(ipa_string, remove_stress=True, merge_spaces=True)`
Normalize IPA string to canonical form.

**Parameters:**
- `ipa_string` (str): Input IPA transcription
- `remove_stress` (bool): Remove stress markers (default: True)
- `merge_spaces` (bool): Merge space-separated phonemes (default: True)

**Returns:** str - Normalized IPA string

#### `ipa_to_arpa(ipa_string)`
Convert canonical IPA to ARPA phoneme sequence.

**Parameters:**
- `ipa_string` (str): IPA transcription (will be normalized first)

**Returns:** str - Space-separated ARPA phonemes

#### `arpa_to_ipa(arpa_string)`
Convert ARPA phoneme sequence to canonical IPA.

**Parameters:**
- `arpa_string` (str): Space-separated ARPA phonemes

**Returns:** str - IPA transcription

#### `ipa_to_artemes(ipa_string, strict=False)`
Convert IPA to arteme (phonetic class) sequence.

**Parameters:**
- `ipa_string` (str): IPA transcription (will be normalized first)
- `strict` (bool): Raise error on unknown symbols (default: False)

**Returns:** str - Arteme sequence

#### `arpa_to_artemes(arpa_string)`
Convert ARPA to artemes via IPA.

**Parameters:**
- `arpa_string` (str): Space-separated ARPA phonemes

**Returns:** str - Arteme sequence

#### `validate_ipa(ipa_string)`
Validate IPA string and return analysis.

**Parameters:**
- `ipa_string` (str): IPA transcription to validate

**Returns:** dict with keys:
- `valid` (bool): Whether string contains only known symbols
- `unknown_symbols` (list): List of unknown symbols found
- `excluded_symbols` (list): List of excluded symbols found (stress, etc.)

## Design Principles

1. **Canonical IPA as Hub**: All conversions flow through normalized IPA, ensuring consistency
2. **Lossy Arteme Conversion**: Artemes reduce phonetic detail for ML/analysis (one-way only)
3. **Stress-Agnostic**: Stress information is normalized away but preserved in ARPA input
4. **Variant Handling**: Multiple IPA transcription styles are normalized to standard form
5. **Error Tolerance**: Non-strict mode allows processing of partially valid data

## Use Cases

- **Speech Recognition**: Convert model outputs (ARPA) to standard IPA
- **G2P Systems**: Validate and normalize grapheme-to-phoneme outputs
- **Phonetic Analysis**: Reduce to articulatory classes for pattern analysis
- **Dataset Processing**: Standardize phonetic annotations across datasets
- **Lexicon Management**: Convert between different lexicon formats

## Limitations

- **Artemes are lossy**: Cannot reverse artemes back to IPA/ARPA
- **Stress information**: Lost in IPA↔ARPA conversion (ARPA stress markers stripped)
- **Dialectal variation**: Maps to American English ARPA phoneset
- **Tone languages**: Tone markers are excluded during normalization

## References

- IPA Chart: https://www.internationalphoneticassociation.org/content/ipa-chart
- ARPABET: https://en.wikipedia.org/wiki/ARPABET
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

## License

Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)

Licensed under the Apache License, Version 2.0