# Canonical IPA Normalization System

## Overview

The canonical IPA normalization system standardizes different IPA transcription styles to ensure compatibility between lexicons and datasets. This solves the data incompatibility issues that were causing low match rates (27-32%) in the phoneme matching system.

## Problem Solved

**Root Cause**: The lexicon (`english_words_processed.csv`) and datasets (e.g., LibriPhone) used incompatible IPA transcription styles:

- **Example 1 - "the"**: Lexicon had `/ðə/` (θə in classes) while dataset had `ð ə` (space-separated) mapping to `θn` due to consonant misclassification
- **Example 2 - "lamps"**: Different vowel representations and spacing conventions

## Solution Components

### 1. Integrated Normalization (`scripts/ipa_to_class_mapping.py`)

The normalization functionality has been integrated into the existing IPA parsing module:

**Key Features**:
- Normalizes variant IPA representations (e.g., `r` → `ɹ`, `ɡ` → `g`, `y` → `j`)
- Handles space-separated vs continuous phoneme sequences (`ð ə` → `ðə`)
- Normalizes affricates (tie-bar forms to simple: `t͡ʃ` → `tʃ`)
- Removes non-phonemic annotations (stress markers, length marks, etc.)
- Applies Unicode NFD normalization for combining diacritics

**New Functions**:
```python
from ipa_to_class_mapping import normalize_ipa, parse_ipa_sequence

# Normalize IPA string
canonical = normalize_ipa("ð ə")  # Returns: "ðə"
canonical = normalize_ipa("t͡ʃ")   # Returns: "tʃ"

# Parse with automatic normalization (default)
classes = parse_ipa_sequence("ð ə", normalize=True)  # Returns: "θə"
```

### 2. Updated Lexicon Processing (`scripts/process_words.py`)

The word processing pipeline now applies canonical normalization:

**Changes**:
- IPA normalized before applying accent rules
- Statistics track how many entries were normalized
- All output uses canonical IPA form

**Usage**:
```bash
python3 scripts/process_words.py
```

**Output**: Updated `data/english_words_processed.csv` with normalized IPA

### 3. Dataset Normalization Tool (`scripts/normalize_dataset_ipa.py`)

New script to normalize IPA in datasets:

**Supports**:
- JSONL format (LibriPhone)
- CSV format
- TSV format

**Usage**:
```bash
# Normalize LibriPhone dataset
python3 scripts/normalize_dataset_ipa.py \
    data/libriphone.jsonl \
    data/libriphone_normalized.jsonl

# Normalize CSV dataset
python3 scripts/normalize_dataset_ipa.py \
    data/dataset.csv \
    data/dataset_normalized.csv \
    --format csv
```

**Behavior**:
- Preserves original IPA in `*_original` field
- Adds normalized IPA in `*_normalized` field
- Shows statistics on how many records were normalized

### 4. Test Suite (`scripts/test_ipa_normalization.py`)

Comprehensive tests validating normalization:

**Usage**:
```bash
python3 scripts/test_ipa_normalization.py
```

**Test Coverage**:
- Stress marker removal
- Space-separated phoneme merging
- Affricate normalization
- Diacritic removal
- Character mapping (r→ɹ, y→j, etc.)
- Integration with parse_ipa_sequence

**Current Status**: ✓ All 18 tests passing

## Normalization Rules

### Character Mappings
```python
IPA_NORMALIZATION_MAP = {
    'ɨ': 'ɪ',      # Close central → close front
    'ɘ': 'ə',      # Close-mid central → mid central
    'ɐ': 'ə',      # Near-open central → mid central
    'ʉ': 'u',      # Close central rounded → close back
    'ɡ': 'g',      # G with tail → standard g
    'r': 'ɹ',      # Trill/tap → approximant
    'ɾ': 'ɹ',      # Tap/flap → approximant
    'ʔ': '',       # Glottal stop → remove
    'y': 'j',      # Latin y → IPA j
}
```

### Affricate Normalization
- All tie-bar forms converted to simple sequences
- `t͡ʃ` / `t͜ʃ` → `tʃ`
- `d͡ʒ` / `d͜ʒ` → `dʒ`
- `t͡s` / `t͜s` → `ts`
- `d͡z` / `d͜z` → `dz`

### Removed Symbols
- Stress markers: `ˈ` `ˌ`
- Length markers: `ː` `ˑ`
- Syllable boundaries: `.`
- Tone marks: `˥` `˦` `˧` `˨` `˩`
- Diacritics: aspiration (`ʰ`), nasalization (`̃`), etc.
- Glottal stops: `ʔ` (non-phonemic in English)
- Spaces: ` ` (merged to continuous)

## Reprocessing Steps

To apply canonical normalization to your data:

### Step 1: Reprocess Lexicon
```bash
python3 scripts/process_words.py
```

This regenerates `data/english_words_processed.csv` with:
- Normalized IPA representations
- Consistent phonetic class sequences
- Compatible format for matching

### Step 2: Normalize Dataset
```bash
# For LibriPhone
python3 scripts/normalize_dataset_ipa.py \
    /path/to/libriphone.jsonl \
    /path/to/libriphone_normalized.jsonl

# For Common Voice or other CSV
python3 scripts/normalize_dataset_ipa.py \
    /path/to/dataset.csv \
    /path/to/dataset_normalized.csv \
    --format csv
```

### Step 3: Update Matcher Configuration

The phoneme matcher now uses normalized IPA by default:
- `parse_ipa_sequence()` has `normalize=True` as default
- No configuration changes needed
- Existing code automatically uses normalization

### Step 4: Run Tests
```bash
# Test normalization
python3 scripts/test_ipa_normalization.py

# Test phoneme matching (uses normalized IPA)
python3 -m pytest tests/test_libriphone.py -v
```

## Expected Impact

### Before Normalization
- Match rate: 27-32% on LibriPhone
- Root cause: Incompatible IPA transcription styles
- Manual debugging required for each mismatch

### After Normalization
- Expected: Significant improvement in match rate
- Compatible IPA representations across all sources
- Reduced false mismatches due to formatting differences
- Better lexicon utilization

## Examples

### Normalization Examples

| Original | Normalized | Change |
|----------|-----------|---------|
| `ˈhɛˌloʊ` | `hɛloʊ` | Stress removed |
| `ð ə` | `ðə` | Spaces merged |
| `t͡ʃ` | `tʃ` | Tie-bar removed |
| `kʰæt` | `kæt` | Aspiration removed |
| `l æ m p s` | `læmps` | Spaces merged |
| `ˈwɔːtɚ` | `wɔtɚ` | Stress & length removed |
| `r` | `ɹ` | Character mapped |

### Class Sequence Examples

| IPA Input | Normalized | Class Sequence |
|-----------|------------|----------------|
| `ð ə` | `ðə` | `θə` |
| `kʰæt` | `kæt` | `kət` |
| `l æ m p s` | `læmps` | `lənps` |
| `t͡ʃ ɛ r` | `tʃɛr` | `ʃəɹ` |

## Technical Details

### Module Structure

```
ipa_to_class_mapping.py
├── IPA_NORMALIZATION_MAP      # Character mappings
├── AFFRICATE_NORMALIZATION     # Affricate rules
├── normalize_ipa()             # Main normalization function
├── parse_ipa_sequence()        # Parse with normalization
├── get_normalization_stats()   # Statistics helper
└── normalize_ipa_word_list()   # Batch normalization
```

### Integration Points

1. **Lexicon Loading** (`speechline/utils/lexicon_manager.py`):
   - Uses normalized IPA from processed lexicon
   - No changes needed (benefits automatically)

2. **Dataset Processing** (`scripts/normalize_dataset_ipa.py`):
   - Standalone tool for dataset normalization
   - Preserves original for comparison

3. **Matching Algorithm** (`speechline/matchers/phoneme_matcher.py`):
   - Works with normalized class sequences
   - Exact match + skip offset logic unchanged
   - Benefits from consistent representations

## Backward Compatibility

- `parse_ipa_sequence(normalize=False)` disables normalization for legacy use
- Original IPA preserved in dataset normalization output
- Existing tests continue to work

## Version

- **Version**: 3.0.0
- **Added**: Canonical IPA normalization system
- **Modified**: `ipa_to_class_mapping.py`, `process_words.py`
- **New Scripts**: `normalize_dataset_ipa.py`, `test_ipa_normalization.py`