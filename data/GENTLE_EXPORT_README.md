# Gentle Lexicon Export

This document explains how to export the English Lexicon Database to Gentle's `align_lexicon.txt` format.

## Overview

The export script converts the lexicon database to Gentle's format:
- **Format**: `word word phoneme_sequence_with_position_markers`
- **Phonemes**: Lowercase CMU ARPAbet without stress markers
- **Position markers**: `_B` (beginning), `_I` (internal), `_E` (end), `_S` (standalone)

## Quick Start

```bash
# Basic export (creates data/align_lexicon.txt)
python scripts/export_gentle_lexicon.py

# Custom output path
python scripts/export_gentle_lexicon.py -o my_lexicon.txt

# Limit to 1 pronunciation per word
python scripts/export_gentle_lexicon.py --max-pron 1
```

## Export Statistics

**From the current database:**
- **Total entries**: 524,892 pronunciations
- **File size**: 22.3 MB
- **Words skipped**: 728 (invalid characters)
- **Coverage**: 387,459 unique words with up to 3 pronunciations each

## Format Details

### Position Markers

Position markers indicate where a phoneme appears in a word:

**Single-phone words** use `_S` (standalone):
```
a a ah_S
i i ay_S
```

**Two-phone words** use `_B` and `_E`:
```
go go g_B ow_E
no no n_B ow_E
```

**Multi-phone words** use `_B`, `_I`, and `_E`:
```
hello hello hh_B aa_I l_E
cat cat k_B ae_I t_E
world world w_B er_I l_I d_E
```

### Stress Marker Removal

CMU ARPAbet stress markers (0, 1, 2) are automatically removed:
- `AE1` → `ae`
- `EY0` → `ey`
- `T` → `t` (consonants have no stress markers)

### Phone Conversion

The export uses the first pronunciation variant from the database:

**Database ARPA** → **Gentle format**
```
K AE1 T                    → cat cat k_B ae_I t_E
HH EH1 L OW0               → hello hello hh_B eh_I l_I ow_E
T AH0 M EY1 T OW0          → tomato tomato t_B ah_I m_I ey_I t_I ow_E
```

## Usage Options

### Command-Line Arguments

```bash
python scripts/export_gentle_lexicon.py [OPTIONS]

Options:
  --db PATH           Path to lexicon database 
                      (default: data/english_lexicon.db)
  
  -o, --output PATH   Output file path 
                      (default: data/align_lexicon.txt)
  
  --max-pron N        Maximum pronunciations per word 
                      (default: 3)
  
  --samples N         Number of sample entries to display 
                      (default: 10)
  
  -h, --help         Show help message
```

### Examples

**1. Export with single pronunciation per word (smaller file):**
```bash
python scripts/export_gentle_lexicon.py --max-pron 1 -o gentle_single.txt
```

**2. Export all variants (larger file):**
```bash
python scripts/export_gentle_lexicon.py --max-pron 10 -o gentle_full.txt
```

**3. Use custom database:**
```bash
python scripts/export_gentle_lexicon.py --db custom.db -o custom_gentle.txt
```

## Using with Gentle

### Installation

1. **Replace Gentle's lexicon:**
   ```bash
   # Backup original
   cp exp/langdir/phones/align_lexicon.txt exp/langdir/phones/align_lexicon.txt.backup
   
   # Copy exported lexicon
   cp data/align_lexicon.txt /path/to/gentle/exp/langdir/phones/align_lexicon.txt
   ```

2. **Restart Gentle server:**
   ```bash
   cd /path/to/gentle
   python serve.py
   ```

3. **No recompilation needed!** Gentle builds language models dynamically.

### Verification

Test with a simple transcript:
```bash
# Test alignment
curl -F "audio=@test.wav" -F "transcript=hello world" http://localhost:8765/transcriptions?async=false
```

## Sample Output

From the exported lexicon:

```
# Header (automatic)
# Gentle Lexicon exported from speechline english_lexicon.db
# Format: word word phoneme_sequence_with_position_markers
# Position markers: _B (begin), _I (internal), _E (end), _S (standalone)

# Sample entries
a a aa_S
a a ae_S
cat cat k_B ae_I t_E
hello hello hh_B eh_I l_I ow_E
world world w_B er_I l_I d_E
tomato tomato t_B ah_I m_I ey_I t_I ow_E
tomato tomato t_B ah_I m_I aa_I t_I ow_E
```

## Comparison with CMU Dictionary

**CMU format** (with stress):
```
HELLO  HH EH1 L OW0
WORLD  W ER1 L D
```

**Gentle format** (exported):
```
hello hello hh_B eh_I l_I ow_E
world world w_B er_I l_I d_E
```

## Data Sources in Export

The export includes pronunciations from all database sources:
1. CMU Pronouncing Dictionary (133K words)
2. Montreal Forced Aligner lexicons (7 variants)
3. English processed words (81K words)
4. WordUniversal dictionary (200K entries)
5. Common Voice G2P (132K words)

All sources are merged with duplicates removed, providing comprehensive coverage.

## Word Validation

Words are validated before export. Skipped if they contain:
- Special characters (except apostrophes, hyphens)
- Non-ASCII characters
- Empty or very long entries (>50 chars)

Valid word patterns:
- `hello` ✓
- `don't` ✓
- `twenty-one` ✓
- `café` ✗ (non-ASCII)
- `hello@world` ✗ (special char)

## Performance

**Export time**: ~10-15 seconds for full database
**File size**: ~22 MB (524K entries)
**Memory usage**: <200 MB

## Troubleshooting

### Missing Words

If specific words are missing from the export:

1. **Check if word exists in database:**
   ```bash
   python scripts/query_lexicon.py lookup yourword
   ```

2. **Check validation:**
   - Word must contain only letters, numbers, apostrophes, hyphens
   - Must have ARPA pronunciation in database

3. **Add to database if missing:**
   - Rebuild database with additional sources, or
   - Manually add to `align_lexicon.txt`

### Invalid Phones

Gentle uses a fixed set of 39 CMU ARPAbet phones. The export only uses standard phones that are in the original CMU set.

### File Size

To reduce file size:
- Use `--max-pron 1` for single pronunciation per word
- Filter by word frequency (requires custom script)

## Python API

Use the exporter programmatically:

```python
from export_gentle_lexicon import GentleExporter

# Create exporter
exporter = GentleExporter("data/english_lexicon.db")

# Export
exporter.export("gentle_lexicon.txt", max_pronunciations=3)

# Show samples
exporter.show_samples("gentle_lexicon.txt", n=20)

# Close
exporter.close()
```

## References

- [Gentle GitHub](https://github.com/lowerquality/gentle)
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- [ARPAbet Phone Set](https://en.wikipedia.org/wiki/ARPABET)
- [Kaldi Documentation](https://kaldi-asr.org/doc/)

## Related Files

- **Export script**: [`scripts/export_gentle_lexicon.py`](../scripts/export_gentle_lexicon.py)
- **Query script**: [`scripts/query_lexicon.py`](../scripts/query_lexicon.py)
- **Database**: [`data/english_lexicon.db`](english_lexicon.db)
- **Database docs**: [`LEXICON_DB_README.md`](LEXICON_DB_README.md)