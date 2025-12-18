# English Lexicon Database

Consolidated SQLite database containing English word pronunciations from multiple sources with automatic phoneme conversion between IPA, ARPA, and Arteme formats.

## Database Location

```
data/english_lexicon.db
```

## Schema

### `english` Table

Single table storing one row per word with multiple pronunciations:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `word` | TEXT | Unique lowercase word |
| `ipa` | TEXT | Space-delimited IPA pronunciations |
| `arpa` | TEXT | Pipe-delimited ARPA pronunciations |
| `arteme` | TEXT | Space-delimited arteme sequences (deduplicated) |
| `created_at` | TIMESTAMP | Creation timestamp |

**Indexes:**
- `idx_word` on `word` column for fast lookups

### `import_stats` Table

Tracks import statistics from each data source.

---

## Data Format

### IPA (International Phonetic Alphabet)
- **Format**: Space-delimited pronunciations
- **Example**: `"ɹid ɹɛd"` (two pronunciations for "read")
- **Normalization**: Stress marks removed, canonical IPA forms

### ARPA (ARPABET)
- **Format**: Pipe-delimited pronunciations (ARPA uses spaces within)
- **Example**: `"R IY D | R EH D"` (two pronunciations)
- **Conversion**: Automatically generated from IPA

### Arteme (Phonetic Classes)
- **Format**: Space-delimited sequences
- **Example**: `"ɹət"` (single deduplicated sequence)
- **Deduplication**: Multiple IPA pronunciations may collapse to same arteme

---

## Data Sources

The database consolidates pronunciations from:

1. **`english_words_processed.csv`** - Base lexicon with IPA and arteme
2. **`worduniversal_en_full.csv`** - Word Universal dataset (semicolon-delimited IPA)
3. **`data/lexicons/english_us_mfa.dict`** - MFA US English lexicon (IPA)
4. **`data/lexicons/english_us_arpa.dict`** - CMU ARPA dictionary
5. **`data/lexicons/english_uk_mfa.dict`** - MFA UK English lexicon
6. **`data/lexicons/english_india_mfa.dict`** - MFA Indian English
7. **`data/lexicons/english_nigeria_mfa.dict`** - MFA Nigerian English  
8. **`data/lexicons/english_nonnative_mfa.dict`** - MFA non-native English
9. **Common Voice validated.tsv** - Extracts missing words with G2P
10. **Transcript files (.txt)** - Extracts words from individual transcript files (NEW)

---

## Building the Database

### Basic Build

```bash
python scripts/build_english_lexicon.py
```

This will:
- Create `data/english_lexicon.db`
- Import all lexicon sources
- Skip Common Voice (optional)

### With Common Voice

```bash
python scripts/build_english_lexicon.py --cv-path /path/to/validated.tsv
```

### Custom Output

```bash
python scripts/build_english_lexicon.py --output my_lexicon.db
```

### Skip Common Voice

```bash
python scripts/build_english_lexicon.py --skip-cv
```

### With Transcript Files (NEW)

Import words from individual transcript files in specified directories:

```bash
# Import from transcript directories
python scripts/build_english_lexicon.py \
    --transcript-dirs \
        "/mnt/Bookbot" \
        "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips"
```

**See [Transcript Import Guide](../docs/LEXICON_TRANSCRIPT_IMPORT.md) for detailed documentation.**

---

## Querying the Database

### Command-Line Interface

#### Lookup a Word

```bash
python scripts/query_lexicon.py lookup hello
```

**Output:**
```
Word: hello
IPA pronunciations (2):
  hɛloʊ
  həloʊ

ARPA pronunciations (2):
  HH AH L OW
  HH EH L OW

Arteme sequences (1):
  hələə
```

#### Database Statistics

```bash
python scripts/query_lexicon.py stats
```

**Output:**
```
============================================================
LEXICON DATABASE STATISTICS
============================================================
Total unique words: 123,456
Total IPA pronunciations: 234,567
Average pronunciations per word: 1.90
Maximum pronunciations for a word: 8
Words with multiple pronunciations: 45,678
...
```

#### Search for Words

```bash
# Find words starting with "cat"
python scripts/query_lexicon.py search "cat%"

# Find words ending with "ing"
python scripts/query_lexicon.py search "%ing"

# Find words containing "tion"
python scripts/query_lexicon.py search "%tion%"
```

#### Export Database

```bash
# Export to ARPA dictionary format
python scripts/query_lexicon.py export --format arpa --output output.dict

# Export to IPA dictionary format
python scripts/query_lexicon.py export --format ipa --output output.dict

# Export to CSV
python scripts/query_lexicon.py export --format csv --output lexicon.csv
```

---

## Python API

### Lookup Words

```python
from scripts.query_lexicon import LexiconQuery

# Open database
query = LexiconQuery("data/english_lexicon.db")

# Lookup a word
result = query.lookup("hello")

if result:
    print(f"Word: {result['word']}")
    print(f"IPA: {result['ipa']}")      # ['hɛloʊ', 'həloʊ']
    print(f"ARPA: {result['arpa']}")    # ['HH EH L OW', 'HH AH L OW']
    print(f"Arteme: {result['arteme']}")# ['hələə']

query.close()
```

### Direct SQL Queries

```python
import sqlite3

conn = sqlite3.connect("data/english_lexicon.db")
conn.row_factory = sqlite3.Row

# Get all pronunciations for a word
row = conn.execute(
    "SELECT ipa, arpa, arteme FROM english WHERE word = ?",
    ("hello",)
).fetchone()

if row:
    ipa_list = row['ipa'].split()           # Split space-delimited
    arpa_list = row['arpa'].split(' | ')    # Split pipe-delimited
    arteme_list = row['arteme'].split()     # Split space-delimited

conn.close()
```

### Batch Lookup

```python
from scripts.query_lexicon import LexiconQuery

query = LexiconQuery("data/english_lexicon.db")

words = ["cat", "dog", "bird"]
for word in words:
    result = query.lookup(word)
    if result:
        print(f"{word}: {result['ipa']}")

query.close()
```

---

## Features

### 1. Automatic Deduplication

- **IPA**: Normalized and deduplicated
- **ARPA**: Converted from unique IPA pronunciations
- **Arteme**: Deduplicated phonetic classes (multiple IPA → same arteme)

**Example: "read"**
```
IPA: "ɹid ɹɛd"          # 2 pronunciations
ARPA: "R EH D | R IY D"  # 2 pronunciations
Arteme: "ɹət"            # 1 sequence (both vowels → ə)
```

### 2. Word Validation

Automatically rejects words with:
- Dashes: `"re-enter"` ✗
- Spaces: `"ice cream"` ✗
- Empty strings ✗

### 3. Intelligent Plural Handling

For Common Voice words ending in 's':
1. Check if base word exists (e.g., "cats" → check "cat")
2. If base exists, generate plural by adding 's' or 'z' phoneme
3. Voiceless final phoneme → add 's'
4. Voiced final phoneme → add 'z'

**Example:**
```
"cat" exists with: kæt
"cats" generated: kæts  (voiceless t → add s)

"dog" exists with: dɔg
"dogs" generated: dɔgz  (voiced g → add z)
```

### 4. G2P Fallback

For unknown words in Common Voice:
- Uses [`speechline.utils.g2p.g2p_en()`](../speechline/utils/g2p.py:27) (gruut)
- Automatically generates IPA pronunciations
- Converts to ARPA and artemes

### 5. Phoneme Conversion

Uses [`speechline.phonetics`](../speechline/phonetics.py:1):
- [`normalize_ipa()`](../speechline/phonetics.py:386) - Canonicalize IPA
- [`ipa_to_arpa()`](../speechline/phonetics.py:573) - IPA → ARPA conversion
- [`arpa_to_ipa()`](../speechline/phonetics.py:536) - ARPA → IPA conversion  
- [`ipa_to_artemes()`](../speechline/phonetics.py:484) - IPA → Arteme conversion

---

## Database Statistics Example

After building with all sources:

```
Total unique words: ~150,000
Total IPA pronunciations: ~300,000
Average pronunciations per word: ~2.0
Words with multiple pronunciations: ~60,000
```

**Import breakdown:**
- processed_csv: ~119,000 words
- worduniversal: ~15,000 new words
- MFA dictionaries: ~8,000 new words
- ARPA dictionary: ~2,000 new words
- Common Voice + G2P: ~6,000 new words

---

## File Structure

```
data/
├── english_lexicon.db           # Main database
├── LEXICON_DB_README.md         # This file
├── english_words_processed.csv  # Source data
├── worduniversal_en_full.csv    # Source data
└── lexicons/                    # Source lexicons
    ├── english_us_mfa.dict
    ├── english_us_arpa.dict
    └── ...

scripts/
├── build_english_lexicon.py     # Database builder
└── query_lexicon.py             # Query utility
```

---

## Performance

- **Lookup by word**: O(1) with index (instant)
- **Database size**: ~50-100 MB (depending on sources)
- **Build time**: 
  - Without Common Voice: ~5-10 minutes
  - With Common Voice: ~15-30 minutes

---

## Troubleshooting

### Build Errors

**"File not found"**
- Ensure all source files exist in `data/` directory
- Check file paths in error messages

**"G2P import error"**
- Install gruut: `pip install gruut`
- Or skip Common Voice: `--skip-cv`

**"Conversion failed"**
- Warnings for invalid IPA symbols are normal
- Errors are logged but don't stop the build

### Query Errors

**"Database not found"**
```bash
# Build database first
python scripts/build_english_lexicon.py
```

**"Word not found"**
- Check spelling
- Try search: `python scripts/query_lexicon.py search "word%"`

---

## Maintenance

### Rebuilding Database

To rebuild from scratch:

```bash
# Remove existing database
rm data/english_lexicon.db

# Rebuild
python scripts/build_english_lexicon.py --cv-path /path/to/validated.tsv
```

### Adding New Sources

Edit [`scripts/build_english_lexicon.py`](../scripts/build_english_lexicon.py:1) and add import in Phase 3:

```python
builder.import_dict_file("path/to/new.dict", "ipa", "source_name")
```

---

## License

Data sources may have different licenses. Check individual source licenses before redistribution.

---

## References

- **IPA**: [International Phonetic Association](https://www.internationalphoneticassociation.org/)
- **ARPABET**: [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- **MFA Lexicons**: [Montreal Forced Aligner](https://mfa-models.readthedocs.io/)
- **Common Voice**: [Mozilla Common Voice](https://commonvoice.mozilla.org/)