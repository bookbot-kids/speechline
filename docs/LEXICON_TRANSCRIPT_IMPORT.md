# Lexicon Building from Transcript Files

Guide for importing words from transcript files into the English lexicon database using the enhanced `build_english_lexicon.py` script.

## Overview

The lexicon builder now supports extracting words directly from transcript `.txt` files in addition to the traditional TSV-based import. This is particularly useful for:

- **Bookbot datasets**: Automatically scans `en-*` subdirectories for transcript files
- **Common Voice clips**: Scans clips directory for individual transcript files
- **Any transcript directory**: Works with any directory containing `.txt` files

## New Feature: Transcript File Import

### What It Does

1. **Scans directories** for `.txt` transcript files
   - Bookbot: Recursively scans `en-*` subdirectories
   - Common Voice: Scans clips directory directly
   - Generic: Adapts to directory structure automatically

2. **Extracts words** from transcript files
   - Reads first line only (text transcript)
   - Tokenizes and normalizes to lowercase
   - Collects unique words across all files

3. **Processes missing words** using existing logic
   - Checks database for existing words
   - Handles plurals intelligently (if base word exists)
   - Uses G2P (gruut) for unknown words
   - Generates IPA → ARPA → Artemes automatically

## Usage

### Basic Usage

```bash
# Import from transcript directories only
python scripts/build_english_lexicon.py \
    --transcript-dirs \
        "/mnt/Bookbot" \
        "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips"
```

### Full Database Build

```bash
# Build complete lexicon from all sources including transcripts
python scripts/build_english_lexicon.py \
    --cv-path "/mnt/Store07/Common Voice/en/validated.tsv" \
    --transcript-dirs \
        "/mnt/Bookbot" \
        "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips"
```

### Skip Traditional Import

```bash
# Only import from transcript files (skip TSV import)
python scripts/build_english_lexicon.py \
    --skip-cv \
    --transcript-dirs "/mnt/Bookbot"
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--transcript-dirs` | List of directory paths to scan for `.txt` files |
| `--output` | Output database path (default: `data/english_lexicon.db`) |
| `--cv-path` | Path to Common Voice `validated.tsv` (optional) |
| `--skip-cv` | Skip traditional TSV-based Common Voice import |

## Directory Structure Support

### Bookbot Structure

```
/mnt/Bookbot/
├── en-US/
│   ├── audio1.wav
│   ├── audio1.txt          # ← Scanned
│   ├── audio2.wav
│   └── audio2.txt          # ← Scanned
├── en-GB/
│   ├── audio3.wav
│   └── audio3.txt          # ← Scanned
└── en-AU/
    └── ...
```

The script automatically:
- Finds all `en-*` subdirectories
- Recursively scans for `.txt` files
- Extracts words from first line of each file

### Common Voice Clips Structure

```
/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips/
├── common_voice_en_1.mp3
├── common_voice_en_1.txt    # ← Scanned
├── common_voice_en_2.mp3
├── common_voice_en_2.txt    # ← Scanned
└── ...
```

The script:
- Scans directory for `.txt` files
- Extracts words from first line
- Ignores audio files

## Output Example

```
PHASE 5: Import from Transcript Files
------------------------------------------------------------
Scanning transcript directories...
  Scanning: /mnt/Bookbot
    Scanned 1000 files, found 8,542 unique words so far...
    Scanned 2000 files, found 12,387 unique words so far...
    Found 2847 transcript files
  Scanning: /mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips
    Scanned 3000 files, found 18,291 unique words so far...
    Found 15234 transcript files

  Total files scanned: 18,081
  Total unique words found: 23,456

  Processed 1000 words, added 234 new words
  Processed 2000 words, added 456 new words
  ...

  ✓ Processed: 23,456 words
  ✓ Added: 1,234 new words
  ✓ Plurals handled: 567
  ✓ G2P used: 667
```

## Features

### 1. Automatic Directory Detection

The script intelligently detects directory structure:
- **Bookbot pattern**: Looks for `en-*` subdirectories
- **Flat pattern**: Scans current directory if no `en-*` found
- **Recursive scanning**: Finds all `.txt` files in subdirectories

### 2. Word Validation

Uses existing validation logic:
- Filters out words with dashes or spaces
- Keeps only alphabetic characters
- Normalizes to lowercase

### 3. Intelligent Plural Handling

Same logic as TSV import:
```python
# If "cats" is unknown but "cat" exists:
"cat" → "kæt" (existing)
"cats" → "kæts" (automatically generated)

# Voicing rules:
"dogs" → base "dog" + "z" (voiced)
"cats" → base "cat" + "s" (voiceless)
```

### 4. G2P Fallback

For truly unknown words:
- Uses gruut's G2P engine
- Generates IPA pronunciation
- Automatically converts to ARPA and Artemes
- Adds to database

### 5. Progress Tracking

- Reports every 1,000 files scanned
- Shows unique word count in real-time
- Displays processing progress
- Final statistics summary

## Performance

### Efficiency
- **Scanning**: ~1,000 files/second
- **Processing**: ~1,000 words/second
- **Memory**: Efficient iterator-based scanning
- **Database**: Batch commits for performance

### Expected Runtime
- **Bookbot (~3K files)**: ~30 seconds
- **Common Voice clips (~15K files)**: ~2 minutes
- **Both combined**: ~3 minutes
- **Full database build**: ~10-15 minutes

## Testing

### Small Test Run

```bash
# Test on single Bookbot directory
python scripts/build_english_lexicon.py \
    --transcript-dirs "/mnt/Bookbot/en-US"
```

### Verify Results

```bash
# Check newly added words
python scripts/query_lexicon.py lookup newword

# Get statistics
python scripts/query_lexicon.py stats
```

### Check Pronunciations

```bash
# Verify IPA/ARPA/Arteme generated correctly
python scripts/query_lexicon.py lookup example_word
```

## Troubleshooting

### "Directory not found"

**Cause**: Path doesn't exist
**Solution**: Verify paths with `ls -la /path/to/directory`

### "No transcript directories provided"

**Cause**: `--transcript-dirs` not specified
**Solution**: Add at least one directory path

### "G2P error for 'word'"

**Cause**: gruut failed to generate pronunciation
**Solution**: Normal for edge cases, automatically skipped

### Low word count added

**Cause**: Most words already in database
**Solution**: Expected behavior - only new words are added

## Related Files

- **Script**: [`scripts/build_english_lexicon.py`](../scripts/build_english_lexicon.py)
- **Database**: [`data/english_lexicon.db`](../data/english_lexicon.db)
- **Query tool**: [`scripts/query_lexicon.py`](../scripts/query_lexicon.py)
- **Main docs**: [`data/LEXICON_DB_README.md`](../data/LEXICON_DB_README.md)

## Best Practices

1. **Run full build first**: Include all traditional sources before transcript import
2. **Test on subset**: Try one directory first before full run
3. **Check results**: Verify new words with query tool
4. **Monitor progress**: Watch for G2P errors (usually harmless)
5. **Backup database**: Copy `english_lexicon.db` before major updates

## Example Workflow

```bash
# 1. Build base lexicon from traditional sources
python scripts/build_english_lexicon.py

# 2. Add transcript-based words
python scripts/build_english_lexicon.py \
    --transcript-dirs \
        "/mnt/Bookbot" \
        "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips"

# 3. Verify results
python scripts/query_lexicon.py stats

# 4. Test specific words
python scripts/query_lexicon.py lookup someword
```

## Notes

- First line of `.txt` file is treated as text transcript
- Second line (if exists) is ignored (may contain phonemes)
- Audio files are ignored
- Empty transcripts are skipped
- Invalid words (with spaces/dashes) are filtered out