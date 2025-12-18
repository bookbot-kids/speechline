# Gentle Phoneme Alignment - Usage Guide

This guide explains how to use the Gentle phoneme alignment system to add phoneme transcriptions to audio datasets.

## Overview

The system uses Gentle forced aligner with a custom lexicon from speechline to:
1. Validate transcripts against audio (find not-found-in-audio words)
2. Add phoneme transcriptions to transcript files

## Setup Status

✅ **Completed Setup Steps:**
- Gentle resources installed at `/mnt/Projects/Projects/AudioProcessing/gentle`
- Original lexicon backed up to `align_lexicon.txt.backup`
- Custom speechline lexicon installed (29.1 MB, 524K+ entries)
- Vocabulary size: 42,156 words
- Test alignment successful

## Quick Start

### 1. Test the Setup

```bash
python scripts/test_gentle_setup.py
```

This will verify:
- Gentle resources load properly
- Custom lexicon is installed
- Basic alignment works

### 2. Validation Mode

Find examples of not-found-in-audio issues:

```bash
# Scan Bookbot dataset
python scripts/gentle_phoneme_alignment.py \
    --mode validate \
    --bookbot-path /mnt/Store07/Bookbot \
    --examples-per-dataset 20

# Scan Common Voice dataset
python scripts/gentle_phoneme_alignment.py \
    --mode validate \
    --cv-clips-path "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips" \
    --examples-per-dataset 20

# Scan both datasets
python scripts/gentle_phoneme_alignment.py \
    --mode validate \
    --bookbot-path /mnt/Store07/Bookbot \
    --cv-clips-path "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips" \
    --examples-per-dataset 20
```

**Output:** Displays 20 examples from each dataset showing:
- File path
- Transcript text
- Words not found in audio

### 3. Phoneme Addition Mode

Process datasets and add phoneme transcriptions:

```bash
# Process Bookbot only
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --review-log bookbot_review.txt

# Process Common Voice only
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --cv-clips-path "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips" \
    --review-log cv_review.txt

# Process both datasets
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --cv-clips-path "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips" \
    --review-log phoneme_alignment_review.txt \
    --log-level INFO
```

## File Format

### Input Format

**Before processing** (1 line):
```
Hello world
```

### Output Formats

**Valid Alignment** (2 lines):
```
Hello world
hh eh l ow w er l d
```

**Invalid Alignment** (2 lines with error marker):
```
The quick brown fox
INVALID_TRANSCRIPT: not-found-in-audio=['fox']
```

## Processing Behavior

### Files Processed

- ✅ Audio files with corresponding `.txt` transcripts
- ✅ Files without existing phoneme transcripts (1-line .txt files)
- ❌ Files already having phoneme transcripts (2+ lines) - **skipped**

### Alignment Results

1. **Valid Alignment:**
   - All words found in audio
   - Phoneme sequence extracted successfully
   - **Action:** Phonemes appended to line 2

2. **Invalid Alignment (not-found-in-audio):**
   - Some words not found in audio
   - **Action:** `INVALID_TRANSCRIPT: not-found-in-audio=[...]` written to line 2
   - **Action:** Entry logged to review file

3. **OOV Error:**
   - Out-of-vocabulary word encountered
   - **Action:** Exception raised, processing stops
   - **Note:** Should not happen with complete lexicon

## Review Log Format

Files with alignment issues are logged to the review file:

```
File: /path/to/audio.wav
Transcript: The quick brown fox
Not Found Words: ['fox']
Reason: Words not found in audio during forced alignment
---

File: /path/to/another.wav
Transcript: Hello world from here
Not Found Words: ['from']
Reason: Words not found in audio during forced alignment
---
```

## Command-Line Options

### Required Arguments

- `--mode {validate,add-phonemes}` - Operating mode

### Optional Arguments

- `--bookbot-path PATH` - Path to Bookbot dataset
- `--cv-clips-path PATH` - Path to Common Voice clips directory
- `--gentle-path PATH` - Path to Gentle installation (default: `/mnt/Projects/Projects/AudioProcessing/gentle`)
- `--review-log PATH` - Review log file (default: `phoneme_alignment_review.txt`)
- `--log-level LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--examples-per-dataset N` - Examples to show in validation mode (default: 20)

**Note:** At least one of `--bookbot-path` or `--cv-clips-path` must be provided.

## Progress Tracking

The script shows progress every 100 files:

```
Processed 100 files...
  Added phonemes: 85
  Added invalid: 12
  Skipped (has phonemes): 3

Processed 200 files...
  Added phonemes: 170
  Added invalid: 25
  Skipped (has phonemes): 5
```

## Final Statistics

At completion, the script displays summary statistics:

```
============================================================
PHONEME ADDITION COMPLETE
============================================================
Total files processed: 1247
Phonemes added: 1050
Invalid markers added: 183
Skipped (already has phonemes): 14
OOV errors: 0
Other errors: 0

Review log: phoneme_alignment_review.txt
```

## Phoneme Format

**Format:** Simple space-separated phonemes without position markers

**Example:**
- Gentle output: `hh_B eh_I l_I ow_E w_B er_I l_I d_E`
- Stored format: `hh eh l ow w er l d`

**Phone Set:** Lowercase CMU ARPAbet without stress markers
- Vowels: `aa`, `ae`, `ah`, `ao`, `aw`, `ay`, `eh`, `er`, `ey`, `ih`, `iy`, `ow`, `oy`, `uh`, `uw`
- Consonants: `b`, `ch`, `d`, `dh`, `f`, `g`, `hh`, `jh`, `k`, `l`, `m`, `n`, `ng`, `p`, `r`, `s`, `sh`, `t`, `th`, `v`, `w`, `y`, `z`, `zh`

## Performance Expectations

### Processing Time

- **Per file:** ~1-5 seconds (depending on audio length)
- **Bookbot:** ~50,000 files × 3 sec = ~40 hours
- **Common Voice:** ~1,000,000 clips × 3 sec = ~800 hours

### Recommendations

1. **Start with validation mode** on a subset to understand data quality
2. **Process incrementally** - the script skips already processed files
3. **Monitor the review log** for problematic files
4. **Run overnight** for large datasets

## Troubleshooting

### Issue: "Gentle resources failed to load"

**Solution:**
- Check Gentle installation at `/mnt/Projects/Projects/AudioProcessing/gentle`
- Run `python scripts/test_gentle_setup.py` to diagnose

### Issue: "Lexicon file not found"

**Solution:**
```bash
cp /mnt/Projects/Projects/AudioProcessing/speechline/data/align_lexicon.txt \
   /mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt
```

### Issue: "OOV Error"

**Cause:** Word not in lexicon (should not happen with complete lexicon)

**Solution:**
1. Check the word in the lexicon:
   ```bash
   python scripts/query_lexicon.py lookup <word>
   ```
2. If missing, add to lexicon database and re-export
3. Report the issue for investigation

### Issue: Too many invalid transcripts

**Possible Causes:**
- Poor audio quality
- Incorrect transcripts
- Background noise
- Mumbled speech

**Solution:**
- Review the invalid files in the review log
- Consider audio quality filtering
- Manual review of problematic transcripts

### Issue: Script too slow

**Options:**
- Process in parallel (future enhancement)
- Filter by file size/duration first
- Process overnight
- Use faster hardware

## File Organization

### Script Files

- [`scripts/gentle_phoneme_alignment.py`](../../scripts/gentle_phoneme_alignment.py) - Main alignment script
- [`scripts/test_gentle_setup.py`](../../scripts/test_gentle_setup.py) - Setup verification
- [`scripts/export_gentle_lexicon.py`](../../scripts/export_gentle_lexicon.py) - Lexicon export tool

### Documentation

- [`docs/guides/gentle_phoneme_alignment_plan.md`](gentle_phoneme_alignment_plan.md) - Implementation plan
- [`docs/guides/gentle_phoneme_alignment_usage.md`](gentle_phoneme_alignment_usage.md) - This guide
- [`data/GENTLE_EXPORT_README.md`](../../data/GENTLE_EXPORT_README.md) - Lexicon export details

### Data Files

- `/mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt` - Active lexicon
- `/mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt.backup` - Original backup
- `phoneme_alignment_review.txt` - Review log (created during processing)

## Examples

### Example 1: Test with Sample Files

```bash
# Create test directory
mkdir -p test_alignment
cd test_alignment

# Copy a few sample files (assuming they exist)
cp /mnt/Store07/Bookbot/en-us/sample*.{wav,txt} .

# Run validation
python ../scripts/gentle_phoneme_alignment.py \
    --mode validate \
    --bookbot-path . \
    --examples-per-dataset 5

# Run phoneme addition
python ../scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path . \
    --review-log test_review.txt

# Check results
cat sample001.txt  # Should now have 2 lines
cat test_review.txt  # Review any issues
```

### Example 2: Incremental Processing

```bash
# Day 1: Process first batch
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --review-log day1_review.txt

# Day 2: Continue (skips already processed files)
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --review-log day2_review.txt
```

### Example 3: Debug Mode

```bash
# Run with debug logging to see detailed alignment info
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --log-level DEBUG \
    --review-log debug_review.txt 2>&1 | tee alignment_debug.log
```

## Next Steps

1. **Run validation mode** to understand data quality:
   ```bash
   python scripts/gentle_phoneme_alignment.py --mode validate --bookbot-path /mnt/Store07/Bookbot
   ```

2. **Review validation results** to assess expected success rate

3. **Start phoneme addition** on a subset or full dataset:
   ```bash
   python scripts/gentle_phoneme_alignment.py --mode add-phonemes --bookbot-path /mnt/Store07/Bookbot
   ```

4. **Monitor review log** for problematic files

5. **Validate results** by spot-checking processed files

## Support

For issues or questions:
- Check the implementation plan: [`gentle_phoneme_alignment_plan.md`](gentle_phoneme_alignment_plan.md)
- Run setup test: `python scripts/test_gentle_setup.py`
- Review debug logs with `--log-level DEBUG`

## Future Enhancements

Planned improvements:
- Parallel processing for faster throughput
- Progress checkpointing for resume capability
- Batch processing with configurable batch sizes
- Quality filtering based on alignment confidence
- Integration with data pipeline