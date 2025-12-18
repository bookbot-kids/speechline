# Gentle Lexicon Update Guide

## Overview

This guide provides a comprehensive procedure for updating the Gentle force aligner lexicon with the expanded speechline lexicon database. The speechline lexicon contains 524K+ entries from multiple sources, providing significantly better coverage than Gentle's original 140K entry lexicon.

## Background

Gentle uses two key locations for lexicon files:
1. **`exp/langdir/phones/`** - Main lexicon location with FST files
2. **`exp/tdnn_7b_chain_online/graph_pp/phones/`** - Graph directory for decoding

Additionally, Gentle uses pre-compiled Finite State Transducer (FST) binary files:
- `L.fst` - Lexicon FST
- `L_disambig.fst` - Disambiguated lexicon FST

Simply updating the text files (`align_lexicon.txt`) is **not sufficient** - the FST files must also be regenerated for changes to take effect.

## Prerequisites

- Gentle installed at: `/mnt/Projects/Projects/AudioProcessing/gentle`
- Speechline lexicon at: `data/align_lexicon.txt`
- Python 3.7+
- Kaldi tools (included with Gentle)
- Bash shell

## Update Procedure

### Step 1: Validate Current State

First, check the current state of the lexicons and identify missing words:

```bash
python scripts/validate_gentle_lexicon.py
```

This script will:
- Check if lexicons are synchronized across both Gentle locations
- Compare MD5 hashes of all lexicon files
- Analyze vocabulary coverage
- Find words from error transcripts that are missing
- Report which words would be added by the update

**Expected Output:**
```
GENTLE LEXICON VALIDATION
======================================================================

1. CHECKING LEXICON FILE SYNC
----------------------------------------------------------------------
Speechline lexicon: data/align_lexicon.txt
  MD5: 86816a6373a453da0bf6a6f4792215c1
  Size: 22.3 MB

Gentle lexicon: exp/langdir/phones/align_lexicon.txt
  MD5: [some_hash]
  Status: ✗ DIFFERENT or ✓ MATCH

...

2. VOCABULARY COVERAGE ANALYSIS
----------------------------------------------------------------------
Speechline lexicon: 387,459 unique words
Gentle langdir: 140,000 unique words (example)

3. ANALYZING ERROR TRANSCRIPTS
----------------------------------------------------------------------
Found 15 unique words marked as NOT_FOUND_IN_AUDIO
Words in Speechline lexicon: 12 (80.0%)
...
```

### Step 2: Test Run (Dry Run)

Before making any changes, do a dry run to see what would happen:

```bash
python scripts/update_gentle_lexicon.py --dry-run
```

This will show all operations without actually performing them.

### Step 3: Execute Update

When ready, run the actual update:

```bash
python scripts/update_gentle_lexicon.py
```

**This script will:**

1. **Create backups** of all existing files:
   - `align_lexicon.txt.backup.[timestamp]`
   - `align_lexicon.int.backup.[timestamp]`
   - `L.fst.backup.[timestamp]`
   - `L_disambig.fst.backup.[timestamp]`

2. **Copy speechline lexicon** to both Gentle locations:
   - `exp/langdir/phones/align_lexicon.txt`
   - `exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt`

3. **Regenerate .int files** using symbol tables:
   - Converts text lexicon to integer format using `words.txt` and `phones.txt`
   - Handles words not in original symbol tables

4. **Rebuild FST files**:
   - Executes `scripts/rebuild_gentle_lexicon.sh`
   - Compiles new L.fst and L_disambig.fst from updated lexicon

5. **Validate** the update:
   - Confirms all files exist
   - Checks file integrity

### Step 4: Verify Update

After the update completes, verify the changes:

```bash
python scripts/validate_gentle_lexicon.py
```

Expected output should now show:
```
✓ All lexicon files are IN SYNC
✓ Speechline lexicon is deployed to all Gentle locations
```

### Step 5: Test Alignment

Test with sample files to ensure alignment still works:

```bash
python scripts/test_gentle_setup.py
```

Test with one of the previous error files:

```bash
# Example using error file
python scripts/gentle_phoneme_alignment.py \
    --mode validate \
    --cv-clips-path "/mnt/Store07/Common Voice/en/clips" \
    --examples-per-dataset 5
```

Check if previously failing words are now found.

## Rollback Procedure

If the update causes issues, rollback using the backups:

```bash
cd /mnt/Projects/Projects/AudioProcessing/gentle

# Find backup timestamp
ls -lt exp/langdir/*.backup.* | head

# Set timestamp (use actual timestamp from ls output)
TIMESTAMP=20250227_143022

# Restore langdir files
cp exp/langdir/L.fst.backup.$TIMESTAMP exp/langdir/L.fst
cp exp/langdir/L_disambig.fst.backup.$TIMESTAMP exp/langdir/L_disambig.fst
cp exp/langdir/phones/align_lexicon.txt.backup.$TIMESTAMP exp/langdir/phones/align_lexicon.txt
cp exp/langdir/phones/align_lexicon.int.backup.$TIMESTAMP exp/langdir/phones/align_lexicon.int

# Restore graph_pp files
cp exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt.backup.$TIMESTAMP \
   exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt
cp exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.int.backup.$TIMESTAMP \
   exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.int
```

## Understanding the Update Process

### Why Both Locations?

Gentle uses two lexicon locations for different purposes:

1. **`exp/langdir/`**: Main language model directory
   - Used during alignment for word-level operations
   - Contains FST files for efficient lookup

2. **`exp/tdnn_7b_chain_online/graph_pp/`**: Decoding graph
   - Used by the neural network decoder
   - Must match langdir for consistency

### Why FST Files Matter

FST (Finite State Transducer) files are binary representations of the lexicon that provide:
- **Fast lookup**: O(1) phone-to-word mapping
- **Disambiguation**: Handles pronunciation variants
- **Integration**: Works with Kaldi's acoustic models

Simply updating text files won't work because Gentle's C++ backend reads the FST files directly.

### Symbol Table Matching

The `.int` files use integer IDs from:
- `words.txt`: Maps words to integer IDs
- `phones.txt`: Maps phones to integer IDs

**Critical**: If a word or phone isn't in these files, it will be skipped during regeneration. The speechline lexicon uses the same phone set as Gentle, but includes many additional words.

## Expected Results

After updating the lexicon, you should see:

### Improved Coverage
- **Before**: ~140K words in vocabulary
- **After**: ~387K words in vocabulary
- **Missing word rate**: Should decrease significantly

### Better Alignment Success
From the documentation, the system was at:
- Bookbot: 83.8% success rate
- Common Voice: 94.8% success rate
- Combined: 89.3% success rate

With the updated lexicon, expect:
- Fewer "not-found-in-audio" errors for legitimate words
- Better handling of technical terms, proper nouns
- Improved phoneme alignment accuracy

### Words Now Covered

Examples of words that should now be found:
- Technical terms (from WordUniversal dictionary)
- Proper nouns (from Common Voice G2P)
- Compound words (from MFA lexicons)
- Specialized vocabulary (from various sources)

## Troubleshooting

### Issue: Symbol mismatch warnings during .int regeneration

**Cause**: Words in speechline lexicon not in Gentle's `words.txt`

**Solution**: This is expected and handled automatically. Words not in the symbol table are skipped but this typically affects <1% of entries.

### Issue: FST compilation fails

**Cause**: OpenFST tools not in PATH or incompatible lexicon format

**Solution**: 
1. Check OpenFST installation:
   ```bash
   which fstcompile
   ```
2. Verify lexicon format (should match CMU ARPAbet with position markers)
3. Check error logs in console output

### Issue: Alignment worse after update

**Cause**: Possible FST corruption or incorrect regeneration

**Solution**: Rollback using backup files and investigate the issue

### Issue: Memory errors during alignment

**Cause**: Larger lexicon requires more memory

**Solution**: 
- Increase available RAM
- Process smaller batches
- Use the `--max-pron` option when exporting the lexicon to reduce size

## Maintenance

### When to Update

Update the lexicon when:
- New words are consistently appearing in error logs
- Coverage analysis shows significant missing vocabulary
- After rebuilding the speechline lexicon database

### Regular Checks

Periodically run:
```bash
python scripts/validate_gentle_lexicon.py
```

To verify:
- Lexicons remain synchronized
- No corruption has occurred
- Coverage remains adequate

## File Locations Reference

```
/mnt/Projects/Projects/AudioProcessing/
├── speechline/
│   ├── data/
│   │   └── align_lexicon.txt          # Source lexicon (22.3 MB)
│   ├── scripts/
│   │   ├── validate_gentle_lexicon.py # Validation tool
│   │   ├── update_gentle_lexicon.py   # Update tool
│   │   └── rebuild_gentle_lexicon.sh  # FST rebuild script
│   └── docs/
│       └── GENTLE_LEXICON_UPDATE_GUIDE.md  # This file
│
└── gentle/
    └── exp/
        ├── langdir/
        │   ├── L.fst                   # Lexicon FST (binary)
        │   ├── L_disambig.fst          # Disambiguated FST (binary)
        │   ├── words.txt               # Word symbol table
        │   ├── phones.txt              # Phone symbol table
        │   └── phones/
        │       ├── align_lexicon.txt   # Text lexicon
        │       └── align_lexicon.int   # Integer lexicon
        │
        └── tdnn_7b_chain_online/
            └── graph_pp/
                ├── words.txt           # Word symbol table
                ├── phones.txt          # Phone symbol table
                └── phones/
                    ├── align_lexicon.txt  # Text lexicon
                    └── align_lexicon.int  # Integer lexicon
```

## Related Documentation

- [`GENTLE_LEXICON_UPDATE_ISSUE.md`](GENTLE_LEXICON_UPDATE_ISSUE.md) - Background on why this update is needed
- [`GENTLE_EXPORT_README.md`](../data/GENTLE_EXPORT_README.md) - How the lexicon was created
- [`LEXICON_DB_README.md`](../data/LEXICON_DB_README.md) - Lexicon database documentation
- [`gentle_phoneme_alignment_usage.md`](guides/gentle_phoneme_alignment_usage.md) - Using Gentle alignment

## Quick Reference

```bash
# 1. Check current state
python scripts/validate_gentle_lexicon.py

# 2. Test update (no changes)
python scripts/update_gentle_lexicon.py --dry-run

# 3. Execute update
python scripts/update_gentle_lexicon.py

# 4. Verify update
python scripts/validate_gentle_lexicon.py

# 5. Test alignment
python scripts/test_gentle_setup.py

# 6. Rollback if needed
# (Use backup files created during update)
```

## Support

For issues or questions:
1. Check the error logs generated during update
2. Review the troubleshooting section
3. Verify file permissions and paths
4. Check system resources (memory, disk space)

## Version History

- **2025-10-27**: Initial documentation created
  - Added validation script
  - Added update script
  - Documented complete procedure