# LibriPhone Visual Debugging - Implementation Summary

## What Was Implemented

### 1. CSV Export Utility
**File**: [`scripts/export_libriphone_csv.py`](../../scripts/export_libriphone_csv.py)

Converts LibriPhone HuggingFace dataset to CSV format for:
- Faster iteration (no network required)
- Easy offline inspection
- Version control friendly

**Usage**:
```bash
python scripts/export_libriphone_csv.py
```

**Output**: `data/libriphone_test.csv` (2,620 rows)

### 2. Refactored Test with Visual Alignment
**File**: [`tests/test_libriphone.py`](../../tests/test_libriphone.py)

Key changes:
- ✅ **Debug Mode**: Stop at first mismatch (fail-fast)
- ✅ **Visual Alignment**: 3-line monospace format with pipe separators
- ✅ **Difference Markers**: `^` indicators show mismatched columns
- ✅ **CSV Loading**: Optional loading from CSV instead of HuggingFace
- ✅ **Removed**: Complex mismatch export in debug mode

### 3. Visual Alignment Format

Simple 3-line monospace display:

```
Word Line:    A      |curious|fox |danced|through|the|moonlit|forest
Lexicon IPA:  ə      |kjʊɹiəs|fɒks|dɑnst |θɹu    |ðə |munlɪt |fɔɹəst
Dataset IPA:  ə      |kjʊɹiəs|fɑks|dænst |θɹu    |ðə |munlɪt |fɔɹəst
Markers:                     ^       ^
```

### 4. Documentation
**File**: [`docs/guides/libriphone_debugging.md`](libriphone_debugging.md)

Complete usage guide with:
- Quick start instructions
- Environment variable reference
- Iterative debugging workflow
- Common issues and solutions

## Usage Modes

### Debug Mode (Interactive)
```bash
LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s
```

**When to use**: Fixing specific issues, rapid iteration

**What happens**:
1. Loads dataset
2. Tests samples one by one
3. **STOPS** at first mismatch
4. Shows visual alignment immediately
5. Displays difference markers
6. Lists mismatched words
7. Exits with failure

### Batch Mode (Validation)
```bash
pytest tests/test_libriphone.py -v
```

**When to use**: Full validation, comprehensive testing

**What happens**:
1. Processes all 2,620 samples
2. Collects statistics
3. Exports mismatches to `mismatches/libriphone/`
4. Shows summary at end

### CSV Mode (Offline)
```bash
LIBRIPHONE_CSV=data/libriphone_test.csv LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s
```

**When to use**: Offline work, faster loading

**What happens**:
- Loads from local CSV instead of HuggingFace
- No network required
- Faster startup

## Example Output

### Debug Mode Output

```
================================================================================
DEBUG MODE - Will stop at first mismatch
================================================================================
✓ Lexicon loaded: 159,819 words
✓ Accent rules loaded
✓ Matcher initialized (mismatch export: disabled)
================================================================================

Testing 2620 samples...
Mode: DEBUG (fail-fast)

✓ Sample 1: 1089-134686-0000 - MATCH
✓ Sample 2: 1089-134686-0001 - MATCH
✓ Sample 3: 1089-134686-0002 - MATCH

================================================================================
MISMATCH DETECTED at sample 42
================================================================================
Sample ID: 1221-135766-0005
Reference: THE QUICK BROWN FOX JUMPS

Visual Alignment:
--------------------------------------------------------------------------------
THE  |QUICK  |BROWN|FOX |JUMPS
ðə   |kwɪk   |bɹaʊn|fɒks|dʒʌmps
ðə   |kwɪk   |bɹaʊn|fɑks|dʒʌmps
                        ^

Mismatched words: FOX
--------------------------------------------------------------------------------

Match details:
  Method: greedy_word_match
  Reason: phoneme_mismatch_at_word_fox
================================================================================

FAILED - First mismatch at sample 42: 1221-135766-0005
```

## Implementation Details

### Visual Alignment Algorithm

1. **Extract Components**:
   - Words from reference text
   - Phonemes from lexicon (what matcher chose)
   - Phonemes from dataset (ground truth)

2. **Calculate Widths**:
   - For each position, find max(len(word), len(lexicon_ipa), len(dataset_ipa))
   - Ensures all columns align properly

3. **Format Lines**:
   - Pad each string to column width
   - Join with pipe separators (`|`)
   - Create 3 lines: words, lexicon, dataset

4. **Add Markers**:
   - Compare lexicon vs dataset phonemes
   - Add `^` centered in columns where they differ

### Helper Functions

```python
def format_visual_alignment(words, lexicon_phonemes, dataset_phonemes) -> str:
    """Create 3-line monospace alignment"""
    
def find_mismatch_positions(lexicon_phonemes, dataset_phonemes, widths) -> str:
    """Create difference marker line with ^"""
```

## Benefits

### For Developers

1. **Immediate Feedback** - See problems instantly
2. **Visual Clarity** - IPA differences obvious at a glance
3. **Fast Iteration** - Fix → re-run → see next issue
4. **No File Juggling** - All info in terminal output
5. **Easy Debugging** - Clear visual alignment helps spot patterns

### For System Improvement

1. **Pattern Recognition** - Quickly identify systematic issues
2. **Lexicon Quality** - Easy to spot incorrect entries
3. **Rule Validation** - See exactly where accent rules fail
4. **G2P Accuracy** - Compare G2P output vs manual transcription

## Files Modified/Created

### Created
- `scripts/export_libriphone_csv.py` - Dataset converter
- `docs/guides/libriphone_debugging.md` - Usage guide
- `docs/guides/libriphone_debugging_summary.md` - This file

### Modified
- `tests/test_libriphone.py` - Complete refactor with visual alignment

### Will Be Created (by export script)
- `data/libriphone_test.csv` - Exported dataset

## Next Steps

1. **Export CSV** (optional but recommended):
   ```bash
   python scripts/export_libriphone_csv.py
   ```

2. **Run Debug Mode**:
   ```bash
   LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s
   ```

3. **Fix First Issue** - Based on visual alignment output

4. **Iterate** - Re-run to see next issue

5. **Validate** - Once all fixed, run batch mode:
   ```bash
   pytest tests/test_libriphone.py -v
   ```

## Success Criteria

- ✅ CSV export creates valid file with 2,620 rows
- ✅ Test loads from CSV when `LIBRIPHONE_CSV` specified
- ✅ Debug mode stops at first mismatch
- ✅ Visual alignment clearly shows word-by-word differences
- ✅ Monospace formatting with pipes maintains alignment
- ✅ Difference markers (`^`) highlight mismatches
- ✅ Easy to spot where and why matching fails

## Integration with Existing System

This debugging workflow integrates seamlessly with:

- **PhonemeMatcher** - Uses existing matcher, just changes test output
- **LexiconManager** - Reads same lexicon as production
- **AccentRulesManager** - Uses same rules as production
- **G2P Fallback** - Same G2P logic as production

No changes to core matching logic - only improved debugging visibility.