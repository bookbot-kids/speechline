# LibriPhone Debugging Workflow

This guide explains how to use the interactive visual debugging workflow for LibriPhone phoneme matching.

## Overview

The LibriPhone debugging workflow provides two modes:

1. **Debug Mode** - Stop at first mismatch with visual alignment (fast iteration)
2. **Batch Mode** - Process all samples and collect statistics (full validation)

## Quick Start

### Export Dataset to CSV (Optional)

For faster iteration and offline work, export LibriPhone to CSV:

```bash
python scripts/export_libriphone_csv.py
```

This creates `data/libriphone_test.csv` with 2,620 samples.

### Debug Mode - Visual Alignment

Stop at the first mismatch and see exactly what's wrong:

```bash
LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s
```

**Output Example:**
```
================================================================================
MISMATCH DETECTED at sample 42
================================================================================
Sample ID: 1089-134686-0002
Reference: AFTER A LONG TURN AT THE LAMPS

Visual Alignment:
--------------------------------------------------------------------------------
AFTER |A |LONG|TURN|AT |THE|LAMPS
æftəɹ |ə |lɔŋ |tɝn |æt |ðə |lænps
æftəɹ |ə |lɔŋ |tɝn |ə  |ðə |lænps
                        ^

Mismatched words: AT
--------------------------------------------------------------------------------

Match details:
  Method: greedy_word_match
  Reason: word_mismatch_at_pos_5
================================================================================
```

### Batch Mode - Full Validation

Process all samples and get comprehensive statistics:

```bash
pytest tests/test_libriphone.py -v
```

**Output Example:**
```
================================================================================
LIBRIPHONE TEST RESULTS
================================================================================
Total samples:    2,620
Matched:          2,580
Mismatched:       40
Match rate:       98.47%
================================================================================

Showing first 10 mismatches:
1. 1089-134686-0002: AFTER A LONG TURN AT THE LAMPS...
2. 1221-135766-0003: THE CURIOUS FOX DANCED THROUGH...
...

All mismatches exported to: mismatches/libriphone/
```

### Use CSV Instead of HuggingFace

Load from local CSV (faster, no network required):

```bash
LIBRIPHONE_CSV=data/libriphone_test.csv pytest tests/test_libriphone.py -v -s
```

## Visual Alignment Format

The 3-line monospace alignment shows:

```
Line 1: Reference text words (from dataset)
Line 2: Generated phonemes (from lexicon/G2P)
Line 3: Actual phonemes (from LibriPhone dataset)
```

**Example:**
```
A      |curious|fox |danced|through|the|moonlit|forest
ə      |kjʊɹiəs|fɒks|dɑnst |θɹu    |ðə |munlɪt |fɔɹəst
ə      |kjʊɹiəs|fɑks|dænst |θɹu    |ðə |munlɪt |fɔɹəst
                   ^       ^
```

Columns are aligned using pipes (`|`) as separators, with `^` markers indicating mismatches.

## Iterative Debugging Workflow

1. **Run debug mode** - See first mismatch with visual alignment
2. **Identify issue** - Check lexicon entry, G2P output, or accent rules
3. **Fix issue** - Update lexicon, rules, or matcher logic
4. **Re-run** - See next mismatch immediately
5. **Repeat** - Continue until all issues resolved

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `LIBRIPHONE_DEBUG` | `true`/`false` | Enable debug mode (stop at first mismatch) |
| `LIBRIPHONE_CSV` | file path | Load from CSV instead of HuggingFace |

## Tips

- **Debug Mode**: Use when fixing specific issues, fastest iteration
- **Batch Mode**: Use for validation and comprehensive testing
- **CSV Export**: Recommended for offline work and version control
- **Visual Alignment**: Makes IPA transcription differences immediately obvious

## Common Issues

### Lexicon Mismatches
If the generated phonemes don't match dataset:
- Check if word exists in `data/english_words_processed.csv`
- Verify IPA transcription is correct
- Consider if accent rules should apply

### G2P Fallbacks
If word not in lexicon (shows `✗ OOV`):
- G2P (gruut) generates phonemes automatically
- May differ from LibriPhone's manual transcription
- Consider adding to lexicon if frequently used

### Vowel Collapsing
If alignment is off by 1-2 phonemes:
- Check if vowel collapsing under-consumed phonemes
- Look for consecutive vowels in lexicon vs dataset
- See vowel collapsing logic in `PhonemeMatcher`

## Files

- **Test**: `tests/test_libriphone.py` - Main test with visual alignment
- **Export**: `scripts/export_libriphone_csv.py` - Dataset to CSV converter
- **CSV**: `data/libriphone_test.csv` - Exported dataset (optional)
- **Matcher**: `speechline/matchers/phoneme_matcher.py` - Core matching logic
- **Lexicon**: `data/english_words_processed.csv` - Word-to-IPA mappings

## Next Steps

After achieving 100% match rate on LibriPhone:
1. Run Common Voice tests: `pytest tests/test_common_voice.py`
2. Test on real audio data
3. Deploy to production