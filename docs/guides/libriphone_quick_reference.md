# LibriPhone Debugging - Quick Reference

## Commands

```bash
# Export dataset to CSV (one-time setup)
python scripts/export_libriphone_csv.py

# Debug mode - stop at first mismatch with visual alignment
LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s

# Debug mode with CSV (faster, offline)
LIBRIPHONE_CSV=data/libriphone_test.csv LIBRIPHONE_DEBUG=true pytest tests/test_libriphone.py -v -s

# Batch mode - test all samples
pytest tests/test_libriphone.py -v

# Batch mode with CSV
LIBRIPHONE_CSV=data/libriphone_test.csv pytest tests/test_libriphone.py -v
```

## Visual Alignment Format

```
Word:     THE  |QUICK  |BROWN|FOX |JUMPS
Lexicon:  ðə   |kwɪk   |bɹaʊn|fɒks|dʒʌmps
Dataset:  ðə   |kwɪk   |bɹaʊn|fɑks|dʒʌmps
Diff:                         ^
```

- Line 1: Reference text words
- Line 2: Phonemes from lexicon/G2P (what matcher chose)
- Line 3: Phonemes from LibriPhone dataset (ground truth)
- Line 4: `^` markers show mismatches

## Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `LIBRIPHONE_DEBUG` | `true` | Stop at first mismatch, show visual alignment |
| `LIBRIPHONE_CSV` | path | Load from CSV instead of HuggingFace |

## Workflow

1. **Run debug mode** → See first mismatch
2. **Identify issue** → Check visual alignment
3. **Fix** → Update lexicon/rules/code
4. **Re-run** → See next mismatch
5. **Repeat** → Until 100% match rate

## Key Files

| File | Purpose |
|------|---------|
| `scripts/export_libriphone_csv.py` | Export dataset to CSV |
| `tests/test_libriphone.py` | Test with visual alignment |
| `data/libriphone_test.csv` | Exported dataset (optional) |
| `data/english_words_processed.csv` | Lexicon (159K words) |
| `speechline/matchers/phoneme_matcher.py` | Core matcher |

## Common Issues

**Lexicon Mismatch**: Word has different IPA in lexicon vs dataset
- Check `data/english_words_processed.csv`
- Update if lexicon is wrong

**G2P Fallback**: Word not in lexicon (shows `✗ OOV`)
- G2P generates phonemes automatically
- May differ from manual transcription
- Add to lexicon if needed

**Vowel Collapsing**: Alignment off by 1-2 phonemes
- Collapsed variation under-consumed
- Check vowel collapsing logic in matcher

## Tips

✅ **Use debug mode** for rapid iteration
✅ **Export to CSV** for faster offline work  
✅ **Check visual alignment** - differences are obvious
✅ **Fix one at a time** - debug mode shows next issue immediately
✅ **Validate with batch mode** once fixed

## Documentation

- **Full Guide**: [`libriphone_debugging.md`](libriphone_debugging.md)
- **Implementation**: [`libriphone_debugging_summary.md`](libriphone_debugging_summary.md)
- **This Card**: [`libriphone_quick_reference.md`](libriphone_quick_reference.md)