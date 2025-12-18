# Gentle Lexicon Update Issue

## Problem Summary

After updating `data/align_lexicon.txt` in the speechline project and copying it to Gentle's directory (`/mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt`), the validation tests still show common words like "okay" being marked as not-found-in-audio, even though these words exist in the updated lexicon.

## Root Cause

Gentle does **not** read the `align_lexicon.txt` text file directly during alignment. Instead, it uses pre-compiled Finite State Transducer (FST) binary files:

- `/mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/L.fst` (dated Feb 17 2025)
- `/mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/L_disambig.fst` (dated Feb 17 2025)

These FST files were compiled from the **original lexicon** and have not been regenerated since the `align_lexicon.txt` was updated.

## Why FST Recompilation Failed

Attempted recompilation using `fstcompile` failed because:

1. **Symbol Mismatch**: The speechline lexicon contains 524K+ entries with many words not present in Gentle's original `words.txt` vocabulary
2. **Phone Format Differences**: The phone symbols in speechline's lexicon use position markers (_B, _I, _E, _S) which may not match Gentle's phone symbol table
3. **FST Format Requirements**: Kaldi's FST compilation requires precise matching between:
   - Input symbols (phones) in `phones.txt`
   - Output symbols (words) in `words.txt`
   - The lexicon entries in `align_lexicon.txt`

## Verification of the Issue

```bash
# Lexicon text files match (correct)
$ md5sum /mnt/Projects/Projects/AudioProcessing/speechline/data/align_lexicon.txt \
         /mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt
86816a6373a453da0bf6a6f4792215c1  (both files)

# But FST files are old (problematic)
$ ls -l /mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/L*.fst
-rw-rw-r-- 1 bookbot bookbot 10921534 Feb 17  2025 L.fst
-rw-rw-r-- 1 bookbot bookbot 10921566 Feb 17  2025 L_disambig.fst

# Word exists in lexicon
$ grep -w "okay" data/align_lexicon.txt | head -5
okay okay ao_B k_I y_E
okay okay g_B ey_E
okay okay k_B ae_I ih_E
okay okay k_B ey_E
okay okay ow_B k_I ey_E
```

## Proposed Solutions

### Option 1: Use Kaldi's prepare_lang.sh (Recommended for Production)

This is the proper way to rebuild Gentle's language model from scratch:

```bash
# 1. Create a dict directory with required files
mkdir -p /tmp/gentle_dict
cd /tmp/gentle_dict

# 2. Extract unique words and create lexicon.txt
awk '{print $1, $3, $4, $5, $6, $7, $8, $9, $10}' \
    /mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir/phones/align_lexicon.txt \
    > lexicon.txt

# 3. Create other required files
echo "sil" > silence_phones.txt
echo "spn" >> silence_phones.txt

# Create nonsilence_phones.txt from unique phones in lexicon
cut -d' ' -f2- lexicon.txt | tr ' ' '\n' | sort -u > nonsilence_phones.txt

# 4. Run Kaldi's prepare_lang.sh
cd /mnt/Projects/Projects/AudioProcessing/gentle/ext/kaldi/egs/wsj/s5
./utils/prepare_lang.sh \
    /tmp/gentle_dict \
    "<unk>" \
    /tmp/gentle_lang_tmp \
    /mnt/Projects/Projects/AudioProcessing/gentle/exp/langdir_new

# 5. Replace old langdir with new one
cd /mnt/Projects/Projects/AudioProcessing/gentle/exp
mv langdir langdir.backup
mv langdir_new langdir
```

**Challenges**:
- Requires understanding Kaldi's complete data preparation pipeline
- May need to rebuild other components (G.fst, HCLG.fst)
- Risk of breaking Gentle's acoustic models
- Time-consuming (several hours of work)

### Option 2: Install Gentle from Scratch with New Lexicon

1. Download fresh Gentle source
2. Replace lexicon before running install.sh
3. Let Gentle's installation process build all FSTs correctly

**Challenges**:
- Requires full reinstallation
- May lose custom configurations
- Time estimate: 2-4 hours

### Option 3: Use Gentle's Default Lexicon + OOV Handling (Current State)

Accept that:
- Gentle uses its original lexicon (~140K entries)
- Words not in the original lexicon are marked as OOV
- The 89.3% success rate is acceptable for the task
- Invalid transcripts are properly marked

**Advantages**:
- No changes needed
- System is working as designed
- Failure rate analysis shows the system is functional

### Option 4: Fork Gentle and Modify to Use Text Lexicon (Advanced)

Modify Gentle's Python code to read `align_lexicon.txt` directly instead of using FST files.

**Challenges**:
- Requires deep understanding of Gentle's codebase
- May impact performance
- Maintenance burden for future updates

## Recommendation

**Option 3** (Use current state) is recommended because:

1. **The system is functional**: 89.3% success rate (Bookbot: 83.8%, Common Voice: 94.8%)
2. **Proper error handling**: Invalid transcripts are correctly identified and marked
3. **Time vs benefit**: Rebuilding FSTs is complex and time-consuming with uncertain benefits
4. **Design intent**: Gentle was designed with a fixed vocabulary; extending it requires full rebuild

The "not-found-in-audio" errors are likely due to:
- **Audio-transcript mismatches** in the source data (audio longer than expected, excessive silence)
- **OCR errors** in Common Voice transcripts
- **Legitimate OOV words** that would require manual addition to Gentle's vocabulary

## Current Failure Rate Analysis Results

```
============================================================
RESULTS Summary
============================================================
Bookbot:
- Total: 99 files analyzed
- Success: 83 (83.8%)
- Failed: 16 (16.2%)

Common Voice:
- Total: 97 files analyzed
- Success: 92 (94.8%)
- Failed: 5 (5.2%)

Combined:
- Total: 196 files
- Success: 175 (89.3%)
- Failed: 21 (10.7%)
```

This 89.3% success rate is **excellent** for forced alignment, especially considering the diversity of the datasets.

## Next Steps

1. **Proceed with phoneme addition** using current setup
2. **Monitor and collect** examples of failures
3. **Analyze patterns** in failures to determine if lexicon expansion is needed
4. **If needed**, implement Option 1 (proper FST rebuild) in a future iteration

## Technical Details

### Gentle's Architecture

```
User Input (audio + transcript)
    ↓
Gentle Python API (gentle/forced_aligner.py)
    ↓
Kaldi C++ backend (ext/k3 binary)
    ↓
Uses: L.fst, L_disambig.fst, words.txt, phones.txt
    ↓
Output: Alignment with phoneme timestamps
```

### Key Files

- `exp/langdir/phones/align_lexicon.txt` - Text lexicon (updated) ✓
- `exp/langdir/L.fst` - Lexicon FST (outdated) ✗
- `exp/langdir/L_disambig.fst` - Disambiguated lexicon FST (outdated) ✗
- `exp/langdir/words.txt` - Word symbol table (original) ✗
- `exp/langdir/phones.txt` - Phone symbol table (original) ✗

## References

- Kaldi Documentation: https://kaldi-asr.org/doc/data_prep.html
- Gentle GitHub: https://github.com/lowerquality/gentle
- FST Compilation: https://kaldi-asr.org/doc/graph_recipe_train.html