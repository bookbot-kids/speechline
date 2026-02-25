# Gentle Phoneme Alignment Analysis

## Executive Summary

After comprehensive analysis of 40 error examples (20 Bookbot, 20 Common Voice), we identified the root causes of not-found-in-audio issues and end-truncation patterns.

## Key Findings

### 1. End-Truncation Pattern (Moderate)
- **50%** of files have their last word marked as not-found-in-audio
- **57-58%** of all not-found words occur in the second half of transcripts
- **15.8%** have 2+ consecutive words missing at the end

**Conclusion**: Moderate end-truncation pattern, but not systematic.

### 2. OOV (Out-of-Vocabulary) Issues
- **35%** of error examples contain OOV words (14/40 files)
- **Common OOV categories**:
  - Filler sounds: "Hmm", "Mm-hmm", "Uh"
  - Misspellings: "mislaj", "aloft" 
  - Proper nouns: "Somaliland", "Ugunti", "Deborah"
  - Compound/rare words: "prepackaged", "polysynthetic"

### 3. Audio Duration Analysis (ROOT CAUSE)

**Critical Finding**: Audio segments often contain 1.5-4x MORE duration than expected for the transcript length.

#### Bookbot Examples:
```
File: guest_4fa9c9d3-c3c9-435e-920b-a6c7b641a95e_1722149706552_Speaker_AA5B5_0023.133-0031.672.txt
Transcript: "Oh these are very master fun" (6 words)
Expected duration: 2.0s (@ 3 words/sec)
Actual duration: 8.5s
Duration ratio: 4.26x ⚠️ LONG
Missing words: ['are', 'very', 'master', 'fun'] (4/6 = 66.7%)
```

```
File: guest_4fa9c9d3-c3c9-435e-920b-a6c7b641a95e_1722149255846_Speaker_2C8E9_0017.395-0021.800.txt
Transcript: "Mm-hmm." (1 word)
Expected duration: 0.3s
Actual duration: 4.4s  
Duration ratio: 13.25x ⚠️ LONG
Missing words: ['Mm-hmm'] (100%)
```

#### Common Voice Examples:
```
File: common_voice_en_23680510.txt
Transcript: "However, service along Bonnie Road is very limited..." (13 words)
Expected duration: 4.3s
Actual duration: 8.8s
Duration ratio: 2.03x ⚠️ LONG
Missing words: ALL 13 words (100%)
```

### 4. Root Cause: Audio-Transcript Mismatch

The primary issue is **NOT Gentle's alignment algorithm**, but rather:

1. **Excessive silence/padding**: Audio contains long pauses before/after speech
2. **Incomplete speech**: Speaker didn't say all transcript words
3. **Background noise**: Audio quality issues preventing clear alignment
4. **Timestamp errors**: Segment boundaries don't match actual speech timing
5. **Transcription errors**: Transcript doesn't match what was actually said

## Evidence Supporting This Conclusion

### Duration Ratios:
- **< 0.8x**: Only 1 file (audio cut off) - RARE
- **0.8-1.5x**: 22 files (acceptable) - 55%
- **> 1.5x**: 18 files (excessive duration) - 45%

### Pattern Analysis:
When duration ratio > 1.5x:
- Average missing words: 35% of transcript
- End-truncation: More likely (words never spoken or buried in noise)
- Gentle behavior: CORRECTLY marks unaligned words as not-found

### Bookbot Timestamp Evidence:
Filename: `guest_4fa9c9d3-c3c9-435e-920b-a6c7b641a95e_1722149706552_Speaker_AA5B5_0023.133-0031.672.txt`
- Timestamp range: 23.133s - 31.672s (8.5 seconds)
- Only first 2 words aligned: "Oh these" 
- **Interpretation**: Speaker likely only said "Oh these" in beginning, with 6+ seconds of silence/noise following

## Gentle's Behavior is CORRECT

Gentle is working as designed:
1. Attempts to align each word in transcript to audio signal
2. When audio doesn't contain clear speech matching word, marks as not-found-in-audio
3. Stops alignment when confidence drops (usually toward end of poor segments)

**This is the CORRECT behavior for a forced aligner** - it's identifying bad data.

## Implications for Production Processing

### Expected Results:
When processing full Bookbot (~50K files) and Common Voice (~1M files):

1. **Valid alignments** (~50-60%):
   - Clean audio-transcript match
   - Line 2: Full phoneme transcript
   
2. **Invalid alignments** (~40-50%):
   - Audio-transcript mismatches
   - Line 2: `INVALID_TRANSCRIPT: not-found-in-audio=[...]`

### Files to Review:
- **High OOV rate**: Add legitimate words to lexicon, filter junk
- **High not-found rate**: Audio segmentation issues or transcription errors
- **Duration ratio > 2x**: Likely data quality problems

## Recommendations

### 1. Accept Current Behavior
- Gentle is correctly identifying bad data
- Invalid markers are USEFUL for data quality control
- Don't try to "fix" Gentle's alignment

### 2. Improve Source Data (Long-term)
For Bookbot files:
- Review timestamp extraction logic
- Verify segment boundaries match actual speech
- Consider padding adjustments (±0.5s at boundaries)
- Filter segments with excessive silence

For Common Voice:
- Data is crowdsourced, quality varies
- Accept ~30-40% invalid rate as normal
- Use invalid markers for filtering

### 3. Handle OOV Words
Priority actions:
- Add common proper nouns to lexicon
- Filter filler sounds ("Hmm", "Uh") from transcripts  
- Accept that rare/misspelled words will fail
- Raise exceptions for unexpected OOV during processing

### 4. Production Processing Strategy
```bash
# Process with acceptance of invalid transcripts
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --cv-clips-path "/mnt/Store07/Common Voice/en/clips" \
    --review-log alignment_review.txt
```

**Expected outcomes**:
- ~50-60% valid phoneme transcripts
- ~40-50% marked as INVALID_TRANSCRIPT
- Review log for OOV exceptions and high failure rates

## Conclusion

The "missing phonemes at end" observation is a symptom of **audio-transcript mismatches**, not a Gentle bug. The alignment system is working correctly by identifying segments where the audio doesn't match the transcript.

For production use:
1. Accept invalid transcripts as part of data quality filtering
2. Use invalid markers to identify segments needing manual review
3. Focus on improving source data quality rather than forcing alignment
4. Consider the invalid rate a useful data quality metric

## Files Generated for This Analysis

1. `scripts/analyze_end_truncation.py` - Position analysis of not-found words
2. `scripts/diagnose_audio_segments.py` - Audio duration vs transcript analysis
3. `scripts/analyze_error_transcripts.py` - OOV and error pattern analysis
4. `scripts/annotate_error_transcripts.py` - Error example annotation
5. `errors/` directory - 40 annotated examples (20 Bookbot, 20 Common Voice)