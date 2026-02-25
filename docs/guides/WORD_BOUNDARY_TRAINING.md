# Word Boundary Training for Parakeet CTC

This guide explains how the training pipeline uses word boundary markers to enable the model to learn both phoneme sequences and word segmentation simultaneously.

## Overview

**Problem:** Standard phoneme-only models predict phoneme sequences but don't indicate where words begin and end.

**Solution:** Add a special `|` (pipe) token to the vocabulary that marks word boundaries. The model learns to predict this token between words, enabling:
- Phoneme-level transcription
- Word-level segmentation
- Accurate word boundaries for downstream tasks
- Better forced alignment for word-level timestamps

## How It Works

### 1. Data Format

**Original dataset format:**
```python
phonemes_ipa: ["ð ɪ s", "ɪ z", "ə", "t ɛ s t"]
# Words are separate list elements, phonemes within words are space-separated
```

**Converted training format:**
```
"ð ɪ s | ɪ z | ə | t ɛ s t"
# Pipe (|) explicitly marks word boundaries
```

### 2. Vocabulary Structure

The vocabulary includes:
```
<blank>   # Index 0 - CTC blank token (required)
<unk>     # Index 1 - Unknown phoneme
|         # Index 2 - Word boundary marker ← NEW!
ɐ         # Index 3 - First IPA phoneme
ɑ         # Index 4 - Second IPA phoneme
...       # Remaining phonemes (sorted alphabetically)
```

### 3. Training Behavior

During training, the CTC model learns:
- **Phoneme tokens:** Predict individual phonemes (ð, ɪ, s, etc.)
- **Word boundaries:** Predict `|` when transitioning between words
- **Blank tokens:** CTC blank for alignment flexibility

**Example prediction sequence:**
```
Audio:     [this is a test]
Target:    ð ɪ s | ɪ z | ə | t ɛ s t
Predicted: ð_ɪ_s_|_ɪ_z_|_ə_|_t_ɛ_s_t  (underscores represent blanks)
```

### 4. Inference & Timestamp Extraction

After training, the model outputs:

**Frame-level predictions:**
```
Frame 0-10:   ð
Frame 11-12:  <blank>
Frame 13-25:  ɪ
Frame 26-27:  <blank>
Frame 28-40:  s
Frame 41-45:  |        ← Word boundary detected!
Frame 46-58:  ɪ
...
```

**Post-processing extracts:**
1. **Phoneme-level timestamps** - Start/end time for each phoneme
2. **Word-level timestamps** - Group phonemes by `|` boundaries
3. **Word segmentation** - Reconstruct words from phoneme sequences

## Modified Scripts

### 1. Vocabulary Building ([`02_build_phoneme_vocabulary.py`](../../scripts/training/02_build_phoneme_vocabulary.py))

**Changes:**
```python
# OLD
vocabulary = ["<blank>", "<unk>", " ", ...phonemes]

# NEW  
vocabulary = ["<blank>", "<unk>", "|", ...phonemes]
```

**Result:** Word boundary token `|` at index 2, ready for training

### 2. Manifest Creation ([`03_create_ctc_manifests.py`](../../scripts/training/03_create_ctc_manifests.py))

**Changes:**
```python
# OLD - Concatenate word phonemes
if isinstance(phonemes, list):
    text = " ".join(str(p) for p in phonemes if p)
    # Result: "ð ɪ s ɪ z ə t ɛ s t"

# NEW - Insert word boundaries
if isinstance(phonemes, list):
    text = " | ".join(str(p) for p in phonemes if p)
    # Result: "ð ɪ s | ɪ z | ə | t ɛ s t"
```

**Result:** Training manifests now include explicit word boundaries

### 3. Timestamp Extraction ([`07_extract_phoneme_timestamps.py`](../../scripts/training/07_extract_phoneme_timestamps.py))

**New function:**
```python
def extract_words_from_phonemes(alignments: List[Dict]) -> List[Dict]:
    """Group phoneme alignments into word boundaries using | markers."""
    # Finds all | tokens and groups phonemes between them into words
    # Returns word-level timestamps with constituent phonemes
```

**Output formats:**

**JSON output:**
```json
{
  "audio_filepath": "sample.wav",
  "phonemes": [
    {"phoneme": "ð", "start": 0.00, "end": 0.05, "duration": 0.05},
    {"phoneme": "ɪ", "start": 0.05, "end": 0.10, "duration": 0.05},
    {"phoneme": "s", "start": 0.10, "end": 0.20, "duration": 0.10},
    {"phoneme": "|", "start": 0.20, "end": 0.22, "duration": 0.02},
    {"phoneme": "ɪ", "start": 0.22, "end": 0.27, "duration": 0.05},
    {"phoneme": "z", "start": 0.27, "end": 0.35, "duration": 0.08}
  ],
  "words": [
    {
      "text": "ð ɪ s",
      "start": 0.00,
      "end": 0.20,
      "duration": 0.20,
      "phonemes": [...]
    },
    {
      "text": "ɪ z", 
      "start": 0.22,
      "end": 0.35,
      "duration": 0.13,
      "phonemes": [...]
    }
  ],
  "num_phonemes": 6,
  "num_words": 2
}
```

**Praat TextGrid output:**
```
TextGrid with 2 tiers:
  Tier 1: "phonemes" - Individual phoneme boundaries
  Tier 2: "words"    - Word-level boundaries ← NEW!
```

## Benefits

### 1. **Improved Forced Alignment**
- Explicit word boundaries improve alignment accuracy
- No need for separate word segmentation step
- Direct integration with NeMo's CTC aligner

### 2. **Better Word-Level Timestamps**
- Accurate word start/end times from model predictions
- Phoneme-to-word mapping maintained
- Useful for subtitle generation, karaoke, speech synthesis

### 3. **Enhanced Error Analysis**
- Separate word error rate (WER) from phoneme error rate (PER)
- Identify word boundary prediction errors
- Improve model training with boundary-aware metrics

### 4. **Flexible Post-Processing**
```python
# Extract just phonemes (remove boundaries)
phonemes_only = [p for p in predictions if p != "|"]

# Extract just word boundaries
word_boundaries = [i for i, p in enumerate(predictions) if p == "|"]

# Group into words
words = " | ".join(predictions).split(" | ")
```

## Training Example

**Input data (manifest entry):**
```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "h ɛ l oʊ | w ɜ ɹ l d",
  "duration": 1.5
}
```

**Model learns:**
- `h`, `ɛ`, `l`, `oʊ` → Part of "hello"
- `|` → Word boundary
- `w`, `ɜ`, `ɹ`, `l`, `d` → Part of "world"

**Inference output:**
```python
{
  "phonemes": ["h", "ɛ", "l", "oʊ", "|", "w", "ɜ", "ɹ", "l", "d"],
  "words": [
    {"text": "h ɛ l oʊ", "start": 0.0, "end": 0.7},
    {"text": "w ɜ ɹ l d", "start": 0.8, "end": 1.5}
  ]
}
```

## Comparison: With vs Without Word Boundaries

| Feature | Without Boundaries | With Boundaries |
|---------|-------------------|-----------------|
| Output | `ð ɪ s ɪ z ə t ɛ s t` | `ð ɪ s \| ɪ z \| ə \| t ɛ s t` |
| Word segmentation | Post-processing required | Built into model |
| Word timestamps | Approximate/heuristic | Direct from model |
| Forced alignment | Requires separate tool | Integrated |
| Training complexity | Simpler | Slightly more complex |
| Inference accuracy | Phoneme-level only | Phoneme + word level |

## Validation

Check word boundary predictions:
```bash
# Extract timestamps from trained model
python scripts/training/07_extract_phoneme_timestamps.py \
  --model final_model.nemo \
  --audio sample.wav \
  --output sample.json

# Inspect output
cat sample.json | jq '.words'
```

Expected output:
- Words reconstructed from phonemes
- Word-level start/end times
- Phoneme sequences within each word

## Best Practices

1. **Ensure consistent boundaries:** All datasets should have word-separated phoneme lists
2. **Validate vocabulary:** Check `|` is at correct index (typically 2)
3. **Monitor boundary accuracy:** Track word boundary prediction during training
4. **Test segmentation:** Verify words are correctly reconstructed from phonemes

## Troubleshooting

**Issue:** Model predicts too many/few word boundaries

**Solutions:**
- Increase training data with diverse word lengths
- Adjust CTC loss weighting
- Fine-tune with boundary-heavy examples

**Issue:** Word boundaries don't align with audio

**Solutions:**
- Check audio-text alignment in training data
- Verify word separations in source datasets
- Increase model capacity if needed

## Summary

Word boundary training enhances the Parakeet CTC model by:
✅ Adding explicit word segmentation capability
✅ Enabling accurate word-level timestamps
✅ Improving forced alignment quality
✅ Maintaining phoneme-level precision
✅ Simplifying post-processing pipelines

The model now outputs both phoneme sequences and word boundaries in a single inference pass, providing a complete phonetic-prosodic representation of speech.