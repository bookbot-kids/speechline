# Alignment Validation Guide

This guide explains how to use Parakeet TDT's alignment validation feature to validate audio recordings against expected ground truth text.

## Overview

Alignment validation detects whether spoken audio phonetically aligns with expected text. It accepts audio where phonemes align well (including homophones and similar sounds) and rejects audio where phonemes don't align (different words or mispronunciations).

### What Gets Accepted ✅

- **Exact matches**: "cat" → "cat"
- **Homophones**: "their" → "there" (same phonemes: /ð ɛ ɹ/)
- **Close minimal pairs**: "pin" → "pen" (similar phonemes)

### What Gets Rejected ❌

- **Different words**: "cat" → "dog" (different phonemes)
- **Mispronunciations**: "three" → "free" (phoneme errors)
- **Missing words**: "the cat sat" → "cat sat" (skipped words)

## How It Works

The validation process uses three components:

1. **Parakeet TDT**: Transcribes audio (what was actually said)
2. **NeMo Forced Aligner**: Aligns ground truth to audio phonetically
3. **Confidence Scores**: Validates alignment quality per token

```
Audio: "Put there books on table"
Ground Truth: "Put their books on the table"

Alignment Results:
✅ "Put" → "Put" (confidence: 0.93)
✅ "their" → "there" (confidence: 0.91, homophone aligned)
✅ "books" → "books" (confidence: 0.89)
✅ "on" → "on" (confidence: 0.88)
❌ "the" → missing (confidence: 0.45, low confidence = no alignment)
✅ "table" → "table" (confidence: 0.87)

Result: 5/6 words aligned (83%) → ACCEPT (above 80% threshold)
```

## Quick Start

### 1. Configuration

Create a config file with validation enabled:

```json
{
    "transcriber": {
        "type": "parakeet_tdt",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "return_timestamps": "word",
        "chunk_length_s": 30,
        "transcriber_device": "cuda",
        "validate_alignment": true,
        "token_confidence_threshold": 0.7,
        "min_alignment_ratio": 0.8,
        "nfa_model": "nvidia/parakeet-ctc-1.1b"
    },
    "segmenter": {
        "type": "word_overlap",
        "minimum_chunk_duration": 0.2
    }
}
```

### 2. Python API Usage

```python
from speechline.transcribers import ParakeetTDTTranscriber
from datasets import Dataset, Audio

# Initialize transcriber
transcriber = ParakeetTDTTranscriber("nvidia/parakeet-tdt-0.6b-v2")

# Prepare dataset
dataset = Dataset.from_dict({
    "audio": ["student1.wav", "student2.wav", "student3.wav"]
}).cast_column("audio", Audio(sampling_rate=16000))

# Ground truth texts
ground_truths = [
    "Put their books on the table",
    "The quick brown fox jumps",
    "I saw a cat in the yard"
]

# Run validation
results = transcriber.predict_with_validation(
    dataset=dataset,
    ground_truth_texts=ground_truths,
    token_confidence_threshold=0.7,  # Min confidence per token
    min_alignment_ratio=0.8  # 80% of tokens must align
)

# Process results
for i, result in enumerate(results):
    print(f"\nFile: student{i+1}.wav")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Transcription: {result['transcription']}")
    print(f"Alignment: {result['alignment_ratio']:.1%}")
    print(f"Confidence: {result['avg_confidence']:.2f}")
    
    if result['is_valid']:
        print(f"✅ ACCEPTED")
        print(f"  Valid tokens: {result['num_valid_tokens']}")
    else:
        print(f"❌ REJECTED")
        print(f"  Rejected tokens: {result['num_rejected_tokens']}")
        print(f"  Rejected: {[t['text'] for t in result['rejected_tokens']]}")
```

### 3. CLI Usage

Run the speechline pipeline with validation:

```bash
python speechline/run.py \
    --input_dir /path/to/audio/files \
    --output_dir /path/to/output \
    --config examples/validation_config.json
```

## Result Format

Each validation result contains:

```python
{
    "is_valid": True,  # Overall pass/fail
    "ground_truth": "Put their books on the table",
    "transcription": "Put there books on table",
    "alignment_ratio": 0.83,  # 83% of tokens aligned
    "avg_confidence": 0.82,  # Average confidence score
    
    "token_alignment": [
        {"text": "Put", "confidence": 0.93, "start_time": 0.0, "end_time": 0.2},
        {"text": "there", "confidence": 0.91, "start_time": 0.2, "end_time": 0.4},
        # ... more tokens
    ],
    
    "valid_tokens": [
        # Tokens with confidence >= threshold
    ],
    
    "rejected_tokens": [
        # Tokens with confidence < threshold
    ],
    
    "num_total_tokens": 6,
    "num_valid_tokens": 5,
    "num_rejected_tokens": 1
}
```

## Output Files

When running via CLI, the following files are created:

### validation_report.json
Complete validation results for all files:

```json
[
    {
        "is_valid": true,
        "ground_truth": "Put their books on the table",
        "transcription": "Put there books on table",
        "alignment_ratio": 0.833,
        "avg_confidence": 0.82
    }
]
```

### rejected_files.csv
List of files that failed validation:

```csv
audio,ground_truth
student2.wav,The quick brown fox jumps
student5.wav,I saw a cat in the yard
```

## Configuration Parameters

### validate_alignment (bool)
Enable/disable validation mode. Default: `false`

```json
{"validate_alignment": true}
```

### token_confidence_threshold (float)
Minimum confidence score (0-1) for a token to be considered aligned. Default: `0.7`

- **0.6**: Very permissive (accepts more variation)
- **0.7**: Balanced (recommended)
- **0.8**: Strict (requires high confidence)

```json
{"token_confidence_threshold": 0.7}
```

### min_alignment_ratio (float)
Minimum fraction (0-1) of tokens that must pass the confidence threshold. Default: `0.8`

- **0.7**: Accept if 70% of tokens align
- **0.8**: Accept if 80% of tokens align (recommended)
- **0.9**: Accept if 90% of tokens align (very strict)

```json
{"min_alignment_ratio": 0.8}
```

### nfa_model (str)
NeMo model for forced alignment. Default: `"nvidia/parakeet-ctc-1.1b"`

Available models:
- `nvidia/parakeet-ctc-1.1b` (recommended)
- `nvidia/parakeet-ctc-0.6b`

```json
{"nfa_model": "nvidia/parakeet-ctc-1.1b"}
```

## Use Cases

### 1. Reading Assessment
Validate that students read the correct text:

```python
# Ground truth: passage from reading book
ground_truth = "The quick brown fox jumps over the lazy dog"

# Student reads (may have errors)
results = transcriber.predict_with_validation(
    dataset=student_recordings,
    ground_truth_texts=[ground_truth] * len(student_recordings),
    token_confidence_threshold=0.75,  # Allow some variation
    min_alignment_ratio=0.85  # Require 85% accuracy
)
```

### 2. Pronunciation Practice
Accept correct pronunciation, regardless of spelling choices:

```python
# Homophones are accepted (same pronunciation)
ground_truth = "I saw their new car there"
# Student says: "I saw there new car their"
# Result: ACCEPTED (phonemes align, even if words are swapped)
```

### 3. Voice Verification
Ensure speaker says expected passphrase:

```python
ground_truth = "My voice is my passport"

results = transcriber.predict_with_validation(
    dataset=verification_audio,
    ground_truth_texts=[ground_truth],
    token_confidence_threshold=0.8,  # High threshold
    min_alignment_ratio=0.95  # Very strict (95% must match)
)
```

## Troubleshooting

### Low Alignment Ratio
**Problem**: Good pronunciation but low alignment scores

**Solutions**:
- Lower `token_confidence_threshold` (e.g., 0.6)
- Check audio quality (background noise affects alignment)
- Verify ground truth text matches expected speech

### False Positives (Homophones)
**Problem**: Accepting wrong homophones (e.g., "there" vs "their")

**Solution**: This is expected behavior - forced alignment validates **pronunciation**, not **word choice**. If you need exact word matching, use free ASR transcription and text comparison:

```python
# Get both validation and transcription
validation = transcriber.predict_with_validation(dataset, ground_truths)
transcription = transcriber.predict(dataset)[0]

# Check exact text match
if transcription.lower() != ground_truth.lower():
    print(f"Wrong word used: {transcription} (expected: {ground_truth})")
```

### NFA Script Not Found
**Problem**: `FileNotFoundError: NeMo Forced Aligner script not found`

**Solution**: Install NeMo toolkit:

```bash
pip install nemo_toolkit[asr]
```

Or specify NFA script path in code:

```python
# Modify parakeet_tdt.py line 387-391 to add your path
nfa_script_paths = [
    "/your/custom/path/NeMo/tools/nemo_forced_aligner/align.py",
    os.path.expanduser("~/NeMo/tools/nemo_forced_aligner/align.py"),
    # ... existing paths
]
```

## Technical Details

### Confidence Score Calculation

Confidence scores come from NeMo Forced Aligner's CTM (time-marked) output:

```
Format: <utt_id> <channel> <start_time> <duration> <token> <confidence>
audio_1  1       0.00       0.15        the        0.95
audio_1  1       0.15       0.12        quick      0.88
audio_1  1       0.27       0.18        brown      0.42
```

Low confidence (< threshold) indicates poor alignment:
- Different phonemes spoken
- Mispronunciation
- Background noise
- Missing words

### Phoneme vs Word Alignment

The system aligns at the **phoneme level** but reports at the **token/word level**:

```
Word: "their"
↓
Phonemes: /ð ɛ ɹ/
↓
Audio phonemes: /ð ɛ ɹ/ (from "there")
↓
Result: HIGH CONFIDENCE (phonemes match)
```

This is why homophones align successfully.

## See Also

- [Parakeet TDT Transcriber Documentation](../reference/transcribers/parakeet_tdt.md)
- [NeMo Forced Aligner Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/forced_alignment.html)
- [Configuration Guide](../config.md)