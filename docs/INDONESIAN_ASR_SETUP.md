# Indonesian ASR Setup Guide

This guide explains how to use the Indonesian (Bahasa Indonesia) ASR model with SpeechLine to transcribe audio files in the `/mnt/Store07/Bookbot/id-*` directories.

## Overview

**Model**: [`bookbot/wav2vec2-xls-r-bookbot-id`](https://huggingface.co/bookbot/wav2vec2-xls-r-bookbot-id)  
**Type**: Wav2Vec2-based ASR model for Indonesian/Bahasa Indonesia  
**Features**: Word-level timestamps, optimized for Indonesian speech recognition  
**Dataset Coverage**: 805K+ Indonesian word tokens in phonetic lexicon

---

## Quick Start

### 1. Basic Indonesian Transcription

Process Indonesian audio files from `/mnt/Store07/Bookbot/id-*` directories:

```bash
cd /mnt/Projects/Projects/AudioProcessing/speechline
./examples/run_indonesian.sh
```

This will:
- Scan all `id-*` subdirectories in `/mnt/Store07/Bookbot`
- Transcribe audio using `bookbot/wav2vec2-xls-r-bookbot-id`
- Generate word-level timestamps
- Save results to `/mnt/Store07/Bookbot_processed/indonesian`

### 2. Multi-GPU Processing (2 GPUs)

For faster processing using 2 GPUs in parallel:

```bash
./examples/run_indonesian_multi_gpu.sh
```

This will:
- Automatically split the workload between GPU 0 and GPU 1
- Process files in parallel for 2x speed improvement
- Merge results into a single output directory
- Create separate logs for each GPU

### 3. Custom Processing

For custom input/output paths:

```bash
python speechline/run.py \
    --input_dir /path/to/indonesian/audio \
    --output_dir /path/to/output \
    --config examples/id_config.json
```

---

## Configuration

The Indonesian configuration is defined in [`examples/id_config.json`](../examples/id_config.json):

```json
{
    "transcriber": {
        "type": "wav2vec2",
        "model": "bookbot/wav2vec2-xls-r-bookbot-id",
        "return_timestamps": "word",
        "chunk_length_s": 30
    }
}
```

### Key Parameters

- **model**: Indonesian ASR model from Hugging Face
- **return_timestamps**: `"word"` for word-level, `"char"` for character-level
- **chunk_length_s**: Audio chunk size for processing (default: 30 seconds)

---

## Phoneme Alignment (Optional)

Add phoneme transcriptions to Indonesian audio files using Gentle forced aligner:

```bash
./examples/run_indonesian_phoneme_alignment.sh
```

This will:
- Process all `id-*` directories in `/mnt/Store07/Bookbot`
- Use Gentle to align text with audio
- Add phoneme transcriptions to transcript files
- Generate review log for any alignment issues

### Manual Phoneme Alignment

```bash
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --language id \
    --review-log indonesian_review.txt \
    --threads 4
```

---

## Directory Structure

Expected structure for Indonesian audio:

```
/mnt/Store07/Bookbot/
├── id-ID/              # Indonesian (Indonesia)
│   ├── audio1.wav
│   ├── audio1.txt      # Text transcript
│   ├── audio2.aac
│   └── audio2.txt
├── id-MY/              # Indonesian (Malaysia) 
└── id-*/               # Other Indonesian variants
```

Each audio file should have a corresponding `.txt` file with the transcript.

---

## Supported Audio Formats

- `.wav` (recommended)
- `.mp3`
- `.aac`
- `.flac`
- `.ogg`
- `.m4a`

---

## Output Format

Processed files include:

### Transcript Files
```
Text transcript (first line)
Phoneme transcript (second line, if phoneme alignment was run)
```

### JSON Output
Word-level timestamps in JSON format:
```json
{
  "text": "halo dunia",
  "words": [
    {"word": "halo", "start": 0.0, "end": 0.5},
    {"word": "dunia", "start": 0.6, "end": 1.2}
  ]
}
```

---

## Comparison with Other Languages

| Language | Model | Config File | Directory Pattern |
|----------|-------|-------------|-------------------|
| **Indonesian** | `bookbot/wav2vec2-xls-r-bookbot-id` | `id_config.json` | `id-*` |
| **English** | `bookbot/bookbot_en_v2` | `bb_config.json` | `en-*` |
| **Swahili** | `bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm` | `sw_config.json` | `sw-*` |

---

## Phonetic Support

The project includes comprehensive Indonesian phonetic support:

- **Lexicon**: [`data/indonesian_words_ipa.csv`](../data/indonesian_words_ipa.csv) (805K+ tokens)
- **Unique Phonemes**: 4 (ʔ, ɲ, c, ɟ)
- **Coverage**: Documented in [`docs/MULTILINGUAL_PHONEME_COVERAGE.md`](MULTILINGUAL_PHONEME_COVERAGE.md)

### Indonesian Phonetic Features

- **Glottal stop**: ʔ (very common in Indonesian)
- **Palatal nasal**: ɲ (written "ny")
- **Palatal stops**: c and ɟ
- See [`speechline/PHONETICS_README.md`](../speechline/PHONETICS_README.md) for full details

---

## Troubleshooting

### Model Not Found

If you see an error like "Model not found":

```bash
# Verify the model exists on Hugging Face
curl -I https://huggingface.co/bookbot/wav2vec2-xls-r-bookbot-id

# The model will be automatically downloaded on first use
```

### Empty Transcripts

Check that:
1. Audio files have corresponding `.txt` transcript files
2. Transcript files are not empty
3. Audio quality is sufficient for recognition

### Out of Memory

If processing fails due to memory:
1. Reduce `chunk_length_s` in config (try 15 or 10)
2. Process fewer files at once
3. Reduce batch size if using custom scripts

---

## Advanced Usage

### Process Multiple Languages

Process both English and Indonesian:

```bash
python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --language en id \
    --threads 4
```

### Custom Model

To use a different Indonesian model:

1. Edit `examples/id_config.json`
2. Change the `model` field to your model path
3. Run processing as normal

---

## Related Documentation

- [Main README](../README.md) - Project overview
- [Wav2Vec2 Transcriber](reference/transcribers/wav2vec2.md) - Technical details
- [Multilingual Phoneme Coverage](MULTILINGUAL_PHONEME_COVERAGE.md) - Phoneme support
- [Phonetics Library](../speechline/PHONETICS_README.md) - Phonetic processing

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/bookbot-kids/speechline/issues
- Model Page: https://huggingface.co/bookbot/wav2vec2-xls-r-bookbot-id

---

**Last Updated**: 2025-10-22  
**Model Version**: bookbot/wav2vec2-xls-r-bookbot-id  
**SpeechLine Version**: v0.0.2+