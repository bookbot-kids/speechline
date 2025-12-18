# Canary-Qwen Integration Guide

This guide explains how to use NVIDIA's Canary-Qwen model with SpeechLine.

## Overview

Canary-Qwen is a 2.5B parameter multilingual speech recognition model that:
- Supports 100+ languages
- Can translate speech to English
- Runs on Apple Silicon (MPS), CUDA, and CPU

## Prerequisites

### 1. Install NeMo Toolkit

Canary-Qwen requires NVIDIA's NeMo framework:

```bash
# Install NeMo with ASR support
pip install nemo_toolkit[asr]

# Install additional required dependencies
pip install sacrebleu

# Note: You may see warnings about torchcodec and FFmpeg, but these
# can be safely ignored as the transcriber loads audio with librosa
```

### 2. System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- ~8GB disk space for model

**Recommended:**
- Apple Silicon Mac (M1/M2/M3/M4) or NVIDIA GPU
- 32GB RAM
- SSD storage

## Usage

### Basic Transcription

```python
from speechline.transcribers import CanaryTranscriber
from datasets import Dataset, Audio

# Initialize transcriber
transcriber = CanaryTranscriber(
    model_checkpoint="nvidia/canary-qwen-2.5b"
)

# Prepare dataset
dataset = Dataset.from_dict({
    "audio": ["sample.wav"]
}).cast_column("audio", Audio(sampling_rate=16000))

# Transcribe
transcripts = transcriber.predict(dataset)
print(transcripts[0])
```

### Using SpeechLine Pipeline

Create a config file (`canary_config.json`):

```json
{
    "do_classify": false,
    "filter_empty_transcript": true,
    "transcriber": {
        "type": "canary",
        "model": "nvidia/canary-qwen-2.5b",
        "return_timestamps": true,
        "chunk_length_s": 30,
        "torch_dtype": "float16"
    },
    "segmenter": {
        "type": "silence",
        "silence_duration": 0.3,
        "minimum_chunk_duration": 0.2
    }
}
```

Run the pipeline:

```bash
python -m speechline.run \
  --config examples/canary_config.json \
  --input_dir path/to/audio \
  --output_dir path/to/output
```

### Test Script

Test with sample files:

```bash
python scripts/archive/test_canary_transcriber.py
```

**Expected output:**
```
======================================================================
Canary-Qwen Transcriber Test
======================================================================

🖥️  Device: MPS (Apple Silicon)
✅ Model loaded successfully on mps

🎙️  Transcribing...
Transcribing Audios: 100%|██████████| 3/3 [04:51<00:00, 97.02s/it]

======================================================================
📝 TRANSCRIPTION RESULTS
======================================================================
[1] File: 164.flac
    Transcript: Wa some watamakbara bade an inda
...
```

## Performance

### Apple Silicon (M1/M2/M3)
- **Speed**: ~2-5x real-time
- **Memory**: 6-8GB during inference
- **Device**: Automatically uses MPS acceleration

### NVIDIA GPU (CUDA)
- **Speed**: ~5-10x real-time (depending on GPU)
- **Memory**: 6-8GB VRAM
- **Device**: Automatically uses CUDA

### CPU Only
- **Speed**: 0.5-1x real-time
- **Memory**: 8-10GB RAM
- **Device**: Falls back to CPU

## Troubleshooting

### NeMo Installation Issues

If `pip install nemo_toolkit[asr]` fails, try:

```bash
# Install Cython first (required for NeMo)
pip install cython

# Install NeMo dependencies separately
pip install pytorch-lightning
pip install omegaconf
pip install hydra-core

# Then install NeMo
pip install nemo_toolkit[asr]
```

### Missing sacrebleu Module

If you see "ModuleNotFoundError: No module named 'sacrebleu'":

```bash
pip install sacrebleu
```

### TorchCodec/FFmpeg Warnings

You may see warnings like:
```
RuntimeError: Could not load libtorchcodec. FFmpeg is not properly installed...
```

**These can be safely ignored.** The transcriber loads audio files using librosa instead of torchcodec, so FFmpeg dependencies are not required.

### Model Download Issues

If model download fails:

```bash
# Pre-download the model
python -c "from nemo.collections.speechlm2.models import SALM; SALM.from_pretrained('nvidia/canary-qwen-2.5b')"
```

### Memory Issues

For large batches or long audio:

1. Reduce `chunk_length_s` in config
2. Process files individually
3. Use CPU offloading if available

## Differences from Other Transcribers

### vs Whisper
- **Canary-Qwen**: More languages, built on NeMo
- **Whisper**: Better English accuracy, HuggingFace ecosystem

### vs Wav2Vec2
- **Canary-Qwen**: Seq2seq model, better for long-form
- **Wav2Vec2**: CTC model, faster but less accurate

### vs Parakeet
- **Canary-Qwen**: Newer, Qwen-based
- **Parakeet**: Also NeMo-based, established

## Technical Details

### Architecture
- **Encoder**: Conformer (32 layers, 1024 dim)
- **LLM**: Qwen3-1.7B
- **Preprocessor**: 128-feature mel spectrogram
- **Sample Rate**: 16kHz

### Model Files
- **Size**: ~5GB
- **Format**: NeMo checkpoint
- **Dtype**: bfloat16 (default)

## References

- [NVIDIA Canary-Qwen HuggingFace](https://huggingface.co/nvidia/canary-qwen-2.5b)
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [NeMo ASR Tutorial](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html)