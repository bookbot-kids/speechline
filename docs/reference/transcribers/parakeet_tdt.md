# ParakeetTDTTranscriber

::: speechline.transcribers.parakeet_tdt.ParakeetTDTTranscriber
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2

## Overview

The ParakeetTDTTranscriber uses NVIDIA's Parakeet TDT (Token-and-Duration Transducer) models for automatic speech recognition. TDT models are designed to predict both tokens and their durations simultaneously, providing efficient inference with word-level timestamps.

## Key Features

- **Fast Inference**: Optimized for efficient streaming and batch processing
- **Word-Level Timestamps**: Accurate word boundaries and durations
- **Multi-Device Support**: Compatible with CUDA, MPS (Apple Silicon), and CPU
- **High Accuracy**: State-of-the-art speech recognition performance
- **Flexible Audio Handling**: Supports various audio lengths and formats

## Supported Models

- `nvidia/parakeet-tdt-0.6b-v2` - 600M parameter TDT model (recommended)
- `nvidia/parakeet-tdt-1.1b` - 1.1B parameter TDT model
- Any compatible NeMo TDT ASR model

## Requirements

```bash
pip install nemo_toolkit[asr]
```

## Configuration Example

```json
{
    "transcriber": {
        "type": "parakeet_tdt",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "return_timestamps": "word",
        "chunk_length_s": 30,
        "transcriber_device": "cuda",
        "torch_dtype": "float16"
    }
}
```

## Usage Example

### Basic Transcription

```python
from speechline.transcribers import ParakeetTDTTranscriber
from datasets import Dataset, Audio

# Initialize transcriber
transcriber = ParakeetTDTTranscriber("nvidia/parakeet-tdt-0.6b-v2")

# Prepare dataset
dataset = Dataset.from_dict({
    "audio": ["sample1.wav", "sample2.wav"]
}).cast_column("audio", Audio(sampling_rate=transcriber.sr))

# Get transcriptions
transcripts = transcriber.predict(dataset)
print(transcripts)
# ["Hello world", "This is a test"]
```

### Transcription with Word Timestamps

```python
# Get word-level timestamps
offsets = transcriber.predict(
    dataset,
    output_offsets=True,
    return_timestamps="word"
)

for audio_offsets in offsets:
    for word_data in audio_offsets:
        print(f"{word_data['text']}: {word_data['start_time']:.2f}s - {word_data['end_time']:.2f}s")
```

### Advanced Configuration

```python
# Custom device and precision
transcriber = ParakeetTDTTranscriber(
    model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
    transcriber_device="cuda",
    torch_dtype="float16"
)

# Process with custom chunk length
results = transcriber.predict(
    dataset,
    chunk_length_s=60,  # 60-second chunks
    output_offsets=True,
    keep_whitespace=False
)
```

## Device Selection

The transcriber automatically selects the best available device:

1. **CUDA** (NVIDIA GPU) - Fastest option if available
2. **MPS** (Apple Silicon) - Optimized for M1/M2/M3 Macs
3. **CPU** - Fallback option, slower but always available

You can override the automatic selection:

```python
# Force CPU usage
transcriber = ParakeetTDTTranscriber(
    model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
    transcriber_device="cpu"
)
```

## Performance Tips

1. **Batch Size**: Process multiple files together for better throughput
2. **Precision**: Use `float16` or `bfloat16` for faster inference on GPU
3. **Chunk Length**: Adjust based on your audio length and memory constraints
4. **Device**: CUDA provides the best performance, followed by MPS

## Comparison with Other Transcribers

| Feature | ParakeetTDT | Wav2Vec2 | Whisper | Canary |
|---------|-------------|----------|---------|--------|
| Word Timestamps | ✅ Native | ✅ Native | ⚠️ Approximate | ⚠️ Sentence-level |
| Streaming | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Languages | English | Varies | 100+ | 100+ |
| Speed | ⚡⚡⚡ Fast | ⚡⚡ Moderate | ⚡ Slower | ⚡⚡ Moderate |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Troubleshooting

### Import Error

```
ImportError: NeMo toolkit is required for Parakeet TDT
```

**Solution**: Install NeMo toolkit
```bash
pip install nemo_toolkit[asr]
```

### CUDA Out of Memory

**Solution**: Reduce batch size or use CPU
```python
transcriber = ParakeetTDTTranscriber(
    model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
    transcriber_device="cpu"
)
```

### Model Download Issues

**Solution**: Download model manually
```python
from nemo.collections.asr.models import ASRModel

# Download and cache the model
model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
```

## See Also

- [Parakeet Transcriber](parakeet.md) - Alternative Parakeet implementation
- [Wav2Vec2 Transcriber](wav2vec2.md) - CTC-based alternative
- [Whisper Transcriber](whisper.md) - Multilingual alternative
- [Configuration Guide](../config.md) - Full configuration options