# Config

## Example Config File

```json title="example_config.json"
{
    "do_classify": true,
    "filter_empty_transcript": true,
    "classifier": {
        "model": "bookbot/distil-wav2vec2-adult-child-cls-52m",
        "max_duration_s": 3.0
    },
    "transcriber": {
        "type": "wav2vec2",
        "model": "bookbot/wav2vec2-bookbot-en-lm",
        "return_timestamps": "word",
        "chunk_length_s": 30
    },
    "do_noise_classify": true,
    "noise_classifier": {
        "model": "bookbot/distil-ast-audioset",
        "minimum_empty_duration": 0.3,
        "threshold": 0.2
    },
    "segmenter": {
        "type": "word_overlap",
        "minimum_chunk_duration": 1.0
    }
}
```

---
::: speechline.config.NoiseClassifierConfig

::: speechline.config.ClassifierConfig

::: speechline.config.TranscriberConfig

::: speechline.config.SegmenterConfig

::: speechline.config.Config