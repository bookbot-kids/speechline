# Config

## Example Config File

```json title="example_config.json"
{
    "do_classify": true,
    "filter_empty_transcript": true,
    "classifier": {
        "model": "bookbot/distil-wav2vec2-adult-child-cls-52m",
        "max_duration_s": 3.0,
        "batch_size": 128
    },
    "transcriber": {
        "type": "wav2vec2",
        "model": "facebook/wav2vec2-base-960h",
        "return_timestamps": "word",
        "chunk_length_s": 30
    },
    "segmenter": {
        "silence_duration": 3.0,
        "minimum_chunk_duration": 0.2
    }
}
```

---

::: speechline.config.ClassifierConfig

::: speechline.config.TranscriberConfig

::: speechline.config.SegmenterConfig

::: speechline.config.Config