# Config

## Example Config File

```json title="example_config.json"
{
    "languages": [
        "en",
        "id"
    ],
    "classifier": {
        "models": {
            "en": "bookbot/distil-wav2vec2-adult-child-cls-52m",
            "id": "bookbot/distil-wav2vec2-adult-child-id-cls-52m"
        },
        "batch_size": 128
    },
    "transcriber": {
        "models": {
            "en": "bookbot/wav2vec2-ljspeech-gruut",
            "id": "bookbot/wav2vec2-ljspeech-gruut"
        },
        "batch_size": 16
    },
    "segmenter": {
        "silence_duration": 0.2
    }
}
```

---

::: speechline.utils.config.Config