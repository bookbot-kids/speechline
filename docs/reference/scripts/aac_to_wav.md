# `aac`-to-`wav` Audio Converter

## Usage

```sh title="example_aac_to_wav.sh"
python scripts/aac_to_wav.py [-h] -i INPUT_DIR [-c CHANNEL] [-r RATE]
```

```
Batch-convert aac audios in a folder to wav format with ffmpeg.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory of input audios to convert.
  -c CHANNEL, --channel CHANNEL
                        Number of audio channels in output.
  -r RATE, --rate RATE  Sample rate of audio output.
```

## Example

```sh
python scripts/aac_to_wav.py --input_dir="dropbox/" -c 1 -r 16000 
```

---

::: scripts.aac_to_wav