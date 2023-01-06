# `aac`-to-`wav` Audio Converter

## Usage

```sh
python speechline/utils/aac_to_wav.py [-h] [-c CHANNEL] [-r RATE] input_dir
```

```
Batch-convert aac audios in a folder to wav format with ffmpeg.

positional arguments:
  input_dir             Directory of input audios to convert.

optional arguments:
  -h, --help            show this help message and exit
  -c CHANNEL, --channel CHANNEL
                        Number of audio channels in output.
  -r RATE, --rate RATE  Sample rate of audio output.
```

## Example

```sh
python speechline/utils/aac_to_wav.py -c 1 -r 16000 dropbox/
```

---

::: speechline.utils.aac_to_wav