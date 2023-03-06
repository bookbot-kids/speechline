# Create HuggingFace Datset

## Usage

```sh title="example_create_hf_dataset.sh"
python scripts/create_hf_dataset.py [-h] -i INPUT_DIR --dataset_name DATASET_NAME [--phonemize PHONEMIZE] [--private PRIVATE]
```

```
Create HuggingFace dataset from SpeechLine outputs.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory of input audios.
  --dataset_name DATASET_NAME
                        HuggingFace dataset repository name.
  --phonemize PHONEMIZE
                        Phonemize text.
  --private PRIVATE     Set HuggingFace dataset to private.
```

## Example

```sh
python scripts/create_hf_dataset.py \
    --input_dir="training/" \
    --dataset_name="myname/mydataset" \
    --private="True" \
    --phonemize="True"
```

---

::: scripts.create_hf_dataset