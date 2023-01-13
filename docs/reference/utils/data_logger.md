# Audio Data Logger

## Usage

```sh title="example_data_logger.sh"
python speechline/utils/data_logger.py [-h] -u URL -i INPUT_DIR -l LABEL
```

```
Log region-grouped total audio duration to AirTable.

optional arguments:
  -h, --help            show this help message and exit
  -u URL, --url URL     AirTable URL.
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory of input audios to log.
  -l LABEL, --label LABEL
                        Log record label. E.g. training/archive.
```

## Example

```sh title="example_data_logger.sh"
export AIRTABLE_API_KEY="AIRTABLE_API_KEY"
export AIRTABLE_URL="AIRTABLE_TABLE_URL"
python speechline/utils/data_logger.py --url $AIRTABLE_URL --input_dir dropbox/ --label archive
python speechline/utils/data_logger.py --url $AIRTABLE_URL --input_dir training/ --label training
```

---

::: speechline.utils.data_logger.DataLogger