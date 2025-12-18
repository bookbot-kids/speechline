export INPUT_DIR="dropbox/"
export OUTPUT_DIR="training/"

python scripts/aac_to_wav.py --input_dir=$INPUT_DIR
python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/config.json"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$OUTPUT_DIR --label="training"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$INPUT_DIR --label="archive"