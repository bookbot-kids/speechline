export HF_DATASETS_CACHE="/home/ec2-user/SageMaker/.cache"
export BUCKET_NAME="bookbot-speech"
export INPUT_DIR="dropbox/"
export OUTPUT_DIR="training/"

python scripts/download_s3_bucket.py --bucket=$BUCKET_NAME --prefix=$INPUT_DIR --output_dir=$INPUT_DIR
python scripts/aac_to_wav.py --input_dir=$INPUT_DIR
python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/config.json"
python scripts/upload_s3_bucket.py --bucket=$BUCKET_NAME --prefix=$OUTPUT_DIR --input_dir=$OUTPUT_DIR
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$OUTPUT_DIR --label="training"
python scripts/data_logger.py --url=$AIRTABLE_URL --input_dir=$INPUT_DIR --label="archive"