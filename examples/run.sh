export INPUT_DIR="commonvoice/"
export OUTPUT_DIR="training/"

python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/config.json"