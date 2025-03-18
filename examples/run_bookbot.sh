export INPUT_DIR="../bookbot_en"
export OUTPUT_DIR="../bookbot_en_training"
export LOG_DIR="./logs"
export SCRIPT_NAME=$(basename "$0")

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

python speechline/run.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --config="examples/bb_config_word.json" \
    --script_name="$SCRIPT_NAME"