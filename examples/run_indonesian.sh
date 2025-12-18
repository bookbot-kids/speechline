#!/bin/bash
# Process Indonesian audio files using speechline with Indonesian ASR model
# Model: bookbot/wav2vec2-xls-r-bookbot-id

export INPUT_DIR="/mnt/Store07/Bookbot"
export OUTPUT_DIR="/mnt/Store07/Bookbot_processed/indonesian"
export LOG_DIR="./logs"
export SCRIPT_NAME=$(basename "$0")

# Create directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "========================================"
echo "Indonesian ASR Processing"
echo "========================================"
echo "Input:  $INPUT_DIR/id-*"
echo "Output: $OUTPUT_DIR"
echo "Model:  bookbot/wav2vec2-xls-r-bookbot-id"
echo "========================================"

python speechline/run.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --config="examples/id_config.json" \
    --script_name="$SCRIPT_NAME"