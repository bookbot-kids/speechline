#!/bin/bash

# Example script to run SpeechLine with Parakeet TDT transcriber
# Usage: ./examples/run_parakeet_tdt.sh

# Configuration
INPUT_DIR="path/to/your/audio/files"
OUTPUT_DIR="output/parakeet_tdt"
CONFIG_FILE="examples/parakeet_tdt_config.json"

# Run SpeechLine pipeline with Parakeet TDT transcriber
python speechline/run.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --script_name "run_parakeet_tdt" \
    --log_dir "logs"

echo "Processing complete! Results saved to: $OUTPUT_DIR"