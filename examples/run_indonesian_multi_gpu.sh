#!/bin/bash
# Process Indonesian audio using 2 GPUs in parallel
# This script splits the workload across GPU 0 and GPU 1

BOOKBOT_PATH="/mnt/Store07/Bookbot"
OUTPUT_DIR="/mnt/Store07/Bookbot_processed/indonesian"
CONFIG="examples/id_config.json"
LOG_DIR="./logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Multi-GPU Indonesian ASR Processing"
echo "========================================"
echo "GPUs: 2 (CUDA:0, CUDA:1)"
echo "Input: $BOOKBOT_PATH/id-*"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Function to get all Indonesian audio files
get_audio_files() {
    find "$BOOKBOT_PATH" -type d -name "id-*" -exec find {} -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.aac" -o -name "*.flac" \) \;
}

# Get total file count
TOTAL_FILES=$(get_audio_files | wc -l)
HALF_FILES=$((TOTAL_FILES / 2))

echo "Total files found: $TOTAL_FILES"
echo "Files per GPU: ~$HALF_FILES"
echo ""

# Split files into two lists
get_audio_files | head -n $HALF_FILES > /tmp/gpu0_files.txt
get_audio_files | tail -n +$((HALF_FILES + 1)) > /tmp/gpu1_files.txt

# Function to process files on a specific GPU
process_on_gpu() {
    local GPU_ID=$1
    local FILE_LIST=$2
    local GPU_OUTPUT="$OUTPUT_DIR/gpu${GPU_ID}"
    
    mkdir -p "$GPU_OUTPUT"
    
    echo "Starting GPU $GPU_ID processing..."
    
    # Set CUDA device and run processing
    CUDA_VISIBLE_DEVICES=$GPU_ID python speechline/run.py \
        --input_dir "$BOOKBOT_PATH" \
        --output_dir "$GPU_OUTPUT" \
        --config "$CONFIG" \
        --log_dir "$LOG_DIR" \
        --script_name "run_indonesian_gpu${GPU_ID}" \
        2>&1 | tee "$LOG_DIR/gpu${GPU_ID}_processing.log" &
}

# Start processing on both GPUs in parallel
process_on_gpu 0 /tmp/gpu0_files.txt
GPU0_PID=$!

process_on_gpu 1 /tmp/gpu1_files.txt
GPU1_PID=$!

echo ""
echo "Processing started:"
echo "  GPU 0 PID: $GPU0_PID"
echo "  GPU 1 PID: $GPU1_PID"
echo ""
echo "Monitoring progress..."
echo "  GPU 0 log: $LOG_DIR/gpu0_processing.log"
echo "  GPU 1 log: $LOG_DIR/gpu1_processing.log"
echo ""

# Wait for both processes to complete
wait $GPU0_PID
GPU0_STATUS=$?

wait $GPU1_PID
GPU1_STATUS=$?

# Cleanup temp files
rm -f /tmp/gpu0_files.txt /tmp/gpu1_files.txt

echo ""
echo "========================================"
echo "Multi-GPU Processing Complete"
echo "========================================"
echo "GPU 0 exit status: $GPU0_STATUS"
echo "GPU 1 exit status: $GPU1_STATUS"
echo ""
echo "Results:"
echo "  GPU 0: $OUTPUT_DIR/gpu0/"
echo "  GPU 1: $OUTPUT_DIR/gpu1/"
echo "========================================"

# Merge results (optional)
if [ $GPU0_STATUS -eq 0 ] && [ $GPU1_STATUS -eq 0 ]; then
    echo ""
    echo "Merging results..."
    
    # Merge manifest files if they exist
    if [ -f "$OUTPUT_DIR/gpu0/audio_segment_manifest.json" ] && [ -f "$OUTPUT_DIR/gpu1/audio_segment_manifest.json" ]; then
        python -c "
import json
with open('$OUTPUT_DIR/gpu0/audio_segment_manifest.json') as f1, \
     open('$OUTPUT_DIR/gpu1/audio_segment_manifest.json') as f2:
    manifest1 = json.load(f1)
    manifest2 = json.load(f2)
    merged = manifest1 + manifest2
    with open('$OUTPUT_DIR/merged_manifest.json', 'w') as out:
        json.dump(merged, out, indent=2)
print('Merged manifest created: $OUTPUT_DIR/merged_manifest.json')
"
    fi
    
    echo "Merge complete!"
fi