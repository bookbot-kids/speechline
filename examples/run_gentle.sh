#!/bin/bash
# Run Gentle forced alignment on audio directory
#
# Usage:
#   ./examples/run_gentle.sh /path/to/audio/dir [/path/to/output/dir]
#
# Requirements:
#   - Audio files with corresponding .txt files (ground truth)
#   - Gentle installation at /mnt/Projects/Projects/AudioProcessing/gentle

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
INPUT_DIR="${1:-$PROJECT_ROOT/data}"
OUTPUT_DIR="${2:-$INPUT_DIR}"
CONFIG="$PROJECT_ROOT/examples/gentle_config.json"

echo "========================================"
echo "Gentle Forced Alignment"
echo "========================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Config file:      $CONFIG"
echo "========================================"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file does not exist: $CONFIG"
    exit 1
fi

# Set PYTHONPATH to include project root and Gentle installation
GENTLE_PATH="/mnt/4090_projects/Projects/AudioProcessing/gentle"
export PYTHONPATH="$PROJECT_ROOT:$GENTLE_PATH:$PYTHONPATH"

# Run speechline with Gentle transcriber
cd "$PROJECT_ROOT"
python speechline/run.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -c "$CONFIG" \
    --log_dir logs \
    --script_name run_gentle

echo ""
echo "========================================"
echo "Gentle alignment complete!"
echo "========================================"
echo "JSON files with alignment results saved next to audio files"
echo "Segmented audio chunks saved in: $OUTPUT_DIR"
echo "Logs saved in: logs/"
echo "========================================"