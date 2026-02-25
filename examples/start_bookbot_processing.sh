#!/bin/bash
# Start Bookbot phoneme processing with 3 threads
# 
# This will process all audio files in /mnt/Store07/Bookbot/en-* directories
# and add phoneme transcriptions to their transcript files.
#
# Usage: bash scripts/start_bookbot_processing.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Starting Bookbot Phoneme Processing"
echo "============================================================"
echo "Threads: 3"
echo "Dataset: /mnt/Store07/Bookbot"
echo "Review log: bookbot_phoneme_review.txt"
echo "Processing log: bookbot_processing.log"
echo ""
echo "This will process all en-* subdirectories and add phonemes"
echo "to transcript files. Files that already have phonemes will"
echo "be skipped automatically."
echo ""
echo "Press Ctrl+C to stop processing (progress will be saved)"
echo "============================================================"
echo ""

# Run the processing
python3 "$SCRIPT_DIR/gentle_phoneme_alignment.py" \
    --mode add-phonemes \
    --bookbot-path /mnt/Store07/Bookbot \
    --threads 3 \
    --review-log bookbot_phoneme_review.txt \
    --log-level INFO 2>&1 | tee bookbot_processing.log

echo ""
echo "============================================================"
echo "Processing Complete!"
echo "============================================================"
echo "Check bookbot_processing.log for full details"
echo "Check bookbot_phoneme_review.txt for invalid transcripts"
echo "============================================================"