#!/bin/bash
# Add phoneme transcriptions to Indonesian audio files using Gentle
# Processes all id-* directories in /mnt/Store07/Bookbot

BOOKBOT_PATH="/mnt/Store07/Bookbot"
REVIEW_LOG="indonesian_phoneme_review.txt"
GENTLE_PATH="/mnt/Projects/Projects/AudioProcessing/gentle"

echo "========================================"
echo "Indonesian Phoneme Alignment"
echo "========================================"
echo "Dataset: $BOOKBOT_PATH/id-*"
echo "Review:  $REVIEW_LOG"
echo "Gentle:  $GENTLE_PATH"
echo "========================================"
echo ""

python scripts/gentle_phoneme_alignment.py \
    --mode add-phonemes \
    --bookbot-path "$BOOKBOT_PATH" \
    --language id \
    --gentle-path "$GENTLE_PATH" \
    --review-log "$REVIEW_LOG" \
    --threads 4 \
    --log-level INFO

echo ""
echo "========================================"
echo "Phoneme alignment complete!"
echo "Review file: $REVIEW_LOG"
echo "========================================"