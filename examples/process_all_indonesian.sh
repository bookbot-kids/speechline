#!/bin/bash
# Process all Indonesian (id-*) directories with multi-GPU transcription

CONFIG="examples/id_config.json"
BASE_DIR="/mnt/Store07/Bookbot"
GPUS="0,1"
MAX_FILES=100000  # Limit per directory to avoid memory issues
SKIP_EXISTING="--skip_existing"  # Skip files with existing transcripts

echo "============================================================"
echo "Processing All Indonesian Directories with Multi-GPU"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Max files per run: $MAX_FILES"
echo "Skip existing: Yes"
echo

# Find all id-* directories
ID_DIRS=$(ls -d $BASE_DIR/id-* 2>/dev/null)

if [ -z "$ID_DIRS" ]; then
    echo "❌ No id-* directories found in $BASE_DIR"
    exit 1
fi

# Count directories
DIR_COUNT=$(echo "$ID_DIRS" | wc -l)
echo "Found $DIR_COUNT Indonesian directories:"
echo "$ID_DIRS"
echo

# Process each directory
CURRENT=1
for DIR in $ID_DIRS; do
    echo "============================================================"
    echo "[$CURRENT/$DIR_COUNT] Processing: $(basename $DIR)"
    echo "============================================================"
    
    # Count audio files
    AUDIO_COUNT=$(find "$DIR" -name "*.aac" 2>/dev/null | wc -l)
    echo "Audio files: $AUDIO_COUNT"
    
    if [ $AUDIO_COUNT -eq 0 ]; then
        echo "⚠️  No audio files found, skipping..."
        CURRENT=$((CURRENT + 1))
        continue
    fi
    
    # Run multi-GPU transcription
    python examples/run_indonesian_multi_gpu.py \
        -i "$DIR" \
        -c "$CONFIG" \
        --gpus $GPUS \
        --max_files $MAX_FILES \
        $SKIP_EXISTING
    
    # Check if successful
    if [ $? -eq 0 ]; then
        TXT_COUNT=$(find "$DIR" -name "*.txt" 2>/dev/null | wc -l)
        echo "✓ Created $TXT_COUNT transcript files"
    else
        echo "❌ Error processing $DIR"
    fi
    
    echo
    CURRENT=$((CURRENT + 1))
done

echo "============================================================"
echo "All Indonesian directories processed!"
echo "============================================================"