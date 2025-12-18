# Processing Large Indonesian Audio Directories

## Problem

The `id-id` directory contains **1.5 million files**, which causes memory issues and stalls during file scanning.

## Solution

Use the optimized multi-GPU script with these options:

### Command Line Options

1. **`--skip_existing`**: Skip files that already have `.txt` transcripts
2. **`--max_files N`**: Limit processing to N files per run (default: unlimited)

### For Large Directories (1M+ files)

Process in batches to avoid memory issues:

```bash
# First run: Process first 100K files
python examples/run_indonesian_multi_gpu.py \
    -i /mnt/Store07/Bookbot/id-id \
    -c examples/id_config.json \
    --gpus 0,1 \
    --max_files 100000 \
    --skip_existing

# Subsequent runs: Only processes remaining files
python examples/run_indonesian_multi_gpu.py \
    -i /mnt/Store07/Bookbot/id-id \
    -c examples/id_config.json \
    --gpus 0,1 \
    --max_files 100000 \
    --skip_existing
```

### Using the Batch Script

The script `scripts/process_all_indonesian.sh` automatically:
- Processes 100K files per directory per run
- Skips files with existing transcripts
- Can be run multiple times until all files are processed

```bash
# Run multiple times for large directories
./scripts/process_all_indonesian.sh
# Wait for completion, then run again if needed
./scripts/process_all_indonesian.sh
```

### Progress Tracking

The script shows:
- Overall progress: `[OVERALL] Progress: 40/40 files (100.0%)`
- Files already processed are skipped
- Only unprocessed files are transcribed

### Memory Management

- Processes files in batches of 50
- Clears GPU memory after each batch
- GPU memory stays under 15 MiB
- No memory leaks

## Performance

- **Small directories** (< 10K files): ~30 seconds
- **Medium directories** (10K-100K files): 5-30 minutes
- **Large directories** (100K+ files): Multiple runs needed
- **Speedup**: ~2x with 2 GPUs vs single GPU