
# Parakeet CTC Phoneme Finetuning Plan - Option 2

**Date:** 2025-10-14  
**Model:** `nvidia/parakeet-ctc-1.1b`  
**Goal:** Direct phoneme-level prediction with accurate timestamps  
**Hardware:** 2x NVIDIA RTX 4090 GPUs  
**Approach:** CTC (Connectionist Temporal Classification) for phoneme sequences

---

## Executive Summary

This plan uses **Parakeet CTC 1.1B** to directly predict IPA phoneme sequences from audio, avoiding the complexity of phoneme-to-word conversion. CTC is ideal for this task as it:

✅ Naturally handles variable-length phoneme sequences  
✅ Provides frame-level alignments for accurate timing  
✅ Works directly with your existing `phonemes_ipa` data  
✅ Proven architecture for phoneme recognition  

**Key Advantage:** No data conversion needed - use phoneme labels directly!

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Phoneme Vocabulary Construction](#phoneme-vocabulary-construction)
4. [NeMo CTC Training Setup](#nemo-ctc-training-setup)
5. [Multi-GPU Training Configuration](#multi-gpu-training-configuration)
6. [Training Procedure](#training-procedure)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Implementation Timeline](#implementation-timeline)

---

## Architecture Overview

### Parakeet CTC Model

**Model:** `nvidia/parakeet-ctc-1.1b`
- **Parameters:** 1.1 billion
- **Architecture:** Conformer encoder + CTC decoder
- **Input:** 16kHz audio (mel spectrograms)
- **Output:** Frame-level phoneme probabilities
- **Timestamps:** Frame-accurate (10ms resolution)

### CTC vs TDT for Phonemes

| Feature | CTC (Chosen) | TDT |
|---------|-------------|-----|
| Output | Character/phoneme sequences | Word tokens |
| Alignment | Frame-level (excellent) | Word-level only |
| Training data | Phonemes directly | Requires words |
| Timing accuracy | ~10ms precision | Word boundaries only |
| Phoneme support | Native | Requires conversion |

**Winner:** CTC is purpose-built for phoneme prediction! ✅

### Training Strategy

```
Input Audio (16kHz)
    ↓
Conformer Encoder (mel features)
    ↓
CTC Loss (phoneme targets)
    ↓
Output: Space-separated IPA phonemes
    ↓
Post-process: Extract frame-level timestamps
```

---

## Dataset Preparation

### 1. Dataset Inventory

All 6 datasets with phoneme annotations:

| Dataset | HuggingFace ID | Format | Estimated Samples |
|---------|---------------|--------|-------------------|
| YouTube EN | `bookbot/en_youtube_w2v-bert-2.0_filtered` | `phonemes_ipa` | 10K-50K |
| GigaSpeech | `bookbot/gigaspeech_w2v-bert-2.0_filtered` | `phonemes_ipa` | 100K+ |
| Common Voice | `bookbot/common_voice_16_1_en_w2v-bert-2.0_filtered` | `phonemes_ipa` | 500K+ |
| LibriPhone | `bookbot/libriphone` | phonemes | 100K+ |
| EN-AU Dean2Zak | `bookbot/en-AU-Dean2Zak` | TBD | 10K-20K |
| LJSpeech | `bookbot/ljspeech_phonemes` | phonemes | 13K |

**Total estimated:** 700K+ samples

### 2. Phoneme Format Requirements

**Input format (from datasets):**
```python
{
    "audio": {"path": "audio.wav", "array": [...], "sampling_rate": 16000},
    "phonemes_ipa": ["ð ɪ s", "ɪ z", "ə", "t ɛ s t"]  # List of phoneme words
}
```

**CTC training format (space-separated):**
```json
{
    "audio_filepath": "/path/to/audio.wav",
    "text": "ð ɪ s ɪ z ə t ɛ s t",
    "duration": 2.45
}
```

**Conversion:** Simply join phoneme lists with spaces!

### 3. Dataset Download Script

**File:** `scripts/training/01_download_phoneme_datasets.py`

```python
#!/usr/bin/env python3
"""
Download all phoneme datasets for Parakeet CTC training.
"""

from datasets import load_dataset
from pathlib import Path
import json

DATASETS = {
    "en_youtube": "bookbot/en_youtube_w2v-bert-2.0_filtered",
    "gigaspeech": "bookbot/gigaspeech_w2v-bert-2.0_filtered",
    "common_voice": "bookbot/common_voice_16_1_en_w2v-bert-2.0_filtered",
    "libriphone": "bookbot/libriphone",
    "en_au_dean2zak": "bookbot/en-AU-Dean2Zak",
    "ljspeech": "bookbot/ljspeech_phonemes"
}

OUTPUT_DIR = Path("data/training/raw_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_and_verify(name: str, dataset_id: str):
    """Download dataset and verify schema."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {name}")
    print(f"   Source: {dataset_id}")
    print(f"{'='*60}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)
        
        # Save to disk
        output_path = OUTPUT_DIR / name
        dataset.save_to_disk(str(output_path))
        
        # Verify schema
        print(f"\n✅ Success!")
        print(f"   Samples: {len(dataset):,}")
        print(f"   Columns: {dataset.column_names}")
        
        # Check for phoneme data
        if "phonemes_ipa" in dataset.column_names:
            sample = dataset[0]
            print(f"   Phonemes format: {type(sample['phonemes_ipa'])}")
            print(f"   Example: {sample['phonemes_ipa'][:3]}...")
        elif "phonemes" in dataset.column_names:
            print(f"   Has 'phonemes' column")
        else:
            print(f"   ⚠️  WARNING: No phoneme column found!")
        
        print(f"   Saved to: {output_path}")
        
        # Save metadata
        metadata = {
            "name": name,
            "source": dataset_id,
            "samples": len(dataset),
            "columns": dataset.column_names,
            "path": str(output_path)
        }
        
        with open(OUTPUT_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Download all datasets."""
    print("🎙️  Parakeet CTC Phoneme Dataset Downloader")
    print("="*60)
    
    results = {}
    for name, dataset_id in DATASETS.items():
        success = download_and_verify(name, dataset_id)
        results[name] = "✅" if success else "❌"
    
    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary:")
    print("="*60)
    for name, status in results.items():
        print(f"  {status} {name}")
    
    successful = sum(1 for v in results.values() if v == "✅")
    print(f"\nTotal: {successful}/{len(DATASETS)} datasets downloaded successfully")

if __name__ == "__main__":
    main()
```

---

## Phoneme Vocabulary Construction

### Why This Matters

CTC requires a fixed vocabulary of output symbols. We need to:
1. Extract all unique IPA phonemes from all datasets
2. Add special tokens (blank, space, unknown)
3. Create vocabulary file for NeMo

### Vocabulary Builder Script

**File:** `scripts/training/02_build_phoneme_vocabulary.py`

```python
#!/usr/bin/env python3
"""
Build unified IPA phoneme vocabulary from all datasets.
"""

from datasets import load_from_disk
from pathlib import Path
from collections import Counter
import json

def extract_phonemes_from_dataset(dataset_path: Path) -> set:
    """Extract unique phonemes from a dataset."""
    print(f"\nProcessing: {dataset_path.name}")
    
    dataset = load_from_disk(str(dataset_path))
    all_phonemes = set()
    
    # Determine phoneme column name
    if "phonemes_ipa" in dataset.column_names:
        phoneme_col = "phonemes_ipa"
    elif "phonemes" in dataset.column_names:
        phoneme_col = "phonemes"
    else:
        print(f"  ⚠️  No phoneme column found!")
        return all_phonemes
    
    # Extract phonemes
    for sample in dataset:
        phonemes = sample[phoneme_col]
        
        # Handle different formats
        if isinstance(phonemes, list):
            # Format: ["ð ɪ s", "ɪ z", "ə"]
            for phoneme_word in phonemes:
                for phoneme in phoneme_word.split():
                    all_phonemes.add(phoneme)
        elif isinstance(phonemes, str):
            # Format: "ð ɪ s ɪ z ə"
            for phoneme in phonemes.split():
                all_phonemes.add(phoneme)
    
    print(f"  Found {len(all_phonemes)} unique phonemes")
    return all_phonemes

def build_vocabulary():
    """Build unified phoneme vocabulary."""
    print("🔤 Building Unified Phoneme Vocabulary")
    print("="*60)
    
    dataset_dir = Path("data/training/raw_datasets")
    all_phonemes = set()
    phoneme_counts = Counter()
    
    # Process each dataset
    for dataset_path in sorted(dataset_dir.iterdir()):
        if dataset_path.is_dir() and not dataset_path.name.endswith("_metadata.json"):
            phonemes = extract_phonemes_from_dataset(dataset_path)
            all_phonemes.update(phonemes)
    
    # Sort phonemes
    sorted_phonemes = sorted(all_phonemes)
    
    # Add special tokens
    vocabulary = [
        "<blank>",  # CTC blank token (index 0)
        "<unk>",    # Unknown phoneme
        " ",        # Space separator
    ] + sorted_phonemes
    
    # Save vocabulary
    output_dir = Path("data/training/vocabulary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_file = output_dir / "ipa_phoneme_vocabulary.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in vocabulary:
            f.write(f"{token}\n")
    
    # Save metadata
    vocab_metadata = {
        "total_tokens": len(vocabulary),
        "phonemes": len(sorted_phonemes),
        "special_tokens": 3,
        "tokens": vocabulary
    }
    
    with open(output_dir / "vocabulary_metadata.json", "w", encoding="utf-8") as f:
        json.dump(vocab_metadata, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print("✅ Vocabulary Built Successfully!")
    print("="*60)
    print(f"Total tokens: {len(vocabulary)}")
    print(f"Phonemes: {len(sorted_phonemes)}")
    print(f"Special tokens: 3 (<blank>, <unk>, space)")
    print(f"\nSample phonemes: {', '.join(sorted_phonemes[:20])}")
    print(f"\nVocabulary saved to: {vocab_file}")
    
    return vocabulary

if __name__ == "__main__":
    build_vocabulary()
```

### Expected Vocabulary Size

Based on English IPA:
- **Consonants:** ~24 (p, b, t, d, k, g, f, v, θ, ð, s, z, ʃ, ʒ, etc.)
- **Vowels:** ~15 (i, ɪ, e, ɛ, æ, ɑ, ɔ, o, ʊ, u, ʌ, ə, ɝ, etc.)
- **Diphthongs:** ~8 (aɪ, aʊ, ɔɪ, eɪ, oʊ, etc.)
- **Other:** ~5 (ŋ, j, w, l, r, etc.)
- **Special tokens:** 3

**Total:** ~50-60 tokens (very manageable for CTC!)

---

## NeMo CTC Training Setup

### 1. Convert to NeMo Manifest Format

**File:** `scripts/training/03_create_ctc_manifests.py`

```python
#!/usr/bin/env python3
"""
Convert phoneme datasets to NeMo CTC manifest format.
"""

from datasets import load_from_disk
from pathlib import Path
import json
import soundfile as sf
from tqdm import tqdm

def convert_to_ctc_manifest(dataset_path: Path, output_manifest: Path):
    """Convert dataset to CTC manifest format."""
    print(f"\n📝 Converting: {dataset_path.name}")
    
    dataset = load_from_disk(str(dataset_path))
    
    # Determine phoneme column
    if "phonemes_ipa" in dataset.column_names:
        phoneme_col = "phonemes_ipa"
    elif "phonemes" in dataset.column_names:
        phoneme_col = "phonemes"
    else:
        print("  ⚠️  No phoneme column found, skipping!")
        return 0
    
    count = 0
    skipped = 0
    
    with open(output_manifest, "a") as f:
        for sample in tqdm(dataset, desc=f"Processing {dataset_path.name}"):
            try:
                # Get audio path
                audio_info = sample["audio"]
                if isinstance(audio_info, dict):
                    audio_path = audio_info.get("path")
                    audio_array = audio_info.get("array")
                    sr = audio_info.get("sampling_rate", 16000)
                else:
                    audio_path = audio_info
                    audio_array = None
                    sr = 16000
                
                # Calculate duration
                if audio_array is not None:
                    duration = len(audio_array) / sr
                elif audio_path and Path(audio_path).exists():
                    info = sf.info(audio_path)
                    duration = info.duration
                else:
                    skipped += 1
                    continue
                
                # Get phoneme text
                phonemes = sample[phoneme_col]
                if isinstance(phonemes, list):
                    # Join list of phoneme words: ["ð ɪ s", "ɪ z"] → "ð ɪ s ɪ z"
                    text = " ".join(phonemes)
                else:
                    text = phonemes
                
                # Skip empty transcriptions
                if not text or not text.strip():
                    skipped += 1
                    continue
                
                # Create manifest entry
                manifest_entry = {
                    "audio_filepath": str(Path(audio_path).absolute()),
                    "text": text.strip(),
                    "duration": round(duration, 3)
                }
                
                # Write to manifest
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                skipped += 1
                continue
    
    print(f"  ✅ Converted: {count} samples")
    print(f"  ⚠️  Skipped: {skipped} samples")
    return count

def main():
    """Convert all datasets to manifests."""
    print("📄 Creating NeMo CTC Manifests")
    print("="*60)
    
    dataset_dir = Path("data/training/raw_datasets")
    manifest_dir = Path("data/training/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined manifest
    combined_manifest = manifest_dir / "all_combined.json"
    if combined_manifest.exists():
        combined_manifest.unlink()
    
    total_samples = 0
    
    # Process each dataset
    for dataset_path in sorted(dataset_dir.iterdir()):
        if dataset_path.is_dir():
            count = convert_to_ctc_manifest(dataset_path, combined_manifest)
            total_samples += count
    
    print("\n" + "="*60)
    print(f"✅ Total samples in combined manifest: {total_samples:,}")
    print(f"   Saved to: {combined_manifest}")
    print("="*60)

if __name__ == "__main__":
    main()
```

### 2. Split into Train/Val/Test

**File:** `scripts/training/04_split_manifests.py`

```python
#!/usr/bin/env python3
"""Split manifest into train/validation/test sets."""

import json
import random
from pathlib import Path

def split_manifest(input_manifest: Path, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05):
    """Split manifest into train/val/test."""
    print(f"\n📊 Splitting manifest: {input_manifest}")
    
    # Read all entries
    with open(input_manifest) as f:
        all_data = [json.loads(line) for line in f]
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_data)
    
    # Calculate split points
    total = len(all_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Cap training set if too large (optional, for faster iteration)
    max_train_samples = 500000
    if len(train_data) > max_train_samples:
        print(f"  ⚠️  Capping training set at {max_train_samples:,} samples")
        train_data = train_data[:max_train_samples]
    
    # Save splits
    manifest_dir = input_manifest.parent
    
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = manifest_dir / f"{split_name}.json"
        with open(output_path, "w") as f:
            for entry in split_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  ✅ {split_name}: {len(split_data):,} samples → {output_path}")
    
    print(f"\n  Total: {len(train_data) + len(val_data) + len(test_data):,} samples")
    print(f"  Split: {len(train_data):,} / {len(val_data):,} / {len(test_data):,}")

if __name__ == "__main__":
    input_manifest = Path("data/training/manifests/all_combined.json")
    split_manifest(input_manifest)
```

---

## NeMo CTC Training Configuration

### Training Config YAML

**File:** `configs/training/parakeet_ctc_phoneme.yaml`

```yaml
# Parakeet CTC Phoneme Finetuning Configuration
name: "parakeet-ctc-1.1b-phoneme-finetuned"

# Model Configuration
model:
  # Load pretrained Parakeet CTC checkpoint
  restore_from: "nvidia/parakeet-ctc-1.1b"
  
  sample_rate: 16000
  labels: null  # Will be loaded from vocabulary file
  
  # Model architecture (from pretrained, do not modify)
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    normalize: "per_feature"
    window_size: 0.025
    window_stride: 0.01
    window: "hann"
    features: 80
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001
    pad_to: 0

  # Spec augmentation
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2
    freq_width: 27
    time_masks: 10
    time_width: 0.05
    
  # Encoder (Conformer - from pretrained)
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    feat_out: -1
    n_layers: 24
    d_model: 1024
    # ... (other params from pretrained model)
    
  # CTC Decoder
  decoder:
    _target_: nemo.collections.asr.modules.ConvASRDecoder
    feat_in: 1024
    num_classes: -1  # Will be set from vocabulary
    vocabulary: null  # Path to vocabulary file
  
  # CTC Loss
  loss:
    _target_: nemo.collections.asr.losses.CTCLoss
    zero_infinity: true
    reduction: "mean_batch"
  
  # Optimizer Configuration
  optim:
    name: adamw
    lr: 3e-5  # Lower learning rate for finetuning
    betas: [0.9, 0.999]
    weight_decay: 0.0001
    
    # Learning rate schedule
    sched:
      name: CosineAnnealing
      warmup_steps: 2000
      warmup_ratio: null
      min_lr: 1e-6
      max_steps: -1  # Computed from epochs

# Training Configuration
trainer:
  devices: 2  # 2x RTX 4090
  accelerator: gpu
  strategy: ddp  # Distributed Data Parallel
  num_nodes: 1
  max_epochs: 30  # Adjust based on convergence
  max_steps: -1
  val_check_interval: 1.0  # Validate every epoch
  check_val_every_n_epoch: 1
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: 16  # Mixed precision (FP16)
  log_every_n_steps: 100
  enable_progress_bar: true
  num_sanity_val_steps: 0
  
# Experiment Manager
exp_manager:
  exp_dir: ./experiments/parakeet_ctc_phoneme
  name: ${name}
  create_tensorboard_logger: true
  create_checkpoint_callback: true
  
  checkpoint_callback_params:
    monitor: "val_wer"  # Actually monitoring PER (Phoneme Error Rate)
    mode: "min"
    save_top_k: 3
    save_last: true
    always_save_nemo: true
    filename: '{epoch}-{val_wer:.4f}'
  
  resume_if_exists: true
  resume_ignore_no_checkpoint: true

# Dataset Configuration
model:
  train_ds:
    manifest_filepath: data/training/manifests/train.json
    sample_rate: 16000
    batch_size: 12  # Adjust based on GPU memory
    shuffle: true
    num_workers: 8
    pin_memory: true
    trim_silence: false
    max_duration: 20.0  # Skip samples longer than 20 seconds
    min_duration: 0.1   # Skip very short samples
    
  validation_ds:
    manifest_filepath: data/training/manifests/val.json
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    num_workers: 8
    pin_memory: true
    
  test_ds:
    manifest_filepath: data/training/manifests/test.json
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    num_workers: 8
```

### Training Launch Script

**File:** `scripts/training/05_train_parakeet_ctc.py`

```python
#!/usr/bin/env python3
"""
Launch Parakeet CTC phoneme finetuning.

Usage:
    python scripts/training/05_train_parakeet_ctc.py

With custom config:
    python scripts/training/05_train_parakeet_ctc.py \
        --config-path configs/training \
        --config-name parakeet_ctc_phoneme \
        trainer.max_epochs=50 \
        model.train_ds.batch_size=8
"""

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra_runner(config_path="../../configs/training", config_name="parakeet_ctc_phoneme")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    logging.info("="*60)
    logging.info("🎙️  Parakeet CTC Phoneme Finetuning")
    logging.info("="*60)
    logging.info("\nConfiguration:")
    logging.info(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Load vocabulary
    vocab_file = Path("data/training/vocabulary/ipa_phoneme_vocabulary.txt")
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_file}\n"
            "Run: python scripts/training/02_build_phoneme_vocabulary.py"
        )
    
    with open(vocab_file) as f:
        vocabulary = [line.strip() for line in f]
    
    logging.info(f"\n📖 Loaded vocabulary: {len(vocabulary)} tokens")
    logging.info(f"   Sample: {vocabulary[:10]}...")
    
    # Load pretrained model
    logging.info(f"\n🔄 Loading pretrained model: {cfg.model.restore_from}")
    model = EncDecCTCModel.from_pretrained(cfg.model.restore_from)
    
    # Update vocabulary (critical step!)
    logging.info("\n🔄 Updating model vocabulary for phonemes...")
    model.change_vocabulary(new_vocabulary=vocabulary)
    logging.info(f"   New vocabulary size: {model.decoder.vocabulary_size}")
    
    # Setup datasets
    logging.info("\n📂 Setting up datasets...")
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    
    if hasattr(cfg.model, 'test_ds'):
        model.setup_test_data(cfg.model.test_ds)
    
    # Setup optimization
    model.setup_optimization(cfg.model.optim)
    
    # Create trainer
    logging.info("\n🏋️  Creating trainer...")
    trainer = pl.Trainer(**cfg.trainer, **cfg.exp_manager)
    
    # Display training info
    logging.info("\n" + "="*60)
    logging.info("Training Configuration:")
    logging.info("="*60)
    logging.info(f"  Model: {cfg.model.restore_from}")
    logging.info(f"  Vocabulary size: {len(vocabulary)}")
    logging.info(f"  GPUs: {cfg.trainer.devices}")
    logging.info(f"  Batch size per GPU: {cfg.model.train_ds.batch_size}")
    logging.info(f"  Max epochs: {cfg.trainer.max_epochs}")
    logging.info(f"  Learning rate: {cfg.model.optim.lr}")
    logging.info(f"  Precision: FP{cfg.trainer.precision}")
    logging.info("="*60)
    
    # Start training
    logging.info("\n🚀 Starting training...\n")
    trainer.fit(model)
    
    # Test on best checkpoint
    logging.info("\n📊 Running final evaluation on test set...")
    if hasattr(cfg.model, 'test_ds'):
        trainer.test(model)
    
    logging.info("\n" + "="*60)
    logging.info("✅ Training Complete!")
    logging.info("="*60)
    
    # Save final model
    output_path = Path(cfg.exp_manager.exp_dir) / "final_model.nemo"
    model.save_to(str(output_path))
    logging.info(f"\n💾 Final model saved to: {output_path}")

if __name__ == "__main__":
    main()
```

---

## Multi-GPU Training Configuration

### Memory Optimization for 2x RTX 4090

**GPU Specifications:**
- Memory: 24GB per GPU
- Total: 48GB

**Model Memory Breakdown:**
- **Model weights:** ~4.5GB (1.1B params in FP32, ~2.2GB in FP16)
- **Optimizer states:** ~4.5GB (AdamW)
- **Batch activations:** ~15GB (batch_size=12, 16kHz audio)
- **Gradients:** ~2GB
- **Buffer/overhead:** ~2GB
- **Total per GPU:** ~22GB ✅ (safe)

### Optimal Training Settings

```yaml
trainer:
  devices: 2
  strategy: ddp
  precision: 16  # Halves memory usage
  
model:
  train_ds:
    batch_size: 12  # Per GPU
    num_workers: 8
    pin_memory: true
```

**Effective batch size:** 12 × 2 GPUs = 24 samples per step

### Training Launch Commands

```bash
# Standard launch (uses config defaults)
python scripts/training/05_train_parakeet