# Parakeet TDT Phoneme Finetuning Plan

**Date:** 2025-10-14  
**Goal:** Finetune `nvidia/parakeet-tdt-0.6b-v2` with phoneme-level datasets for improved phoneme recognition accuracy and timing information  
**Hardware:** 2x NVIDIA RTX 4090 GPUs

---

## Table of Contents

1. [Critical Architectural Considerations](#critical-architectural-considerations)
2. [Dataset Preparation Pipeline](#dataset-preparation-pipeline)
3. [NeMo Training Setup](#nemo-training-setup)
4. [Multi-GPU Configuration](#multi-gpu-configuration)
5. [Training Procedure](#training-procedure)
6. [Evaluation Strategy](#evaluation-strategy)
7. [Timeline and Resources](#timeline-and-resources)
8. [Risk Mitigation](#risk-mitigation)

---

## Critical Architectural Considerations

### 🚨 **MAJOR DECISION POINT: Model Architecture Mismatch**

**Problem Identified:**
- **Parakeet TDT** is a Token-and-Duration Transducer that predicts **word-level tokens**
- Your datasets contain **IPA phoneme sequences** (`phonemes_ipa` field)
- **TDT models cannot directly predict phonemes** - they use word-piece tokenization

**Three Solution Paths:**

#### Option 1: Convert Phonemes to Words (RECOMMENDED)
**Approach:** Convert IPA phoneme sequences back to words using a pronunciation dictionary.

**Pros:**
- Uses TDT's native word prediction capability
- Leverages TDT's timestamp prediction for word boundaries
- Maintains model architecture strengths
- Training is more stable

**Cons:**
- Loses direct phoneme-level supervision
- Requires accurate phoneme-to-word conversion
- May not perfectly align with original audio

**Implementation:**
```python
# Example conversion
phonemes_ipa = ["ð ɪ s", "ɪ z", "ə", "t ɛ s t"]  # Dataset format
# Convert to: "this is a test"
# Use lexicon: data/english_words_ipa.csv
```

#### Option 2: Use CTC-Based Model Instead (ALTERNATIVE)
**Approach:** Use `nvidia/parakeet-ctc-1.1b` which can predict phoneme sequences directly.

**Pros:**
- Direct phoneme prediction without conversion
- Better phoneme-level alignment
- Natural fit for phoneme datasets

**Cons:**
- Different model architecture than requested
- CTC may have less accurate timestamps than TDT
- Larger model (1.1B vs 0.6B parameters)

#### Option 3: Modify TDT for Phoneme Output (EXPERIMENTAL)
**Approach:** Retrain TDT's output layer to predict phoneme tokens instead of word-pieces.

**Pros:**
- Direct phoneme prediction with TDT timestamps
- Optimal for your use case

**Cons:**
- Requires significant model architecture changes
- Untested approach, high risk
- May need to retrain from scratch (weeks of compute)

### 📋 **RECOMMENDATION**

**Use Option 1** (Convert Phonemes to Words) for initial training:
1. Achieves your goals with lowest risk
2. Faster to implement and debug
3. Can transition to Option 2 or 3 if needed
4. Post-processing can extract phoneme timing from word predictions

---

## Dataset Preparation Pipeline

### 1. Dataset Inventory

| Dataset | HuggingFace ID | Schema | Estimated Size |
|---------|---------------|---------|----------------|
| YouTube EN | `bookbot/en_youtube_w2v-bert-2.0_filtered` | ✅ Has `phonemes_ipa` | 10K-50K |
| GigaSpeech | `bookbot/gigaspeech_w2v-bert-2.0_filtered` | ✅ Has `phonemes_ipa` | 100K+ |
| Common Voice 16.1 | `bookbot/common_voice_16_1_en_w2v-bert-2.0_filtered` | ✅ Has `phonemes_ipa` | 500K+ |
| LibriPhone | `bookbot/libriphone` | ✅ Has phonemes | 100K+ |
| EN-AU Dean2Zak | `bookbot/en-AU-Dean2Zak` | ❓ Check schema | Unknown |
| LJSpeech Phonemes | `bookbot/ljspeech_phonemes` | ❓ Check schema | ~13K |

**Common Schema:**
```python
{
    "audio": {"path": str, "array": np.ndarray, "sampling_rate": 16000},
    "language": str,  # "en"
    "id": str,
    "phonemes_ipa": list  # ["w ɝ d", "w ɪ θ", "phonemes"]
}
```

### 2. Dataset Download and Verification

**Script:** `scripts/training/01_download_datasets.py`

```python
#!/usr/bin/env python3
"""Download and verify all phoneme datasets."""

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

OUTPUT_DIR = Path("data/training/raw")

for name, dataset_id in DATASETS.items():
    print(f"\n📥 Downloading {name}...")
    dataset = load_dataset(dataset_id, split="train")
    
    # Save to disk
    output_path = OUTPUT_DIR / name
    dataset.save_to_disk(str(output_path))
    
    # Verify schema
    print(f"  ✅ {len(dataset)} samples")
    print(f"  Schema: {dataset.column_names}")
    print(f"  Saved to: {output_path}")
```

### 3. Phoneme-to-Word Conversion

**Script:** `scripts/training/02_convert_phonemes_to_words.py`

**Strategy:**
1. Load pronunciation lexicon (`data/english_words_ipa.csv`)
2. Build reverse mapping: IPA phonemes → word
3. For each sample's `phonemes_ipa`, look up corresponding words
4. Handle multi-word sequences and OOV (out-of-vocabulary) phonemes

```python
def convert_phonemes_to_text(phoneme_list: List[str], lexicon: Dict) -> str:
    """
    Convert IPA phoneme sequence to text.
    
    Args:
        phoneme_list: ["ð ɪ s", "ɪ z", "ə", "t ɛ s t"]
        lexicon: {("ð", "ɪ", "s"): "this", ...}
    
    Returns:
        "this is a test"
    """
    words = []
    for phoneme_word in phoneme_list:
        phonemes = tuple(phoneme_word.split())
        word = lexicon.get(phonemes, "<UNK>")
        words.append(word)
    return " ".join(words)
```

### 4. NeMo Manifest Format Conversion

**NeMo Training Format:**
```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcribed text here", "duration": 3.45}
{"audio_filepath": "/path/to/audio2.wav", "text": "another sample", "duration": 2.13}
```

**Script:** `scripts/training/03_create_nemo_manifests.py`

```python
#!/usr/bin/env python3
"""Convert datasets to NeMo manifest format."""

import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def create_manifest(dataset, output_path, lexicon):
    """Create NeMo manifest from dataset."""
    with open(output_path, 'w') as f:
        for item in tqdm(dataset):
            # Get audio path and duration
            audio_path = item["audio"]["path"]
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            duration = len(audio_array) / sr
            
            # Convert phonemes to text
            phonemes = item["phonemes_ipa"]
            text = convert_phonemes_to_text(phonemes, lexicon)
            
            # Skip if conversion failed
            if "<UNK>" in text:
                continue
            
            # Write manifest entry
            manifest_entry = {
                "audio_filepath": str(Path(audio_path).absolute()),
                "text": text.lower(),
                "duration": round(duration, 3)
            }
            f.write(json.dumps(manifest_entry) + "\n")
```

### 5. Dataset Splitting

**Split Strategy:**
- **Training:** 90% (max ~500K samples to keep manageable)
- **Validation:** 5%
- **Test:** 5%

**Script:** `scripts/training/04_split_datasets.py`

```python
from sklearn.model_selection import train_test_split

# Read full manifest
with open("data/training/manifests/all_combined.json") as f:
    all_data = [json.loads(line) for line in f]

# Split
train, temp = train_test_split(all_data, test_size=0.1, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Cap training set if too large
if len(train) > 500000:
    train = train[:500000]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

---

## NeMo Training Setup

### 1. Installation and Environment

**Requirements:**
```bash
# Install NeMo with ASR
pip install nemo_toolkit[asr]

# Additional dependencies
pip install pytorch-lightning==1.9.5
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0

# Verify CUDA and PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. Training Configuration YAML

**File:** `configs/training/parakeet_tdt_finetune.yaml`

```yaml
# Parakeet TDT Finetuning Configuration
name: "parakeet-tdt-0.6b-v2-phoneme-finetuned"

# Model Architecture
model:
  # Load pretrained checkpoint
  restore_from: "nvidia/parakeet-tdt-0.6b-v2"
  
  # Training parameters
  sample_rate: 16000
  log_prediction_samples: true
  
  # Data augmentation
  spec_augment:
    freq_masks: 2
    freq_width: 27
    time_masks: 10
    time_width: 0.05

  # Optimizer
  optim:
    name: adamw
    lr: 5e-5  # Lower for finetuning
    betas: [0.9, 0.999]
    weight_decay: 0.0001
    sched:
      name: WarmupAnnealing
      warmup_steps: 1000
      warmup_ratio: null
      min_lr: 1e-6

# Data Configuration  
trainer:
  devices: 2  # 2x RTX 4090
  accelerator: gpu
  strategy: ddp
  num_nodes: 1
  max_epochs: 50
  max_steps: -1  # Unlimited, use epochs
  val_check_interval: 1.0  # Check validation every epoch
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  precision: 16  # Mixed precision training
  
  # Logging
  logger: true
  log_every_n_steps: 100
  
  # Checkpointing
  enable_checkpointing: true
  
# Data Configuration
exp_manager:
  exp_dir: ./experiments/parakeet_tdt_finetune
  name: ${name}
  create_tensorboard_logger: true
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: "val_wer"
    mode: "min"
    save_top_k: 3
    save_best_model: true
    always_save_nemo: true
  
  resume_if_exists: true
  resume_ignore_no_checkpoint: true

# Dataset Configuration
model:
  train_ds:
    manifest_filepath: data/training/manifests/train.json
    sample_rate: 16000
    batch_size: 16  # Adjust based on GPU memory
    shuffle: true
    num_workers: 8
    pin_memory: true
    trim_silence: false
    
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

### 3. Training Script

**File:** `scripts/training/05_train_parakeet_tdt.py`

```python
#!/usr/bin/env python3
"""
Launch Parakeet TDT finetuning with NeMo.
"""

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf
import torch

@hydra_runner(config_path="configs/training", config_name="parakeet_tdt_finetune")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    logging.info("Configuration:")
    logging.info(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(42)
    
    # Load pretrained model
    logging.info(f"Loading model: {cfg.model.restore_from}")
    model = EncDecRNNTModel.from_pretrained(cfg.model.restore_from)
    
    # Update model configuration for finetuning
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    model.setup_optimization(cfg.model.optim)
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Start training
    logging.info("Starting training...")
    trainer.fit(model)
    
    # Test on best checkpoint
    logging.info("Running final evaluation on test set...")
    model.setup_test_data(cfg.model.test_ds)
    trainer.test(model)
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main()
```

---

## Multi-GPU Configuration

### DDP (Distributed Data Parallel) Setup

**For 2x RTX 4090:**
```yaml
trainer:
  devices: 2
  accelerator: gpu
  strategy: ddp  # DistributedDataParallel
  num_nodes: 1
```

**Memory Optimization:**
- **Batch size per GPU:** 8-16 (adjust based on 24GB VRAM)
- **Gradient accumulation:** 1-2 steps if needed
- **Mixed precision:** fp16 (reduces memory by ~50%)

**Expected Memory Usage:**
- Model: ~2.5GB
- Batch (16 samples): ~18GB
- Gradients & optimizer: ~3GB
- **Total per GPU:** ~20-22GB (safe for 24GB)

### Training Launch Command

```bash
# Single-node multi-GPU training
python scripts/training/05_train_parakeet_tdt.py \
    --config-name=parakeet_tdt_finetune \
    trainer.devices=2 \
    trainer.strategy=ddp \
    model.train_ds.batch_size=12 \
    model.optim.lr=5e-5

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/training/05_train_parakeet_tdt.py
```

---

## Training Procedure

### 1. Pre-Flight Checklist

- [ ] All datasets downloaded and verified
- [ ] Phoneme-to-word conversion completed (>95% success rate)
- [ ] NeMo manifests created (train/val/test)
- [ ] Training config validated
- [ ] GPUs tested and available
- [ ] Disk space sufficient (>500GB for checkpoints)
- [ ] TensorBoard accessible

### 2. Training Phases

**Phase 1: Initial Finetuning (Epochs 1-10)**
- Learning rate: 5e-5
- Monitor: Validation WER
- Expected: Rapid initial improvement

**Phase 2: Stabilization (Epochs 11-30)**
- Learning rate: Decays to 1e-5
- Monitor: WER plateaus
- Expected: Gradual refinement

**Phase 3: Final Polish (Epochs 31-50)**
- Learning rate: 1e-6
- Monitor: Overfitting risk
- Early stopping if val WER increases

### 3. Monitoring

**TensorBoard Metrics:**
```bash
tensorboard --logdir experiments/parakeet_tdt_finetune --port 6006
```

**Key Metrics:**
- `train/loss` - Should decrease steadily
- `val_wer` - Word Error Rate (target: <10%)
- `val_loss` - Should follow train loss
- `learning_rate` - Verify schedule

**Alerts:**
- Val loss increases while train loss decreases → Overfitting
- Both losses plateau early → Increase learning rate
- Loss spikes → Reduce batch size or learning rate

### 4. Checkpointing Strategy

**Checkpoint Schedule:**
- Save top 3 models by validation WER
- Save every 5 epochs as backup
- Keep final epoch checkpoint

**Directory Structure:**
```
experiments/parakeet_tdt_finetune/
├── checkpoints/
│   ├── epoch=10-val_wer=0.15.ckpt
│   ├── epoch=25-val_wer=0.08.ckpt  ← Best
│   └── epoch=50-val_wer=0.09.ckpt
├── logs/
│   └── tensorboard/
└── nemo_experiments/
    └── parakeet-tdt-0.6b-v2-phoneme-finetuned.nemo  ← Final model
```

---

## Evaluation Strategy

### 1. Phoneme-Level Evaluation

Since we trained on word targets, we need to extract phoneme predictions for evaluation.

**Approach:**
1. Run inference on test set
2. Convert predicted words back to phonemes using G2P
3. Compare against original `phonemes_ipa` labels
4. Calculate Phoneme Error Rate (PER)

**Script:** `scripts/training/06_evaluate_phoneme_accuracy.py`

```python
#!/usr/bin/env python3
"""Evaluate phoneme accuracy after word-level finetuning."""

from nemo.collections.asr.models import EncDecRNNTModel
from gruut import sentences
from datasets import load_from_disk
import jiwer

def word_to_phonemes(word: str) -> List[str]:
    """Convert word to IPA phonemes using G2P."""
    phonemes = []
    for sent in sentences(word, lang="en-us"):
        for w in sent:
            if w.phonemes:
                phonemes.extend(w.phonemes)
    return phonemes

def evaluate_phoneme_accuracy(model_path, test_manifest):
    """Evaluate PER (Phoneme Error Rate)."""
    model = EncDecRNNTModel.restore_from(model_path)
    
    # Load test data
    with open(test_manifest) as f:
        test_data = [json.loads(line) for line in f]
    
    total_per = 0
    for item in tqdm(test_data):
        # Get prediction
        audio_path = item["audio_filepath"]
        predicted_words = model.transcribe([audio_path])[0]
        
        # Convert to phonemes
        pred_phonemes = []
        for word in predicted_words.split():
            pred_phonemes.extend(word_to_phonemes(word))
        
        # Get ground truth phonemes
        # Note: Need to map back to original dataset
        true_phonemes = get_true_phonemes(item["audio_filepath"])
        
        # Calculate PER
        per = jiwer.wer(
            " ".join(true_phonemes),
            " ".join(pred_phonemes)
        )
        total_per += per
    
    avg_per = total_per / len(test_data)
    print(f"Average Phoneme Error Rate: {avg_per:.2%}")
    return avg_per
```

### 2. Timing Accuracy Evaluation

**Metrics:**
- **Boundary Accuracy:** How close are predicted word boundaries to true phoneme boundaries?
- **Duration Correlation:** Do predicted durations correlate with phoneme count?

**Script:** `scripts/training/07_evaluate_timing.py`

```python
def evaluate_timing_accuracy(model, test_samples):
    """Evaluate timestamp accuracy."""
    timing_errors = []
    
    for sample in test_samples:
        # Get predictions with timestamps
        result = model.transcribe(
            [sample["audio"]],
            return_hypotheses=True,
            timestamps=True
        )[0]
        
        # Extract word boundaries
        word_boundaries = []
        for word_info in result.timestamp['word']:
            word_boundaries.append({
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"]
            })
        
        # Compare with phoneme boundaries (ground truth)
        # This requires complex alignment - see full implementation
        error = calculate_boundary_error(word_boundaries, sample["phonemes"])
        timing_errors.append(error)
    
    return {
        "mean_error_ms": np.mean(timing_errors),
        "median_error_ms": np.median(timing_errors),
        "std_error_ms": np.std(timing_errors)
    }
```

---

## Timeline and Resources

### Estimated Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Setup** | 2-3 days | Environment, downloads, verification |
| **Data Prep** | 3-5 days | Phoneme conversion, manifest creation |
| **Training** | 3-7 days | 50 epochs on 2x RTX 4090 |
| **Evaluation** | 2-3 days | Testing, analysis, documentation |
| **Total** | **10-18 days** | Complete pipeline |

### Training Time Estimates

**Per Epoch:**
- Dataset size: ~500K samples
- Batch size: 12 per GPU (24 total)
- Samples per second: ~100 (DDP on 2x 4090)
- **Time per epoch:** ~2-3 hours
- **Total training (50 epochs):** ~100-150 hours (4-6 days)

### Resource Requirements

**Storage:**
- Raw datasets: ~200GB
- Processed manifests: ~500MB
- Checkpoints: ~10GB per checkpoint × 10 = 100GB
- Logs: ~5GB
- **Total:** ~305GB

**Compute:**
- GPU memory: 20-22GB per GPU (48GB total)
- System RAM: 64GB recommended
- CPU cores: 16+ for data loading

---

## Risk Mitigation

### Risk 1: Phoneme-to-Word Conversion Failures

**Problem:** Many phoneme sequences may not map to known words.

**Mitigation:**
- Build comprehensive pronunciation lexicon
- Use multiple G2P systems (Gruut, eSpeak, CMUDict)
- Allow fuzzy matching for similar phonemes
- Filter out samples with <80% conversion confidence

**Fallback:** If conversion rate <70%, switch to Option 2 (CTC model)

### Risk 2: Training Instability

**Problem:** Model may diverge or overfit quickly.

**Mitigation:**
- Start with very low learning rate (5e-5)
- Use gradient clipping (1.0)
- Monitor validation metrics closely
- Implement early stopping
- Save frequent checkpoints

### Risk 3: Poor Phoneme Timing Extraction

**Problem:** Word-level timestamps may not align well with phoneme boundaries.

**Mitigation:**
- Post-process with forced alignment (NFA)
- Use word duration to interpolate phoneme positions
- Validate against known phoneme durations
- Consider hybrid approach (TDT + CTC)

### Risk 4: GPU Memory Issues

**Problem:** 2x RTX 4090 may struggle with large batches.

**Mitigation:**
- Reduce batch size to 8 per GPU
- Enable gradient accumulation
- Use mixed precision (fp16)
- Clear cache between batches

---

## Next Steps

### Immediate Actions (Week 1)

1. **Create project structure:**
   ```bash
   mkdir -p scripts/training configs/training data/training/{raw,manifests}
   mkdir -p experiments/parakeet_tdt_finetune
   ```

2. **Install dependencies:**
   ```bash
   pip install nemo_toolkit[asr] pytorch-lightning hydra-core omegaconf
   ```

3. **Download test dataset (GigaSpeech filtered):**
   ```bash
   python scripts/training/01_download_datasets.py --dataset gigaspeech --limit 1000
   ```

4. **Test phoneme-to-word conversion:**
   ```bash
   python scripts/training/02_convert_phonemes_to_words.py --test-mode
   ```

5. **Verify GPU setup:**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

### Questions to Resolve Before Implementation

1. ✅ **Model choice:** Parakeet TDT 0.6B confirmed
2. ⚠️ **Architecture approach:** Phoneme-to-word conversion (Option 1) - confirm acceptance
3. ❓ **Conversion quality threshold:** What minimum success rate is acceptable? (Recommend >80%)
4. ❓ **Training budget:** How many epochs/days available for training?
5. ❓ **Evaluation priority:** Phoneme accuracy vs. timing accuracy - which is more critical?

---

## Appendix

### A. Alternative: Using Parakeet CTC for Direct Phoneme Prediction

If phoneme-to-word conversion proves problematic, switch to CTC model:

```python
# Use Parakeet CTC instead
model = EncDecCTCModel.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Modify vocabulary to include all IPA phonemes
phoneme_vocab = extract_unique_phonemes(all_datasets)
model.change_vocabulary(new_vocabulary=phoneme_vocab)

# Train directly on phonemes
manifest_entry = {
    "audio_filepath": "/path/to/audio.wav",
    "text": "ð ɪ s ɪ z ə t ɛ s t",  # Space-separated phonemes
    "duration": 2.5
}
```

### B. Useful NeMo Commands

```bash
# Check model configuration
nemo-asr model_info --model nvidia/parakeet-tdt-0.6b-v2

# Convert checkpoint to .nemo format
python scripts/convert_ckpt_to_nemo.py --checkpoint epoch=25.ckpt

# Resume training from checkpoint
python scripts/training/05_train_parakeet_tdt.py \
    trainer.resume_from_checkpoint=path/to/checkpoint.ckpt
```

### C. Monitoring Commands

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f experiments/parakeet_tdt_finetune/logs/train.log

# Check TensorBoard
tensorboard --logdir experiments/ --port 6006 --bind_all
```

---

## Conclusion

This plan provides a comprehensive roadmap for finetuning Parakeet TDT on phoneme datasets. The critical decision is whether to proceed with phoneme-to-word conversion (Option 1) or switch to a CTC-based model (Option 2). 

**Recommended path:** Start with Option 1, validate conversion quality, and fall back to Option 2 if needed.

Once you confirm the approach, we can proceed with implementation in Code mode.