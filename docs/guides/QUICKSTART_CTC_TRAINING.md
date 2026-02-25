# Quick Start: Parakeet CTC Phoneme Training

**🎯 Goal:** Train Parakeet CTC 1.1B for direct IPA phoneme prediction  
**⚙️ Hardware:** 2x RTX 4090 GPUs  
**⏱️ Timeline:** 10-15 days

---

## Quick Setup (5 commands)

```bash
# 1. Install dependencies
pip install nemo_toolkit[asr] pytorch-lightning hydra-core

# 2. Download datasets (takes 2-4 hours)
python scripts/training/01_download_phoneme_datasets.py

# 3. Build phoneme vocabulary (~50-60 tokens)
python scripts/training/02_build_phoneme_vocabulary.py

# 4. Create NeMo manifests (convert to training format)
python scripts/training/03_create_ctc_manifests.py
python scripts/training/04_split_manifests.py

# 5. Start training (3-5 days on 2x 4090)
python scripts/training/05_train_parakeet_ctc.py
```

---

## Expected Results

### Phoneme Error Rate (PER)
- **Baseline (no training):** 40-60%
- **After 10 epochs:** 15-25%
- **After 30 epochs:** 8-15%
- **Target:** <10%

### Timing Accuracy
- **Frame resolution:** 10ms (CTC frame-level)
- **Phoneme boundaries:** ±20-30ms accuracy
- **Word boundaries:** ±50ms accuracy

### Training Time
- **Per epoch:** ~2-3 hours (500K samples)
- **Total (30 epochs):** ~60-90 hours (3-4 days)

---

## Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir experiments/parakeet_ctc_phoneme --port 6006

# Training logs
tail -f experiments/parakeet_ctc_phoneme/*/logs/train.log
```

**Key Metrics:**
- `train_loss` - Should decrease steadily
- `val_wer` - Actually PER (Phoneme Error Rate)
- `learning_rate` - Follows cosine schedule

---

## Evaluation & Testing

### 1. Phoneme Accuracy Test

```python
from nemo.collections.asr.models import EncDecCTCModel

# Load best checkpoint
model = EncDecCTCModel.restore_from(
    "experiments/parakeet_ctc_phoneme/checkpoints/best.ckpt"
)

# Transcribe test audio
audio_files = ["test1.wav", "test2.wav"]
predictions = model.transcribe(audio_files)

for audio, pred in zip(audio_files, predictions):
    print(f"{audio}: {pred}")
    # Output: "ð ɪ s ɪ z ə t ɛ s t"  (space-separated IPA)
```

### 2. Extract Phoneme Timestamps

```python
# Get frame-level alignments
predictions = model.transcribe(
    audio_files,
    return_hypotheses=True,
    timestamps=True
)

for hyp in predictions:
    # Extract phoneme timing
    for phoneme_info in hyp.timestep:
        phoneme = phoneme_info['char']
        start = phoneme_info['timestep'] * 0.01  # 10ms frames
        print(f"{phoneme}: {start:.3f}s")
```

### 3. Calculate Phoneme Error Rate

```python
import jiwer

# Ground truth (from dataset)
reference = "ð ɪ s ɪ z ə t ɛ s t"

# Prediction
hypothesis = model.transcribe(["audio.wav"])[0]

# Calculate PER (same as WER but for phonemes)
per = jiwer.wer(reference, hypothesis)
print(f"Phoneme Error Rate: {per:.2%}")
```

---

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution:** Reduce batch size
```bash
python scripts/training/05_train_parakeet_ctc.py \
    model.train_ds.batch_size=8
```

### Issue: Training too slow

**Solution:** Reduce dataset size
```python
# In 04_split_manifests.py
max_train_samples = 200000  # Use subset
```

### Issue: High PER (>20%)

**Possible causes:**
- Not enough epochs (train longer)
- Learning rate too high (reduce to 1e-5)
- Vocabulary mismatch (rebuild vocab)
- Data quality issues (filter bad samples)

### Issue: Vocabulary errors

**Solution:** Ensure all phonemes in vocabulary
```bash
# Rebuild vocabulary
python scripts/training/02_build_phoneme_vocabulary.py

# Verify coverage
python scripts/training/verify_vocabulary_coverage.py
```

---

## Next Steps After Training

### 1. Export Final Model

```python
# Convert checkpoint to .nemo format
model = EncDecCTCModel.restore_from("best.ckpt")
model.save_to("parakeet-ctc-phoneme-finetuned.nemo")
```

### 2. Upload to HuggingFace

```bash
# Install hub
pip install huggingface_hub

# Upload model
huggingface-cli upload \
    your-username/parakeet-ctc-phoneme-en \
    parakeet-ctc-phoneme-finetuned.nemo
```

### 3. Integrate into Speechline

```python
# Use in speechline pipeline
from speechline.transcribers import ParakeetCTCTranscriber

transcriber = ParakeetCTCTranscriber(
    model_checkpoint="your-username/parakeet-ctc-phoneme-en"
)

# Get phoneme transcriptions
results = transcriber.predict(
    dataset,
    output_offsets=True,
    return_timestamps="char"  # Frame-level phoneme timestamps
)
```

---

## File Structure

```
speechline/
├── scripts/training/
│   ├── 01_download_phoneme_datasets.py
│   ├── 02_build_phoneme_vocabulary.py
│   ├── 03_create_ctc_manifests.py
│   ├── 04_split_manifests.py
│   └── 05_train_parakeet_ctc.py
├── configs/training/
│   └── parakeet_ctc_phoneme.yaml
├── data/training/
│   ├── raw_datasets/
│   ├── vocabulary/
│   │   └── ipa_phoneme_vocabulary.txt
│   └── manifests/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── experiments/parakeet_ctc_phoneme/
    ├── checkpoints/
    └── logs/
```

---

## Important Notes

### Why CTC over TDT?
- ✅ **Direct phoneme prediction** (no conversion needed)
- ✅ **Frame-level timestamps** (10ms resolution)
- ✅ **Natural fit** for variable-length sequences
- ✅ **Proven architecture** for phoneme recognition

### Dataset Format
- Input: `phonemes_ipa: ["ð ɪ s", "ɪ z", "ə"]`
- Training: `text: "ð ɪ s ɪ z ə"` (space-separated)
- Output: Same format as training

### Memory Requirements
- Model: ~4.5GB
- Training: ~22GB per GPU
- Total: Safe for 24GB RTX 4090

---

## Summary

**Option 2 (Parakeet CTC)** is the optimal choice for phoneme finetuning:

1. ✅ No phoneme-to-word conversion needed
2. ✅ Direct IPA phoneme prediction  
3. ✅ Accurate frame-level timestamps
4. ✅ Well-tested for phoneme tasks
5. ✅ Faster to implement than Option 1

**Ready to implement?** All scripts are provided in the full guide:
- [`docs/guides/parakeet_ctc_phoneme_finetuning.md`](parakeet_ctc_phoneme_finetuning.md)

**Switch to Code mode** when ready to create the training scripts!