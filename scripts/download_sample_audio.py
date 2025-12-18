#!/usr/bin/env python3
"""
Download sample audio files from Hugging Face dataset.
"""

import os
from pathlib import Path
import json
from datasets import load_dataset
import soundfile as sf
import numpy as np

def download_samples(
    dataset_name: str = "bookbot/en_youtube_w2v-bert-2.0_filtered",
    num_samples: int = 20,
    output_dir: str = "sampler"
):
    """
    Download audio samples from Hugging Face dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        num_samples: Number of samples to download
        output_dir: Output directory for samples
    """
    print(f"📥 Loading dataset: {dataset_name}")
    
    # Load dataset with streaming to avoid loading entire dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created output directory: {output_path}")
    
    # Download samples
    print(f"🎵 Downloading {num_samples} samples...")
    metadata = []
    
    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break
        
        # Get audio data
        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        sampling_rate = audio_data["sampling_rate"]
        
        # Get metadata
        sample_id = sample.get("id", f"sample_{idx}")
        language = sample.get("language", "en")
        phonemes_ipa = sample.get("phonemes_ipa", [])
        
        # Save audio file
        audio_filename = f"{idx:03d}_{sample_id}.wav"
        audio_path = output_path / audio_filename
        sf.write(str(audio_path), audio_array, sampling_rate)
        
        # Save metadata
        meta_filename = f"{idx:03d}_{sample_id}.json"
        meta_path = output_path / meta_filename
        
        metadata_entry = {
            "id": sample_id,
            "audio_file": audio_filename,
            "language": language,
            "phonemes_ipa": phonemes_ipa,
            "sampling_rate": sampling_rate,
            "duration_seconds": len(audio_array) / sampling_rate
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata_entry, f, indent=2)
        
        metadata.append(metadata_entry)
        print(f"  ✅ Downloaded: {audio_filename} ({metadata_entry['duration_seconds']:.2f}s)")
    
    # Save combined metadata
    combined_meta_path = output_path / "metadata.json"
    with open(combined_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Successfully downloaded {len(metadata)} samples to {output_path}")
    print(f"📊 Total duration: {sum(m['duration_seconds'] for m in metadata):.2f} seconds")
    print(f"📝 Metadata saved to: {combined_meta_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample audio from HuggingFace dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bookbot/en_youtube_w2v-bert-2.0_filtered",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sampler",
        help="Output directory for samples"
    )
    
    args = parser.parse_args()
    
    download_samples(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )