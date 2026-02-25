#!/usr/bin/env python3
"""
Multi-GPU Indonesian ASR Transcription
Splits workload across available GPUs using threading
"""

import os
import sys
import argparse
import threading
import gc
import warnings
from pathlib import Path
from typing import List, Dict, Union
import torch

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='librosa.core.audio.__audioread_load')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.config import Config
from speechline.transcribers import Wav2Vec2Transcriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_txt


class MultiGPUTranscriber:
    """Transcriber that splits work across multiple GPUs using threading"""
    
    def __init__(self, config: Config, gpu_ids: List[int]):
        """
        Initialize multi-GPU transcriber
        
        Args:
            config: SpeechLine Config object
            gpu_ids: List of GPU IDs to use (e.g., [0, 1])
        """
        self.config = config
        self.gpu_ids = gpu_ids
        self.transcribers = []
        
        # Progress tracking
        self.progress_lock = threading.Lock()
        self.total_processed = 0
        self.total_files = 0
        
        # Create a transcriber instance for each GPU
        for gpu_id in gpu_ids:
            transcriber = Wav2Vec2Transcriber(
                config.transcriber.model,
                device=gpu_id
            )
            self.transcribers.append((gpu_id, transcriber))
            print(f"✓ Initialized transcriber on GPU {gpu_id}")
    
    def update_progress(self, processed: int):
        """Thread-safe progress update"""
        with self.progress_lock:
            self.total_processed += processed
            percentage = (self.total_processed / self.total_files) * 100
            print(f"\r[OVERALL] Progress: {self.total_processed}/{self.total_files} files ({percentage:.1f}%)", end='', flush=True)
    
    def transcribe_batch(
        self,
        gpu_id: int,
        transcriber: Wav2Vec2Transcriber,
        df_chunk,
        chunk_id: int,
        results: List,
        batch_size: int = 50
    ):
        """
        Transcribe a batch of audio files on a specific GPU with memory management
        
        Args:
            gpu_id: GPU device ID
            transcriber: Transcriber instance
            df_chunk: DataFrame chunk to process
            chunk_id: Chunk identifier
            results: Shared list to store results
            batch_size: Process files in smaller batches to avoid memory buildup
        """
        try:
            total_in_chunk = len(df_chunk)
            print(f"\n[GPU {gpu_id}] Starting chunk {chunk_id} with {total_in_chunk} files")
            
            processed_count = 0
            
            # Process in smaller batches to avoid memory leak
            for batch_start in range(0, total_in_chunk, batch_size):
                batch_end = min(batch_start + batch_size, total_in_chunk)
                batch = df_chunk.iloc[batch_start:batch_end]
                
                # Format dataset
                dataset = format_audio_dataset(batch, sampling_rate=transcriber.sampling_rate)
                
                # Predict
                output_offsets = transcriber.predict(
                    dataset=dataset,
                    chunk_length_s=self.config.transcriber.chunk_length_s,
                    output_offsets=True,
                    return_timestamps=self.config.transcriber.return_timestamps,
                    keep_whitespace=self.config.segmenter.keep_whitespace,
                )
                
                # Export to TXT files
                for audio_path, offsets in zip(batch["audio"], output_offsets):
                    if offsets:
                        txt_path = Path(audio_path).with_suffix(".txt")
                        export_transcripts_txt(str(txt_path), offsets)
                
                processed_count += len(batch)
                
                # Update global progress
                self.update_progress(len(batch))
                
                # Clear memory after each batch
                del dataset
                del output_offsets
                torch.cuda.empty_cache()
                gc.collect()
            
            # Store results
            results[chunk_id] = {
                'gpu_id': gpu_id,
                'files_processed': processed_count,
                'success': True
            }
            
            print(f"\n[GPU {gpu_id}] Completed chunk {chunk_id}: {processed_count} files")
            
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error in chunk {chunk_id}: {str(e)}")
            results[chunk_id] = {
                'gpu_id': gpu_id,
                'files_processed': 0,
                'success': False,
                'error': str(e)
            }
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def transcribe_parallel(self, df):
        """
        Transcribe entire dataset in parallel across GPUs
        
        Args:
            df: DataFrame with audio files
        """
        num_gpus = len(self.gpu_ids)
        self.total_files = len(df)
        self.total_processed = 0
        
        print(f"\n{'='*60}")
        print(f"Multi-GPU Transcription Setup")
        print(f"{'='*60}")
        print(f"Total files: {self.total_files}")
        print(f"GPUs: {num_gpus} ({self.gpu_ids})")
        print(f"Files per GPU: ~{self.total_files // num_gpus}")
        print(f"{'='*60}\n")
        
        # Split dataframe into chunks
        chunk_size = self.total_files // num_gpus
        chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            if i == num_gpus - 1:
                # Last chunk gets remaining files
                end_idx = self.total_files
            else:
                end_idx = (i + 1) * chunk_size
            
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)
            print(f"GPU {self.gpu_ids[i]}: Will process files {start_idx} to {end_idx-1} ({len(chunk)} files)")
        
        # Create threads
        threads = []
        results = [None] * num_gpus
        
        for i, (gpu_id, transcriber) in enumerate(self.transcribers):
            thread = threading.Thread(
                target=self.transcribe_batch,
                args=(gpu_id, transcriber, chunks[i], i, results)
            )
            threads.append(thread)
        
        # Start all threads
        print(f"\n{'='*60}")
        print("Starting parallel transcription...")
        print(f"{'='*60}\n")
        
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Print summary
        print(f"\n{'='*60}")
        print("Transcription Complete - Summary")
        print(f"{'='*60}")
        
        final_count = 0
        for i, result in enumerate(results):
            if result and result['success']:
                print(f"\nGPU {result['gpu_id']}: ✓ {result['files_processed']} files")
                final_count += result['files_processed']
            else:
                error = result.get('error', 'Unknown') if result else 'No result'
                print(f"\nGPU {self.gpu_ids[i]}: ✗ Failed - {error}")
        
        print(f"\nTotal files processed: {final_count}/{self.total_files}")
        print(f"{'='*60}\n")
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Indonesian ASR Transcription"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="examples/id_config.json",
        help="Configuration file"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU IDs (e.g., '0,1' or '0,1,2,3')"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (useful for large directories)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have .txt transcripts"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs available")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    if max(gpu_ids) >= available_gpus:
        print(f"ERROR: Requested GPU {max(gpu_ids)} but only {available_gpus} GPUs available")
        sys.exit(1)
    
    print(f"Using GPUs: {gpu_ids}")
    
    # Load config
    config = Config(args.config)
    
    # Optimized file scanning for large directories
    print(f"\nScanning for audio files in {args.input_dir}...")
    from glob import glob
    import pandas as pd
    
    audio_pattern = f"{args.input_dir}/**/*.{config.audio_extension}"
    print(f"Searching pattern: {audio_pattern}")
    print("This may take a while for large directories...")
    
    audios = sorted(glob(audio_pattern, recursive=True))
    print(f"Found {len(audios)} total audio files")
    
    # Filter by size
    audios = [a for a in audios if Path(a).stat().st_size > 0]
    print(f"After size filter: {len(audios)} files")
    
    # Skip files with existing transcripts if requested
    if args.skip_existing:
        print("Filtering out files with existing transcripts...")
        audios_no_txt = []
        for audio_path in audios:
            txt_path = Path(audio_path).with_suffix(".txt")
            if not txt_path.exists():
                audios_no_txt.append(audio_path)
        audios = audios_no_txt
        print(f"Files without transcripts: {len(audios)} files")
    
    # Limit max files if specified
    if args.max_files and len(audios) > args.max_files:
        print(f"Limiting to {args.max_files} files (use --max_files to change)")
        audios = audios[:args.max_files]
    
    if len(audios) == 0:
        print("No audio files to process!")
        sys.exit(0)
    
    # Create DataFrame
    print(f"\nCreating DataFrame with {len(audios)} files...")
    df = pd.DataFrame({"audio": audios})
    df["id"] = df["audio"].apply(lambda f: Path(f).stem)
    df["language_code"] = df["audio"].apply(lambda f: Path(f).parent.name)
    df["language"] = df["language_code"].apply(lambda f: f.split("-")[0])
    df["ground_truth"] = ""  # We're only creating transcripts, not checking them
    
    print(f"Ready to process: {len(df)} files")
    
    # Create multi-GPU transcriber
    transcriber = MultiGPUTranscriber(config, gpu_ids)
    
    # Run parallel transcription
    transcriber.transcribe_parallel(df)


if __name__ == "__main__":
    main()