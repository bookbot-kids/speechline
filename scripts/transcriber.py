#!/usr/bin/env python3
"""
Parallel Audio Transcription Script
Transcribes audio files using multiple GPUs with optimized performance

Supports multiple transcriber models:
- parakeet_tdt (default) - NVIDIA Parakeet TDT 0.6B
- parakeet - NVIDIA Parakeet CTC
- whisper - OpenAI Whisper
- wav2vec2 - Facebook Wav2Vec2
- canary - NVIDIA Canary

Features:
- Auto-detects available GPUs
- CUDA graphs support for Parakeet models (requires cuda-python>=12.3)
- Recursive directory scanning
- Multiple audio format support (.mp3, .aac, .wav, .flac, .m4a)
- Resume capability (skips already transcribed files)
- Progress bar with real-time speed tracking
- Detailed logging

Performance: 1,500-2,000+ files/sec with CUDA graphs enabled (Parakeet TDT)
"""

import os
import sys
from pathlib import Path
import multiprocessing as mp
import argparse
import time
import traceback
import logging
from tqdm import tqdm
from datetime import datetime

# Setup logging
def setup_logging(output_dir: str):
    """Setup logging to file in output directory"""
    log_file = os.path.join(output_dir, f"transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def worker_process(
    worker_id: int,
    gpu_id: int,
    file_queue: mp.Queue,
    batch_size: int,
    progress_queue: mp.Queue,
    ready_queue: mp.Queue,
    log_queue: mp.Queue,
    transcriber_type: str,
    transcriber_model: str,
    enable_cuda_graphs: bool,
    debug: bool = False
):
    """
    Worker process for parallel transcription.
    
    Args:
        worker_id: Unique worker identifier
        gpu_id: GPU device ID
        file_queue: Queue to receive batches of audio file paths
        batch_size: Number of files to process per batch
        progress_queue: Queue for reporting progress
        ready_queue: Queue for signaling model load completion
        log_queue: Queue for logging messages
        transcriber_type: Type of transcriber (parakeet_tdt, whisper, etc.)
        transcriber_model: Model checkpoint/name
        enable_cuda_graphs: Whether to enable CUDA graphs (Parakeet only)
    """
    # Create debug log file for this worker if debug enabled
    debug_log = None
    if debug:
        debug_log_path = f"/tmp/transcriber_worker_{worker_id}_debug.log"
        debug_log = open(debug_log_path, 'w', buffering=1)
        debug_log.write(f"Worker {worker_id} (GPU{gpu_id}): Starting debug logging\n")
        debug_log.flush()
    
    try:
        # CRITICAL: Redirect stderr to suppress ALL internal progress bars
        # This ensures only the main progress bar is shown
        # UNLESS debug mode is enabled
        import sys as worker_sys
        if not debug:
            worker_sys.stderr = open(os.devnull, 'w')
        else:
            if debug_log:
                debug_log.write(f"Worker {worker_id} (GPU{gpu_id}): Debug mode - stderr NOT redirected\n")
                debug_log.flush()
        
        # Set GPU BEFORE any torch/CUDA imports
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Add project root to Python path
        project_root = Path(__file__).parent.parent
        worker_sys.path.insert(0, str(project_root))
        
        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Initializing...")
        
        # Import after GPU isolation
        import torch
        from datasets import Dataset, Audio
        
        # Import appropriate transcriber
        if transcriber_type == "parakeet_tdt":
            from speechline.transcribers import ParakeetTDTTranscriber
        elif transcriber_type == "parakeet":
            from speechline.transcribers import ParakeetTranscriber
        elif transcriber_type == "whisper":
            from speechline.transcribers import WhisperTranscriber
        elif transcriber_type == "wav2vec2":
            from speechline.transcribers import Wav2Vec2Transcriber
        elif transcriber_type == "canary":
            from speechline.transcribers import CanaryTranscriber
        else:
            log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): ERROR - Unknown transcriber type: {transcriber_type}")
            ready_queue.put({'worker_id': worker_id, 'status': 'failed'})
            return
        
        # Verify GPU
        if not torch.cuda.is_available():
            log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): ERROR - CUDA not available!")
            ready_queue.put({'worker_id': worker_id, 'status': 'failed'})
            return
        
        actual_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(actual_device)
        vram_total = torch.cuda.get_device_properties(actual_device).total_memory / 1e9
        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Using device {actual_device} ({device_name}, {vram_total:.1f}GB VRAM)")
        
        # Load model
        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Loading {transcriber_type} model: {transcriber_model}")
        
        try:
            # Set environment variable for CUDA graphs before loading model (Parakeet only)
            if transcriber_type in ["parakeet_tdt", "parakeet"] and enable_cuda_graphs:
                os.environ["NEMO_ENABLE_CUDA_GRAPHS"] = "1"
                
                # Check if cuda-python is available
                try:
                    import cuda
                    # Try to get version, but don't crash if it's not available
                    try:
                        cuda_version = cuda.__version__
                        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): CUDA graphs enabled with cuda-python {cuda_version}")
                    except AttributeError:
                        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): CUDA graphs enabled with cuda-python (version unknown)")
                except ImportError:
                    log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): WARNING - cuda-python not found, CUDA graphs may not work optimally")
            else:
                os.environ["NEMO_ENABLE_CUDA_GRAPHS"] = "0"
            
            # Initialize transcriber based on type
            if transcriber_type == "parakeet_tdt":
                transcriber = ParakeetTDTTranscriber(
                    model_checkpoint=transcriber_model,
                    transcriber_device=f"cuda:{actual_device}"
                )
            elif transcriber_type == "parakeet":
                transcriber = ParakeetTranscriber(
                    model_checkpoint=transcriber_model,
                    transcriber_device=f"cuda:{actual_device}"
                )
            elif transcriber_type == "whisper":
                transcriber = WhisperTranscriber(model_checkpoint=transcriber_model)
            elif transcriber_type == "wav2vec2":
                transcriber = Wav2Vec2Transcriber(model_checkpoint=transcriber_model)
            elif transcriber_type == "canary":
                transcriber = CanaryTranscriber(
                    model_checkpoint=transcriber_model,
                    torch_dtype="float16"
                )
            
            vram_used = torch.cuda.memory_allocated(actual_device) / 1e9
            log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Model loaded! (Using {vram_used:.1f}GB VRAM)")
            
            # Signal ready
            ready_queue.put({'worker_id': worker_id, 'status': 'ready'})
            
        except Exception as e:
            error_msg = f"Worker {worker_id} (GPU{gpu_id}): Failed to load model: {str(e)}\n{traceback.format_exc()}"
            log_queue.put(error_msg)
            ready_queue.put({'worker_id': worker_id, 'status': 'failed'})
            return
        
        # Process batches
        files_processed = 0
        
        while True:
            # Get next batch
            batch_files = file_queue.get()
            
            if batch_files is None:  # Shutdown signal
                log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Received shutdown signal")
                break
            
            try:
                # Filter files that need processing
                files_to_process = []
                for audio_path in batch_files:
                    # Determine transcript path based on audio format
                    ext = os.path.splitext(audio_path)[1]
                    transcript_path = audio_path.replace(ext, '.txt')
                    if not os.path.exists(transcript_path):
                        files_to_process.append(audio_path)
                
                if not files_to_process:
                    # All files in batch already processed
                    progress_queue.put({
                        'worker_id': worker_id,
                        'files_processed': len(batch_files),
                        'skipped': len(batch_files)
                    })
                    continue
                
                # Create dataset with proper audio decoding
                # For AAC files, we need to ensure ffmpeg is used
                try:
                    dataset = Dataset.from_dict({
                        "audio": files_to_process
                    }).cast_column("audio", Audio(sampling_rate=16000, decode=True))
                    
                    # Transcribe
                    transcripts = transcriber.predict(dataset)
                except Exception as audio_error:
                    # If Audio column fails, try loading audio files individually
                    log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Audio loading failed, trying alternative method: {str(audio_error)}")
                    
                    # Use librosa or torchaudio as fallback
                    import librosa
                    import numpy as np
                    
                    audio_arrays = []
                    for audio_path in files_to_process:
                        try:
                            # Load audio using librosa which supports AAC via ffmpeg
                            # Force float32 to avoid dtype mixing issues
                            audio_array, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
                            audio_arrays.append(audio_array)
                        except Exception as e:
                            log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Failed to load {audio_path}: {e}")
                            # Skip this file with consistent dtype
                            audio_arrays.append(np.array([], dtype=np.float32))
                    
                    # Create dataset from loaded audio arrays
                    dataset = Dataset.from_dict({
                        "audio": [{"array": arr, "sampling_rate": 16000} for arr in audio_arrays]
                    })
                    
                    # Filter out empty arrays
                    valid_indices = [i for i, arr in enumerate(audio_arrays) if len(arr) > 0]
                    if not valid_indices:
                        raise Exception("No valid audio files in batch")
                    
                    dataset = dataset.select(valid_indices)
                    files_to_process = [files_to_process[i] for i in valid_indices]
                    
                    # Transcribe
                    transcripts = transcriber.predict(dataset)
                
                # Save transcripts
                for audio_path, transcript in zip(files_to_process, transcripts):
                    ext = os.path.splitext(audio_path)[1]
                    transcript_path = audio_path.replace(ext, '.txt')
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                
                files_processed += len(files_to_process)
                
                # Report progress
                progress_queue.put({
                    'worker_id': worker_id,
                    'files_processed': len(files_to_process),
                    'skipped': len(batch_files) - len(files_to_process)
                })
                
            except Exception as e:
                error_msg = f"Worker {worker_id} (GPU{gpu_id}): Batch processing error: {str(e)}\n{traceback.format_exc()}"
                log_queue.put(error_msg)
                # Report as skipped to keep progress moving
                progress_queue.put({
                    'worker_id': worker_id,
                    'files_processed': 0,
                    'skipped': len(batch_files)
                })
        
        log_queue.put(f"Worker {worker_id} (GPU{gpu_id}): Completed {files_processed:,} files")
        
    except Exception as e:
        error_msg = f"Worker {worker_id} (GPU{gpu_id}): Fatal error: {str(e)}\n{traceback.format_exc()}"
        log_queue.put(error_msg)
        
        # Write to debug log if available
        if debug_log:
            debug_log.write(f"\nFATAL ERROR:\n{error_msg}\n")
            debug_log.flush()
        
        # Also write directly to file as fallback
        try:
            with open(f"/tmp/transcriber_worker_{worker_id}_crash.log", 'w') as f:
                f.write(error_msg)
        except:
            pass
        
        try:
            ready_queue.put({'worker_id': worker_id, 'status': 'failed'})
        except:
            pass
    finally:
        if debug_log:
            debug_log.write(f"Worker {worker_id} (GPU{gpu_id}): Shutting down\n")
            debug_log.close()


def log_writer_process(log_queue: mp.Queue, log_file: str, debug: bool = False):
    """Dedicated process for writing logs to file"""
    try:
        # Write startup marker
        with open(log_file, 'a', encoding='utf-8') as f:
            startup_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log writer started\n"
            f.write(startup_msg)
            f.flush()
        
        if debug:
            print(f"DEBUG: Log writer started, writing to {log_file}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            while True:
                try:
                    msg = log_queue.get(timeout=60)
                    if msg is None:  # Shutdown signal
                        if debug:
                            print("DEBUG: Log writer received shutdown signal")
                        break
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_line = f"[{timestamp}] {msg}\n"
                    f.write(log_line)
                    f.flush()
                except Exception as e:
                    # Write error to file
                    error_line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LOG WRITER ERROR: {str(e)}\n"
                    f.write(error_line)
                    f.flush()
                    if debug:
                        print(f"DEBUG: Log writer error: {e}")
    except Exception as e:
        # Write to emergency log file
        try:
            with open("/tmp/transcriber_log_writer_crash.log", 'w') as f:
                f.write(f"Log writer crashed: {str(e)}\n{traceback.format_exc()}")
        except:
            pass
        if debug:
            print(f"DEBUG: Log writer crashed: {e}")
            traceback.print_exc()


def detect_gpus():
    """Detect number of available GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except:
        pass
    return 0


# Default model checkpoints for each transcriber type
DEFAULT_MODELS = {
    "parakeet_tdt": "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet": "nvidia/parakeet-ctc-0.6b",
    "whisper": "openai/whisper-large-v3",
    "wav2vec2": "facebook/wav2vec2-large-960h-lv60-self",
    "canary": "nvidia/canary-1b"
}


def main():
    """Main orchestration with CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='Parallel transcription of audio files with multiple transcriber support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe with default Parakeet TDT (auto-detect GPUs)
  python scripts/transcriber.py /path/to/audio

  # Use Whisper transcriber
  python scripts/transcriber.py /path/to/audio --transcriber whisper

  # Use specific model checkpoint
  python scripts/transcriber.py /path/to/audio --transcriber parakeet_tdt --model nvidia/parakeet-tdt-1.1b

  # Use specific number of GPUs with custom settings
  python scripts/transcriber.py /path/to/audio --num-gpus 2 --workers-per-gpu 1 --batch-size 100

  # Disable CUDA graphs (Parakeet models only)
  python scripts/transcriber.py /path/to/audio --no-cuda-graphs

Supported transcribers:
  - parakeet_tdt (default): NVIDIA Parakeet TDT (fastest with CUDA graphs)
  - parakeet: NVIDIA Parakeet CTC
  - whisper: OpenAI Whisper
  - wav2vec2: Facebook Wav2Vec2
  - canary: NVIDIA Canary

Supported audio formats: .mp3, .aac, .wav, .flac, .m4a
Transcripts are saved as .txt files alongside the original audio files.
        """
    )
    
    parser.add_argument(
        'audio_dir',
        type=str,
        help='Directory containing audio files to transcribe (scanned recursively)'
    )
    
    parser.add_argument(
        '--transcriber',
        type=str,
        choices=['parakeet_tdt', 'parakeet', 'whisper', 'wav2vec2', 'canary'],
        default='parakeet_tdt',
        help='Transcriber model to use (default: parakeet_tdt)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model checkpoint (default: auto-selected based on transcriber type)'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (default: auto-detect all available GPUs)'
    )
    
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=2,
        help='Number of worker processes per GPU (default: 2)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of files to process per batch (default: 50)'
    )
    
    parser.add_argument(
        '--no-cuda-graphs',
        action='store_true',
        help='Disable CUDA graphs for Parakeet models (may reduce speed by 10-30%%)'
    )
    
    parser.add_argument(
        '--audio-formats',
        type=str,
        nargs='+',
        default=['.mp3', '.aac', '.wav', '.flac', '.m4a'],
        help='Audio file extensions to process (default: .mp3 .aac .wav .flac .m4a)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging and error messages'
    )
    
    args = parser.parse_args()
    
    # Select model checkpoint
    if args.model is None:
        transcriber_model = DEFAULT_MODELS[args.transcriber]
    else:
        transcriber_model = args.model
    
    # Validate audio directory
    if not os.path.exists(args.audio_dir):
        print(f"❌ Error: Directory not found: {args.audio_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.audio_dir):
        print(f"❌ Error: Not a directory: {args.audio_dir}")
        sys.exit(1)
    
    # Detect or use specified number of GPUs
    available_gpus = detect_gpus()
    if args.num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(args.num_gpus, available_gpus)
    
    if num_gpus == 0:
        print("❌ Error: No GPUs detected. This script requires CUDA-enabled GPUs.")
        sys.exit(1)
    
    total_workers = num_gpus * args.workers_per_gpu
    enable_cuda_graphs = not args.no_cuda_graphs
    
    # Setup logging
    log_file = setup_logging(args.audio_dir)
    
    print("=" * 80)
    print("🎙️  Parallel Audio Transcription (Optimized)")
    print("=" * 80)
    print(f"Audio directory: {args.audio_dir}")
    print(f"\n⚙️  Configuration:")
    print(f"   Transcriber: {args.transcriber}")
    print(f"   Model: {transcriber_model}")
    print(f"   GPUs available: {available_gpus}")
    print(f"   GPUs to use: {num_gpus}")
    print(f"   Workers per GPU: {args.workers_per_gpu}")
    print(f"   Total workers: {total_workers}")
    print(f"   Batch size: {args.batch_size} files/batch")
    if args.transcriber in ["parakeet_tdt", "parakeet"]:
        print(f"   CUDA graphs: {'Enabled' if enable_cuda_graphs else 'Disabled'}")
    print(f"   Audio formats: {', '.join(args.audio_formats)}")
    print(f"   Log file: {log_file}")
    
    if args.transcriber in ["parakeet_tdt", "parakeet"] and enable_cuda_graphs:
        # Check for cuda-python
        try:
            import cuda
            print(f"   ✅ cuda-python installed: CUDA graphs will accelerate decoding")
        except ImportError:
            print(f"   ⚠️  cuda-python not found: Install with 'pip install cuda-python>=12.3' for 10-30% speedup")
    
    if args.transcriber == "parakeet_tdt":
        print(f"\n🎯 Performance target: 1,500-2,000+ files/sec")
    
    if args.debug:
        print("\n🐛 DEBUG MODE ENABLED")
        print(f"   - Worker stderr will NOT be redirected")
        print(f"   - Debug logs will be written to /tmp/transcriber_worker_*_debug.log")
        print(f"   - Crash logs will be written to /tmp/transcriber_worker_*_crash.log")
        print(f"   - Log writer errors will be written to /tmp/transcriber_log_writer_crash.log")
    
    # Set multiprocessing start method
    if args.debug:
        print(f"\nDEBUG: Setting multiprocessing start method to 'spawn'")
    mp.set_start_method('spawn', force=True)
    
    # Create queues
    progress_queue = mp.Queue()
    ready_queue = mp.Queue()
    log_queue = mp.Queue()
    file_queues = [mp.Queue() for _ in range(total_workers)]
    
    # Start log writer process
    if args.debug:
        print(f"DEBUG: Starting log writer process for {log_file}")
    try:
        log_writer = mp.Process(target=log_writer_process, args=(log_queue, log_file, args.debug))
        log_writer.start()
        if args.debug:
            print(f"DEBUG: Log writer process started (PID: {log_writer.pid})")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to start log writer: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Scan for audio files recursively
    print("\n📂 Scanning for audio files...")
    log_queue.put(f"Scanning {args.audio_dir} for audio files...")
    
    audio_files = []
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if any(file.endswith(ext) for ext in args.audio_formats):
                audio_files.append(os.path.join(root, file))
    
    audio_files = sorted(audio_files)
    
    if not audio_files:
        print(f"❌ No audio files found in {args.audio_dir}")
        print(f"   Looking for formats: {', '.join(args.audio_formats)}")
        log_queue.put(None)
        log_writer.join()
        sys.exit(1)
    
    print(f"✅ Found {len(audio_files):,} audio files")
    log_queue.put(f"Found {len(audio_files):,} audio files")
    
    # Check existing transcripts
    existing_transcripts = 0
    for audio_path in audio_files:
        ext = os.path.splitext(audio_path)[1]
        transcript_path = audio_path.replace(ext, '.txt')
        if os.path.exists(transcript_path):
            existing_transcripts += 1
    
    files_to_process = len(audio_files) - existing_transcripts
    
    print(f"📝 Existing transcripts: {existing_transcripts:,}")
    print(f"🔄 Files to process: {files_to_process:,}")
    log_queue.put(f"Existing: {existing_transcripts:,}, To process: {files_to_process:,}")
    
    if files_to_process == 0:
        print("\n✅ All files already transcribed!")
        log_queue.put(None)
        log_writer.join()
        return
    
    # Split files among workers
    files_per_worker = len(audio_files) // total_workers
    worker_file_lists = []
    for i in range(total_workers):
        start_idx = i * files_per_worker
        end_idx = start_idx + files_per_worker if i < total_workers - 1 else len(audio_files)
        worker_file_lists.append(audio_files[start_idx:end_idx])
    
    print(f"\n📊 File distribution:")
    for i, file_list in enumerate(worker_file_lists):
        gpu_id = i // args.workers_per_gpu
        print(f"   Worker {i} (GPU{gpu_id}): {len(file_list):,} files")
    log_queue.put(f"File distribution: {[len(fl) for fl in worker_file_lists]}")
    
    # Start worker processes
    print(f"\n🚀 Starting {total_workers} workers with parallel model loading...")
    print("=" * 80)
    log_queue.put(f"Starting {total_workers} workers...")
    
    if args.debug:
        print(f"DEBUG: About to start {total_workers} worker processes")
    
    processes = []
    for i in range(total_workers):
        gpu_id = i // args.workers_per_gpu
        if args.debug:
            print(f"DEBUG: Starting worker {i} on GPU {gpu_id}")
        try:
            p = mp.Process(
                target=worker_process,
                args=(i, gpu_id, file_queues[i], args.batch_size, progress_queue, ready_queue, log_queue, args.transcriber, transcriber_model, enable_cuda_graphs, args.debug)
            )
            p.start()
            processes.append(p)
            if args.debug:
                print(f"DEBUG: Worker {i} started (PID: {p.pid})")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to start worker {i}: {e}")
            traceback.print_exc()
            # Try to clean up
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            log_queue.put(None)
            log_writer.join()
            sys.exit(1)
    
    # Monitor initialization
    print("\n📥 Waiting for all workers to load models...")
    workers_ready = 0
    worker_failed = False
    
    while workers_ready < total_workers and not worker_failed:
        try:
            msg = ready_queue.get(timeout=300)
            if msg['status'] == 'ready':
                workers_ready += 1
                print(f"✅ Worker {msg['worker_id']} ready ({workers_ready}/{total_workers})")
            else:
                print(f"❌ Worker {msg['worker_id']} failed to load")
                worker_failed = True
        except Exception as e:
            print(f"❌ Timeout or error during initialization: {e}")
            worker_failed = True
            break
    
    if worker_failed or not all(p.is_alive() for p in processes):
        print("\n❌ One or more workers failed during initialization!")
        log_queue.put("Worker initialization failed!")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        log_queue.put(None)
        log_writer.join()
        sys.exit(1)
    
    print(f"\n✅ All {total_workers} workers loaded successfully!")
    log_queue.put(f"All {total_workers} workers ready")
    
    # Send batches to workers
    print("\n📤 Distributing work to workers...")
    for i, file_list in enumerate(worker_file_lists):
        # Split into batches
        for batch_idx in range(0, len(file_list), args.batch_size):
            batch = file_list[batch_idx:batch_idx + args.batch_size]
            file_queues[i].put(batch)
        # Send shutdown signal
        file_queues[i].put(None)
    
    # Monitor progress with progress bar
    print("\n📊 Processing transcriptions...")
    print("=" * 80)
    log_queue.put("Starting transcription processing...")
    
    start_time = time.time()
    total_files = len(audio_files)
    
    # Create progress bar
    pbar = tqdm(
        total=total_files,
        desc="Transcribing",
        unit="files",
        unit_scale=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # Update progress bar to reflect existing transcripts
    pbar.update(existing_transcripts)
    
    active_workers = total_workers
    total_processed = existing_transcripts
    total_skipped = existing_transcripts
    
    while active_workers > 0:
        try:
            msg = progress_queue.get(timeout=30)
            
            files_done = msg['files_processed']
            skipped = msg.get('skipped', 0)
            
            total_processed += files_done
            total_skipped += skipped
            
            # Update progress bar
            pbar.update(files_done + skipped)
            
            # Update progress bar description with speed
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = (total_processed - existing_transcripts) / elapsed
                pbar.set_description(f"Transcribing ({speed:.0f} files/sec)")
            
        except Exception as e:
            # Check if workers are still alive
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count == 0:
                print("\n⚠️  All workers have stopped")
                log_queue.put("All workers stopped")
                break
            # If timeout but workers alive, they might be finishing up
            active_workers = alive_count
    
    pbar.close()
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()
    
    # Stop log writer
    log_queue.put(None)
    log_writer.join()
    
    elapsed = time.time() - start_time
    actual_processed = total_processed - existing_transcripts
    
    print("\n" + "=" * 80)
    print("🎉 Transcription Complete!")
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"📝 Files processed: {actual_processed:,}/{files_to_process:,}")
    print(f"⏭️  Files skipped: {total_skipped:,}")
    if elapsed > 0 and actual_processed > 0:
        print(f"⚡ Average speed: {actual_processed/elapsed:.2f} files/sec")
    print(f"📋 Detailed logs: {log_file}")
    print(f"💾 Transcripts saved in: {args.audio_dir} (alongside audio files)")
    print("=" * 80)


if __name__ == "__main__":
    main()