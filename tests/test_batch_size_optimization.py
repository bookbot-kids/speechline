#!/usr/bin/env python3
"""
Batch Size Optimization Test Script

This script tests different batch sizes to find the optimal configuration
for the fastest transcription on the current machine.

It will:
1. Generate synthetic test audio files
2. Test multiple batch sizes
3. Measure throughput (files/sec) and GPU memory usage
4. Provide recommendations for optimal settings
"""

import os
import sys
import time
import json
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_test_audio_files(output_dir: str, num_files: int = 100, duration: float = 3.0):
    """
    Generate synthetic audio files for testing.
    
    Args:
        output_dir: Directory to save test files
        num_files: Number of audio files to generate
        duration: Duration of each audio file in seconds
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("❌ Error: soundfile is required. Install with: pip install soundfile")
        sys.exit(1)
    
    print(f"\n📁 Generating {num_files} test audio files ({duration}s each)...")
    os.makedirs(output_dir, exist_ok=True)
    
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    for i in range(num_files):
        # Generate simple white noise or tone as test audio
        # Mix of silence, noise, and tones to simulate speech patterns
        audio = np.zeros(num_samples, dtype=np.float32)
        
        # Add some random noise bursts to simulate speech
        for _ in range(np.random.randint(3, 8)):
            start = np.random.randint(0, num_samples - 1000)
            length = np.random.randint(500, 3000)
            end = min(start + length, num_samples)
            
            # Random frequency for variety
            freq = np.random.uniform(200, 500)
            t = np.linspace(0, length / sample_rate, end - start)
            audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * t)
            # Add some noise
            audio[start:end] += 0.1 * np.random.randn(end - start)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save as WAV file
        output_path = os.path.join(output_dir, f"test_audio_{i:04d}.wav")
        sf.write(output_path, audio, sample_rate)
    
    print(f"✅ Generated {num_files} test files in {output_dir}")


def run_batch_size_test(
    audio_dir: str,
    batch_size: int,
    num_gpus: int,
    workers_per_gpu: int,
    transcriber: str,
    model: str,
    enable_cuda_graphs: bool = True,
    timeout: int = 300
) -> Tuple[float, float, bool]:
    """
    Test a specific batch size and measure performance.
    
    Args:
        audio_dir: Directory containing test audio files
        batch_size: Batch size to test
        num_gpus: Number of GPUs to use
        workers_per_gpu: Workers per GPU
        transcriber: Transcriber type
        model: Model checkpoint
        enable_cuda_graphs: Whether to enable CUDA graphs
        timeout: Maximum time to wait (seconds)
    
    Returns:
        Tuple of (throughput in files/sec, peak memory in GB, success flag)
    """
    import subprocess
    
    # Clean up any existing transcripts
    for txt_file in Path(audio_dir).glob("*.txt"):
        txt_file.unlink()
    
    print(f"\n🧪 Testing batch_size={batch_size}...")
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "transcriber.py"),
        audio_dir,
        "--transcriber", transcriber,
        "--model", model,
        "--num-gpus", str(num_gpus),
        "--workers-per-gpu", str(workers_per_gpu),
        "--batch-size", str(batch_size),
    ]
    
    if not enable_cuda_graphs:
        cmd.append("--no-cuda-graphs")
    
    # Run transcription and measure time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"   ❌ Failed with return code {result.returncode}")
            print(f"   Error: {result.stderr[-500:]}")  # Last 500 chars
            return 0.0, 0.0, False
        
        # Parse output for statistics
        throughput = 0.0
        peak_memory = 0.0
        
        # Look for speed in output
        for line in result.stdout.split('\n'):
            if 'files/sec' in line.lower():
                # Extract number before 'files/sec'
                try:
                    parts = line.split('files/sec')[0].strip().split()
                    throughput = float(parts[-1].replace(',', ''))
                except:
                    pass
        
        # Count processed files
        txt_files = list(Path(audio_dir).glob("*.txt"))
        num_processed = len(txt_files)
        
        # Calculate throughput if not found in output
        if throughput == 0.0 and elapsed > 0 and num_processed > 0:
            throughput = num_processed / elapsed
        
        # Try to get GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
                torch.cuda.reset_peak_memory_stats()
        except:
            pass
        
        success = num_processed > 0
        
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Throughput: {throughput:.2f} files/sec")
        print(f"   💾 Files processed: {num_processed}")
        if peak_memory > 0:
            print(f"   🎮 Peak GPU memory: {peak_memory:.2f} GB")
        
        return throughput, peak_memory, success
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after {timeout}s")
        return 0.0, 0.0, False
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return 0.0, 0.0, False


def run_optimization_test(
    test_dir: str,
    num_test_files: int,
    batch_sizes: List[int],
    num_gpus: int,
    workers_per_gpu: int,
    transcriber: str,
    model: str,
    enable_cuda_graphs: bool,
    audio_duration: float,
    test_workers: bool = False,
    worker_configs: List[int] = None
) -> Dict:
    """
    Run the full optimization test across multiple batch sizes and optionally worker configurations.
    
    Args:
        test_workers: If True, test different worker configurations instead of just batch sizes
        worker_configs: List of worker-per-gpu values to test
    
    Returns:
        Dictionary with results
    """
    print("=" * 80)
    if test_workers:
        print("🔬 Worker Configuration Optimization Test")
    else:
        print("🔬 Batch Size Optimization Test")
    print("=" * 80)
    print(f"Test Configuration:")
    print(f"  📁 Test directory: {test_dir}")
    print(f"  🎵 Test files: {num_test_files} files ({audio_duration}s each)")
    print(f"  🎯 Transcriber: {transcriber}")
    print(f"  🤖 Model: {model}")
    print(f"  🎮 GPUs: {num_gpus}")
    if test_workers:
        print(f"  👷 Worker configs to test: {worker_configs}")
        print(f"  📊 Batch size: {batch_sizes[0]} (fixed)")
    else:
        print(f"  👷 Workers per GPU: {workers_per_gpu}")
        print(f"  📊 Batch sizes to test: {batch_sizes}")
    print(f"  ⚡ CUDA graphs: {'Enabled' if enable_cuda_graphs else 'Disabled'}")
    print("=" * 80)
    
    # Generate test audio files
    generate_test_audio_files(test_dir, num_test_files, audio_duration)
    
    # Test each configuration
    results = []
    
    if test_workers:
        # Test different worker configurations with fixed batch size
        fixed_batch_size = batch_sizes[0]
        for workers in worker_configs:
            throughput, memory, success = run_batch_size_test(
                audio_dir=test_dir,
                batch_size=fixed_batch_size,
                num_gpus=num_gpus,
                workers_per_gpu=workers,
                transcriber=transcriber,
                model=model,
                enable_cuda_graphs=enable_cuda_graphs,
                timeout=300
            )
            
            results.append({
                'batch_size': fixed_batch_size,
                'workers_per_gpu': workers,
                'throughput': throughput,
                'memory_gb': memory,
                'success': success
            })
            
            # Small delay between tests
            time.sleep(2)
    else:
        # Original behavior: test batch sizes
        for batch_size in batch_sizes:
            throughput, memory, success = run_batch_size_test(
                audio_dir=test_dir,
                batch_size=batch_size,
                num_gpus=num_gpus,
                workers_per_gpu=workers_per_gpu,
                transcriber=transcriber,
                model=model,
                enable_cuda_graphs=enable_cuda_graphs,
                timeout=300
            )
            
            results.append({
                'batch_size': batch_size,
                'workers_per_gpu': workers_per_gpu,
                'throughput': throughput,
                'memory_gb': memory,
                'success': success
            })
            
            # Small delay between tests
            time.sleep(2)
    
    return {
        'test_config': {
            'num_files': num_test_files,
            'audio_duration': audio_duration,
            'transcriber': transcriber,
            'model': model,
            'num_gpus': num_gpus,
            'workers_per_gpu': workers_per_gpu if not test_workers else 'variable',
            'enable_cuda_graphs': enable_cuda_graphs,
            'test_mode': 'workers' if test_workers else 'batch_size'
        },
        'results': results
    }


def analyze_results(results: Dict) -> Dict:
    """
    Analyze results and provide recommendations.
    
    Returns:
        Dictionary with analysis and recommendations
    """
    print("\n" + "=" * 80)
    print("📊 Results Analysis")
    print("=" * 80)
    
    test_results = results['results']
    test_mode = results['test_config'].get('test_mode', 'batch_size')
    
    # Filter successful results
    successful = [r for r in test_results if r['success'] and r['throughput'] > 0]
    
    if not successful:
        print("❌ No successful tests!")
        return {'optimal_batch_size': None, 'recommendation': 'All tests failed'}
    
    # Print results table
    if test_mode == 'workers':
        print("\n📈 Performance by Worker Configuration:")
        print("-" * 80)
        print(f"{'Workers/GPU':<12} {'Throughput':<20} {'Memory':<15} {'Status':<10}")
    else:
        print("\n📈 Performance by Batch Size:")
        print("-" * 80)
        print(f"{'Batch Size':<12} {'Throughput':<20} {'Memory':<15} {'Status':<10}")
    print("-" * 80)
    
    for r in test_results:
        status = "✅ OK" if r['success'] else "❌ FAIL"
        throughput_str = f"{r['throughput']:.2f} files/sec" if r['throughput'] > 0 else "N/A"
        memory_str = f"{r['memory_gb']:.2f} GB" if r['memory_gb'] > 0 else "N/A"
        
        if test_mode == 'workers':
            print(f"{r['workers_per_gpu']:<12} {throughput_str:<20} {memory_str:<15} {status:<10}")
        else:
            print(f"{r['batch_size']:<12} {throughput_str:<20} {memory_str:<15} {status:<10}")
    
    # Find optimal configuration (highest throughput)
    optimal = max(successful, key=lambda x: x['throughput'])
    
    # Calculate efficiency metrics
    avg_throughput = sum(r['throughput'] for r in successful) / len(successful)
    improvement = (optimal['throughput'] - avg_throughput) / avg_throughput * 100
    
    print("-" * 80)
    print(f"\n🎯 Optimal Configuration:")
    if test_mode == 'workers':
        print(f"  Workers per GPU: {optimal['workers_per_gpu']}")
        print(f"  Batch Size: {optimal['batch_size']}")
    else:
        print(f"  Batch Size: {optimal['batch_size']}")
        if 'workers_per_gpu' in optimal:
            print(f"  Workers per GPU: {optimal['workers_per_gpu']}")
    print(f"  Throughput: {optimal['throughput']:.2f} files/sec")
    if optimal['memory_gb'] > 0:
        print(f"  GPU Memory: {optimal['memory_gb']:.2f} GB")
    print(f"  Improvement over average: {improvement:+.1f}%")
    
    # Generate recommendation text
    if test_mode == 'workers':
        recommendation = f"""
Recommended Configuration:
--------------------------
Workers per GPU: {optimal['workers_per_gpu']}
Batch Size: {optimal['batch_size']}
Expected Throughput: {optimal['throughput']:.2f} files/sec

To use this configuration, run:
python scripts/transcriber.py /path/to/audio \\
    --transcriber {results['test_config']['transcriber']} \\
    --num-gpus {results['test_config']['num_gpus']} \\
    --workers-per-gpu {optimal['workers_per_gpu']} \\
    --batch-size {optimal['batch_size']}
"""
    else:
        workers_str = optimal.get('workers_per_gpu', 'N/A')
        recommendation = f"""
Recommended Configuration:
--------------------------
Batch Size: {optimal['batch_size']}
Workers per GPU: {workers_str}
Expected Throughput: {optimal['throughput']:.2f} files/sec

To use this configuration, run:
python scripts/transcriber.py /path/to/audio \\
    --transcriber {results['test_config']['transcriber']} \\
    --num-gpus {results['test_config']['num_gpus']} \\
    --workers-per-gpu {workers_str} \\
    --batch-size {optimal['batch_size']}
"""
    
    if not results['test_config']['enable_cuda_graphs']:
        recommendation += "    --no-cuda-graphs\n"
    
    print(recommendation)
    
    # Additional insights
    print("\n💡 Insights:")
    
    if test_mode == 'workers':
        # Analyze worker scaling
        sorted_by_workers = sorted(successful, key=lambda x: x['workers_per_gpu'])
        sorted_by_throughput = sorted(successful, key=lambda x: x['throughput'], reverse=True)
        
        if len(successful) > 1:
            # Check scaling efficiency
            min_workers = sorted_by_workers[0]
            max_workers = sorted_by_workers[-1]
            
            if min_workers['workers_per_gpu'] < max_workers['workers_per_gpu']:
                worker_ratio = max_workers['workers_per_gpu'] / min_workers['workers_per_gpu']
                throughput_ratio = max_workers['throughput'] / min_workers['throughput']
                efficiency = (throughput_ratio / worker_ratio) * 100
                
                print(f"  • Worker scaling efficiency: {efficiency:.1f}%")
                if efficiency > 80:
                    print("    → Excellent scaling - workers are well-utilized")
                elif efficiency > 60:
                    print("    → Good scaling - consider testing more workers")
                else:
                    print("    → Diminishing returns - current config is near optimal")
        
        if sorted_by_workers[-1]['workers_per_gpu'] == optimal['workers_per_gpu']:
            print("  • More workers are beneficial - consider testing even higher values")
        elif sorted_by_workers[0]['workers_per_gpu'] == optimal['workers_per_gpu']:
            print("  • Fewer workers perform better - may be memory constrained")
    else:
        # Original batch size insights
        # Check if larger batch sizes are better
        sorted_by_size = sorted(successful, key=lambda x: x['batch_size'])
        sorted_by_throughput = sorted(successful, key=lambda x: x['throughput'], reverse=True)
        
        if sorted_by_size[-1]['batch_size'] == optimal['batch_size']:
            print("  • Larger batch sizes are beneficial - consider testing even larger values")
        elif sorted_by_size[0]['batch_size'] == optimal['batch_size']:
            print("  • Smaller batch sizes perform better - system may be memory constrained")
        else:
            print("  • Mid-range batch size is optimal - good balance achieved")
    
    # Memory efficiency
    if optimal['memory_gb'] > 0:
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                utilization = (optimal['memory_gb'] / total_memory) * 100
                print(f"  • GPU memory utilization: {utilization:.1f}%")
                if utilization < 50:
                    print("    → Low utilization - could potentially use larger batch sizes")
                elif utilization > 90:
                    print("    → High utilization - close to optimal memory usage")
        except:
            pass
    
    return {
        'optimal_batch_size': optimal['batch_size'],
        'optimal_workers_per_gpu': optimal.get('workers_per_gpu'),
        'optimal_throughput': optimal['throughput'],
        'optimal_memory': optimal['memory_gb'],
        'recommendation': recommendation
    }


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal batch size for transcription on current machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  python tests/test_batch_size_optimization.py
  
  # Test specific transcriber
  python tests/test_batch_size_optimization.py --transcriber whisper
  
  # Custom batch size range
  python tests/test_batch_size_optimization.py --batch-sizes 10 25 50 100 200 500
  
  # More test files for better accuracy
  python tests/test_batch_size_optimization.py --num-files 200
  
  # Test with multiple GPUs
  python tests/test_batch_size_optimization.py --num-gpus 2 --workers-per-gpu 1
        """
    )
    
    parser.add_argument(
        '--transcriber',
        type=str,
        choices=['parakeet_tdt', 'parakeet', 'whisper', 'wav2vec2', 'canary'],
        default='parakeet_tdt',
        help='Transcriber to test (default: parakeet_tdt)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model checkpoint (default: auto-selected based on transcriber)'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use (default: 1)'
    )
    
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=2,
        help='Workers per GPU (default: 2)'
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[10, 25, 50, 100, 200, 400],
        help='Batch sizes to test (default: 10 25 50 100 200 400)'
    )
    
    parser.add_argument(
        '--num-files',
        type=int,
        default=100,
        help='Number of test audio files to generate (default: 100)'
    )
    
    parser.add_argument(
        '--audio-duration',
        type=float,
        default=3.0,
        help='Duration of each test audio file in seconds (default: 3.0)'
    )
    
    parser.add_argument(
        '--no-cuda-graphs',
        action='store_true',
        help='Disable CUDA graphs (for Parakeet models)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file (default: None)'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Directory for test files (default: temporary directory)'
    )
    
    parser.add_argument(
        '--test-workers',
        action='store_true',
        help='Test different worker-per-gpu configurations instead of batch sizes'
    )
    
    parser.add_argument(
        '--worker-configs',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help='Worker-per-gpu values to test when --test-workers is used (default: 1 2 3 4)'
    )
    
    args = parser.parse_args()
    
    # Default models
    DEFAULT_MODELS = {
        'parakeet_tdt': 'nvidia/parakeet-tdt-0.6b-v2',
        'parakeet': 'nvidia/parakeet-ctc-0.6b',
        'whisper': 'openai/whisper-large-v3',
        'wav2vec2': 'facebook/wav2vec2-large-960h-lv60-self',
        'canary': 'nvidia/canary-1b',
    }
    
    model = args.model if args.model else DEFAULT_MODELS[args.transcriber]
    enable_cuda_graphs = not args.no_cuda_graphs
    
    # Verify CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Error: CUDA is not available. This test requires a GPU.")
            sys.exit(1)
        
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available")
            args.num_gpus = available_gpus
        
        # Print GPU info
        print(f"\n🎮 GPU Information:")
        for i in range(min(args.num_gpus, available_gpus)):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    except ImportError:
        print("❌ Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    # Sort batch sizes and worker configs
    batch_sizes = sorted(args.batch_sizes)
    worker_configs = sorted(args.worker_configs) if args.test_workers else None
    
    # If testing workers, use a reasonable batch size
    if args.test_workers:
        if not args.batch_sizes or args.batch_sizes == [10, 25, 50, 100, 200, 400]:
            # User didn't specify batch size, use a good default
            batch_sizes = [100]
            print(f"ℹ️  Testing workers with fixed batch size: {batch_sizes[0]}")
    
    # Create or use test directory
    cleanup_temp = False
    if args.test_dir:
        test_dir = args.test_dir
        os.makedirs(test_dir, exist_ok=True)
    else:
        test_dir = tempfile.mkdtemp(prefix='transcribe_batch_test_')
        cleanup_temp = True
    
    try:
        # Run optimization test
        results = run_optimization_test(
            test_dir=test_dir,
            num_test_files=args.num_files,
            batch_sizes=batch_sizes,
            num_gpus=args.num_gpus,
            workers_per_gpu=args.workers_per_gpu,
            transcriber=args.transcriber,
            model=model,
            enable_cuda_graphs=enable_cuda_graphs,
            audio_duration=args.audio_duration,
            test_workers=args.test_workers,
            worker_configs=worker_configs
        )
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Combine results and analysis
        full_results = {
            **results,
            'analysis': analysis
        }
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(full_results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        print("\n" + "=" * 80)
        print("✅ Optimization test complete!")
        print("=" * 80)
        
    finally:
        # Cleanup temporary directory
        if cleanup_temp and os.path.exists(test_dir):
            print(f"\n🧹 Cleaning up temporary directory: {test_dir}")
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    main()

