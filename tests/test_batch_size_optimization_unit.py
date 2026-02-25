#!/usr/bin/env python3
"""
Unit tests for batch size optimization test script
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_batch_size_optimization import (
    generate_test_audio_files,
    analyze_results
)


def test_generate_test_audio_files():
    """Test that audio file generation works correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_files = 5
        duration = 2.0
        
        # Generate files
        generate_test_audio_files(tmpdir, num_files, duration)
        
        # Check files were created
        audio_files = list(Path(tmpdir).glob("*.wav"))
        assert len(audio_files) == num_files
        
        # Check files are not empty
        for audio_file in audio_files:
            assert audio_file.stat().st_size > 0


def test_analyze_results_with_successful_tests():
    """Test result analysis with successful test data"""
    mock_results = {
        'test_config': {
            'num_files': 100,
            'audio_duration': 3.0,
            'transcriber': 'parakeet_tdt',
            'model': 'nvidia/parakeet-tdt-0.6b-v2',
            'num_gpus': 1,
            'workers_per_gpu': 2,
            'enable_cuda_graphs': True
        },
        'results': [
            {'batch_size': 10, 'throughput': 45.0, 'memory_gb': 2.3, 'success': True},
            {'batch_size': 25, 'throughput': 67.0, 'memory_gb': 3.1, 'success': True},
            {'batch_size': 50, 'throughput': 98.0, 'memory_gb': 4.5, 'success': True},
            {'batch_size': 100, 'throughput': 112.0, 'memory_gb': 6.7, 'success': True},
            {'batch_size': 200, 'throughput': 105.0, 'memory_gb': 10.2, 'success': True},
        ]
    }
    
    analysis = analyze_results(mock_results)
    
    # Check optimal batch size is correctly identified
    assert analysis['optimal_batch_size'] == 100
    assert analysis['optimal_throughput'] == 112.0
    assert 'recommendation' in analysis


def test_analyze_results_with_failures():
    """Test result analysis with some failed tests"""
    mock_results = {
        'test_config': {
            'num_files': 100,
            'audio_duration': 3.0,
            'transcriber': 'parakeet_tdt',
            'model': 'nvidia/parakeet-tdt-0.6b-v2',
            'num_gpus': 1,
            'workers_per_gpu': 2,
            'enable_cuda_graphs': True
        },
        'results': [
            {'batch_size': 10, 'throughput': 45.0, 'memory_gb': 2.3, 'success': True},
            {'batch_size': 50, 'throughput': 98.0, 'memory_gb': 4.5, 'success': True},
            {'batch_size': 100, 'throughput': 0.0, 'memory_gb': 0.0, 'success': False},
            {'batch_size': 200, 'throughput': 0.0, 'memory_gb': 0.0, 'success': False},
        ]
    }
    
    analysis = analyze_results(mock_results)
    
    # Should find optimal among successful tests
    assert analysis['optimal_batch_size'] == 50
    assert analysis['optimal_throughput'] == 98.0


def test_analyze_results_all_failed():
    """Test result analysis when all tests fail"""
    mock_results = {
        'test_config': {
            'num_files': 100,
            'audio_duration': 3.0,
            'transcriber': 'parakeet_tdt',
            'model': 'nvidia/parakeet-tdt-0.6b-v2',
            'num_gpus': 1,
            'workers_per_gpu': 2,
            'enable_cuda_graphs': True
        },
        'results': [
            {'batch_size': 10, 'throughput': 0.0, 'memory_gb': 0.0, 'success': False},
            {'batch_size': 50, 'throughput': 0.0, 'memory_gb': 0.0, 'success': False},
        ]
    }
    
    analysis = analyze_results(mock_results)
    
    # Should handle all failures gracefully
    assert analysis['optimal_batch_size'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

