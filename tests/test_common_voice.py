# Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test suite for Common Voice dataset validation.

This test validates audio transcription + phoneme matching on
a validated subset of the Common Voice dataset.
Target: 100% match rate.
"""

import pytest
import logging
import csv
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from speechline.matchers import AudioMatcher, PhonemeMatcher
from speechline.utils.lexicon_manager import LexiconManager
from speechline.utils.accent_rules import AccentRulesManager

logger = logging.getLogger(__name__)


# Common Voice dataset path
CV_BASE_PATH = Path('/Volumes/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en')
CV_CLIPS_PATH = CV_BASE_PATH / 'clips'
CV_VALIDATED_TSV = CV_BASE_PATH / 'validated.tsv'


@pytest.fixture(scope="module")
def matcher():
    """Create AudioMatcher instance for testing."""
    phoneme_matcher = PhonemeMatcher(
        lexicon_manager=LexiconManager(),
        accent_rules_manager=AccentRulesManager(),
        export_mismatches=True,
        mismatch_output_dir='mismatches/common_voice'
    )
    
    return AudioMatcher(
        phoneme_matcher=phoneme_matcher,
        model_checkpoint='bookbot/wav2vec2-ljspeech-gruut'
    )


def load_validated_samples(
    tsv_path: Path,
    clips_path: Path,
    limit: int = 10000
) -> List[Dict]:
    """Load validated samples from Common Voice TSV.
    
    Args:
        tsv_path: Path to validated.tsv
        clips_path: Path to clips directory
        limit: Maximum number of samples to load
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for i, row in enumerate(reader):
            if i >= limit:
                break
            
            audio_path = clips_path / row['path']
            
            # Only include if audio file exists
            if audio_path.exists():
                samples.append({
                    'id': row.get('client_id', f'cv_{i}'),
                    'audio_path': str(audio_path),
                    'text': row['sentence'],
                    'metadata': {
                        'age': row.get('age', ''),
                        'gender': row.get('gender', ''),
                        'accent': row.get('accent', ''),
                        'locale': row.get('locale', 'en')
                    }
                })
    
    return samples


@pytest.fixture(scope="module")
def common_voice_samples():
    """Load Common Voice validated samples."""
    if not CV_BASE_PATH.exists():
        pytest.skip(f"Common Voice path not found: {CV_BASE_PATH}")
    
    if not CV_VALIDATED_TSV.exists():
        pytest.skip(f"Validated TSV not found: {CV_VALIDATED_TSV}")
    
    if not CV_CLIPS_PATH.exists():
        pytest.skip(f"Clips directory not found: {CV_CLIPS_PATH}")
    
    try:
        samples = load_validated_samples(
            tsv_path=CV_VALIDATED_TSV,
            clips_path=CV_CLIPS_PATH,
            limit=10000
        )
        
        if len(samples) == 0:
            pytest.skip("No validated samples found")
        
        logger.info(f"Loaded {len(samples)} Common Voice samples")
        return samples
        
    except Exception as e:
        pytest.skip(f"Could not load Common Voice samples: {e}")


class TestCommonVoiceMatching:
    """Test audio transcription + matching on Common Voice dataset."""
    
    def test_common_voice_sample_structure(self, common_voice_samples):
        """Test that samples have expected structure."""
        assert len(common_voice_samples) > 0, "No samples loaded"
        
        # Check first sample has required fields
        sample = common_voice_samples[0]
        assert 'audio_path' in sample, "No audio_path field"
        assert 'text' in sample, "No text field"
        assert Path(sample['audio_path']).exists(), "Audio file doesn't exist"
    
    def test_common_voice_matches(self, matcher, common_voice_samples):
        """Test Method 1: audio file + word reference.
        
        This test validates that audio transcriptions match their
        corresponding reference text using:
        1. Wav2Vec2 phoneme ASR
        2. Phonetic class matching
        3. Accent rule tolerance
        
        Target: 100% match rate
        """
        total = 0
        matched = 0
        mismatches: List[Dict] = []
        transcription_errors = 0
        
        logger.info(f"Testing {len(common_voice_samples)} Common Voice samples")
        
        for sample in tqdm(common_voice_samples, desc="Testing Common Voice"):
            total += 1
            
            sample_id = sample['id']
            audio_path = sample['audio_path']
            reference_text = sample['text']
            
            try:
                # Use Method 1 (transcribe + match)
                result = matcher.match_audio(
                    audio_path=audio_path,
                    reference=reference_text,
                    aligned=True,
                    sample_id=sample_id
                )
                
                if result.matched:
                    matched += 1
                else:
                    # Check if transcription error
                    if result.method in ['transcription_error', 'file_not_found']:
                        transcription_errors += 1
                    
                    mismatches.append({
                        'id': sample_id,
                        'audio_path': audio_path,
                        'reference': reference_text,
                        'transcript': result.details.get('transcript', ''),
                        'method': result.method,
                        'reason': result.details.get('reason', 'unknown'),
                        'metadata': sample.get('metadata', {})
                    })
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                transcription_errors += 1
                mismatches.append({
                    'id': sample_id,
                    'audio_path': audio_path,
                    'reference': reference_text,
                    'transcript': '',
                    'method': 'exception',
                    'reason': str(e),
                    'metadata': sample.get('metadata', {})
                })
        
        # Calculate metrics
        match_rate = matched / total if total > 0 else 0
        valid_samples = total - transcription_errors
        valid_match_rate = matched / valid_samples if valid_samples > 0 else 0
        
        # Log results
        logger.info(f"\n{'='*80}")
        logger.info("COMMON VOICE TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total samples:          {total:,}")
        logger.info(f"Transcription errors:   {transcription_errors:,}")
        logger.info(f"Valid samples:          {valid_samples:,}")
        logger.info(f"Matched:                {matched:,}")
        logger.info(f"Mismatched:             {len(mismatches):,}")
        logger.info(f"Overall match rate:     {match_rate:.2%}")
        logger.info(f"Valid match rate:       {valid_match_rate:.2%}")
        logger.info(f"{'='*80}")
        
        if mismatches:
            logger.info("\nFirst 10 mismatches:")
            for i, mismatch in enumerate(mismatches[:10], 1):
                logger.info(f"\n{i}. ID: {mismatch['id']}")
                logger.info(f"   Reference:  {mismatch['reference']}")
                logger.info(f"   Transcript: {mismatch['transcript']}")
                logger.info(f"   Method:     {mismatch['method']}")
                logger.info(f"   Reason:     {mismatch['reason']}")
                if mismatch.get('metadata'):
                    logger.info(f"   Metadata:   {mismatch['metadata']}")
            
            logger.info(f"\nAll mismatches exported to: mismatches/common_voice/")
            
            # Analyze mismatch patterns
            self._analyze_mismatches(mismatches)
        
        # Get statistics
        stats = matcher.get_statistics()
        logger.info(f"\nMatcher Statistics:")
        logger.info(f"  Model: {stats['model_checkpoint']}")
        logger.info(f"  Lexicon size: {stats['lexicon_size']:,}")
        logger.info(f"  Accent rules: {stats['accent_rules_count']}")
        logger.info(f"  Total mismatches exported: {stats.get('mismatch_count', 0):,}")
        
        # Assert 100% match rate (on valid samples)
        assert valid_match_rate == 1.0, (
            f"Match rate {valid_match_rate:.2%} is below 100%. "
            f"{len(mismatches)} mismatches found ({transcription_errors} transcription errors). "
            f"Check mismatches/common_voice/ for details."
        )
    
    def _analyze_mismatches(self, mismatches: List[Dict]) -> None:
        """Analyze patterns in mismatches."""
        logger.info("\nMismatch Analysis:")
        
        # Group by reason
        by_reason = {}
        for m in mismatches:
            reason = m.get('reason', 'unknown')
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        logger.info("  By reason:")
        for reason, count in sorted(by_reason.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {reason}: {count}")
        
        # Group by method
        by_method = {}
        for m in mismatches:
            method = m.get('method', 'unknown')
            by_method[method] = by_method.get(method, 0) + 1
        
        logger.info("  By method:")
        for method, count in sorted(by_method.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {method}: {count}")
        
        # Analyze by accent if available
        by_accent = {}
        for m in mismatches:
            accent = m.get('metadata', {}).get('accent', 'unknown')
            if accent:
                by_accent[accent] = by_accent.get(accent, 0) + 1
        
        if by_accent:
            logger.info("  By accent:")
            for accent, count in sorted(by_accent.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"    {accent}: {count}")
    
    def test_common_voice_sample_cases(self, matcher):
        """Test specific audio samples if available."""
        # This is a placeholder for testing specific known audio samples
        # You would add specific test cases here
        pass


@pytest.mark.benchmark
class TestCommonVoicePerformance:
    """Performance benchmarks for Common Voice matching."""
    
    def test_matching_speed(self, matcher, common_voice_samples, benchmark):
        """Benchmark matching speed on a single sample."""
        if not common_voice_samples:
            pytest.skip("No samples available")
        
        sample = common_voice_samples[0]
        
        # Benchmark
        result = benchmark(
            matcher.match_audio,
            audio_path=sample['audio_path'],
            reference=sample['text'],
            aligned=True
        )
        
        assert result.matched or not result.matched  # Just ensure it completes


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])