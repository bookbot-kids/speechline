#!/usr/bin/env python3
"""
Measure the failure rate of Gentle alignment on a sample of files
"""

import sys
import logging
import argparse
from pathlib import Path
import random

# Add Gentle to Python path
GENTLE_PATH = "/mnt/Store07/Projects/gentle"
sys.path.insert(0, GENTLE_PATH)

import gentle

class FailureRateAnalyzer:
    """Analyze failure rate on sample of files"""
    
    def __init__(self):
        self.resources = gentle.Resources()
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
    def align(self, audio_path: str, transcript: str) -> dict:
        """Perform alignment and return results"""
        try:
            with gentle.resampled(audio_path) as wavfile:
                aligner = gentle.ForcedAligner(
                    self.resources, 
                    transcript,
                    nthreads=1,
                    disfluency=False,
                    conservative=False
                )
                result = aligner.transcribe(wavfile, logging=logging)
                
            words = result.words
            not_found_words = [w.word for w in words if w.not_found_in_audio()]
            
            return {
                'success': len(not_found_words) == 0,
                'not_found_count': len(not_found_words),
                'total_words': len(words),
                'not_found_words': not_found_words
            }
            
        except Exception as e:
            logging.error(f"Alignment failed for {audio_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_file_pairs(self, path: str, pattern: str = "en-*") -> list:
        """Get list of (audio, transcript) pairs"""
        path = Path(path)
        pairs = []
        
        if pattern:
            # Bookbot: scan en-* subdirectories
            for en_dir in path.glob(pattern):
                if not en_dir.is_dir():
                    continue
                for audio_file in en_dir.rglob("*"):
                    if audio_file.suffix.lower() in self.audio_extensions:
                        txt_file = audio_file.with_suffix('.txt')
                        if txt_file.exists():
                            pairs.append((str(audio_file), str(txt_file)))
        else:
            # Common Voice: scan clips directly
            for audio_file in path.glob("*"):
                if audio_file.suffix.lower() in self.audio_extensions:
                    txt_file = audio_file.with_suffix('.txt')
                    if txt_file.exists():
                        pairs.append((str(audio_file), str(txt_file)))
        
        return pairs
    
    def read_transcript(self, path: str) -> str:
        """Read first line of transcript"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.readline().strip()
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            return ""
    
    def analyze_dataset(self, path: str, pattern: str, name: str, sample_size: int = 100):
        """Analyze failure rate on dataset"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name}")
        print(f"{'='*60}")
        
        # Get all file pairs
        print(f"Scanning {path}...")
        all_pairs = self.get_file_pairs(path, pattern)
        print(f"Found {len(all_pairs)} files with transcripts")
        
        if len(all_pairs) == 0:
            print("No files found!")
            return None
        
        # Sample files
        sample_size = min(sample_size, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, sample_size)
        print(f"Analyzing random sample of {sample_size} files...\n")
        
        # Analyze each file
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'error': 0,
            'not_found_examples': []
        }
        
        for i, (audio_path, txt_path) in enumerate(sampled_pairs, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{sample_size}")
            
            transcript = self.read_transcript(txt_path)
            if not transcript:
                continue
            
            result = self.align(audio_path, transcript)
            results['total'] += 1
            
            if result.get('error'):
                results['error'] += 1
            elif result['success']:
                results['success'] += 1
            else:
                results['failed'] += 1
                if len(results['not_found_examples']) < 5:
                    results['not_found_examples'].append({
                        'file': audio_path,
                        'transcript': transcript,
                        'not_found': result['not_found_words'],
                        'not_found_count': result['not_found_count'],
                        'total_words': result['total_words']
                    })
        
        # Calculate statistics
        if results['total'] > 0:
            success_pct = (results['success'] / results['total']) * 100
            failed_pct = (results['failed'] / results['total']) * 100
            error_pct = (results['error'] / results['total']) * 100
            
            print(f"\n{'='*60}")
            print(f"RESULTS: {name}")
            print(f"{'='*60}")
            print(f"Total analyzed: {results['total']}")
            print(f"Success: {results['success']} ({success_pct:.1f}%)")
            print(f"Failed (not-found-in-audio): {results['failed']} ({failed_pct:.1f}%)")
            print(f"Errors: {results['error']} ({error_pct:.1f}%)")
            
            if results['not_found_examples']:
                print(f"\nExample failures:")
                for i, ex in enumerate(results['not_found_examples'], 1):
                    print(f"\n  {i}. {Path(ex['file']).name}")
                    print(f"     Transcript: {ex['transcript'][:60]}...")
                    print(f"     Not found: {ex['not_found_count']}/{ex['total_words']} words")
                    print(f"     Missing: {ex['not_found'][:5]}")
            
            return {
                'name': name,
                'total': results['total'],
                'success_count': results['success'],
                'success_pct': success_pct,
                'failed_count': results['failed'],
                'failed_pct': failed_pct,
                'error_count': results['error'],
                'error_pct': error_pct
            }
        
        return None

def main():
    parser = argparse.ArgumentParser(description='Measure Gentle alignment failure rate')
    parser.add_argument('--bookbot-path', type=str, help='Path to Bookbot dataset')
    parser.add_argument('--cv-clips-path', type=str, help='Path to Common Voice clips')
    parser.add_argument('--sample-size', type=int, default=100, help='Sample size per dataset (default: 100)')
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    if not args.bookbot_path and not args.cv_clips_path:
        parser.error("At least one of --bookbot-path or --cv-clips-path must be provided")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("GENTLE ALIGNMENT FAILURE RATE ANALYSIS")
    print("="*60)
    
    analyzer = FailureRateAnalyzer()
    all_results = []
    
    # Analyze Bookbot
    if args.bookbot_path:
        result = analyzer.analyze_dataset(
            args.bookbot_path,
            "en-*",
            "Bookbot",
            args.sample_size
        )
        if result:
            all_results.append(result)
    
    # Analyze Common Voice
    if args.cv_clips_path:
        result = analyzer.analyze_dataset(
            args.cv_clips_path,
            None,
            "Common Voice",
            args.sample_size
        )
        if result:
            all_results.append(result)
    
    # Overall summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        
        total = sum(r['total'] for r in all_results)
        success = sum(r['success_count'] for r in all_results)
        failed = sum(r['failed_count'] for r in all_results)
        errors = sum(r['error_count'] for r in all_results)
        
        print(f"Combined total: {total}")
        print(f"Success: {success} ({success/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Errors: {errors} ({errors/total*100:.1f}%)")
        print()

if __name__ == '__main__':
    main()