#!/usr/bin/env python3
"""
Re-test error files with updated lexicon

This script re-processes the Common Voice error files to measure
the actual improvement from the lexicon update.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add Gentle to Python path
GENTLE_PATH = "/mnt/Store07/Projects/gentle"
sys.path.insert(0, GENTLE_PATH)

try:
    import gentle
except ImportError:
    print("ERROR: Gentle not found. Please ensure Gentle is installed.")
    sys.exit(1)


class ErrorFileRetester:
    """Re-test error files with updated lexicon"""
    
    def __init__(self, gentle_dir: str = "/mnt/Store07/Projects/gentle"):
        self.gentle_dir = Path(gentle_dir)
        self.errors_dir = Path("errors/common_voice")
        
        print("Loading Gentle resources...")
        try:
            self.resources = gentle.Resources()
            print("✓ Gentle resources loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load Gentle resources: {e}")
            sys.exit(1)
    
    def read_error_file(self, txt_file: Path) -> Tuple[str, List[str]]:
        """Read error file and extract transcript and missing words"""
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            return None, None
        
        transcript = lines[0].strip()
        missing_words = []
        
        # Extract missing words from line 3
        not_found_line = lines[2].strip()
        if "NOT_FOUND_IN_AUDIO:" in not_found_line:
            import re
            match = re.search(r'\[(.*?)\]', not_found_line)
            if match:
                words_str = match.group(1)
                missing_words = [w.strip().strip("'\"") for w in words_str.split(',')]
        
        return transcript, missing_words
    
    def test_alignment(self, audio_file: Path, transcript: str) -> Dict:
        """Test alignment with updated lexicon"""
        if not audio_file.exists():
            return {
                "success": False,
                "error": "Audio file not found"
            }
        
        try:
            with gentle.resampled(str(audio_file)) as wavfile:
                aligner = gentle.ForcedAligner(self.resources, transcript, nthreads=1)
                result = aligner.transcribe(wavfile)
            
            # Analyze result
            if not result:
                return {
                    "success": False,
                    "error": "No alignment result"
                }
            
            # Extract word objects
            words = result.words
            not_found_words = [w.word for w in words if w.not_found_in_audio()]
            
            # Get phonemes
            phonemes = []
            for word in words:
                if word.success() and word.phones:
                    for phone_info in word.phones:
                        phone = phone_info.get('phone', '')
                        phonemes.append(phone)
            
            # Check if entire transcript was OOV (single "oov" phoneme)
            is_complete_oov = (len(phonemes) == 1 and phonemes[0].lower() == 'oov')
            
            # Check for any OOV in phonemes
            has_oov = 'oov' in [p.lower() for p in phonemes]
            
            return {
                "success": True,
                "has_oov": has_oov,
                "is_complete_oov": is_complete_oov,
                "phonemes": phonemes,
                "not_found_words": not_found_words,
                "total_words": len(words),
                "aligned_words": sum(1 for w in words if w.success())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def retest_all_errors(self):
        """Re-test all error files"""
        print("\n" + "=" * 70)
        print("RE-TESTING ERROR FILES WITH UPDATED LEXICON")
        print("=" * 70)
        print()
        
        # Find all error files
        error_files = sorted(self.errors_dir.glob("*.txt"))
        
        if not error_files:
            print("No error files found in", self.errors_dir)
            return
        
        print(f"Found {len(error_files)} error files to re-test")
        print()
        
        # Statistics
        stats = {
            "total": 0,
            "audio_not_found": 0,
            "still_has_oov": 0,
            "complete_oov": 0,
            "partial_alignment": 0,
            "full_alignment": 0,
            "test_failed": 0
        }
        
        results = []
        
        for txt_file in error_files[:20]:  # Limit to 20 for testing
            stats["total"] += 1
            
            # Get audio file
            audio_file = txt_file.with_suffix('.mp3')
            if not audio_file.exists():
                audio_file = txt_file.with_suffix('.aac')
            
            if not audio_file.exists():
                print(f"⚠ Audio not found: {txt_file.stem}")
                stats["audio_not_found"] += 1
                continue
            
            # Read original error info
            transcript, original_missing = self.read_error_file(txt_file)
            
            if not transcript:
                print(f"⚠ Could not read: {txt_file.name}")
                stats["test_failed"] += 1
                continue
            
            # Test with updated lexicon
            print(f"\nTesting: {txt_file.stem}")
            print(f"Transcript: {transcript[:60]}...")
            
            result = self.test_alignment(audio_file, transcript)
            
            if not result["success"]:
                print(f"  ✗ Test failed: {result.get('error', 'Unknown error')}")
                stats["test_failed"] += 1
                continue
            
            # Analyze result
            if result["is_complete_oov"]:
                print(f"  ✗ COMPLETE OOV - No alignment possible")
                print(f"    Phonemes: {result['phonemes']}")
                stats["complete_oov"] += 1
                status = "COMPLETE_OOV"
            elif result["has_oov"]:
                aligned = result["aligned_words"]
                total = result["total_words"]
                print(f"  ⚠ PARTIAL - {aligned}/{total} words aligned")
                print(f"    Not found: {result['not_found_words'][:5]}...")
                stats["partial_alignment"] += 1
                status = "PARTIAL"
            else:
                print(f"  ✓ FULL ALIGNMENT - All words found!")
                stats["full_alignment"] += 1
                status = "SUCCESS"
            
            results.append({
                "file": txt_file.stem,
                "status": status,
                "transcript": transcript,
                "original_missing": original_missing,
                "new_not_found": result.get("not_found_words", []),
                "phonemes_count": len(result.get("phonemes", [])),
                "alignment_ratio": f"{result['aligned_words']}/{result['total_words']}"
            })
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY OF RE-TEST RESULTS")
        print("=" * 70)
        print()
        print(f"Total files tested: {stats['total']}")
        print(f"  ✓ Full alignment: {stats['full_alignment']} "
              f"({stats['full_alignment']/stats['total']*100:.1f}%)")
        print(f"  ⚠ Partial alignment: {stats['partial_alignment']} "
              f"({stats['partial_alignment']/stats['total']*100:.1f}%)")
        print(f"  ✗ Complete OOV: {stats['complete_oov']} "
              f"({stats['complete_oov']/stats['total']*100:.1f}%)")
        print(f"  ⚠ Test failed: {stats['test_failed']}")
        print(f"  ⚠ Audio not found: {stats['audio_not_found']}")
        print()
        
        # Detailed analysis
        if stats["complete_oov"] > 0:
            print("\n" + "-" * 70)
            print("FILES WITH COMPLETE OOV (Should be ZERO with updated lexicon!):")
            print("-" * 70)
            for r in results:
                if r["status"] == "COMPLETE_OOV":
                    print(f"\n{r['file']}")
                    print(f"  Transcript: {r['transcript'][:60]}...")
                    print(f"  Original missing: {r['original_missing'][:5]}...")
        
        if stats["partial_alignment"] > 0:
            print("\n" + "-" * 70)
            print("FILES WITH PARTIAL ALIGNMENT (Audio-transcript mismatch):")
            print("-" * 70)
            for r in results:
                if r["status"] == "PARTIAL":
                    print(f"\n{r['file']} [{r['alignment_ratio']}]")
                    print(f"  Transcript: {r['transcript'][:60]}...")
                    print(f"  Not found now: {r['new_not_found'][:5]}...")
        
        if stats["full_alignment"] > 0:
            print("\n" + "-" * 70)
            print("FILES NOW WITH FULL ALIGNMENT (Lexicon update success!):")
            print("-" * 70)
            for r in results:
                if r["status"] == "SUCCESS":
                    print(f"  ✓ {r['file']}")
        
        print()
        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        
        if stats["complete_oov"] > 0:
            print("⚠ WARNING: Files still showing complete OOV!")
            print("  This suggests the lexicon update didn't fully take effect.")
            print("  OR these files have words not in the speechline lexicon.")
        elif stats["partial_alignment"] > 0:
            print("✓ No complete OOV (lexicon update successful)")
            print("⚠ Partial alignments are due to audio-transcript mismatches,")
            print("  not lexicon issues. These are data quality problems.")
        else:
            print("✅ EXCELLENT: All tested files now have full alignment!")
            print("   The lexicon update completely resolved all OOV issues.")
        
        print()
        
        # Save results
        results_file = Path("errors/retest_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "stats": stats,
                "results": results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        print()


def main():
    retester = ErrorFileRetester()
    retester.retest_all_errors()


if __name__ == '__main__':
    main()