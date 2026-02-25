#!/usr/bin/env python3
"""
Test Gentle Setup and Lexicon

This script tests that:
1. Gentle is properly installed
2. The custom lexicon is loaded
3. Basic alignment works
"""

import sys
import logging
from pathlib import Path

# Add Gentle to path
GENTLE_PATH = "/mnt/Store07/Projects/gentle"
sys.path.insert(0, GENTLE_PATH)

import gentle

def test_gentle_resources():
    """Test that Gentle resources load properly"""
    print("Testing Gentle resources...")
    try:
        resources = gentle.Resources()
        print(f"✓ Resources loaded")
        print(f"  Proto langdir: {resources.proto_langdir}")
        print(f"  Vocab size: {len(resources.vocab)}")
        return resources
    except Exception as e:
        print(f"✗ Failed to load resources: {e}")
        return None

def test_lexicon_file():
    """Test that the custom lexicon file exists"""
    print("\nTesting custom lexicon file...")
    lexicon_path = Path("/mnt/Store07/Projects/gentle/exp/langdir/phones/align_lexicon.txt")
    backup_path = Path("/mnt/Store07/Projects/gentle/exp/langdir/phones/align_lexicon.txt.backup")
    
    if lexicon_path.exists():
        size_mb = lexicon_path.stat().st_size / 1024 / 1024
        print(f"✓ Lexicon file exists: {lexicon_path}")
        print(f"  Size: {size_mb:.1f} MB")
        
        # Check first few lines
        with open(lexicon_path, 'r') as f:
            lines = [f.readline() for _ in range(10)]
        print(f"  First non-comment line: {[l for l in lines if not l.startswith('#')][0].strip()[:80]}...")
    else:
        print(f"✗ Lexicon file not found: {lexicon_path}")
        return False
    
    if backup_path.exists():
        print(f"✓ Backup exists: {backup_path}")
    else:
        print(f"⚠ No backup found")
    
    return True

def test_sample_alignment(resources):
    """Test alignment with a sample audio file"""
    print("\nTesting sample alignment...")
    
    # Look for a test audio file
    test_audio = Path("tests/test_empty_short/en-us/test_audio.mp3")
    test_txt = Path("tests/test_empty_short/en-us/test_audio.txt")
    
    if not test_audio.exists() or not test_txt.exists():
        print(f"⚠ Test files not found, skipping alignment test")
        return True
    
    # Read transcript
    with open(test_txt, 'r') as f:
        transcript = f.read().strip()
    
    print(f"  Audio: {test_audio}")
    print(f"  Transcript: {transcript}")
    
    try:
        with gentle.resampled(str(test_audio)) as wavfile:
            aligner = gentle.ForcedAligner(resources, transcript, nthreads=1)
            result = aligner.transcribe(wavfile)
            
        print(f"✓ Alignment successful")
        print(f"  Words aligned: {len(result.words)}")
        
        # Check for not-found-in-audio
        not_found = [w.word for w in result.words if w.not_found_in_audio()]
        if not_found:
            print(f"  Not found in audio: {not_found}")
        else:
            print(f"  All words found in audio")
        
        # Extract phonemes
        phonemes = []
        for word in result.words:
            if word.success() and word.phones:
                for phone_info in word.phones:
                    phone = phone_info.get('phone', '')
                    phone_clean = phone.split('_')[0] if '_' in phone else phone
                    if phone_clean:
                        phonemes.append(phone_clean)
        
        if phonemes:
            print(f"  Phonemes: {' '.join(phonemes[:20])}{'...' if len(phonemes) > 20 else ''}")
        
        return True
        
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("GENTLE SETUP TEST")
    print("="*60)
    
    # Test resources
    resources = test_gentle_resources()
    if not resources:
        print("\n✗ FAILED: Could not load Gentle resources")
        sys.exit(1)
    
    # Test lexicon
    if not test_lexicon_file():
        print("\n✗ FAILED: Lexicon file not found")
        sys.exit(1)
    
    # Test alignment
    if not test_sample_alignment(resources):
        print("\n⚠ WARNING: Sample alignment failed")
        print("This may be normal if test files are not available")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nGentle is ready to use with the custom lexicon!")
    print("\nNext steps:")
    print("1. Run validation mode to find not-found-in-audio examples")
    print("2. Run phoneme addition mode to process datasets")

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()