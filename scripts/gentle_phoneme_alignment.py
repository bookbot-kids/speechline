#!/usr/bin/env python3
"""
Gentle Phoneme Alignment Script

This script uses Gentle forced aligner to:
1. Validate transcripts against audio (find not-found-in-audio words)
2. Add phoneme transcriptions to transcript files

Usage:
    # Validation mode - find not-found-in-audio examples
    python scripts/gentle_phoneme_alignment.py \
        --mode validate \
        --bookbot-path /mnt/Store07/Bookbot \
        --cv-clips-path "/mnt/Store07/Common Voice/cv-corpus-22.0-2025-06-20/en/clips"
    
    # Phoneme addition mode
    python scripts/gentle_phoneme_alignment.py \
        --mode add-phonemes \
        --bookbot-path /mnt/Store07/Bookbot \
        --review-log phoneme_alignment_review.txt
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Optional
from queue import Queue
from threading import Thread, Lock

# Add Gentle to Python path
GENTLE_PATH = "/mnt/Projects/Projects/AudioProcessing/gentle"
sys.path.insert(0, GENTLE_PATH)

import gentle


class OOVError(Exception):
    """Raised when Out-Of-Vocabulary word is found"""
    def __init__(self, word: str, audio_path: str):
        self.word = word
        self.audio_path = audio_path
        super().__init__(
            f"OOV word '{word}' found in {audio_path}. "
            f"This should not happen with align_lexicon.txt!"
        )


class GentleAligner:
    """Wrapper for Gentle forced aligner"""
    
    def __init__(self, gentle_path: str = GENTLE_PATH):
        """Initialize with path to Gentle installation"""
        self.gentle_path = gentle_path
        # Initialize Gentle resources
        self.resources = gentle.Resources()
        logging.info(f"Gentle resources initialized from {gentle_path}")
        
    def align(self, audio_path: str, transcript: str) -> Dict:
        """
        Perform forced alignment
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript
            
        Returns:
            Dictionary with alignment results including:
            - 'words': List of word alignment info
            - 'phonemes': Space-separated phoneme sequence (if successful)
            - 'not_found_words': List of words not found in audio
            - 'is_valid': Boolean indicating if alignment is valid
        """
        logging.debug(f"Aligning: {audio_path}")
        
        try:
            # Use Gentle's resampled context manager for audio conversion
            with gentle.resampled(audio_path) as wavfile:
                # Create forced aligner
                aligner = gentle.ForcedAligner(
                    self.resources, 
                    transcript,
                    nthreads=1,  # Single thread for reliability
                    disfluency=False,
                    conservative=False
                )
                
                # Perform alignment
                result = aligner.transcribe(wavfile, logging=logging)
                
        except Exception as e:
            logging.error(f"Alignment failed for {audio_path}: {e}")
            return {
                'words': [],
                'phonemes': None,
                'not_found_words': [],
                'is_valid': False,
                'error': str(e)
            }
        
        # Extract results
        words = result.words
        not_found_words = [w.word for w in words if w.not_found_in_audio()]
        
        # Extract phonemes only from successfully aligned words
        phonemes = []
        for word in words:
            if word.success() and word.phones:
                for phone_info in word.phones:
                    phone = phone_info.get('phone', '')
                    # Strip position markers (_B, _I, _E, _S)
                    phone_clean = phone.split('_')[0] if '_' in phone else phone
                    if phone_clean:
                        phonemes.append(phone_clean)
        
        phoneme_string = ' '.join(phonemes) if phonemes else None
        is_valid = len(not_found_words) == 0 and phoneme_string is not None
        
        return {
            'words': [w.as_dict() for w in words],
            'phonemes': phoneme_string,
            'not_found_words': not_found_words,
            'is_valid': is_valid
        }
    
    def check_oov(self, result: Dict) -> Optional[str]:
        """Check if result contains OOV words (detected as 'oov' phoneme)"""
        if result.get('phonemes'):
            phoneme_list = result['phonemes'].split()
            if 'oov' in phoneme_list:
                # Find which word caused the OOV
                for word_dict in result.get('words', []):
                    if word_dict.get('phones'):
                        for phone in word_dict['phones']:
                            if phone.get('phone', '').lower() == 'oov':
                                return word_dict.get('word', 'unknown')
                return 'unknown'
        return None


class DatasetScanner:
    """Scan directories for audio/transcript pairs"""
    
    def __init__(self):
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
    def scan_bookbot(self, base_path: str, language_prefixes: List[str] = None) -> Iterator[Tuple[str, str]]:
        """
        Yield (audio_path, transcript_path) for Bookbot language dirs
        
        Args:
            base_path: Base path to Bookbot directory
            language_prefixes: List of language prefixes (e.g., ['en', 'id', 'sw'])
                              Defaults to ['en'] for backward compatibility
            
        Yields:
            (audio_path, transcript_path) tuples
        """
        if language_prefixes is None:
            language_prefixes = ['en']
        
        base_path = Path(base_path)
        
        if not base_path.exists():
            logging.warning(f"Bookbot path does not exist: {base_path}")
            return
        
        # Find all matching language subdirectories
        for prefix in language_prefixes:
            for lang_dir in base_path.glob(f"{prefix}-*"):
                if not lang_dir.is_dir():
                    continue
                
                logging.info(f"Scanning Bookbot {prefix.upper()} directory: {lang_dir}")
                
                # Find all audio files
                for audio_file in lang_dir.rglob("*"):
                    if audio_file.suffix.lower() in self.audio_extensions:
                        # Look for corresponding .txt file
                        txt_file = audio_file.with_suffix('.txt')
                        if txt_file.exists():
                            yield (str(audio_file), str(txt_file))
    
    def scan_common_voice(self, clips_path: str) -> Iterator[Tuple[str, str]]:
        """
        Yield (audio_path, transcript_path) for Common Voice clips
        
        Args:
            clips_path: Path to Common Voice clips directory
            
        Yields:
            (audio_path, transcript_path) tuples
        """
        clips_path = Path(clips_path)
        
        if not clips_path.exists():
            logging.warning(f"Common Voice path does not exist: {clips_path}")
            return
        
        logging.info(f"Scanning Common Voice directory: {clips_path}")
        
        # Find all audio files
        for audio_file in clips_path.glob("*"):
            if audio_file.suffix.lower() in self.audio_extensions:
                # Look for corresponding .txt file
                txt_file = audio_file.with_suffix('.txt')
                if txt_file.exists():
                    yield (str(audio_file), str(txt_file))
    
    def has_phoneme_transcript(self, transcript_path: str) -> bool:
        """Check if transcript file already has phonemes (2+ lines)"""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return len(lines) >= 2
        except Exception as e:
            logging.error(f"Error reading {transcript_path}: {e}")
            return False


class TranscriptManager:
    """Manage transcript file operations"""
    
    def read_transcript(self, path: str) -> str:
        """Read text transcript from file (first line only)"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.readline().strip()
        except Exception as e:
            logging.error(f"Error reading transcript {path}: {e}")
            return ""
    
    def has_phonemes(self, path: str) -> bool:
        """Check if file has 2+ lines (text + phonemes)"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return len(lines) >= 2
        except Exception as e:
            logging.error(f"Error checking {path}: {e}")
            return False
    
    def add_phonemes(self, path: str, phonemes: str):
        """Append phoneme line to transcript file (atomic write)"""
        try:
            # Read existing content
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline().rstrip('\n')
            
            # Write to temp file in same directory (avoid cross-device link error)
            path_obj = Path(path)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.txt',
                dir=path_obj.parent,
                text=True
            )
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    f.write(first_line + '\n')
                    f.write(phonemes + '\n')
                
                # Atomic rename
                os.replace(temp_path, path)
                logging.debug(f"Added phonemes to {path}")
                
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise
                
        except Exception as e:
            logging.error(f"Error adding phonemes to {path}: {e}")
            raise
    
    def add_invalid_marker(self, path: str, not_found_words: List[str]):
        """Add INVALID_TRANSCRIPT marker to transcript file"""
        try:
            # Read existing content
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline().rstrip('\n')
            
            # Create invalid marker
            invalid_line = f"INVALID_TRANSCRIPT: not-found-in-audio={not_found_words}"
            
            # Write to temp file in same directory (avoid cross-device link error)
            path_obj = Path(path)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.txt',
                dir=path_obj.parent,
                text=True
            )
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    f.write(first_line + '\n')
                    f.write(invalid_line + '\n')
                
                # Atomic rename
                os.replace(temp_path, path)
                logging.debug(f"Added invalid marker to {path}")
                
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise
                
        except Exception as e:
            logging.error(f"Error adding invalid marker to {path}: {e}")
            raise


class ValidationMode:
    """Validation mode: Find not-found-in-audio examples"""
    
    def __init__(self, aligner: GentleAligner, scanner: DatasetScanner,
                 transcript_mgr: TranscriptManager, output_dir: str = "errors"):
        self.aligner = aligner
        self.scanner = scanner
        self.transcript_mgr = transcript_mgr
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.bookbot_dir = self.output_dir / "bookbot"
        self.cv_dir = self.output_dir / "common_voice"
        self.bookbot_dir.mkdir(parents=True, exist_ok=True)
        self.cv_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_example_files(self, audio_path: str, result: Dict, output_dir: Path):
        """Copy audio and create annotated transcript in output directory"""
        audio_path = Path(audio_path)
        
        # Copy audio file
        audio_dest = output_dir / audio_path.name
        shutil.copy2(audio_path, audio_dest)
        
        # Create annotated transcript
        txt_dest = audio_dest.with_suffix('.txt')
        transcript = result['transcript']
        not_found = result['not_found']
        phonemes = result.get('phonemes', '')
        
        with open(txt_dest, 'w', encoding='utf-8') as f:
            f.write(f"{transcript}\n")
            if phonemes:
                f.write(f"{phonemes}\n")
            else:
                f.write("oov\n")
            f.write(f"NOT_FOUND_IN_AUDIO: {not_found}\n")
    
    def run(self, bookbot_path: Optional[str], cv_clips_path: Optional[str],
            examples_per_dataset: int = 20):
        """Run validation mode and display not-found-in-audio examples"""
        
        results = {
            'bookbot': [],
            'common_voice': []
        }
        
        # Scan Bookbot
        if bookbot_path:
            lang_prefixes = getattr(self.scanner, 'language_prefixes', ['en'])
            print(f"\n{'='*60}")
            print(f"SCANNING BOOKBOT DATASET (languages: {lang_prefixes})")
            print(f"{'='*60}\n")
            
            count = 0
            for audio_path, txt_path in self.scanner.scan_bookbot(bookbot_path, lang_prefixes):
                if count >= examples_per_dataset:
                    break
                
                transcript = self.transcript_mgr.read_transcript(txt_path)
                if not transcript:
                    continue
                
                result = self.aligner.align(audio_path, transcript)
                
                if result['not_found_words']:
                    example = {
                        'file': audio_path,
                        'transcript': transcript,
                        'not_found': result['not_found_words'],
                        'phonemes': result.get('phonemes', '')
                    }
                    results['bookbot'].append(example)
                    
                    # Copy files to errors directory
                    self.copy_example_files(audio_path, example, self.bookbot_dir)
                    
                    count += 1
                    print(f"Found example {count}/{examples_per_dataset}")
            
            print(f"\nFound {count} Bookbot examples with not-found-in-audio words")
        
        # Scan Common Voice
        if cv_clips_path:
            print(f"\n{'='*60}")
            print(f"SCANNING COMMON VOICE DATASET")
            print(f"{'='*60}\n")
            
            count = 0
            for audio_path, txt_path in self.scanner.scan_common_voice(cv_clips_path):
                if count >= examples_per_dataset:
                    break
                
                transcript = self.transcript_mgr.read_transcript(txt_path)
                if not transcript:
                    continue
                
                result = self.aligner.align(audio_path, transcript)
                
                if result['not_found_words']:
                    example = {
                        'file': audio_path,
                        'transcript': transcript,
                        'not_found': result['not_found_words'],
                        'phonemes': result.get('phonemes', '')
                    }
                    results['common_voice'].append(example)
                    
                    # Copy files to errors directory
                    self.copy_example_files(audio_path, example, self.cv_dir)
                    
                    count += 1
                    print(f"Found example {count}/{examples_per_dataset}")
            
            print(f"\nFound {count} Common Voice examples with not-found-in-audio words")
        
        # Display results
        self.display_results(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FILES COPIED TO:")
        print(f"  Bookbot: {self.bookbot_dir}")
        print(f"  Common Voice: {self.cv_dir}")
        print(f"{'='*60}\n")
    
    def display_results(self, results: Dict):
        """Display validation results"""
        
        # Bookbot results
        if results['bookbot']:
            print(f"\n{'='*60}")
            print(f"BOOKBOT DATASET - Not-Found-In-Audio Examples")
            print(f"{'='*60}\n")
            
            for i, example in enumerate(results['bookbot'], 1):
                print(f"Example {i}:")
                print(f"  File: {example['file']}")
                print(f"  Transcript: {example['transcript']}")
                print(f"  Not Found: {example['not_found']}")
                print()
        
        # Common Voice results
        if results['common_voice']:
            print(f"\n{'='*60}")
            print(f"COMMON VOICE DATASET - Not-Found-In-Audio Examples")
            print(f"{'='*60}\n")
            
            for i, example in enumerate(results['common_voice'], 1):
                print(f"Example {i}:")
                print(f"  File: {example['file']}")
                print(f"  Transcript: {example['transcript']}")
                print(f"  Not Found: {example['not_found']}")
                print()


class PhonemeAdditionMode:
    """Phoneme addition mode: Add phoneme transcripts to files"""
    
    def __init__(self, aligner: GentleAligner, scanner: DatasetScanner,
                 transcript_mgr: TranscriptManager, review_log: str, threads: int = 1):
        self.aligner = aligner
        self.scanner = scanner
        self.transcript_mgr = transcript_mgr
        self.review_log = review_log
        self.threads = threads
        self.stats = {
            'processed': 0,
            'skipped_has_phonemes': 0,
            'added_phonemes': 0,
            'added_invalid': 0,
            'oov_errors': 0,
            'other_errors': 0
        }
        self.stats_lock = Lock()
        self.review_lock = Lock()
    
    def run(self, bookbot_path: Optional[str], cv_clips_path: Optional[str]):
        """Run phoneme addition mode"""
        
        # Open review log
        review_file = open(self.review_log, 'w', encoding='utf-8') if self.review_log else None
        
        try:
            # Process Bookbot
            if bookbot_path:
                lang_prefixes = getattr(self.scanner, 'language_prefixes', ['en'])
                print(f"\n{'='*60}")
                print(f"PROCESSING BOOKBOT DATASET (languages: {lang_prefixes})")
                print(f"{'='*60}\n")
                self.process_dataset(self.scanner.scan_bookbot(bookbot_path, lang_prefixes), review_file)
            
            # Process Common Voice
            if cv_clips_path:
                print(f"\n{'='*60}")
                print(f"PROCESSING COMMON VOICE DATASET")
                print(f"{'='*60}\n")
                self.process_dataset(self.scanner.scan_common_voice(cv_clips_path), review_file)
        
        finally:
            if review_file:
                review_file.close()
        
        # Display final statistics
        self.display_stats()
    
    def process_dataset(self, file_iterator: Iterator[Tuple[str, str]], review_file):
        """Process a dataset with multi-threading support"""
        
        if self.threads == 1:
            # Single-threaded processing (original behavior)
            for audio_path, txt_path in file_iterator:
                self._process_file(audio_path, txt_path, review_file)
        else:
            # Multi-threaded processing
            self._process_dataset_threaded(file_iterator, review_file)
    
    def _process_dataset_threaded(self, file_iterator: Iterator[Tuple[str, str]], review_file):
        """Process dataset using multiple threads"""
        
        # Create work queue
        work_queue = Queue()
        
        # Fill queue with files
        file_count = 0
        for audio_path, txt_path in file_iterator:
            work_queue.put((audio_path, txt_path))
            file_count += 1
        
        print(f"Found {file_count} files to process with {self.threads} threads")
        
        # Add sentinel values to signal thread completion
        for _ in range(self.threads):
            work_queue.put(None)
        
        # Start worker threads
        threads = []
        for i in range(self.threads):
            thread = Thread(
                target=self._worker_thread,
                args=(work_queue, review_file, i),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    def _worker_thread(self, work_queue: Queue, review_file, thread_id: int):
        """Worker thread for processing files"""
        logging.info(f"Thread {thread_id} started")
        
        while True:
            item = work_queue.get()
            if item is None:
                # Sentinel value - exit thread
                logging.info(f"Thread {thread_id} finished")
                break
            
            audio_path, txt_path = item
            try:
                self._process_file(audio_path, txt_path, review_file)
            finally:
                work_queue.task_done()
    
    def _process_file(self, audio_path: str, txt_path: str, review_file):
        """Process a single file (thread-safe)"""
        
        with self.stats_lock:
            self.stats['processed'] += 1
            processed_count = self.stats['processed']
            
            # Log progress every 100 files
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
                print(f"  Added phonemes: {self.stats['added_phonemes']}")
                print(f"  Added invalid: {self.stats['added_invalid']}")
                print(f"  Skipped (has phonemes): {self.stats['skipped_has_phonemes']}")
        
        try:
            # Skip if already has phonemes
            if self.transcript_mgr.has_phonemes(txt_path):
                with self.stats_lock:
                    self.stats['skipped_has_phonemes'] += 1
                return
            
            # Read transcript
            transcript = self.transcript_mgr.read_transcript(txt_path)
            if not transcript:
                logging.warning(f"Empty transcript: {txt_path}")
                return
            
            # Perform alignment
            result = self.aligner.align(audio_path, transcript)
            
            # Check for OOV - mark as oov instead of raising exception
            oov_word = self.aligner.check_oov(result)
            if oov_word:
                # Add "oov" marker to file
                self.transcript_mgr.add_phonemes(txt_path, "oov")
                with self.stats_lock:
                    self.stats['oov_errors'] += 1
                
                # Log to review file (thread-safe)
                if review_file:
                    with self.review_lock:
                        review_file.write(f"File: {audio_path}\n")
                        review_file.write(f"Transcript: {transcript}\n")
                        review_file.write(f"OOV Word: {oov_word}\n")
                        review_file.write(f"Reason: Out-of-vocabulary word found\n")
                        review_file.write("---\n\n")
                        review_file.flush()
                
                logging.warning(f"OOV word '{oov_word}' in {audio_path} - marked as oov")
                return
            
            # Check for not-found-in-audio words
            if result['not_found_words']:
                # Add invalid marker to file
                self.transcript_mgr.add_invalid_marker(txt_path, result['not_found_words'])
                
                with self.stats_lock:
                    self.stats['added_invalid'] += 1
                
                # Log to review file (thread-safe)
                if review_file:
                    with self.review_lock:
                        review_file.write(f"File: {audio_path}\n")
                        review_file.write(f"Transcript: {transcript}\n")
                        review_file.write(f"Not Found Words: {result['not_found_words']}\n")
                        review_file.write(f"Reason: Words not found in audio during forced alignment\n")
                        review_file.write("---\n\n")
                        review_file.flush()
                
                return
            
            # Add phonemes to file
            if result['phonemes']:
                self.transcript_mgr.add_phonemes(txt_path, result['phonemes'])
                with self.stats_lock:
                    self.stats['added_phonemes'] += 1
            else:
                logging.warning(f"No phonemes extracted for {audio_path}")
                
        except Exception as e:
            with self.stats_lock:
                self.stats['other_errors'] += 1
            logging.error(f"Error processing {audio_path}: {e}")
    
    def display_stats(self):
        """Display final statistics"""
        print(f"\n{'='*60}")
        print(f"PHONEME ADDITION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {self.stats['processed']}")
        print(f"Phonemes added: {self.stats['added_phonemes']}")
        print(f"Invalid markers added: {self.stats['added_invalid']}")
        print(f"Skipped (already has phonemes): {self.stats['skipped_has_phonemes']}")
        print(f"OOV errors: {self.stats['oov_errors']}")
        print(f"Other errors: {self.stats['other_errors']}")
        print(f"\nReview log: {self.review_log}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Gentle Phoneme Alignment Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode',
        required=True,
        choices=['validate', 'add-phonemes'],
        help='Operating mode'
    )
    parser.add_argument(
        '--bookbot-path',
        type=str,
        help='Path to Bookbot dataset'
    )
    parser.add_argument(
        '--cv-clips-path',
        type=str,
        help='Path to Common Voice clips directory'
    )
    parser.add_argument(
        '--gentle-path',
        type=str,
        default=GENTLE_PATH,
        help=f'Path to Gentle installation (default: {GENTLE_PATH})'
    )
    parser.add_argument(
        '--review-log',
        type=str,
        default='phoneme_alignment_review.txt',
        help='Review log file for invalid transcripts (default: phoneme_alignment_review.txt)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--examples-per-dataset',
        type=int,
        default=20,
        help='Number of examples to show per dataset in validation mode (default: 20)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of threads for parallel processing (default: 1)'
    )
    parser.add_argument(
        '--language',
        type=str,
        nargs='+',
        default=['en'],
        help='Language prefixes to process (e.g., en id sw). Default: en'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate inputs
    if not args.bookbot_path and not args.cv_clips_path:
        parser.error("At least one of --bookbot-path or --cv-clips-path must be provided")
    
    # Initialize components
    logging.info("Initializing Gentle aligner...")
    aligner = GentleAligner(args.gentle_path)
    scanner = DatasetScanner()
    transcript_mgr = TranscriptManager()
    
    # Run selected mode
    if args.mode == 'validate':
        logging.info(f"Running validation mode for languages: {args.language}...")
        validator = ValidationMode(aligner, scanner, transcript_mgr, output_dir="errors")
        # Store language prefixes for scanner to use
        scanner.language_prefixes = args.language
        validator.run(
            args.bookbot_path,
            args.cv_clips_path,
            args.examples_per_dataset
        )
    else:  # add-phonemes
        logging.info(f"Running phoneme addition mode with {args.threads} thread(s) for languages: {args.language}...")
        processor = PhonemeAdditionMode(aligner, scanner, transcript_mgr, args.review_log, args.threads)
        # Store language prefixes for scanner to use
        scanner.language_prefixes = args.language
        processor.run(args.bookbot_path, args.cv_clips_path)


if __name__ == '__main__':
    main()