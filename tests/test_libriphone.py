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

"""Test suite for LibriPhone dataset validation.

This test validates phoneme matching on the LibriPhone dataset,
which provides gold-standard word and phoneme transcriptions.
Target: 100% match rate.
"""

import pytest
import logging
import csv
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from speechline.utils.lexicon_manager import LexiconManager
from scripts.ipa_to_class_mapping import normalize_ipa, parse_ipa_sequence

# Configure logging for debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def lexicon_manager():
    """Create LexiconManager instance for testing."""
    logger.info("="*80)
    logger.info("INITIALIZING LEXICON MANAGER WITH G2P ENABLED")
    logger.info("="*80)
    
    lexicon_mgr = LexiconManager(use_g2p_fallback=True)
    logger.info(f"✓ Lexicon loaded: {len(lexicon_mgr)} words")
    logger.info(f"✓ G2P fallback: ENABLED")
    logger.info("="*80)
    
    return lexicon_mgr


def load_libriphone_csv(csv_path: str = 'data/libriphone_test.csv') -> List[Dict]:
    """Load LibriPhone dataset from CSV with pipe-delimited words.
    
    CSV format:
        id,text,ipa
        1089-134686-0000,HE HOPED THERE,h i|h oʊ p t|ð ɛ ɹ
    
    Returns list of dicts with:
        - id: sample id
        - text: reference text
        - words: list of words
        - ipa_words: list of IPA strings (one per word, spaces removed)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"LibriPhone CSV not found: {csv_path}")
    
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Split text into words
            words = row['text'].split()
            
            # Split IPA by pipes and remove spaces within each word
            ipa_words = [ipa_word.replace(' ', '') for ipa_word in row['ipa'].split('|')]
            
            # Verify word count matches
            if len(words) != len(ipa_words):
                logger.warning(f"Word count mismatch in {row['id']}: {len(words)} words, {len(ipa_words)} IPA segments")
                continue
            
            samples.append({
                'id': row['id'],
                'text': row['text'],
                'words': words,
                'ipa_words': ipa_words
            })
    
    return samples


def format_word_alignment(
    words: List[str],
    expected_classes: List[str],
    actual_classes: List[str],
    mismatch_indices: List[int] = None
) -> str:
    """Format word-by-word alignment with class sequences.
    
    Format:
        word1|word2|word3|...
        cls1 |cls2 |cls3 |...  (expected from lexicon)
        cls1 |cls2 |cls3 |...  (actual from dataset)
    
    Segments lines at 80 characters and shows all mismatches.
    """
    if mismatch_indices is None:
        mismatch_indices = []
    
    # Calculate max width for each column
    max_widths = []
    for i in range(len(words)):
        width = max(
            len(words[i]),
            len(expected_classes[i]) if i < len(expected_classes) else 0,
            len(actual_classes[i]) if i < len(actual_classes) else 0
        )
        max_widths.append(width)
    
    # Build lines
    all_lines = []
    
    # Segment into chunks that fit 80 characters
    current_pos = 0
    start_idx = 0
    
    while start_idx < len(words):
        # Find how many words fit in 80 chars
        line_length = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(words)):
            word_width = max_widths[i] + 1  # +1 for pipe separator
            if line_length + word_width > 80 and i > start_idx:
                break
            line_length += word_width
            end_idx = i + 1
        
        # Format this segment
        segment_words = words[start_idx:end_idx]
        segment_widths = max_widths[start_idx:end_idx]
        
        # Line 1: Words
        word_line = '|'.join(
            segment_words[i].ljust(segment_widths[i])
            for i in range(len(segment_words))
        )
        
        # Line 2: Expected classes
        expected_line = '|'.join(
            (expected_classes[start_idx + i] if start_idx + i < len(expected_classes) else '').ljust(segment_widths[i])
            for i in range(len(segment_words))
        )
        
        # Line 3: Actual classes
        actual_line = '|'.join(
            (actual_classes[start_idx + i] if start_idx + i < len(actual_classes) else '').ljust(segment_widths[i])
            for i in range(len(segment_words))
        )
        
        all_lines.append(word_line)
        all_lines.append(expected_line)
        all_lines.append(actual_line)
        
        # Add marker line if there are mismatches in this segment
        segment_has_mismatch = any(start_idx <= idx < end_idx for idx in mismatch_indices)
        if segment_has_mismatch:
            marker_parts = []
            for i in range(len(segment_words)):
                global_idx = start_idx + i
                if global_idx in mismatch_indices:
                    marker_parts.append('^' * segment_widths[i])
                else:
                    marker_parts.append(' ' * segment_widths[i])
            marker_line = '|'.join(marker_parts)
            all_lines.append(marker_line)
        
        all_lines.append('')  # Blank line between segments
        start_idx = end_idx
    
    return '\n'.join(all_lines)


def collapse_vowels(class_seq: str) -> str:
    """Collapse consecutive vowel sounds by removing one.
    
    Vowels in phonetic class system: 'ə' (schwa/vowel class)
    If 2+ consecutive vowels, reduce by 1. Example: 'əəə' -> 'əə'
    """
    if not class_seq or 'ə' not in class_seq:
        return class_seq
    
    result = []
    i = 0
    while i < len(class_seq):
        if class_seq[i] == 'ə':
            # Count consecutive vowels
            vowel_count = 0
            j = i
            while j < len(class_seq) and class_seq[j] == 'ə':
                vowel_count += 1
                j += 1
            
            # If 2 or more vowels, reduce by 1
            if vowel_count >= 2:
                result.extend(['ə'] * (vowel_count - 1))
            else:
                result.append('ə')
            
            i = j
        else:
            result.append(class_seq[i])
            i += 1
    
    return ''.join(result)


def compare_words(
    lexicon_manager: LexiconManager,
    sample: Dict
) -> Dict:
    """Compare word-by-word using phonetic class sequences.
    
    Checks all lexicon variations and applies vowel collapsing.
    Shows the actual matching variation (not just first one).
    
    Returns dict with:
        - matched: bool
        - mismatch_word_indices: list of int (indices of mismatched words)
        - expected_classes: list of class sequences that matched (or first if no match)
        - actual_classes: list of class sequences from dataset
    """
    words = sample['words']
    actual_ipa = sample['ipa_words']
    expected_classes_display = []  # For display (matched variation or first)
    actual_classes = []
    mismatch_indices = []
    
    # Convert actual IPA to class sequences first
    for i, ipa in enumerate(actual_ipa):
        # Normalize IPA
        normalized_ipa = normalize_ipa(ipa, remove_stress=True, merge_spaces=True)
        # Convert to class sequence
        class_seq = parse_ipa_sequence(normalized_ipa, strict=False, normalize=False)
        actual_classes.append(class_seq)
    
    # Compare word by word - check ALL lexicon variations with vowel collapsing
    for i, word in enumerate(words):
        word_lower = word.lower()
        variations = lexicon_manager.get_phoneme_variations(word_lower)
        
        if not variations:
            # No lexicon entry (shouldn't happen with G2P fallback)
            expected_classes_display.append('???')
            mismatch_indices.append(i)
            continue
        
        if i >= len(actual_classes):
            # Dataset has fewer words than expected
            expected_classes_display.append('???')
            mismatch_indices.append(i)
            continue
        
        # Try all variations (with and without vowel collapsing)
        actual_class = actual_classes[i]
        word_matched = False
        matched_class = None  # Track which variation matched
        
        for variation in variations:
            # Remove spaces from lexicon IPA
            variation_clean = variation.replace(' ', '')
            # Normalize IPA
            normalized_ipa = normalize_ipa(variation_clean, remove_stress=True, merge_spaces=True)
            # Convert to class sequence
            expected_class = parse_ipa_sequence(normalized_ipa, strict=False, normalize=False)
            
            # Try exact match
            if expected_class == actual_class:
                word_matched = True
                matched_class = expected_class
                break
            
            # Try with vowel collapsing on expected
            expected_collapsed = collapse_vowels(expected_class)
            if expected_collapsed == actual_class:
                word_matched = True
                matched_class = expected_collapsed
                break
            
            # Try actual with vowel collapsing
            actual_collapsed = collapse_vowels(actual_class)
            if expected_class == actual_collapsed:
                word_matched = True
                matched_class = expected_class
                break
            
            # Try both collapsed
            if expected_collapsed == actual_collapsed:
                word_matched = True
                matched_class = expected_collapsed
                break
        
        # Store the matched variation for display, or first if no match
        if matched_class is not None:
            expected_classes_display.append(matched_class)
        else:
            # No match found - use first variation
            first_var = variations[0].replace(' ', '')
            first_norm = normalize_ipa(first_var, remove_stress=True, merge_spaces=True)
            first_class = parse_ipa_sequence(first_norm, strict=False, normalize=False)
            expected_classes_display.append(first_class)
        
        # Track if no match found
        if not word_matched:
            mismatch_indices.append(i)
    
    # Return result
    matched = len(mismatch_indices) == 0
    return {
        'matched': matched,
        'mismatch_word_indices': mismatch_indices,
        'expected_classes': expected_classes_display,
        'actual_classes': actual_classes,
        'reason': f'word_mismatches_at_{mismatch_indices}' if not matched else None
    }


class TestLibriPhoneMatching:
    """Test phoneme matching on LibriPhone dataset."""
    
    def test_libriphone_csv_structure(self):
        """Test that CSV has expected structure."""
        samples = load_libriphone_csv()
        assert len(samples) > 0, "Dataset is empty"
        
        # Check first sample
        sample = samples[0]
        assert 'id' in sample
        assert 'text' in sample
        assert 'words' in sample
        assert 'ipa_words' in sample
        assert len(sample['words']) == len(sample['ipa_words'])
    
    def test_libriphone_count_all_mismatches(self, lexicon_manager):
        """Count all mismatches in entire LibriPhone dataset.
        
        This test:
        1. Loads LibriPhone CSV with pipe-delimited words
        2. Compares each word's IPA with lexicon
        3. Counts ALL mismatches (doesn't stop)
        4. Shows summary statistics
        
        Target: Get full picture of mismatch rate
        """
        samples = load_libriphone_csv()
        
        logger.info("\n" + "="*80)
        logger.info(f"COUNTING ALL MISMATCHES - {len(samples)} samples")
        logger.info("="*80)
        
        total_samples = 0
        matched_samples = 0
        total_words = 0
        matched_words = 0
        mismatched_words = {}  # word -> count
        
        for sample in tqdm(samples, desc="Counting Mismatches"):
            total_samples += 1
            
            # Compare word by word
            result = compare_words(lexicon_manager, sample)
            
            # Count words
            total_words += len(sample['words'])
            matched_words += len(sample['words']) - len(result['mismatch_word_indices'])
            
            if result['matched']:
                matched_samples += 1
            else:
                # Track mismatched words
                for idx in result['mismatch_word_indices']:
                    word = sample['words'][idx].upper()
                    mismatched_words[word] = mismatched_words.get(word, 0) + 1
        
        # Calculate rates
        sample_match_rate = (matched_samples / total_samples * 100) if total_samples > 0 else 0
        word_match_rate = (matched_words / total_words * 100) if total_words > 0 else 0
        
        # Show summary
        logger.info("\n" + "="*80)
        logger.info("MISMATCH SUMMARY")
        logger.info("="*80)
        logger.info(f"\nSample-level:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Matched samples: {matched_samples}")
        logger.info(f"  Mismatched samples: {total_samples - matched_samples}")
        logger.info(f"  Sample match rate: {sample_match_rate:.2f}%")
        
        logger.info(f"\nWord-level:")
        logger.info(f"  Total words: {total_words}")
        logger.info(f"  Matched words: {matched_words}")
        logger.info(f"  Mismatched words: {total_words - matched_words}")
        logger.info(f"  Word match rate: {word_match_rate:.2f}%")
        
        if mismatched_words:
            logger.info(f"\nTop 20 most frequent mismatched words:")
            sorted_mismatches = sorted(mismatched_words.items(), key=lambda x: x[1], reverse=True)
            for word, count in sorted_mismatches[:20]:
                logger.info(f"  {word}: {count} occurrences")
        
        logger.info("="*80)
        
        # This test always passes - it's just for counting
        logger.info(f"\nTest complete. Word match rate: {word_match_rate:.2f}%")

    def test_libriphone_word_by_word_matches(self, lexicon_manager):
        """Test word-by-word phoneme matching.
        
        This test:
        1. Loads LibriPhone CSV with pipe-delimited words
        2. Compares each word's IPA with lexicon
        3. Stops at first mismatch
        4. Shows visual alignment
        
        Target: 100% match rate
        """
        samples = load_libriphone_csv()
        
        logger.info("\n" + "="*80)
        logger.info(f"STARTING LIBRIPHONE WORD-BY-WORD TEST - {len(samples)} samples")
        logger.info("="*80)
        
        total = 0
        matched = 0
        
        for sample in tqdm(samples, desc="Testing LibriPhone"):
            total += 1
            
            # Compare word by word
            result = compare_words(lexicon_manager, sample)
            
            if result['matched']:
                matched += 1
            else:
                # STOP AT FIRST MISMATCH - Show visual alignment
                logger.error("\n" + "="*80)
                logger.error(f"MISMATCH DETECTED - Sample: {sample['id']}")
                logger.error("="*80)
                logger.error(f"\nReference Text: {sample['text']}")
                
                # Show mismatched words
                mismatch_words = [sample['words'][i] for i in result['mismatch_word_indices']]
                logger.error(f"Mismatched words: {', '.join(mismatch_words)}")
                logger.error(f"Mismatch indices: {result['mismatch_word_indices']}")
                
                logger.error(f"\nWord-by-Word Class Alignment (80-char segments):")
                logger.error("-"*80)
                logger.error("")  # Add blank line before alignment for proper formatting
                
                alignment = format_word_alignment(
                    words=sample['words'],
                    expected_classes=result['expected_classes'],
                    actual_classes=result['actual_classes'],
                    mismatch_indices=result['mismatch_word_indices']
                )
                logger.error(alignment)
                
                logger.error("-"*80)
                logger.error("="*80)
                
                # FAIL FAST - stop at first mismatch
                pytest.fail(
                    f"First mismatch found at sample {sample['id']}, "
                    f"{len(result['mismatch_word_indices'])} word(s) mismatched: "
                    f"{', '.join(mismatch_words)}"
                )
        
        # Calculate metrics
        match_rate = matched / total if total > 0 else 0
        
        # Log results
        logger.info(f"\n{'='*80}")
        logger.info("LIBRIPHONE TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total samples:    {total:,}")
        logger.info(f"Matched:          {matched:,}")
        logger.info(f"Mismatched:       {total - matched:,}")
        logger.info(f"Match rate:       {match_rate:.2%}")
        logger.info(f"{'='*80}")
        
        # Assert 100% match rate
        assert match_rate == 1.0, (
            f"Match rate {match_rate:.2%} is below 100%. "
            f"{total - matched} mismatches found."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])