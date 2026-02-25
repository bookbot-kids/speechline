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

import csv
import logging
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from .g2p import g2p_en

logger = logging.getLogger(__name__)


class LexiconManager:
    """Manages word-to-phoneme mappings from english_words_processed.csv.
    
    This class loads a lexicon containing words with their IPA transcriptions
    and pre-computed phonetic class sequences. It provides efficient lookup
    of all phoneme variations for a word and falls back to G2P for OOV words.
    
    Attributes:
        lexicon: Dict mapping words to list of (ipa, class_sequence) tuples
        csv_path: Path to the lexicon CSV file
    """
    
    def __init__(self, csv_path: str = 'data/english_words_processed.csv', use_g2p_fallback: bool = True):
        """Initialize lexicon manager and load lexicon.
        
        Args:
            csv_path: Path to lexicon CSV file relative to project root
            use_g2p_fallback: Whether to use G2P for OOV words (default: True)
        """
        self.csv_path = Path(csv_path)
        self.use_g2p_fallback = use_g2p_fallback
        self.lexicon: Dict[str, List[Dict[str, str]]] = {}
        self._accent_rules_manager = None  # Cached accent rules manager for G2P variations
        self._load_lexicon()
        logger.info(f"Loaded {len(self.lexicon)} words with {self._count_variations()} total variations")
        if not use_g2p_fallback:
            logger.info("G2P fallback is DISABLED")
    
    def _load_lexicon(self) -> None:
        """Load lexicon from CSV file.
        
        Expected CSV format:
            word,ipa,class_sequence
            hello,hɛˈloʊ,hələ
            hello,həˈloʊ,hələ
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {self.csv_path}")
        
        # Use defaultdict to group by word
        lexicon_temp = defaultdict(list)
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                word = row['word'].lower().strip()
                ipa = row['ipa'].strip()
                class_seq = row['class_sequence'].strip()
                
                # Store as dict for clarity
                lexicon_temp[word].append({
                    'ipa': ipa,
                    'class_sequence': class_seq
                })
        
        # Convert to regular dict
        self.lexicon = dict(lexicon_temp)
    
    def get_phoneme_variations(self, word: str) -> List[str]:
        """Get all IPA phoneme variations for a word.
        
        Args:
            word: Word to look up (case-insensitive)
            
        Returns:
            List of IPA transcriptions for the word. If word not in lexicon,
            returns G2P prediction in a list.
            
        Example:
            >>> manager = LexiconManager()
            >>> manager.get_phoneme_variations("hello")
            ['h ɛ l oʊ', 'h ə l oʊ']
        """
        word_lower = word.lower().strip()
        
        if word_lower in self.lexicon:
            return [entry['ipa'] for entry in self.lexicon[word_lower]]
        else:
            if self.use_g2p_fallback:
                # Fall back to G2P
                logger.debug(f"Word '{word}' not in lexicon, using G2P")
                phonemes = g2p_en(word)
                return [' '.join(phonemes)]
            else:
                # Return empty list for OOV words
                logger.debug(f"Word '{word}' not in lexicon (G2P disabled)")
                return []
    
    def get_class_sequences(self, word: str) -> List[str]:
        """Get all phonetic class sequences for a word.
        
        Args:
            word: Word to look up (case-insensitive)
            
        Returns:
            List of class sequences for the word. If word not in lexicon,
            computes from G2P output and generates variations using accent rules.
            
        Example:
            >>> manager = LexiconManager()
            >>> manager.get_class_sequences("hello")
            ['hələ', 'hələ']
        """
        word_lower = word.lower().strip()
        
        if word_lower in self.lexicon:
            return [entry['class_sequence'] for entry in self.lexicon[word_lower]]
        else:
            if self.use_g2p_fallback:
                # Fall back to G2P and compute class sequence
                # NOTE: Temporarily disabled accent rule variations for G2P due to performance
                from speechline.phonetics import ipa_to_artemes
                
                logger.debug(f"Word '{word}' not in lexicon, using G2P")
                phonemes = g2p_en(word)
                phoneme_str = ''.join(phonemes)
                class_seq = ipa_to_artemes(phoneme_str)
                return [class_seq]
            else:
                # Return empty list for OOV words
                logger.debug(f"Word '{word}' not in lexicon (G2P disabled)")
                return []
    
    def text_to_phonemes(self, text: str) -> List[List[str]]:
        """Convert text to word-level phoneme variations.
        
        Args:
            text: Input text (sentence or phrase)
            
        Returns:
            List of lists, where each inner list contains all phoneme
            variations for that word position.
            
        Example:
            >>> manager = LexiconManager()
            >>> manager.text_to_phonemes("hello world")
            [
                ['h ɛ l oʊ', 'h ə l oʊ'],
                ['w ɝ l d', 'w ɔ l d']
            ]
        """
        words = text.lower().strip().split()
        return [self.get_phoneme_variations(word) for word in words]
    
    def text_to_class_sequences(self, text: str) -> List[List[str]]:
        """Convert text to word-level class sequence variations.
        
        Args:
            text: Input text (sentence or phrase)
            
        Returns:
            List of lists, where each inner list contains all class sequence
            variations for that word position.
            
        Example:
            >>> manager = LexiconManager()
            >>> manager.text_to_class_sequences("hello world")
            [
                ['hələ', 'hələ'],
                ['əɹlt', 'əlt']
            ]
        """
        words = text.lower().strip().split()
        return [self.get_class_sequences(word) for word in words]
    
    def get_all_class_sequence_combinations(self, text: str) -> Set[str]:
        """Get all possible class sequence combinations for a text.
        
        This generates all possible sentence-level class sequences by combining
        all word-level variations.
        
        Args:
            text: Input text (sentence or phrase)
            
        Returns:
            Set of all possible class sequences (concatenated, space-separated)
            
        Example:
            >>> manager = LexiconManager()
            >>> manager.get_all_class_sequence_combinations("hello world")
            {'hələ əɹlt', 'hələ əlt', 'hələ əɹlt', 'hələ əlt'}
        """
        word_class_variations = self.text_to_class_sequences(text)
        
        if not word_class_variations:
            return set()
        
        # Generate all combinations using iterative product
        combinations = {''}
        for word_variations in word_class_variations:
            new_combinations = set()
            for existing in combinations:
                for variation in word_variations:
                    if existing:
                        new_combinations.add(f"{existing} {variation}")
                    else:
                        new_combinations.add(variation)
            combinations = new_combinations
        
        return combinations
    
    def _count_variations(self) -> int:
        """Count total number of variations in lexicon."""
        return sum(len(variations) for variations in self.lexicon.values())
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in lexicon."""
        return word.lower().strip() in self.lexicon
    
    def __len__(self) -> int:
        """Return number of words in lexicon."""
        return len(self.lexicon)