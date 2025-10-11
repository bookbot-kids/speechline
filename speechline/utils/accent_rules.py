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
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from scripts.ipa_to_class_mapping import IPA_TO_CLASS

logger = logging.getLogger(__name__)


@dataclass
class AccentRule:
    """Represents a single accent rule.
    
    Attributes:
        name: Rule name/description
        country_codes: Applicable country codes
        examples: Example transformations
        scope: Rule scope (inter/intra/both)
        match_regex: Regular expression pattern to match
        op: Operation type (S=substitute, D=delete, A=add)
        out_phone: Output phoneme(s)
    """
    name: str
    country_codes: str
    examples: str
    scope: str
    match_regex: str
    op: str
    out_phone: str
    
    def __repr__(self) -> str:
        return f"AccentRule({self.name}, op={self.op})"


class AccentRulesManager:
    """Manages accent rules for validating phoneme differences.
    
    This class loads inter-word accent rules and provides functionality
    to determine if differences between transcription and reference
    can be explained by legitimate accent variation.
    """
    
    def __init__(self, rules_path: str = 'data/accent_rules_inter.csv'):
        """Initialize accent rules manager and load rules.
        
        Args:
            rules_path: Path to accent rules CSV file
        """
        self.rules_path = Path(rules_path)
        self.rules: List[AccentRule] = []
        self._load_rules()
        logger.info(f"Loaded {len(self.rules)} accent rules")
    
    def _load_rules(self) -> None:
        """Load accent rules from CSV file.
        
        Expected CSV format:
            name,country_codes,examples,scope,match_regex,op,out_phone
        """
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Accent rules file not found: {self.rules_path}")
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                rule = AccentRule(
                    name=row['name'],
                    country_codes=row['country_codes'],
                    examples=row['examples'],
                    scope=row['scope'],
                    match_regex=row['match_regex'],
                    op=row['op'],
                    out_phone=row['out_phone']
                )
                self.rules.append(rule)
    
    def _get_vowels_and_consonants(self) -> Tuple[set, set]:
        """Extract vowels and consonants from IPA_TO_CLASS mapping.
        
        Returns:
            Tuple of (vowel_chars, consonant_chars)
        """
        vowels = set()
        consonants = set()
        
        for ipa_char, class_char in IPA_TO_CLASS.items():
            # Skip multi-character sequences
            if len(ipa_char) > 1:
                continue
            
            if class_char == 'ə':  # Vowel class (includes glides)
                vowels.add(ipa_char)
            else:  # All other classes are consonants
                consonants.add(ipa_char)
        
        return vowels, consonants
    
    def _expand_regex_pattern(self, pattern: str) -> str:
        """Expand phonetic class placeholders in regex patterns.
        
        V = vowels
        C = consonants
        # = word boundary
        
        Args:
            pattern: Pattern with placeholders (e.g., "(?<= V ) h (?= V)")
            
        Returns:
            Expanded regex pattern
        """
        vowels, consonants = self._get_vowels_and_consonants()
        
        # Create regex character classes
        vowel_pattern = '[' + ''.join(sorted(vowels)) + ']'
        consonant_pattern = '[' + ''.join(sorted(consonants)) + ']'
        
        # Word boundary (space in our context since we work with space-separated phonemes)
        boundary = r'(?:^|$| )'
        
        # Replace placeholders
        expanded = pattern.replace('V', vowel_pattern)
        expanded = expanded.replace('C', consonant_pattern)
        expanded = expanded.replace('#', boundary)
        
        return expanded
    
    def can_explain_difference(
        self,
        position: int,
        transcript_phonemes: str,
        reference_phonemes: str
    ) -> Tuple[bool, Optional[AccentRule]]:
        """Check if a difference at a position can be explained by accent rules.
        
        Args:
            position: Character position where difference occurs
            transcript_phonemes: Transcribed phoneme sequence (space-separated)
            reference_phonemes: Reference phoneme sequence (space-separated)
            
        Returns:
            Tuple of (can_explain, rule) where rule is the explaining rule or None
        """
        # Extract context around the position
        context_size = 20
        start = max(0, position - context_size)
        end = min(len(reference_phonemes), position + context_size)
        context = reference_phonemes[start:end]
        
        # Try each rule
        for rule in self.rules:
            try:
                expanded_pattern = self._expand_regex_pattern(rule.match_regex)
                
                # Check if pattern matches in context
                if re.search(expanded_pattern, context):
                    # This rule could explain the difference
                    return True, rule
                    
            except Exception as e:
                logger.debug(f"Rule {rule.name} pattern failed: {e}")
                continue
        
        return False, None
    
    def apply_rule(
        self,
        phonemes: str,
        rule: AccentRule
    ) -> Tuple[str, bool]:
        """Apply an accent rule to phoneme sequence.
        
        Args:
            phonemes: Input phoneme sequence (space-separated)
            rule: Accent rule to apply
            
        Returns:
            Tuple of (modified_phonemes, was_modified)
        """
        try:
            # Expand regex pattern
            pattern = self._expand_regex_pattern(rule.match_regex)
            
            # Check if pattern matches
            if not re.search(pattern, phonemes):
                return phonemes, False
            
            # Apply operation based on type
            if rule.op == 'S':  # Substitute
                modified = re.sub(pattern, rule.out_phone, phonemes)
            elif rule.op == 'D':  # Delete
                modified = re.sub(pattern, '', phonemes)
            elif rule.op == 'A':  # Add
                modified = re.sub(pattern, lambda m: m.group(0) + ' ' + rule.out_phone, phonemes)
            else:
                return phonemes, False
            
            # Check if actually modified
            return modified, modified != phonemes
            
        except Exception as e:
            logger.debug(f"Failed to apply rule {rule.name}: {e}")
            return phonemes, False
    
    def generate_variations(
        self,
        reference_phonemes: str,
        max_rules: int = 3
    ) -> List[str]:
        """Generate phoneme variations by applying accent rules.
        
        This can be used to expand reference phonemes to include likely
        accent variations.
        
        Args:
            reference_phonemes: Base phoneme sequence
            max_rules: Maximum number of rules to apply in combination
            
        Returns:
            List of phoneme variations including original
        """
        variations = {reference_phonemes}  # Start with original
        
        # Apply each rule individually
        for rule in self.rules:
            modified, was_modified = self.apply_rule(reference_phonemes, rule)
            if was_modified:
                variations.add(modified)
        
        return list(variations)
    
    def __len__(self) -> int:
        """Return number of rules loaded."""
        return len(self.rules)