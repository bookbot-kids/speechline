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
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MismatchExporter:
    """Exports mismatched samples for further analysis.
    
    This class maintains both CSV and JSON exports of mismatches to enable
    detailed analysis and system improvement.
    """
    
    def __init__(self, output_dir: str = 'mismatches'):
        """Initialize mismatch exporter.
        
        Args:
            output_dir: Directory for mismatch exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file
        self.csv_path = self.output_dir / 'mismatches.csv'
        self.json_path = self.output_dir / 'mismatches_detailed.jsonl'
        
        self._init_csv()
        logger.info(f"Mismatch exporter initialized: {self.output_dir}")
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'sample_id',
                    'reference_text',
                    'reference_words',
                    'transcript_phonemes',
                    'transcript_classes',
                    'reference_class_variations',
                    'matched',
                    'reason',
                    'audio_path',
                    'alignment_type'
                ])
    
    def export_mismatch(
        self,
        sample_id: str,
        reference: Union[str, List[str]],
        transcript_phonemes: str,
        transcript_classes: str,
        reference_class_variations: List[str],
        reason: str = 'unknown',
        audio_path: Optional[str] = None,
        alignment_type: str = 'aligned',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export a mismatch to CSV and JSON.
        
        Args:
            sample_id: Unique identifier for the sample
            reference: Reference text (word string or list of words)
            transcript_phonemes: Transcribed phoneme sequence
            transcript_classes: Phonetic class sequence from transcript
            reference_class_variations: All possible class sequences from reference
            reason: Reason for mismatch
            audio_path: Optional path to audio file
            alignment_type: 'aligned' or 'misaligned'
            metadata: Additional metadata to include in JSON
        """
        timestamp = datetime.now().isoformat()
        
        # Convert reference to string if it's a list
        if isinstance(reference, list):
            reference_text = ' '.join(reference)
            reference_words = reference
        else:
            reference_text = reference
            reference_words = reference.split()
        
        # Export to CSV
        self._export_to_csv(
            timestamp=timestamp,
            sample_id=sample_id,
            reference_text=reference_text,
            reference_words=reference_words,
            transcript_phonemes=transcript_phonemes,
            transcript_classes=transcript_classes,
            reference_class_variations=reference_class_variations,
            reason=reason,
            audio_path=audio_path,
            alignment_type=alignment_type
        )
        
        # Export to JSON with full details
        self._export_to_json(
            timestamp=timestamp,
            sample_id=sample_id,
            reference_text=reference_text,
            reference_words=reference_words,
            transcript_phonemes=transcript_phonemes,
            transcript_classes=transcript_classes,
            reference_class_variations=reference_class_variations,
            reason=reason,
            audio_path=audio_path,
            alignment_type=alignment_type,
            metadata=metadata or {}
        )
        
        logger.debug(f"Exported mismatch: {sample_id}")
    
    def _export_to_csv(
        self,
        timestamp: str,
        sample_id: str,
        reference_text: str,
        reference_words: List[str],
        transcript_phonemes: str,
        transcript_classes: str,
        reference_class_variations: List[str],
        reason: str,
        audio_path: Optional[str],
        alignment_type: str
    ) -> None:
        """Append mismatch to CSV file."""
        with open(self.csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                sample_id,
                reference_text,
                '|'.join(reference_words),
                transcript_phonemes,
                transcript_classes,
                '|'.join(reference_class_variations[:10]),  # Limit to first 10
                'False',
                reason,
                audio_path or '',
                alignment_type
            ])
    
    def _export_to_json(
        self,
        timestamp: str,
        sample_id: str,
        reference_text: str,
        reference_words: List[str],
        transcript_phonemes: str,
        transcript_classes: str,
        reference_class_variations: List[str],
        reason: str,
        audio_path: Optional[str],
        alignment_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Append mismatch to JSONL file with full details."""
        record = {
            'timestamp': timestamp,
            'sample_id': sample_id,
            'reference': {
                'text': reference_text,
                'words': reference_words,
                'class_variations': reference_class_variations
            },
            'transcript': {
                'phonemes': transcript_phonemes,
                'classes': transcript_classes
            },
            'matched': False,
            'reason': reason,
            'audio_path': audio_path,
            'alignment_type': alignment_type,
            'metadata': metadata
        }
        
        with open(self.json_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def get_mismatch_count(self) -> int:
        """Get total number of mismatches exported.
        
        Returns:
            Number of mismatches in CSV file
        """
        if not self.csv_path.exists():
            return 0
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # Subtract header row
    
    def get_mismatches(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load mismatches from JSON file.
        
        Args:
            limit: Maximum number of mismatches to return
            
        Returns:
            List of mismatch records
        """
        if not self.json_path.exists():
            return []
        
        mismatches = []
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                mismatches.append(json.loads(line))
        
        return mismatches
    
    def clear(self) -> None:
        """Clear all mismatch exports."""
        if self.csv_path.exists():
            self.csv_path.unlink()
        if self.json_path.exists():
            self.json_path.unlink()
        self._init_csv()
        logger.info("Cleared all mismatch exports")