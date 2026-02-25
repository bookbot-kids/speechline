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

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

from datasets import Dataset
from tqdm.auto import tqdm

# Add Gentle to Python path
GENTLE_PATH_DEFAULT = "/mnt/Projects/Projects/AudioProcessing/gentle"


class GentleTranscriber:
    """
    Gentle forced aligner for validating existing transcripts.
    
    Unlike traditional transcribers, Gentle doesn't generate transcriptions.
    Instead, it performs forced alignment between audio and existing ground truth text,
    providing word-level and phoneme-level confidence metrics.
    
    Args:
        gentle_path (str, optional):
            Path to Gentle installation. Defaults to "/mnt/Projects/Projects/AudioProcessing/gentle".
        output_phonemes (bool, optional):
            Include phoneme sequences in output. Defaults to True.
        output_word_boundaries (bool, optional):
            Include word boundary timestamps in output. Defaults to True.
    """
    
    def __init__(
        self, 
        gentle_path: str = GENTLE_PATH_DEFAULT,
        output_phonemes: bool = True,
        output_word_boundaries: bool = True
    ) -> None:
        self.gentle_path = gentle_path
        self.output_phonemes = output_phonemes
        self.output_word_boundaries = output_word_boundaries
        
        # Add Gentle to path and import
        sys.path.insert(0, gentle_path)
        
        try:
            import gentle
            self.gentle = gentle
            self.resources = gentle.Resources()
            logging.info(f"Gentle resources initialized from {gentle_path}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import Gentle from {gentle_path}. "
                f"Ensure Gentle is installed at this location. Error: {e}"
            )
        
        # Sampling rate for Gentle (always 8000 Hz)
        self.sampling_rate = 8000
    
    def predict(
        self,
        dataset: Dataset,
        output_dir: str = None,
        **kwargs
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Performs forced alignment on dataset using ground truth from .txt files.
        
        Args:
            dataset (Dataset):
                Dataset containing audio files with ground_truth column.
            output_dir (str, optional):
                Output directory (not used, files written next to audio). Defaults to None.
            **kwargs:
                Additional arguments (for compatibility with other transcribers).
        
        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of offsets for each audio file, compatible with segmenters.
                Each offset contains: {"text": word, "start_time": float, "end_time": float}
        """
        results = []
        
        for idx in tqdm(range(len(dataset)), desc="Gentle Forced Alignment"):
            # Handle both dictionary and direct path access
            audio_item = dataset[idx]['audio']
            if isinstance(audio_item, dict):
                audio_path = audio_item['path']
            else:
                audio_path = audio_item  # Already a string path
            
            ground_truth = dataset[idx].get('ground_truth', '')
            
            # Skip if no ground truth
            if not ground_truth or not ground_truth.strip():
                logging.warning(f"No ground truth for {audio_path}, skipping")
                results.append([])
                continue
            
            try:
                # Perform alignment
                alignment = self._align_single(audio_path, ground_truth)
                
                # Create JSON output next to audio file
                json_path = Path(audio_path).with_suffix('.json')
                self._create_json_output(alignment, json_path)
                
                # Create Audacity label file next to audio file with .labels.txt extension
                audio_path_obj = Path(audio_path)
                label_path = audio_path_obj.with_suffix('').with_suffix('.labels.txt')
                self._create_audacity_labels(alignment, label_path)
                
                # Convert to offsets for segmentation
                offsets = self._format_to_offsets(alignment)
                results.append(offsets)
                
            except Exception as e:
                logging.error(f"Alignment failed for {audio_path}: {e}")
                results.append([])
        
        return results
    
    def _align_single(self, audio_path: str, ground_truth: str) -> Dict:
        """
        Perform Gentle alignment for a single audio file.
        
        Args:
            audio_path (str):
                Path to audio file.
            ground_truth (str):
                Ground truth transcript text.
        
        Returns:
            Dict:
                Alignment results with words, phonemes, and statistics.
        """
        try:
            # Use Gentle's resampled context manager for audio conversion
            with self.gentle.resampled(audio_path) as wavfile:
                # Create forced aligner
                aligner = self.gentle.ForcedAligner(
                    self.resources,
                    ground_truth,
                    nthreads=1,
                    disfluency=False,
                    conservative=False
                )
                
                # Perform alignment
                result = aligner.transcribe(wavfile, logging=logging)
        
        except Exception as e:
            logging.error(f"Gentle alignment failed for {audio_path}: {e}")
            return {
                'ground_truth': ground_truth,
                'alignment_success': False,
                'words': [],
                'phoneme_sequence': '',
                'word_boundaries': [],
                'statistics': {
                    'total_words': len(ground_truth.split()),
                    'aligned_words': 0,
                    'not_found_words': ground_truth.split(),
                    'alignment_ratio': 0.0
                },
                'error': str(e)
            }
        
        # Extract word-level information
        words = []
        word_boundaries = []
        not_found_words = []
        aligned_words = 0
        all_phonemes = []
        
        for word_obj in result.words:
            word_text = word_obj.word
            
            if word_obj.success():
                # Word successfully aligned
                aligned_words += 1
                
                # Extract phoneme information with calculated timestamps
                phones = []
                if word_obj.phones:
                    # Calculate phone timestamps from word start time and phone durations
                    current_time = word_obj.start if word_obj.start is not None else 0.0
                    
                    for phone_dict in word_obj.phones:
                        # phone_dict is a dictionary with 'phone' and 'duration'
                        phone = phone_dict.get('phone', '')
                        # Strip position markers (_B, _I, _E, _S)
                        phone_clean = phone.split('_')[0] if '_' in phone else phone
                        if phone_clean:
                            duration = phone_dict.get('duration', 0.0)
                            phone_start = current_time
                            phone_end = current_time + duration
                            
                            phones.append({
                                'phone': phone_clean,
                                'start': round(phone_start, 3),
                                'end': round(phone_end, 3),
                                'duration': round(duration, 3)
                            })
                            all_phonemes.append(phone_clean)
                            
                            # Move current time forward by this phone's duration
                            current_time = phone_end
                
                word_data = {
                    'word': word_text,
                    'start_time': round(word_obj.start, 3) if word_obj.start is not None else None,
                    'end_time': round(word_obj.end, 3) if word_obj.end is not None else None,
                    'aligned': True
                }
                
                # Add phones as space-separated string if enabled
                if self.output_phonemes and phones:
                    word_data['phones'] = ' '.join([p['phone'] for p in phones])
                
                # Add to word boundaries
                if self.output_word_boundaries and word_obj.start is not None:
                    word_boundaries.append({
                        'word': word_text,
                        'start': round(word_obj.start, 3),
                        'end': round(word_obj.end, 3)
                    })
            else:
                # Word not found in audio
                not_found_words.append(word_text)
                word_data = {
                    'word': word_text,
                    'aligned': False
                }
            
            words.append(word_data)
        
        # Calculate statistics
        total_words = len(result.words)
        alignment_ratio = aligned_words / total_words if total_words > 0 else 0.0
        
        return {
            'ground_truth': ground_truth,
            'words': words
        }
    
    def _create_json_output(self, alignment: Dict, output_path: Path) -> None:
        """
        Create comprehensive JSON output file with alignment details.
        
        Args:
            alignment (Dict):
                Alignment results from _align_single.
            output_path (Path):
                Path to output JSON file.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alignment, f, indent=2, ensure_ascii=False)
            logging.debug(f"Created alignment JSON: {output_path}")
        except Exception as e:
            logging.error(f"Failed to write JSON to {output_path}: {e}")
    
    def _create_audacity_labels(self, alignment: Dict, output_path: Path) -> None:
        """
        Create Audacity label file with aligned words only.
        
        Format: start_time\tend_time\tlabel
        
        Args:
            alignment (Dict):
                Alignment results from _align_single.
            output_path (Path):
                Path to output label file (.txt).
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for word_data in alignment['words']:
                    # Only include successfully aligned words with timestamps
                    if (word_data.get('aligned', False) and
                        word_data.get('start_time') is not None and
                        word_data.get('end_time') is not None):
                        start = word_data['start_time']
                        end = word_data['end_time']
                        word = word_data['word']
                        # Audacity label format: start\tend\tlabel
                        f.write(f"{start}\t{end}\t{word}\n")
            logging.debug(f"Created Audacity label file: {output_path}")
        except Exception as e:
            logging.error(f"Failed to write Audacity labels to {output_path}: {e}")
    
    def _format_to_offsets(self, alignment: Dict) -> List[Dict[str, Union[str, float]]]:
        """
        Convert Gentle alignment to offset format compatible with segmenters.
        
        Args:
            alignment (Dict):
                Alignment results from _align_single.
        
        Returns:
            List[Dict[str, Union[str, float]]]:
                List of offsets with format: {"text": word, "start_time": float, "end_time": float}
        """
        offsets = []
        
        for word_data in alignment['words']:
            # Only include successfully aligned words
            if word_data.get('aligned', False) and word_data.get('start_time') is not None:
                offsets.append({
                    'text': word_data['word'],
                    'start_time': word_data['start_time'],
                    'end_time': word_data['end_time']
                })
        
        return offsets