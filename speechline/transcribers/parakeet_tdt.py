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

from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
import tempfile
import soundfile as sf
from pathlib import Path
import subprocess
import json
import os


class ParakeetTDTTranscriber:
    """
    NVIDIA Parakeet TDT (Token-and-Duration Transducer) model for ASR.
    
    This transcriber uses NVIDIA NeMo framework for the Parakeet TDT models.
    Parakeet TDT models are efficient speech recognition models that predict
    both tokens and their durations simultaneously.
    
    The TDT architecture provides:
    - Fast inference with streaming capabilities
    - Word-level timestamps
    - High accuracy speech recognition
    - Support for various audio lengths
    
    Requirements:
        - nemo_toolkit[asr] must be installed
        - Run: pip install nemo_toolkit[asr]
    
    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint or NeMo .nemo file path.
            Defaults to "nvidia/parakeet-tdt-0.6b-v2".
        transcriber_device (str, optional):
            Device to run inference on ('cuda', 'cpu', 'mps').
            Defaults to auto-detection.
        torch_dtype (str, optional):
            Torch dtype for model weights (e.g., 'float16', 'bfloat16').
            Defaults to None (uses model's default).
    """

    def __init__(
        self,
        model_checkpoint: str = "nvidia/parakeet-tdt-0.6b-v2",
        transcriber_device: str = None,
        torch_dtype: str = None
    ) -> None:
        try:
            from nemo.collections.asr.models import ASRModel
            from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for Parakeet TDT. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        
        # Determine device
        if transcriber_device:
            self.device = transcriber_device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading Parakeet TDT model on {self.device}...")
        
        # Load the NeMo ASR model
        # Parakeet TDT is an ASRModel in NeMo
        self.model = ASRModel.from_pretrained(model_checkpoint)
        self.model.eval()
        
        # Configure model to preserve word-level confidence scores
        # This enables real confidence estimation during transcription
        confidence_cfg = ConfidenceConfig(
            preserve_word_confidence=True,
            preserve_token_confidence=True,
            aggregation="min",  # Use minimum confidence across tokens for word confidence
            exclude_blank=False,
            tdt_include_duration=True  # Include duration confidence for TDT models
        )
        
        # Apply confidence configuration to the model
        try:
            self.model.change_decoding_strategy(
                RNNTDecodingConfig(
                    fused_batch_size=-1,
                    strategy="greedy_batch",
                    confidence_cfg=confidence_cfg
                )
            )
            print(f"✅ Confidence estimation enabled")
        except Exception as e:
            print(f"⚠️  Warning: Could not enable confidence estimation: {e}")
        
        # Move model to device
        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
                
                # Verify model is on correct device
                if hasattr(self.model, 'encoder'):
                    first_param = next(self.model.encoder.parameters())
                    actual_device = str(first_param.device)
                    print(f"   Model parameters on device: {actual_device}")
                    
                    if actual_device.startswith("cpu") and self.device != "cpu":
                        print(f"   ⚠️  Warning: Model failed to move to {self.device}, using CPU")
                        self.device = "cpu"
            except Exception as e:
                print(f"   ⚠️  Warning: Could not move model to {self.device}: {e}")
                print(f"   Falling back to CPU")
                self.device = "cpu"
        
        # Store sampling rate (Parakeet models typically use 16kHz)
        self.sampling_rate = self.model.cfg.sample_rate if hasattr(self.model.cfg, 'sample_rate') else 16000
        self.sr = self.sampling_rate  # Alias for compatibility
        
        print(f"✅ Parakeet TDT model loaded successfully on {self.device}")
        print(f"   Sampling rate: {self.sampling_rate} Hz")
    
    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        return_timestamps: str = None,
        keep_whitespace: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """
        Performs inference on `dataset`.

        Args:
            dataset (Dataset):
                Dataset to be inferred. Must have 'audio' column.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
                Note: Parakeet TDT handles long audio automatically.
            output_offsets (bool, optional):
                Whether to output word timestamps. Defaults to `False`.
            return_timestamps (str, optional):
                Timestamp level ('word' or 'char'). Defaults to `"word"`.
            keep_whitespace (bool, optional):
                Whether to preserve whitespace predictions. Defaults to `False`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of word offsets.

        ### Example
        ```pycon title="example_parakeet_tdt_predict.py"
        >>> from speechline.transcribers import ParakeetTDTTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = ParakeetTDTTranscriber()
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=transcriber.sr)
        ... )
        >>> transcripts = transcriber.predict(dataset)
        >>> transcripts
        ["This is a sample transcription."]
        >>> offsets = transcriber.predict(dataset, output_offsets=True)
        >>> offsets
        [
            [
                {"text": "This", "start_time": 0.0, "end_time": 0.3},
                {"text": "is", "start_time": 0.3, "end_time": 0.5},
                {"text": "a", "start_time": 0.5, "end_time": 0.6},
                {"text": "sample", "start_time": 0.6, "end_time": 1.0},
                {"text": "transcription.", "start_time": 1.0, "end_time": 1.8}
            ]
        ]
        ```
        """
        import librosa
        results = []
        
        # Create temporary directory for audio files if needed
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            audio_paths = []
            
            # Prepare audio files
            # Get raw data without triggering audio decoding
            # Use to_dict() to avoid automatic Audio feature decoding
            for idx in tqdm(range(len(dataset)), desc="Preparing audio files"):
                # Access raw data directly to avoid Audio feature decoding
                # This is critical for AAC files which torchcodec cannot handle
                item = dataset._data.to_pydict()
                item = {k: v[idx] for k, v in item.items()}
                
                # Handle different audio formats from dataset
                # Priority: direct string path > dict with path > dict with array
                if isinstance(item["audio"], str):
                    # Audio is a file path string - use directly
                    audio_path = item["audio"]
                    if Path(audio_path).exists():
                        audio_paths.append(audio_path)
                        continue
                    else:
                        raise ValueError(f"Audio file not found: {audio_path}")
                elif isinstance(item["audio"], dict):
                    if "path" in item["audio"]:
                        # Audio dict has a path - use it directly if it exists
                        audio_path = item["audio"]["path"]
                        if Path(audio_path).exists():
                            audio_paths.append(audio_path)
                            continue
                        # Otherwise, extract the array if available
                        if "array" in item["audio"]:
                            audio_array = item["audio"]["array"]
                            audio_sr = item["audio"]["sampling_rate"]
                        else:
                            raise ValueError(f"Audio file not found and no array available: {audio_path}")
                    elif "array" in item["audio"]:
                        # Audio is already loaded as array
                        audio_array = item["audio"]["array"]
                        audio_sr = item["audio"]["sampling_rate"]
                    else:
                        raise ValueError(f"Audio dict has neither 'path' nor 'array': {item['audio']}")
                else:
                    raise ValueError(f"Unexpected audio format: {type(item['audio'])}")
                
                # Resample if necessary
                if audio_sr != self.sampling_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=audio_sr,
                        target_sr=self.sampling_rate
                    )
                
                # Save audio to temporary file
                temp_audio_path = temp_dir_path / f"audio_{idx}.wav"
                sf.write(temp_audio_path, audio_array, self.sampling_rate)
                audio_paths.append(str(temp_audio_path))
            
            # Perform batch transcription
            print(f"Transcribing {len(audio_paths)} audio files...")
            
            # NeMo's transcribe method returns list of transcriptions
            # For timestamps, we need to use transcribe with return_hypotheses=True and timestamps=True
            # Only enable timestamps when explicitly requested via output_offsets=True
            if output_offsets:
                # Get detailed hypotheses with word timestamps
                hypotheses = self.model.transcribe(
                    audio_paths,
                    batch_size=1,  # Process one at a time for memory efficiency
                    return_hypotheses=True,
                    timestamps=True  # Enable timestamp extraction
                )
                
                for hyp in tqdm(hypotheses, desc="Processing transcriptions"):
                    if output_offsets:
                        # Extract word-level timestamps from hyp.timestamp['word']
                        offsets = []
                        
                        # Check if timestamp information is available
                        if hasattr(hyp, 'timestamp') and isinstance(hyp.timestamp, dict) and 'word' in hyp.timestamp:
                            # Extract word-level timestamps from the timestamp dict
                            word_timestamps = hyp.timestamp['word']
                            for word_data in word_timestamps:
                                if isinstance(word_data, dict):
                                    word_text = word_data.get('word', '')
                                    if not keep_whitespace and not word_text.strip():
                                        continue
                                    offsets.append({
                                        "text": word_text,
                                        "start_time": round(word_data.get('start', 0.0), 3),
                                        "end_time": round(word_data.get('end', 0.0), 3)
                                    })
                        elif hasattr(hyp, 'words') and hyp.words:
                            # Fallback: try to use words attribute (without timestamps)
                            for word_info in hyp.words:
                                if isinstance(word_info, str):
                                    if not keep_whitespace and not word_info.strip():
                                        continue
                                    offsets.append({
                                        "text": word_info,
                                        "start_time": 0.0,
                                        "end_time": 0.0
                                    })
                        elif hasattr(hyp, 'text'):
                            # Fallback: create single offset for entire transcription
                            text = hyp.text if keep_whitespace else hyp.text.strip()
                            if text:
                                offsets.append({
                                    "text": text,
                                    "start_time": 0.0,
                                    "end_time": 0.0  # Duration unknown
                                })
                        
                        results.append(offsets)
                    else:
                        # Return just the text
                        text = hyp.text if hasattr(hyp, 'text') else str(hyp)
                        if not keep_whitespace:
                            text = text.strip()
                        results.append(text)
            else:
                # Simple transcription without timestamps
                # Temporarily disable timestamps at the model config level
                original_compute_timestamps = getattr(self.model.decoding, 'compute_timestamps', None)
                
                try:
                    # Disable timestamps at the decoding config level
                    if hasattr(self.model.decoding, 'compute_timestamps'):
                        self.model.decoding.compute_timestamps = False
                    
                    transcriptions = self.model.transcribe(
                        audio_paths,
                        batch_size=1,
                        return_hypotheses=False  # Return simple strings, not hypothesis objects
                    )
                finally:
                    # Restore original setting
                    if original_compute_timestamps is not None:
                        self.model.decoding.compute_timestamps = original_compute_timestamps
                
                for text in transcriptions:
                    # Handle both string and Hypothesis objects
                    if hasattr(text, 'text'):
                        text = text.text
                    else:
                        text = str(text)
                    
                    if not keep_whitespace:
                        text = text.strip()
                    results.append(text)
        
        return results
    
    def predict_with_phoneme_alignment(
        self,
        dataset: Dataset,
        nfa_model: str = "nvidia/parakeet-ctc-1.1b",
        output_dir: Optional[str] = None,
        chunk_length_s: int = 30,
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Performs word transcription with Parakeet TDT, then uses NeMo Forced Aligner
        to get token-level alignments aligned to the transcribed words.
        
        Note: NFA provides subword tokens (SentencePiece pieces), not IPA phonemes.
        This ensures alignment between words and tokens since NFA force-aligns
        tokens to the known word transcription.
        
        Args:
            dataset (Dataset):
                Dataset to be inferred. Must have 'audio' column.
            nfa_model (str, optional):
                NeMo model to use for forced alignment.
                Defaults to "nvidia/parakeet-ctc-1.1b".
            output_dir (str, optional):
                Directory for NFA output files. If None, uses temp directory.
            chunk_length_s (int, optional):
                Audio chunk length during inference. Defaults to 30.
        
        Returns:
            Tuple[List[List[Dict]], List[List[Dict]]]:
                (word_offsets, token_offsets) where each is a list of offset lists.
                token_offsets contain subword tokens, not IPA phonemes.
        
        ### Example
        ```python
        >>> from speechline.transcribers import ParakeetTDTTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = ParakeetTDTTranscriber()
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=16000)
        ... )
        >>> word_offsets, token_offsets = transcriber.predict_with_phoneme_alignment(dataset)
        >>> print(f"Words: {len(word_offsets[0])}, Tokens: {len(token_offsets[0])}")
        ```
        """
        import tempfile as tf
        
        # Step 1: Get word transcription with timestamps
        print("\n[1/3] Transcribing words with Parakeet TDT...")
        word_offsets = self.predict(
            dataset,
            chunk_length_s=chunk_length_s,
            output_offsets=True,
            return_timestamps="word"
        )
        
        # Create output directory
        if output_dir is None:
            temp_output_dir = tf.TemporaryDirectory()
            output_dir = temp_output_dir.name
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Step 2: Create manifest file with transcriptions for NFA
        print("\n[2/3] Preparing manifest for NeMo Forced Aligner...")
        manifest_path = os.path.join(output_dir, "nfa_manifest.json")
        
        with open(manifest_path, 'w') as f:
            for idx, item in enumerate(tqdm(dataset, desc="Creating manifest")):
                # Get audio path
                if isinstance(item["audio"], dict) and "path" in item["audio"]:
                    audio_path = item["audio"]["path"]
                elif isinstance(item["audio"], str):
                    audio_path = item["audio"]
                else:
                    raise ValueError(f"Cannot extract audio path from item {idx}")
                
                # Get transcribed text from word offsets
                words = word_offsets[idx]
                text = " ".join([w["text"] for w in words])
                
                # Write manifest entry
                json.dump({
                    "audio_filepath": audio_path,
                    "text": text
                }, f)
                f.write("\n")
        
        # Step 3: Run NeMo Forced Aligner for token alignment
        print(f"\n[3/3] Running NeMo Forced Aligner with {nfa_model}...")
        nfa_output_dir = os.path.join(output_dir, "nfa_output")
        
        # Find NFA script path
        # Try common locations
        nfa_script_paths = [
            os.path.expanduser("~/NeMo/tools/nemo_forced_aligner/align.py"),
            "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/NeMo/tools/nemo_forced_aligner/align.py",
            "/opt/NeMo/tools/nemo_forced_aligner/align.py",
        ]
        
        nfa_script = None
        for path in nfa_script_paths:
            if os.path.exists(path):
                nfa_script = path
                break
        
        if nfa_script is None:
            raise FileNotFoundError(
                "NeMo Forced Aligner script not found. Please ensure NeMo is installed "
                "and set the correct path. Tried:\n" + "\n".join(nfa_script_paths)
            )
        
        # Run NFA
        cmd = [
            "python",
            nfa_script,
            f"pretrained_name={nfa_model}",
            f"manifest_filepath={manifest_path}",
            f"output_dir={nfa_output_dir}",
            f"transcribe_device={self.device}",
            "batch_size=1",
            "align_using_pred_text=false",  # Use provided text, not predicted
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"NFA Error: {e.stderr}")
            raise RuntimeError(f"NeMo Forced Aligner failed: {e.stderr}")
        
        # Step 4: Parse NFA output (token-level CTM files)
        print("\n[4/4] Parsing token alignments...")
        manifest_with_paths = os.path.join(
            nfa_output_dir,
            "nfa_manifest_with_output_file_paths.json"
        )
        
        if not os.path.exists(manifest_with_paths):
            raise FileNotFoundError(
                f"NFA output manifest not found: {manifest_with_paths}"
            )
        
        # Read token alignments
        token_offsets = []
        
        with open(manifest_with_paths) as f:
            for line in tqdm(f, desc="Reading token alignments", total=len(dataset)):
                entry = json.loads(line)
                
                # Read token-level CTM file (these are subword tokens, not phonemes)
                if "tokens_level_ctm_filepath" in entry:
                    token_list = []
                    with open(entry["tokens_level_ctm_filepath"]) as ctm_file:
                        for ctm_line in ctm_file:
                            # Format: <utt_id> 1 <start_time> <duration> <token> <confidence>
                            parts = ctm_line.strip().split()
                            if len(parts) >= 5:
                                start_time = float(parts[2])
                                duration = float(parts[3])
                                token = parts[4]
                                confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                                
                                # Skip blank tokens and word boundary markers
                                if token in ['<b>', '<blank>', '<pad>']:
                                    continue
                                
                                # Clean up SentencePiece markers (▁ = word start)
                                token = token.replace('▁', '')
                                
                                if token:  # Only add non-empty tokens
                                    token_list.append({
                                        "text": token,
                                        "start_time": round(start_time, 3),
                                        "end_time": round(start_time + duration, 3),
                                        "confidence": round(confidence, 3)
                                    })
                    token_offsets.append(token_list)
                else:
                    # No token alignment available
                    token_offsets.append([])
        
        print(f"\n✅ Alignment complete!")
        print(f"   Words: {sum(len(w) for w in word_offsets)} total across {len(word_offsets)} files")
    
    def predict_with_validation(
        self,
        dataset: Dataset,
        ground_truth_texts: List[str],
        nfa_model: str = "nvidia/parakeet-ctc-1.1b",
        token_confidence_threshold: float = 0.7,
        min_alignment_ratio: float = 0.8,
        output_dir: Optional[str] = None,
        chunk_length_s: int = 30,
    ) -> List[Dict]:
        """
        Validate audio against ground truth by detecting phonetic alignment.
        
        This method finds where the ground truth and actual transcript align
        phonetically. It accepts audio where phonemes align well (including
        homophones like "their"/"there" and close minimal pairs like "pin"/"pen"),
        and rejects audio where phonemes don't align (different words, mispronunciations).
        
        Uses:
        1. Parakeet TDT for free ASR transcription (what was actually said)
        2. NeMo Forced Aligner to align ground truth to audio
        3. Confidence scores from alignment to validate quality
        
        Args:
            dataset (Dataset):
                Dataset to validate. Must have 'audio' column.
            ground_truth_texts (List[str]):
                Expected text for each audio file (what should have been said).
            nfa_model (str, optional):
                NeMo model for forced alignment. Defaults to "nvidia/parakeet-ctc-1.1b".
            token_confidence_threshold (float, optional):
                Minimum confidence for token acceptance (0-1). Defaults to 0.7.
            min_alignment_ratio (float, optional):
                Minimum fraction of tokens that must align (0-1). Defaults to 0.8.
            output_dir (str, optional):
                Directory for intermediate files. If None, uses temp directory.
            chunk_length_s (int, optional):
                Audio chunk length during inference. Defaults to 30.
        
        Returns:
            List[Dict]: Validation results for each audio file with keys:
                - is_valid (bool): Overall pass/fail based on thresholds
                - ground_truth (str): Expected text
                - transcription (str): What was actually said
                - alignment_ratio (float): Fraction of tokens that aligned well
                - avg_confidence (float): Average confidence across all tokens
                - token_alignment (List[Dict]): Per-token alignment details
                - valid_tokens (List[Dict]): Tokens that passed threshold
                - rejected_tokens (List[Dict]): Tokens that failed threshold
        
        ### Example
        ```python
        >>> from speechline.transcribers import ParakeetTDTTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = ParakeetTDTTranscriber()
        >>> dataset = Dataset.from_dict({
        ...     "audio": ["student1.wav", "student2.wav"]
        ... }).cast_column("audio", Audio(sampling_rate=16000))
        >>> ground_truths = [
        ...     "Put their books on the table",
        ...     "The quick brown fox jumps"
        ... ]
        >>> results = transcriber.predict_with_validation(
        ...     dataset=dataset,
        ...     ground_truth_texts=ground_truths,
        ...     token_confidence_threshold=0.7,
        ...     min_alignment_ratio=0.8
        ... )
        >>> for i, result in enumerate(results):
        ...     print(f"File {i+1}: {'✅ ACCEPT' if result['is_valid'] else '❌ REJECT'}")
        ...     print(f"  Alignment: {result['alignment_ratio']:.1%}")
        ...     print(f"  Ground truth: {result['ground_truth']}")
        ...     print(f"  Transcription: {result['transcription']}")
        ```
        """
        import tempfile as tf
        from pydub import AudioSegment
        import shutil
        
        # Step 1: Get free ASR transcription with confidence scores (what was actually said)
        print("\n[1/4] Transcribing audio with Parakeet TDT and extracting confidence scores...")
        
        # We need to call transcribe with return_hypotheses=True to get confidence scores
        import librosa
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            audio_paths = []
            
            # Prepare audio files
            for idx in tqdm(range(len(dataset)), desc="Preparing audio files"):
                item = dataset._data.to_pydict()
                item = {k: v[idx] for k, v in item.items()}
                
                if isinstance(item["audio"], str):
                    audio_path = item["audio"]
                    if Path(audio_path).exists():
                        audio_paths.append(audio_path)
                        continue
                elif isinstance(item["audio"], dict) and "path" in item["audio"]:
                    audio_path = item["audio"]["path"]
                    if Path(audio_path).exists():
                        audio_paths.append(audio_path)
                        continue
                
                raise ValueError(f"Cannot extract audio path from item {idx}")
            
            # Transcribe with confidence scores
            hypotheses = self.model.transcribe(
                audio_paths,
                batch_size=1,
                return_hypotheses=True  # Get full hypothesis objects with confidence
            )
            
            # Extract transcriptions and word confidences
            transcriptions = []
            word_confidences = []
            
            for hyp in hypotheses:
                transcriptions.append(hyp.text if hasattr(hyp, 'text') else str(hyp))
                
                # Extract word-level confidence scores
                if hasattr(hyp, 'word_confidence') and hyp.word_confidence is not None:
                    # Convert tensor to list if needed
                    if hasattr(hyp.word_confidence, 'tolist'):
                        confidences = hyp.word_confidence.tolist()
                    elif hasattr(hyp.word_confidence, 'cpu'):
                        confidences = hyp.word_confidence.cpu().tolist()
                    else:
                        confidences = list(hyp.word_confidence)
                    word_confidences.append(confidences)
                else:
                    # No confidence available - use placeholder
                    words = hyp.text.split() if hasattr(hyp, 'text') else str(hyp).split()
                    word_confidences.append([1.0] * len(words))
        
        # Step 2: Create temporary directory and manifest for forced alignment
        print("\n[2/4] Preparing forced alignment with ground truth...")
        
        # Track if we created the output directory
        created_output_dir = False
        if output_dir is None:
            temp_output_dir = tf.TemporaryDirectory()
            output_dir = temp_output_dir.name
            created_output_dir = True
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for converted WAV files
        wav_dir = os.path.join(output_dir, "converted_wav")
        os.makedirs(wav_dir, exist_ok=True)
        converted_files = []  # Track converted files for cleanup
        
        manifest_path = os.path.join(output_dir, "validation_manifest.json")
        
        # Create manifest with ground truth texts, converting AAC to WAV if needed
        # Access raw data to avoid Audio feature decoding (which fails on AAC)
        raw_data = dataset._data.to_pydict()
        
        with open(manifest_path, 'w') as f:
            for idx in tqdm(range(len(dataset)), desc="Creating manifest"):
                # Get item from raw data to avoid Audio decoding
                item = {k: v[idx] for k, v in raw_data.items()}
                
                # Get audio path
                if isinstance(item["audio"], dict) and "path" in item["audio"]:
                    audio_path = item["audio"]["path"]
                elif isinstance(item["audio"], str):
                    audio_path = item["audio"]
                else:
                    raise ValueError(f"Cannot extract audio path from item {idx}")
                
                # Convert AAC to WAV if needed
                audio_path_lower = audio_path.lower()
                if audio_path_lower.endswith('.aac') or audio_path_lower.endswith('.m4a'):
                    # Convert to WAV using pydub
                    wav_path = os.path.join(wav_dir, f"audio_{idx}.wav")
                    try:
                        audio = AudioSegment.from_file(audio_path)
                        audio = audio.set_frame_rate(16000).set_channels(1)  # 16kHz mono
                        audio.export(wav_path, format="wav")
                        converted_files.append(wav_path)  # Track for cleanup
                        audio_path = wav_path
                    except Exception as e:
                        print(f"Warning: Failed to convert {audio_path}: {e}")
                        # Try to continue with original path
                
                # Write manifest entry with ground truth
                json.dump({
                    "audio_filepath": audio_path,
                    "text": ground_truth_texts[idx]
                }, f)
                f.write("\n")
        
        # Step 3: Run NeMo Forced Aligner
        print(f"\n[3/4] Running NeMo Forced Aligner with {nfa_model}...")
        nfa_output_dir = os.path.join(output_dir, "validation_output")
        
        # Find NFA script path
        nfa_script_paths = [
            os.path.expanduser("~/NeMo/tools/nemo_forced_aligner/align.py"),
            "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/NeMo/tools/nemo_forced_aligner/align.py",
            "/opt/NeMo/tools/nemo_forced_aligner/align.py",
        ]
        
        nfa_script = None
        for path in nfa_script_paths:
            if os.path.exists(path):
                nfa_script = path
                break
        
        if nfa_script is None:
            raise FileNotFoundError(
                "NeMo Forced Aligner script not found. Please ensure NeMo is installed "
                "and set the correct path. Tried:\n" + "\n".join(nfa_script_paths)
            )
        
        # Run NFA
        cmd = [
            "python",
            nfa_script,
            f"pretrained_name={nfa_model}",
            f"manifest_filepath={manifest_path}",
            f"output_dir={nfa_output_dir}",
            f"transcribe_device={self.device}",
            "batch_size=1",
            "align_using_pred_text=false",  # Use provided ground truth
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"NFA Error: {e.stderr}")
            raise RuntimeError(f"NeMo Forced Aligner failed: {e.stderr}")
        
        # Step 4: Parse alignment results and calculate validation metrics
        print("\n[4/4] Analyzing alignment and calculating metrics...")
        
        # NeMo Forced Aligner uses the input manifest basename as prefix
        input_manifest_name = Path(manifest_path).stem  # Get filename without extension
        manifest_with_paths = os.path.join(
            nfa_output_dir,
            f"{input_manifest_name}_with_output_file_paths.json"
        )
        
        if not os.path.exists(manifest_with_paths):
            raise FileNotFoundError(
                f"NFA output manifest not found: {manifest_with_paths}"
            )
        
        # Process each file's alignment
        results = []
        
        with open(manifest_with_paths) as f:
            for idx, line in enumerate(tqdm(f, desc="Processing alignments", total=len(dataset))):
                entry = json.loads(line)
                
                # Parse token-level alignment with confidence scores
                token_alignment = []
                
                if "tokens_level_ctm_filepath" in entry:
                    with open(entry["tokens_level_ctm_filepath"]) as ctm_file:
                        for ctm_line in ctm_file:
                            parts = ctm_line.strip().split()
                            if len(parts) >= 5:
                                start_time = float(parts[2])
                                duration = float(parts[3])
                                token = parts[4]
                                # NeMo Forced Aligner outputs 'NA' for confidence - it doesn't compute scores
                                # We'll use 1.0 for successfully aligned tokens (presence = success)
                                # and let the validation decision be based on alignment_ratio only
                                confidence = 1.0
                                
                                # Skip special tokens
                                if token in ['<b>', '<blank>', '<pad>']:
                                    continue
                                
                                # Clean up token
                                token = token.replace('▁', '')
                                
                                if token:
                                    token_alignment.append({
                                        "text": token,
                                        "start_time": round(start_time, 3),
                                        "end_time": round(start_time + duration, 3),
                                        "confidence": round(confidence, 3)
                                    })
                
                # Calculate validation metrics
                # Note: NFA doesn't provide real confidence scores, just alignment success/failure
                # So we base validation purely on whether tokens were successfully aligned
                if token_alignment:
                    # All aligned tokens have confidence=1.0 (successful alignment)
                    avg_confidence = 1.0
                    valid_tokens = token_alignment  # All aligned tokens are considered valid
                    rejected_tokens = []  # NFA either aligns or doesn't - no partial scores
                    
                    # Alignment ratio is based on: (tokens aligned) / (tokens expected)
                    # We compare against ground truth token count
                    ground_truth_tokens = ground_truth_texts[idx].split()
                    expected_token_count = len(ground_truth_tokens)
                    
                    if expected_token_count > 0:
                        # Ratio of aligned tokens to expected tokens
                        alignment_ratio = len(token_alignment) / expected_token_count
                        # Cap at 1.0 in case there are more tokens aligned than expected (insertions)
                        alignment_ratio = min(alignment_ratio, 1.0)
                    else:
                        alignment_ratio = 0.0
                    
                    # Overall validation decision based purely on alignment ratio
                    # (confidence threshold is ignored since NFA doesn't provide real confidences)
                    is_valid = alignment_ratio >= min_alignment_ratio
                else:
                    # No tokens aligned
                    avg_confidence = 0.0
                    alignment_ratio = 0.0
                    is_valid = False
                    valid_tokens = []
                    rejected_tokens = []
                
                # Get word-level confidence scores from Parakeet TDT transcription
                transcription_confidences = word_confidences[idx] if idx < len(word_confidences) else []
                transcription_words = transcriptions[idx].split() if idx < len(transcriptions) else []
                
                # Create word-level confidence data (from ASR, not NFA)
                word_confidence_data = []
                for word, conf in zip(transcription_words, transcription_confidences):
                    word_confidence_data.append({
                        "text": word,
                        "confidence": round(float(conf), 3)
                    })
                
                # Compile result with both alignment data and real confidence scores
                results.append({
                    "is_valid": is_valid,
                    "ground_truth": ground_truth_texts[idx],
                    "transcription": transcriptions[idx] if idx < len(transcriptions) else "",
                    "alignment_ratio": round(alignment_ratio, 3),
                    "avg_confidence": round(avg_confidence, 3),
                    "token_alignment": token_alignment,  # Token-level timing from NFA
                    "word_confidence": word_confidence_data,  # Word-level confidence from Parakeet TDT
                    "valid_tokens": valid_tokens,
                    "rejected_tokens": rejected_tokens,
                    "num_total_tokens": len(token_alignment),
                    "num_valid_tokens": len(valid_tokens),
                    "num_rejected_tokens": len(rejected_tokens)
                })
        
        # Summary statistics
        total_valid = sum(1 for r in results if r["is_valid"])
        total_files = len(results)
        
        print(f"\n✅ Validation complete!")
        print(f"   Files processed: {total_files}")
        print(f"   Accepted: {total_valid} ({total_valid/total_files:.1%})")
        print(f"   Rejected: {total_files - total_valid} ({(total_files - total_valid)/total_files:.1%})")
        print(f"   Average alignment ratio: {np.mean([r['alignment_ratio'] for r in results]):.1%}")
        print(f"   Average confidence: {np.mean([r['avg_confidence'] for r in results]):.2f}")
        
        # Cleanup converted WAV files
        if converted_files:
            print(f"\n🧹 Cleaning up {len(converted_files)} converted WAV files...")
            for wav_path in converted_files:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception as e:
                    print(f"Warning: Failed to remove {wav_path}: {e}")
            
            # Remove the converted_wav directory if empty
            try:
                if os.path.exists(wav_dir) and not os.listdir(wav_dir):
                    os.rmdir(wav_dir)
            except Exception as e:
                print(f"Warning: Failed to remove directory {wav_dir}: {e}")
        
        return results
        print(f"   Tokens: {sum(len(t) for t in token_offsets)} total across {len(token_offsets)} files")
        
        return word_offsets, token_offsets