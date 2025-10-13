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
            for idx in tqdm(range(len(dataset)), desc="Preparing audio files"):
                item = dataset[idx]
                
                # Check if audio is already loaded or is a file path
                if isinstance(item["audio"], dict) and "path" in item["audio"]:
                    # Audio has a path - use it directly if it exists
                    if Path(item["audio"]["path"]).exists():
                        audio_paths.append(item["audio"]["path"])
                        continue
                    # Otherwise, extract the array
                    audio_array = item["audio"]["array"]
                    audio_sr = item["audio"]["sampling_rate"]
                elif isinstance(item["audio"], dict) and "array" in item["audio"]:
                    # Audio is already loaded
                    audio_array = item["audio"]["array"]
                    audio_sr = item["audio"]["sampling_rate"]
                elif isinstance(item["audio"], str):
                    # Audio is a file path - use directly
                    audio_paths.append(item["audio"])
                    continue
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
                transcriptions = self.model.transcribe(
                    audio_paths,
                    batch_size=1
                )
                
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
                            # Format: <utt_id> 1 <start_time> <duration> <token> ...
                            parts = ctm_line.strip().split()
                            if len(parts) >= 5:
                                start_time = float(parts[2])
                                duration = float(parts[3])
                                token = parts[4]
                                
                                # Skip blank tokens and word boundary markers
                                if token in ['<b>', '<blank>', '<pad>']:
                                    continue
                                
                                # Clean up SentencePiece markers (▁ = word start)
                                token = token.replace('▁', '')
                                
                                if token:  # Only add non-empty tokens
                                    token_list.append({
                                        "text": token,
                                        "start_time": round(start_time, 3),
                                        "end_time": round(start_time + duration, 3)
                                    })
                    token_offsets.append(token_list)
                else:
                    # No token alignment available
                    token_offsets.append([])
        
        print(f"\n✅ Alignment complete!")
        print(f"   Words: {sum(len(w) for w in word_offsets)} total across {len(word_offsets)} files")
        print(f"   Tokens: {sum(len(t) for t in token_offsets)} total across {len(token_offsets)} files")
        
        return word_offsets, token_offsets