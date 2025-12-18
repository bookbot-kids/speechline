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

from typing import Dict, List, Union, Optional
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm


class GraniteTranscriber:
    """
    IBM Granite Speech model for multilingual ASR with word timestamps.
    
    Granite Speech is IBM's open-source speech recognition model supporting:
    - Multiple languages
    - Word-level timestamps
    - Optimized for CUDA and CPU
    
    Requirements:
        - transformers library
        - Run: pip install transformers
    
    Args:
        model_checkpoint (str):
            HuggingFace model checkpoint.
            Defaults to "ibm-granite/granite-speech-3.3-8b".
        transcriber_device (str, optional):
            Device to run inference on ('cuda', 'cpu').
            Defaults to auto-detection.
        torch_dtype (str, optional):
            Torch dtype for model weights.
            Defaults to None (uses model's default).
    """

    def __init__(
        self,
        model_checkpoint: str = "ibm-granite/granite-speech-3.3-8b",
        transcriber_device: str = None,
        torch_dtype: str = None
    ) -> None:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise ImportError(
                "Transformers library is required for Granite Speech. "
                "Install with: pip install transformers"
            )
        
        # Determine device
        if transcriber_device:
            self.device = transcriber_device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Determine dtype
        if torch_dtype:
            self.torch_dtype = getattr(torch, torch_dtype)
        elif self.device == "cuda":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        print(f"Loading Granite Speech model on {self.device}...")
        print(f"   Using dtype: {self.torch_dtype}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_checkpoint,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        # Move model to device
        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
                print(f"   Model moved to {self.device}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not move model to {self.device}: {e}")
                print(f"   Falling back to CPU")
                self.device = "cpu"
        
        # Store sampling rate
        # GraniteSpeechProcessor uses feature_extractor_class instead of feature_extractor
        if hasattr(self.processor, 'feature_extractor'):
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            # Default to 16kHz for Granite Speech models
            self.sampling_rate = 16000
        self.sr = self.sampling_rate  # Alias for compatibility
        
        print(f"✅ Granite Speech model loaded successfully on {self.device}")
        print(f"   Sampling rate: {self.sampling_rate} Hz")
    
    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        return_timestamps: str = "word",
        keep_whitespace: bool = False,
        batch_size: int = 1,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """
        Performs inference on `dataset`.

        Args:
            dataset (Dataset):
                Dataset to be inferred. Must have 'audio' column.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
            output_offsets (bool, optional):
                Whether to output word timestamps. Defaults to `False`.
            return_timestamps (str, optional):
                Timestamp level ('word' or 'char'). Defaults to `"word"`.
            keep_whitespace (bool, optional):
                Whether to preserve whitespace predictions. Defaults to `False`.
            batch_size (int, optional):
                Batch size for inference. Defaults to `1`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of word offsets.
        """
        import librosa
        results = []
        
        for idx in tqdm(range(len(dataset)), desc="Transcribing with Granite"):
            item = dataset[idx]
            
            # Extract audio
            if isinstance(item["audio"], dict) and "array" in item["audio"]:
                audio_array = item["audio"]["array"]
                audio_sr = item["audio"]["sampling_rate"]
            elif isinstance(item["audio"], str):
                audio_array, audio_sr = librosa.load(
                    item["audio"],
                    sr=self.sampling_rate,
                    mono=True
                )
            else:
                raise ValueError(f"Unexpected audio format: {type(item['audio'])}")
            
            # Resample if necessary
            if audio_sr != self.sampling_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=audio_sr,
                    target_sr=self.sampling_rate
                )
            
            # Process audio
            # GraniteSpeechProcessor requires both audio and text parameters
            # For ASR (audio-to-text), pass empty string for text
            inputs = self.processor(
                audio=audio_array,
                text="",
                return_tensors="pt"
            )
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                if output_offsets and return_timestamps == "word":
                    # Generate transcription
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=448,
                    )
                    
                    # Decode with word offsets
                    transcription = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True,
                        output_word_offsets=True
                    )[0]
                    
                    # Extract word-level offsets
                    if isinstance(transcription, dict) and "chunks" in transcription:
                        offsets = []
                        for chunk in transcription["chunks"]:
                            if not keep_whitespace and not chunk["text"].strip():
                                continue
                            offsets.append({
                                "text": chunk["text"],
                                "start_time": round(chunk["timestamp"][0], 3) if chunk["timestamp"][0] is not None else 0.0,
                                "end_time": round(chunk["timestamp"][1], 3) if chunk["timestamp"][1] is not None else 0.0
                            })
                        results.append(offsets)
                    elif isinstance(transcription, dict) and "text" in transcription:
                        # Fallback: single offset for entire transcription
                        text = transcription["text"]
                        if not keep_whitespace:
                            text = text.strip()
                        results.append([{
                            "text": text,
                            "start_time": 0.0,
                            "end_time": round(len(audio_array) / self.sampling_rate, 3)
                        }])
                    else:
                        # Text only
                        text = transcription if isinstance(transcription, str) else str(transcription)
                        if not keep_whitespace:
                            text = text.strip()
                        results.append([{
                            "text": text,
                            "start_time": 0.0,
                            "end_time": round(len(audio_array) / self.sampling_rate, 3)
                        }])
                else:
                    # Generate without timestamps
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=448,
                    )
                    
                    # Decode
                    transcription = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )[0]
                    
                    if not keep_whitespace:
                        transcription = transcription.strip()
                    
                    results.append(transcription)
        
        return results