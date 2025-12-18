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

from typing import Dict, List, Union
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
import tempfile
import soundfile as sf
from pathlib import Path


class CanaryTranscriber:
    """
    NVIDIA Canary-Qwen model for multilingual ASR and translation.
    
    This transcriber uses NVIDIA NeMo framework instead of HuggingFace pipelines
    because Canary-Qwen is built on NeMo's neural modules architecture.
    
    Canary-Qwen is a 2.5B parameter speech-to-text model supporting:
    - Automatic Speech Recognition (ASR) in 100+ languages
    - Speech translation to English
    - Optimized for Apple Silicon (MPS), CUDA, and CPU

    Requirements:
        - nemo_toolkit[asr] must be installed
        - Run: pip install nemo_toolkit[asr]

    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint or NeMo .nemo file path.
            Defaults to "nvidia/canary-qwen-2.5b".
        torch_dtype (str, optional):
            Torch dtype for model weights (e.g., 'float16' for Apple Silicon).
            Note: Canary-Qwen uses bfloat16 by default.
    """

    def __init__(
        self,
        model_checkpoint: str = "nvidia/canary-qwen-2.5b",
        torch_dtype: str = None
    ) -> None:
        try:
            from nemo.collections.speechlm2.models import SALM
        except ImportError:
            raise ImportError(
                "NeMo toolkit with speechlm2 is required for Canary-Qwen. "
                "Install with: pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git'"
            )
        
        # Determine device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading Canary-Qwen SALM model on {self.device}...")
        
        # Load the NeMo SALM model
        # Canary-Qwen is a Speech-Augmented Language Model (SALM)
        self.model = SALM.from_pretrained(model_checkpoint)
        self.model.eval()
        
        # Move model to device and verify placement
        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
                
                # Verify model is on correct device by checking first parameter
                first_param = next(self.model.parameters())
                actual_device = str(first_param.device)
                print(f"   Model parameters on device: {actual_device}")
                
                if actual_device.startswith("cpu") and self.device != "cpu":
                    print(f"   ⚠️  Warning: Model failed to move to {self.device}, using CPU")
                    print(f"   This is a known limitation with NeMo models and MPS")
                    self.device = "cpu"
            except Exception as e:
                print(f"   ⚠️  Warning: Could not move model to {self.device}: {e}")
                print(f"   Falling back to CPU")
                self.device = "cpu"
        
        # Store sampling rate (Canary models use 16kHz)
        self.sampling_rate = 16000  # Standard for Canary models
        self.sr = self.sampling_rate  # Alias for compatibility
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"   Sampling rate: {self.sampling_rate} Hz")
    
    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        return_timestamps: bool = True,
        keep_whitespace: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """
        Performs inference on `dataset`.

        Args:
            dataset (Dataset):
                Dataset to be inferred. Must have 'audio' column with array and sampling_rate.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
                Note: Long audio is automatically handled by NeMo.
            output_offsets (bool, optional):
                Whether to output timestamps. Defaults to `False`.
            return_timestamps (bool, optional):
                Whether to return timestamps. Defaults to `True`.
            keep_whitespace (bool, optional):
                Whether to preserve whitespace predictions. Defaults to `False`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of text offsets.

        ### Example
        ```pycon title="example_canary_predict.py"
        >>> from speechline.transcribers import CanaryTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = CanaryTranscriber()
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
                {
                    "text": "This is a sample transcription.",
                    "start_time": 0.0,
                    "end_time": 2.5,
                }
            ]
        ]
        ```
        """
        import librosa
        results = []
        
        # Create temporary directory for audio files
        # SALM models work with file paths in prompts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Get audio column - handle both file paths and loaded audio
            for idx in tqdm(range(len(dataset)), desc="Transcribing Audios"):
                item = dataset[idx]
                
                # Check if audio is already loaded or is a file path
                if isinstance(item["audio"], dict) and "array" in item["audio"]:
                    # Audio is already loaded
                    audio_array = item["audio"]["array"]
                    audio_sr = item["audio"]["sampling_rate"]
                elif isinstance(item["audio"], str):
                    # Audio is a file path - load it with librosa
                    audio_array, audio_sr = librosa.load(
                        item["audio"],
                        sr=self.sampling_rate,
                        mono=True
                    )
                else:
                    raise ValueError(f"Unexpected audio format: {type(item['audio'])}")
                
                # Save audio to temporary file
                temp_audio_path = temp_dir_path / f"audio_{idx}.wav"
                sf.write(temp_audio_path, audio_array, self.sampling_rate)
                
                # Transcribe using SALM generate API
                # Format: [{"role": "user", "content": "prompt", "audio": ["path"]}]
                answer_ids = self.model.generate(
                    prompts=[
                        [{
                            "role": "user",
                            "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                            "audio": [str(temp_audio_path)]
                        }]
                    ],
                    max_new_tokens=128,
                )
                
                # Decode the generated tokens
                if answer_ids is not None and answer_ids.numel() > 0:
                    transcription = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                else:
                    transcription = ""
                
                if not keep_whitespace:
                    transcription = transcription.strip()
                
                if output_offsets:
                    # For offsets, estimate timestamps
                    # SALM models don't provide word-level timestamps by default
                    audio_duration = len(audio_array) / self.sampling_rate
                    offset = [{
                        "text": transcription,
                        "start_time": 0.0,
                        "end_time": round(audio_duration, 3)
                    }]
                    results.append(offset)
                else:
                    results.append(transcription)
        
        return results