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
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class ClassifierConfig:
    """
    Audio classifier config.

    Args:
        model (str):
            HuggingFace Hub model hub checkpoint.
        max_duration_s (float, optional):
            Maximum audio duration for padding. Defaults to `3.0` seconds.
        batch_size (int, optional):
            Batch size during inference. Defaults to `1`.
    """

    model: str
    max_duration_s: float = 3.0
    batch_size: int = 1


@dataclass
class NoiseClassifierConfig:
    """
    Noise classifier config.

    Args:
        model (str):
            HuggingFace Hub model hub checkpoint.
        min_empty_duration (float, optional):
            Minimum non-transcribed segment duration to be segmented,
            and passed to noise classifier.
            Defaults to `1.0` seconds.
        threshold (float, optional):
            The probability threshold for the multi label classification.
            Defaults to `0.3`.
        batch_size (int, optional):
            Batch size during inference. Defaults to `1`.

    """

    model: str
    minimum_empty_duration: float = 1.0
    threshold: float = 0.3
    batch_size: int = 1


@dataclass
class TranscriberConfig:
    """
    Audio transcriber config.

    Args:
        type (str):
            Transcriber model architecture type.
        model (str):
            HuggingFace Hub model hub checkpoint.
        return_timestamps (Union[str, bool]):
            `return_timestamps` argument in `AutomaticSpeechRecognitionPipeline`'s
            `__call__` method. Use `"char"` for CTC-based models and
            `True` for Whisper-based models.
        chunk_length_s (int):
            Audio chunk length in seconds.
    """

    type: str
    model: str
    return_timestamps: Union[str, bool]
    chunk_length_s: Optional[int] = None 
    transcriber_device: str = "cuda"

    def __post_init__(self):
        SUPPORTED_MODELS = {"wav2vec2", "whisper", "parakeet"}
        WAV2VEC_TIMESTAMPS = {"word", "char"}
        PARAKEET_TIMESTAMPS = {"word"}
        
        if self.type not in SUPPORTED_MODELS:
            raise ValueError(f"Transcriber of type {self.type} is not yet supported!")

        if self.type == "wav2vec2" and self.return_timestamps not in WAV2VEC_TIMESTAMPS:
            raise ValueError("wav2vec2 only supports `'word'` or `'char'` timestamps!")
        elif self.type == "parakeet" and self.return_timestamps not in PARAKEET_TIMESTAMPS:
            raise ValueError("parakeet only supports `word` timestamps!")
        elif self.type == "whisper" and self.return_timestamps is not True:
            raise ValueError("Whisper only supports `True` timestamps!")
        
        # Add validation for chunk_length_s requirement
        if self.type in {"wav2vec2", "whisper"} and self.chunk_length_s is None:
            raise ValueError(f"chunk_length_s is required for {self.type} models")


@dataclass
class SegmenterConfig:
    """
    Audio segmenter config.

    Args:
        silence_duration (float, optional):
            Minimum in-between silence duration (in seconds) to consider as gaps.
            Defaults to `3.0` seconds.
        minimum_chunk_duration (float, optional):
            Minimum chunk duration (in seconds) to be exported.
            Defaults to 0.2 second.
        lexicon_path (str, optional):
            Path to lexicon file. Defaults to `None`.
        keep_whitespace (bool, optional):
            Whether to keep whitespace in transcript. Defaults to `False`.
    """

    type: str
    silence_duration: float = 0.0
    minimum_chunk_duration: float = 0.2
    lexicon_path: str = None
    keep_whitespace: bool = False
    segment_with_ground_truth: bool = False

    def __post_init__(self):
        SUPPORTED_TYPES = {"silence", "word_overlap", "phoneme_overlap"}

        if self.type not in SUPPORTED_TYPES:
            raise ValueError(f"Segmenter of type {self.type} is not yet supported!")


@dataclass
class Config:
    """
    Main SpeechLine config, contains all other subconfigs.

    Args:
        path (str):
            Path to JSON config file.
    """

    path: str

    def __post_init__(self):
        config = json.load(open(self.path))
        self.do_classify = config.get("do_classify", False)
        self.do_noise_classify = config.get("do_noise_classify", False)
        self.filter_empty_transcript = config.get("filter_empty_transcript", False)

        if self.do_classify:
            self.classifier = ClassifierConfig(**config["classifier"])

        if self.do_noise_classify:
            self.noise_classifier = NoiseClassifierConfig(**config["noise_classifier"])

        self.transcriber = TranscriberConfig(**config["transcriber"])
        self.segmenter = SegmenterConfig(**config["segmenter"])
