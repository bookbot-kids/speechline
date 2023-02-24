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
from typing import Union


@dataclass
class ClassifierConfig:
    model: str
    max_duration_s: float = 3.0
    batch_size: int = 1


@dataclass
class TranscriberConfig:
    type: str
    model: str
    return_timestamps: Union[str, bool]
    chunk_length_s: int

    def __post_init__(self):
        SUPPORTED_MODELS = {"wav2vec2", "whisper"}
        WAV2VEC_TIMESTAMPS = {"word", "char"}

        if self.type not in SUPPORTED_MODELS:
            raise ValueError(f"Transcriber of type {self.type} is not yet supported!")

        if self.type == "wav2vec2" and self.return_timestamps not in WAV2VEC_TIMESTAMPS:
            raise ValueError("wav2vec2 only supports `'word'` or `'char'` timestamps!")
        elif self.type == "whisper" and self.return_timestamps is not True:
            raise ValueError("Whisper only supports `True` timestamps!")


@dataclass
class SegmenterConfig:
    silence_duration: float = 3.0
    minimum_chunk_duration: float = 0.2


@dataclass
class Config:
    """
    SpeechLine Config: languages, models, parameters.

    Args:
        path (str):
            Path to JSON config file.
    """

    path: str

    def __post_init__(self):
        config = json.load(open(self.path))
        self.do_classify = config.get("do_classify", False)
        self.classifier = ClassifierConfig(**config["classifier"])
        self.transcriber = TranscriberConfig(**config["transcriber"])
        self.segmenter = SegmenterConfig(**config["segmenter"])
