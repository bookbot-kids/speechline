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

from typing import Optional, Any
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer, SequenceFeatureExtractor
from datasets import Dataset, Audio
import torch


class AudioModule:
    """Base AudioModule. Inherit this class for other audio models.
    Contains implemented methods for audio dataset formatting and preprocessing.

    Args:
        model (`PreTrainedModel`):
            Pre-trained audio model for inference.
        feature_extractor (`SequenceFeatureExtractor`):
            Pre-trained feature extractor for inference.
        tokenizer (`PreTrainedTokenizer`, optional):
            Optional pre-trained tokenizer for logits decoding. Defaults to None.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizer = None,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sr = self.feature_extractor.sampling_rate

    def format_audio_dataset(self, df: pd.DataFrame) -> Dataset:
        """Formats Pandas `DataFrame` as a datasets `Dataset`.
        Converts `audio` path column to audio arrays and resamples accordingly.

        Args:
            df (`pd.DataFrame`): Pandas

        Returns:
            `Dataset`: datasets `Dataset` usable for batch inference.
        """
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sr))
        return dataset

    def preprocess_function(
        self,
        batch: Dataset,
        max_duration: Optional[int] = None,
    ) -> Any:
        """Audio dataset preprocessing function. Extracts audio waves as features.
        Allows for optional truncation given a maximum duration.

        Args:
            batch (`Dataset`):
                Batch audio dataset to be preprocessed.
            max_duration (`Optional[int]`, optional):
                Maximum audio duration in seconds.
                Truncates audio if specified. Defaults to None.

        Returns:
            `Any`: Batch of preprocessed audio features.
        """
        max_length = int(self.sr * max_duration) if max_duration else None
        truncation = True if max_duration else False
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sr,
            max_length=max_length,
            truncation=truncation,
        )
        return inputs

    def clear_memory(self) -> None:
        """Clears model from memory.
        Optionally also clears CUDA cache, if GPU is available.
        Source: [PyTorch Forums](https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879).
        """  # noqa: E501
        del self.model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
