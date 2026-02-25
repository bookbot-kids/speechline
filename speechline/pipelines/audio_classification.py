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


import numpy as np
import torch
from transformers import AudioClassificationPipeline


class AudioClassificationWithPaddingPipeline(AudioClassificationPipeline):
    """
    Subclass of `AudioClassificationPipeline`.
    Pads/truncates audio array to maximum length before performing audio classification.
    """

    def __init__(self, *args, **kwargs):
        self.max_duration_s = kwargs.get("max_duration_s")
        super().__init__(*args, **kwargs)

    def preprocess(self, inputs: np.ndarray) -> torch.Tensor:
        """
        Pre-process `inputs` to a maximum length used during model's training.
        Let `max_length = int(sampling_rate * max_duration_s)`.
        Audio arrays shorter than `max_length` will be padded to `max_length`,
        while arrays longer than `max_length` will be truncated to `max_length`.


        Args:
            inputs (np.ndarray):
                Input audio array.

        Returns:
            torch.Tensor:
                Pre-processed audio array as PyTorch tensors.
        """
        processed = self.feature_extractor(
            inputs,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration_s),
            truncation=True,
        )
        return processed
