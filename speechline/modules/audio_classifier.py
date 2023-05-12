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

from typing import List

import torch
import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import pipeline

from ..pipelines import AudioClassificationWithPaddingPipeline
from .audio_module import AudioModule


class AudioClassifier(AudioModule):
    """
    Generic AudioClassifier Module. Performs padded audio classification.

    Args:
        model_checkpoint (str):
            HuggingFace Hub model checkpoint.
    """

    def __init__(self, model_checkpoint: str, **kwargs) -> None:
        classifier = pipeline(
            "audio-classification",
            model=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
            pipeline_class=AudioClassificationWithPaddingPipeline,
            **kwargs,
        )
        super().__init__(pipeline=classifier)

    def inference(self, dataset: Dataset) -> List[str]:
        """
        Inference function for audio classification.

        Args:
            dataset (Dataset):
                Dataset to be inferred.

        Returns:
            List[str]:
                List of predicted labels.
        """

        def _get_audio_array(
            dataset: Dataset,
        ) -> np.ndarray:
            for item in dataset:
                yield item["audio"]["array"]

        results = []

        for out in tqdm(
            self.pipeline(_get_audio_array(dataset), top_k=1),
            total=len(dataset),
            desc="Classifying Audios",
        ):
            prediction = out[0]["label"]
            results.append(prediction)

        return results
