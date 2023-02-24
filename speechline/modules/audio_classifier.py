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
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

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

    def inference(self, batch: Dataset, batch_size: int = 1) -> List[str]:
        """
        Inference function for batched audio classification.

        Args:
            batch (Dataset):
                Dataset to be inferred.
            batch_size (int, optional):
                Batch size during inference. Defaults to `1`.

        Returns:
            List[str]:
                List of predicted labels.
        """
        prediction = [
            o["label"]
            for out in tqdm(
                self.pipeline(
                    KeyDataset(batch["audio"], key="array"),
                    batch_size=batch_size,
                    top_k=1,
                ),
                total=len(batch),
                desc="Classifying Audios",
            )
            for o in out
        ]

        return prediction
