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
from transformers import pipeline

from ..pipelines import AudioMultiLabelClassificationPipeline
from .audio_module import AudioModule


class AudioMultiLabelClassifier(AudioModule):
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
            feature_extractor=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
            pipeline_class=AudioMultiLabelClassificationPipeline,
            **kwargs,
        )
        super().__init__(pipeline=classifier)

    def inference(
        self, batch: Dataset, threshold: float = 0.5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Inference function for audio classification.

        Args:
            batch (Dataset):
                Dataset to be inferred.
            threshold (float):
                Threshold probability for predicted labels.
                Anything above this threshold will be considered as a valid prediction.

        Returns:
            List[Dict[str, Union[str, float]]]:
                List of predictions in the format of dictionaries,
                consisting of the predicted label and probability.
        """

        prediction = []
        outputs = self.pipeline(batch["audio"]["array"])
        ids = np.where(outputs >= threshold)[0].tolist()

        if len(ids) > 0:
            prediction = [
                {
                    "label": self.pipeline.model.config.id2label[id],
                    "score": outputs[id],
                }
                for id in ids
            ]

        batch["prediction"] = prediction

        return batch
