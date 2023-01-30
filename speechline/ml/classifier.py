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

from typing import List, Union

import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import AudioClassificationPipeline, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from .module import AudioModule


class AudioClassificationWithPaddingPipeline(AudioClassificationPipeline):
    def __init__(self, *args, **kwargs):
        self.max_duration_s = kwargs.get("max_duration_s", None)
        super().__init__(*args, **kwargs)

    def preprocess(self, inputs):
        sampling_rate = self.feature_extractor.sampling_rate
        max_length = (
            int(sampling_rate * self.max_duration_s) if self.max_duration_s else None
        )
        truncation = self.max_duration_s is not None

        processed = self.feature_extractor(
            inputs,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            max_length=max_length,
            truncation=truncation,
        )
        return processed


class AudioClassifier(AudioModule):
    def __init__(self, model_checkpoint: str, max_duration_s: float = None) -> None:
        self.classifier = pipeline(
            "audio-classification",
            model=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
            max_duration_s=max_duration_s,
            pipeline_class=AudioClassificationWithPaddingPipeline,
        )
        self.sr = self.classifier.feature_extractor.sampling_rate

    def inference(self, batch: Dataset, batch_size: int = 1) -> List[str]:
        prediction = [
            o["label"]
            for out in tqdm(
                self.classifier(
                    KeyDataset(batch["audio"], key="array"),
                    batch_size=batch_size,
                    top_k=1,
                ),
                total=len(batch),
            )
            for o in out
        ]

        return prediction


class Wav2Vec2Classifier(AudioClassifier):
    """Audio classifier with feature extractor.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str, max_duration_s: float = None) -> None:
        super().__init__(model_checkpoint, max_duration_s=max_duration_s)

    def predict(
        self, dataset: Union[Dataset, DatasetDict], batch_size: int = 1
    ) -> List[str]:
        """Performs batch audio classification (inference) on `dataset`.
        Preprocesses datasets, performs batch inference, then returns predictions.

        Args:
            dataset (Union[Dataset, DatasetDict]): Dataset to be inferred.
            batch_size (int, optional): Per device batch size. Defaults to 1.

        Returns:
            List[str]: List of predictions (in string of labels).
        """

        predictions = self.inference(dataset, batch_size=batch_size)
        return predictions
