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

from datasets import Dataset

from ..modules import AudioClassifier


class Wav2Vec2Classifier(AudioClassifier):
    """
    Audio classifier with feature extractor.

    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint.
        max_duration_s (float):
            Maximum audio duration in seconds.
    """

    def __init__(self, model_checkpoint: str, max_duration_s: float) -> None:
        super().__init__(model_checkpoint, max_duration_s=max_duration_s)

    def predict(self, dataset: Dataset) -> List[str]:
        """
        Performs audio classification (inference) on `dataset`.
        Preprocesses datasets, performs inference, then returns predictions.

        Args:
            dataset (Dataset):
                Dataset to be inferred.

        Returns:
            List[str]:
                List of predictions (as strings of labels).
        """

        dataset = dataset.map(self.inference, desc="Classifying Audios")
        return dataset["prediction"]
