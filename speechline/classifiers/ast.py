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

from datasets import Dataset

from ..modules import AudioMultiLabelClassifier


class ASTClassifier(AudioMultiLabelClassifier):
    """
    Audio classifier with feature extractor.

    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        super().__init__(model_checkpoint)

    def predict(
        self, dataset: Dataset, threshold: float = 0.5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Performs audio classification (inference) on `dataset`.
        Preprocesses datasets, performs inference, then returns predictions.

        Args:
            dataset (Dataset):
                Dataset to be inferred.
            threshold (float):
                Threshold probability for predicted labels.
                Anything above this threshold will be considered as a valid prediction.

        Returns:
            List[Dict[str, Union[str, float]]]:
                List of predictions in the format of dictionaries,
                consisting of the predicted label and probability.
        """

        dataset = dataset.map(
            self.inference,
            fn_kwargs={"threshold": threshold},
            desc="Classifying Audios",
        )
        return dataset["prediction"]
