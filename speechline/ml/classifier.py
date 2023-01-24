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
from functools import partial
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import numpy as np

from speechline.ml.module import AudioModule


class Wav2Vec2Classifier(AudioModule):
    """Audio classifier with feature extractor.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForAudioClassification.from_pretrained(model_checkpoint)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        super().__init__(model, feature_extractor)

    def predict(self, dataset: Dataset, batch_size: int = 128) -> List[str]:
        """Performs batch audio classification (inference) on `dataset`.
        Preprocesses datasets, performs batch inference, then returns predictions.

        Args:
            dataset (Dataset): Dataset to be inferred.
            batch_size (int, optional): Per device batch size. Defaults to 128.

        Returns:
            List[str]: List of predictions (in string of labels).
        """

        encoded_dataset = dataset.map(
            partial(self.preprocess_function, max_duration=3.0),
            batched=True,
            desc="Preprocessing Dataset",
        )

        args = TrainingArguments(output_dir="./", per_device_eval_batch_size=batch_size)

        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.feature_extractor,
        )

        logits, *_ = trainer.predict(encoded_dataset)
        predicted_ids = np.argmax(logits, axis=1).tolist()
        predictions = [self.model.config.id2label[p] for p in predicted_ids]
        return predictions
