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
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, Audio
import torch
import numpy as np
import pandas as pd


class AudioClassifier:
    def __init__(self, model_checkpoint: str) -> None:
        """Initializes a HuggingFace audio classifier and feature extractor.

        Args:
            model_checkpoint (str): HuggingFace model hub checkpoint.
        """
        self.model = AutoModelForAudioClassification.from_pretrained(model_checkpoint)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        self.sr = self.feature_extractor.sampling_rate

    def format_audio_dataset(self, df: pd.DataFrame) -> Dataset:
        """Formats Pandas `DataFrame` as a datasets `Dataset`.
        Converts `audio` path column to audio arrays and resamples accordingly.

        Args:
            df (pd.DataFrame): Pandas

        Returns:
            Dataset: datasets `Dataset` usable for batch inference.
        """
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sr))
        return dataset

    def preprocess_function(self, examples: Dataset) -> torch.Tensor:
        """Preprocess function for audio classification.
        Truncates audio arrays into the designated maximum audio duration used in training.

        Args:
            examples (Dataset): Audio `Dataset`.

        Returns:
            torch.Tensor: Batch of preprocessed audio tensors.
        """
        max_duration = 3.0  # seconds
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=int(self.feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs

    def predict(self, dataset: Dataset, batch_size: int = 128) -> List[str]:
        """Performs batch audio classification (inference) on `dataset`.
        First preprocesses dataset into tensors, performs batch inference, then returns predictions.

        Args:
            dataset (Dataset): Dataset to be inferred.
            batch_size (int, optional): Per device evaluation batch size. Defaults to 128.

        Returns:
            List[str]: List of predictions (in string of labels).
        """
        encoded_dataset = dataset.map(
            self.preprocess_function,
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
        predictions = np.argmax(logits, axis=1).tolist()
        predictions = [self.model.config.id2label[p] for p in predictions]
        return predictions
