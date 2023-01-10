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
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import torch
import numpy as np
from itertools import groupby

from speechline.ml.module import AudioModule


class Wav2Vec2Transcriber(AudioModule):
    def __init__(self, model_checkpoint: str) -> None:
        """Initializes a Wav2Vec2-CTC model for phoneme recognition and its processor.

        Args:
            model_checkpoint (str): HuggingFace model hub checkpoint.
        """
        model = AutoModelForCTC.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        super().__init__(model, processor.feature_extractor, processor.tokenizer)

    def decode_phonemes(self, ids: torch.Tensor) -> str:
        """CTC-like decoding. First removes consecutive duplicates, then removes special tokens.

        Args:
            ids (torch.Tensor): Predicted token ids to be decoded.

        Returns:
            str: Decoded phoneme transcription.
        """
        # removes consecutive duplicates
        ids = [id_ for id_, _ in groupby(ids)]

        special_token_ids = self.tokenizer.all_special_ids + [
            self.tokenizer.word_delimiter_token_id
        ]
        # converts id to token, skipping special tokens
        phonemes = [
            self.tokenizer.decode(id_) for id_ in ids if id_ not in special_token_ids
        ]

        # joins phonemes
        prediction = " ".join(phonemes)

        return prediction

    def predict(self, dataset: Dataset, batch_size: int = 128) -> List[str]:
        """Performs batched inference on `dataset`.

        Args:
            dataset (Dataset): Dataset to be inferred.
            batch_size (int, optional): Batch size during inference. Defaults to 128.

        Returns:
            List[str]: List of transcriptions.
        """
        encoded_dataset = dataset.map(
            self.preprocess_function, batched=True, desc="Preprocessing Dataset"
        )

        args = TrainingArguments(
            output_dir="./",
            per_device_eval_batch_size=batch_size,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.feature_extractor,
        )

        logits, *_ = trainer.predict(encoded_dataset)
        predicted_ids = np.argmax(logits, axis=-1)
        transcription = [self.decode_phonemes(id) for id in predicted_ids]
        return transcription


class WhisperTranscriber(AudioModule):
    def __init__(self, model_checkpoint: str) -> None:
        """Initializes a Whisper model for speech recognition and its processor.

        Args:
            model_checkpoint (str): HuggingFace model hub checkpoint.
        """
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        super().__init__(model, processor.feature_extractor, processor.tokenizer)

    def predict(self, dataset: Dataset, batch_size: int = 128):
        """Performs batched inference on `dataset`.

        Args:
            dataset (Dataset): Dataset to be inferred.
            batch_size (int, optional): Batch size during inference. Defaults to 128.

        Returns:
            List[str]: List of transcriptions.
        """
        encoded_dataset = dataset.map(
            self.preprocess_function, batched=True, desc="Preprocessing Dataset"
        )

        args = Seq2SeqTrainingArguments(
            output_dir="./",
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            tokenizer=self.feature_extractor,
        )

        predicted_ids, *_ = trainer.predict(encoded_dataset)
        transcription = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription
