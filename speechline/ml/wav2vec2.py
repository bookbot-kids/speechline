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

from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
from itertools import groupby

from speechline.ml.transcriber import AudioTranscriber


class Wav2Vec2Transcriber(AudioTranscriber):
    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForCTC.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        super().__init__(model, processor)

    def decode_phonemes(
        self,
        ids: torch.Tensor,
        ignore_stress: bool = True,
    ) -> str:
        """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
        # removes consecutive duplicates
        ids = [id_ for id_, _ in groupby(ids)]

        special_token_ids = self.processor.tokenizer.all_special_ids + [
            self.processor.tokenizer.word_delimiter_token_id
        ]
        # converts id to token, skipping special tokens
        phonemes = [
            self.processor.decode(id_) for id_ in ids if id_ not in special_token_ids
        ]

        # joins phonemes
        prediction = " ".join(phonemes)

        # whether to ignore IPA stress marks
        if ignore_stress:
            prediction = prediction.replace("ˈ", "").replace("ˌ", "")

        return prediction

    def predict(self, dataset, batch_size: int = 128):
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
