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
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from speechline.ml.transcriber import AudioTranscriber

class WhisperTranscriber(AudioTranscriber):
    def __init__(self, model_checkpoint: str) -> None:
        model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
        processor = WhisperProcessor.from_pretrained(model_checkpoint)
        super().__init__(model, processor)

    def predict(self, dataset, batch_size: int = 128):
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
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription