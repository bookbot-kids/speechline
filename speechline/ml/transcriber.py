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

from typing import Any, Dict, List, Union

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from .module import AudioModule


class Wav2Vec2Transcriber(AudioModule):
    """Wav2Vec2-CTC model for phoneme recognition with its processor.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForCTC.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        super().__init__(model, feature_extractor, tokenizer)
        self.time_offset = self.model.config.inputs_to_logits_ratio / self.sr

    def predict(
        self,
        dataset: Union[Dataset, DatasetDict],
        batch_size: int = 1,
        output_phoneme_offsets: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Any]]]]:
        """Performs batched inference on `dataset`.

        Args:
            dataset (Union[Dataset, DatasetDict]):
                Dataset to be inferred.
            batch_size (int, optional):
                Batch size during inference. Defaults to 1.
                Using a batch size >1 may hurt the performance of the model.
            output_phoneme_offsets (bool, optional):
                Whether to output phoneme-level timestamps. Defaults to False.

        Returns:
            Union[List[str], List[List[Dict[str, Any]]]]:
                Defaults to list of transcriptions.
                If `output_phoneme_offsets` is `True`, return list of phoneme offsets.

        ### Example
        ```pycon title="example_transcriber_predict.py"
        >>> from speechline.ml.transcriber import Wav2Vec2Transcriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = Wav2Vec2Transcriber("bookbot/wav2vec2-ljspeech-gruut")
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=transcriber.sr)
        ... )
        >>> transcripts = transcriber.predict(dataset)
        >>> transcripts
        ["ɪ t ɪ z n oʊ t ʌ p"]
        >>> phoneme_offsets = transcriber.predict(dataset, output_phoneme_offsets=True)
        >>> phoneme_offsets
        [
            [
                {"phoneme": "ɪ", "start_time": 0.0, "end_time": 0.02},
                {"phoneme": "t", "start_time": 0.26, "end_time": 0.3},
                {"phoneme": "ɪ", "start_time": 0.34, "end_time": 0.36},
                {"phoneme": "z", "start_time": 0.42, "end_time": 0.44},
                {"phoneme": "n", "start_time": 0.5, "end_time": 0.54},
                {"phoneme": "oʊ", "start_time": 0.54, "end_time": 0.58},
                {"phoneme": "t", "start_time": 0.58, "end_time": 0.62},
                {"phoneme": "ʌ", "start_time": 0.76, "end_time": 0.78},
                {"phoneme": "p", "start_time": 0.92, "end_time": 0.94},
            ]
        ]
        ```
        """
        encoded_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Preprocessing Dataset",
            fn_kwargs={
                "feature_extractor": self.feature_extractor,
            },
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
        outputs = self.tokenizer.batch_decode(predicted_ids, output_char_offsets=True)

        phoneme_offsets: List[List[Dict[str, Any]]] = [
            [
                {
                    "phoneme": o["char"],
                    "start_time": round(o["start_offset"] * self.time_offset, 3),
                    "end_time": round(o["end_offset"] * self.time_offset, 3),
                }
                for o in offset
                if o["char"] != " "
            ]
            for offset in outputs.char_offsets
        ]

        transcripts: List[str] = [
            " ".join(o["phoneme"] for o in offset) for offset in phoneme_offsets
        ]

        if output_phoneme_offsets:
            return phoneme_offsets
        else:
            return transcripts


class WhisperTranscriber(AudioModule):
    """Whisper model for seq2seq speech recognition with its processor.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        super().__init__(model, feature_extractor, tokenizer)

    def predict(
        self, dataset: Union[Dataset, DatasetDict], batch_size: int = 1
    ) -> List[str]:
        """Performs batched inference on `dataset`.

        Args:
            dataset (Union[Dataset, DatasetDict]): Dataset to be inferred.
            batch_size (int, optional): Batch size during inference. Defaults to 1.

        Returns:
            List[str]: List of transcriptions.
        """
        encoded_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Preprocessing Dataset",
            fn_kwargs={
                "feature_extractor": self.feature_extractor,
            },
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
        transcription: List[str] = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription
