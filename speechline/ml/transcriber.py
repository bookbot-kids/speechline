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

from typing import List, Dict, Any, Union
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from ctc_segmentation import (
    CtcSegmentationParameters,
    prepare_text,
    ctc_segmentation,
    determine_utterance_segments,
)
from datasets import Dataset
import torch
import numpy as np
from itertools import groupby

from speechline.ml.module import AudioModule


class Wav2Vec2Transcriber(AudioModule):
    """Wav2Vec2-CTC model for phoneme recognition with its processor.

    Args:
        model_checkpoint (`str`): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForCTC.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        super().__init__(model, feature_extractor, tokenizer)

        self.ctc_config = CtcSegmentationParameters(char_list=char_list)
        self.ctc_config.index_duration = model.config.inputs_to_logits_ratio / self.sr

    def decode_phonemes(self, ids: torch.Tensor) -> str:
        """CTC-like decoding of phonemes.
        First removes consecutive duplicates, then removes special tokens.

        Args:
            ids (`torch.Tensor`): Predicted token ids to be decoded.

        Returns:
            `str`: Decoded phoneme transcription.
        """
        # removes consecutive duplicates
        deduplicated_ids = [id_ for id_, _ in groupby(ids)]

        special_token_ids = self.tokenizer.all_special_ids + [
            self.tokenizer.word_delimiter_token_id
        ]
        # converts id to token, skipping special tokens
        phonemes = [
            self.tokenizer.decode(id_)
            for id_ in deduplicated_ids
            if id_ not in special_token_ids
        ]

        # joins phonemes
        prediction = " ".join(phonemes)

        return prediction

    def decode_phoneme_offsets(
        self, phonemes: List[str], probabilities: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Perform CTC-based segmentation on transcribed phonemes and probabilities.
        Returns a list of dictionaries containing predicted phoneme and start-end times.
        Source: [CTC-Segmentation](https://github.com/lumaku/ctc-segmentation#usage).

        Args:
            phonemes (`List[str]`):
                List of transcribed phonemes.
            probabilities (`np.ndarray`):
                Output probabilities per timestep from wav2vec2.

        Returns:
            `List[Dict[str, Any]]`: List of phoneme-level timestamps.
        """
        ground_truth_mat, utt_begin_indices = prepare_text(self.ctc_config, phonemes)
        timings, char_probs, _ = ctc_segmentation(
            self.ctc_config, probabilities, ground_truth_mat
        )
        segments = determine_utterance_segments(
            self.ctc_config, utt_begin_indices, char_probs, timings, phonemes
        )
        phoneme_offsets = [
            {"phoneme": p, "start_time": round(s[0], 3), "end_time": round(s[1], 3)}
            for p, s in zip(phonemes, segments)
        ]
        return phoneme_offsets

    def predict(
        self,
        dataset: Dataset,
        batch_size: int = 128,
        output_phoneme_offsets: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Any]]]]:
        """Performs batched inference on `dataset`.

        Args:
            dataset (`Dataset`):
                Dataset to be inferred.
            batch_size (`int`, optional):
                Batch size during inference. Defaults to 128.
            output_phoneme_offsets (`bool`, optional):
                Whether to output phoneme-level timestamps. Defaults to False.

        Returns:
            `Union[List[str], List[List[Dict[str, Any]]]]`:
                Defaults to list of transcriptions.
                If `output_phoneme_offsets` is `True`, return list of phoneme offsets.
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
        logits = torch.from_numpy(logits)

        probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()
        predicted_ids = np.argmax(logits, axis=-1)

        transcription: List[str] = [self.decode_phonemes(id) for id in predicted_ids]
        phonemes: List[List[str]] = [t.split() for t in transcription]

        if output_phoneme_offsets:
            phoneme_offsets: List[List[Dict[str, Any]]] = [
                self.decode_phoneme_offsets(phn, prob)
                for phn, prob in zip(phonemes, probabilities)
            ]
            return phoneme_offsets
        else:
            return transcription


class WhisperTranscriber(AudioModule):
    """Whisper model for seq2seq speech recognition with its processor.

    Args:
        model_checkpoint (`str`): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        super().__init__(model, feature_extractor, tokenizer)

    def predict(self, dataset: Dataset, batch_size: int = 128) -> List[str]:
        """Performs batched inference on `dataset`.

        Args:
            dataset (`Dataset`): Dataset to be inferred.
            batch_size (`int`, optional): Batch size during inference. Defaults to 128.

        Returns:
            `List[str]`: List of transcriptions.
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
        transcription: List[str] = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription
