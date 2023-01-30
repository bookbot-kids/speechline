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

from typing import Dict, List, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import pipeline

from .module import AudioModule


class AudioTranscriber(AudioModule):
    """Generic AudioTranscriber class for speech/phoneme recognition.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
        )
        self.sr = self.asr.feature_extractor.sampling_rate

    def inference(
        self,
        batch: Dataset,
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        offset_key: str = "text",
        return_timestamps: Union[str, bool] = True,
    ) -> Dataset:
        """Inference/prediction function to be mapped to a dataset.

        Args:
            batch (Dataset):
                Batch of dataset.
            chunk_length_s (int, optional):
                Audio chunk length in seconds. Defaults to `30`.
            output_offsets (bool, optional):
                Whether to output offsets. Defaults to `False`.
            offset_key (str, optional):
                Offset dictionary key. Defaults to `"text"`.
            return_timestamps (Union[str, bool], optional):
                `return_timestamps` argument in `AutomaticSpeechRecognitionPipeline`'s
                `__call__` method. Use `"char"` for CTC-based models and
                `True` for Whisper-based models.
                Defaults to `True`.

        Returns:
            Dataset:
                Dataset with inferred predictions in `prediction` column.
        """

        def _format_timestamps_to_offsets(
            timestamps: Dict[
                str, Union[str, List[Dict[str, Union[str, Tuple[float, float]]]]]
            ],
            offset_key: str = "text",
        ) -> List[Dict[str, Union[str, float]]]:
            """Formats `AutomaticSpeechRecognitionPipeline`'s timestamp outputs to
            a list of offsets with the following format:

            ```json
            [
                {
                    "{offset_key}": {text},
                    "start_time": {start_time},
                    "end_time": {end_time}
                },
                {
                    "{offset_key}": {text},
                    "start_time": {start_time},
                    "end_time": {end_time}
                },
                ...
            ]
            ```

            Args:
                timestamps (Dict[str, Union[str, List[Dict[str, Union[str, Tuple[float, float]]]]]]):  # noqa: E501
                    Output timestamps from `AutomaticSpeechRecognitionPipeline`.

            Returns:
                List[Dict[str, Union[str, float]]]:
                    List of offsets.
            """
            return [
                {
                    offset_key: o["text"],
                    "start_time": round(o["timestamp"][0], 3),
                    "end_time": round(o["timestamp"][1], 3),
                }
                for o in timestamps["chunks"]
                if o["text"] != " "
            ]

        def _format_timestamps_to_transcript(
            timestamps: Dict[
                str, Union[str, List[Dict[str, Union[str, Tuple[float, float]]]]]
            ],
        ) -> str:
            """Formats `AutomaticSpeechRecognitionPipeline`'s timestamp outputs to
            a transcript string.

            Args:
                timestamps (Dict[str, Union[str, List[Dict[str, Union[str, Tuple[float, float]]]]]]):  # noqa: E501
                    Output timestamps from `AutomaticSpeechRecognitionPipeline`.

            Returns:
                str:
                    Transcript string.
            """
            return " ".join(
                [o["text"] for o in timestamps["chunks"] if o["text"] != " "]
            )

        prediction = self.asr(
            batch["audio"]["array"],
            chunk_length_s=chunk_length_s,
            return_timestamps=return_timestamps,
        )

        if output_offsets:
            batch["prediction"] = _format_timestamps_to_offsets(
                prediction, offset_key=offset_key
            )
        else:
            batch["prediction"] = _format_timestamps_to_transcript(prediction)

        return batch


class Wav2Vec2Transcriber(AudioTranscriber):
    """Wav2Vec2-CTC model for phoneme recognition.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        super().__init__(model_checkpoint)

    def predict(
        self,
        dataset: Union[Dataset, DatasetDict],
        chunk_length_s: int = 30,
        output_offsets: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """Performs inference on `dataset`.

        Args:
            dataset (Union[Dataset, DatasetDict]):
                Dataset to be inferred.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
            output_offsets (bool, optional):
                Whether to output phoneme-level timestamps. Defaults to `False`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of phoneme offsets.

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
        >>> phoneme_offsets = transcriber.predict(dataset, output_offsets=True)
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

        dataset = dataset.map(
            self.inference,
            desc="Performing Inference",
            fn_kwargs={
                "chunk_length_s": chunk_length_s,
                "output_offsets": output_offsets,
                "offset_key": "phoneme",
                "return_timestamps": "char",
            },
        )

        return dataset["prediction"]


class WhisperTranscriber(AudioTranscriber):
    """Whisper model for seq2seq speech recognition with its processor.

    Args:
        model_checkpoint (str): HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        super().__init__(model_checkpoint)

    def predict(
        self,
        dataset: Union[Dataset, DatasetDict],
        chunk_length_s: int = 30,
        output_offsets: bool = False,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """Performs inference on `dataset`.

        Args:
            dataset (Union[Dataset, DatasetDict]):
                Dataset to be inferred.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
            output_offsets (bool, optional):
                Whether to output phoneme-level timestamps. Defaults to `False`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of text offsets.

        ### Example
        ```pycon title="example_transcriber_predict.py"
        >>> from speechline.ml.transcriber import WhisperTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = WhisperTranscriber("openai/whisper-tiny")
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=transcriber.sr)
        ... )
        >>> transcripts = transcriber.predict(dataset)
        >>> transcripts
        [" Her red umbrella is just the best."]
        >>> offsets = transcriber.predict(dataset, output_offsets=True)
        >>> offsets
        [
            [
                {
                    "text": " Her red umbrella is just the best.",
                    "start_time": 0.0,
                    "end_time": 3.0,
                }
            ]
        ]
        ```
        """
        dataset = dataset.map(
            self.inference,
            desc="Performing Inference",
            fn_kwargs={
                "chunk_length_s": chunk_length_s,
                "output_offsets": output_offsets,
                "offset_key": "text",
                "return_timestamps": True,
            },
        )

        return dataset["prediction"]
