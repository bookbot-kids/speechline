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
from datasets import Dataset
from transformers import pipeline

from .audio_module import AudioModule


class AudioTranscriber(AudioModule):
    """
    Generic AudioTranscriber class for speech/phoneme recognition.

    Args:
        model_checkpoint (str):
            HuggingFace Hub model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
        )
        super().__init__(pipeline=asr)

    def inference(
        self,
        batch: Dataset,
        chunk_length_s: int = 0,
        output_offsets: bool = False,
        offset_key: str = "text",
        return_timestamps: Union[str, bool] = True,
        **kwargs,
    ) -> Dataset:
        """
        Inference/prediction function to be mapped to a dataset.

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
            """
            Formats `AutomaticSpeechRecognitionPipeline`'s timestamp outputs to
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
                    offset_key: o["text"].strip(),
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
            """
            Formats `AutomaticSpeechRecognitionPipeline`'s timestamp outputs
            to a transcript string.

            Args:
                timestamps (Dict[str, Union[str, List[Dict[str, Union[str, Tuple[float, float]]]]]]):  # noqa: E501
                    Output timestamps from `AutomaticSpeechRecognitionPipeline`.

            Returns:
                str:
                    Transcript string.
            """
            return " ".join(
                [o["text"].strip() for o in timestamps["chunks"] if o["text"] != " "]
            )

        prediction = self.pipeline(
            batch["audio"]["array"],
            chunk_length_s=chunk_length_s,
            return_timestamps=return_timestamps,
            **kwargs,
        )

        if output_offsets:
            batch["prediction"] = _format_timestamps_to_offsets(
                prediction, offset_key=offset_key
            )
        else:
            batch["prediction"] = _format_timestamps_to_transcript(prediction)

        return batch
