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

from ..modules import AudioTranscriber


class WhisperTranscriber(AudioTranscriber):
    """
    Whisper model for seq2seq speech recognition with its processor.

    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        super().__init__(model_checkpoint)

    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 0,
        output_offsets: bool = False,
        return_timestamps: bool = True,
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """
        Performs inference on `dataset`.

        Args:
            dataset (Dataset):
                Dataset to be inferred.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `0`.
            output_offsets (bool, optional):
                Whether to output timestamps. Defaults to `False`.
            return_timestamps (bool, optional):
                Returned timestamp level. Defaults to `True`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of text offsets.

        ### Example
        ```pycon title="example_transcriber_predict.py"
        >>> from speechline.transcribers import WhisperTranscriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = WhisperTranscriber("openai/whisper-tiny")
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=transcriber.sr)
        ... )
        >>> transcripts = transcriber.predict(dataset)
        >>> transcripts
        ["Her red umbrella is just the best."]
        >>> offsets = transcriber.predict(dataset, output_offsets=True)
        >>> offsets
        [
            [
                {
                    "text": "Her red umbrella is just the best.",
                    "start_time": 0.0,
                    "end_time": 3.0,
                }
            ]
        ]
        ```
        """
        dataset = dataset.map(
            self.inference,
            desc="Transcribing Audios",
            fn_kwargs={
                "chunk_length_s": chunk_length_s,
                "output_offsets": output_offsets,
                "offset_key": "text",
                "return_timestamps": return_timestamps,
                "generate_kwargs": {"max_new_tokens": 448},
            },
        )

        return dataset["prediction"]
