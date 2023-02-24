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


class Wav2Vec2Transcriber(AudioTranscriber):
    """
    Wav2Vec2-CTC model for speech recognition.

    Args:
        model_checkpoint (str):
            HuggingFace model hub checkpoint.
    """

    def __init__(self, model_checkpoint: str) -> None:
        super().__init__(model_checkpoint)

    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        return_timestamps: str = "word",
    ) -> Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
        """
        Performs inference on `dataset`.

        Args:
            dataset (Dataset):
                Dataset to be inferred.
            chunk_length_s (int):
                Audio chunk length during inference. Defaults to `30`.
            output_offsets (bool, optional):
                Whether to output timestamps. Defaults to `False`.
            return_timestamps (str, optional):
                Returned timestamp level. Defaults to `"word"`.

        Returns:
            Union[List[str], List[List[Dict[str, Union[str, float]]]]]:
                Defaults to list of transcriptions.
                If `output_offsets` is `True`, return list of offsets.

        ### Example
        ```pycon title="example_transcriber_predict.py"
        >>> from speechline.transcribers import Wav2Vec2Transcriber
        >>> from datasets import Dataset, Audio
        >>> transcriber = Wav2Vec2Transcriber("bookbot/wav2vec2-ljspeech-gruut")
        >>> dataset = Dataset.from_dict({"audio": ["sample.wav"]}).cast_column(
        ...     "audio", Audio(sampling_rate=transcriber.sr)
        ... )
        >>> transcripts = transcriber.predict(dataset)
        >>> transcripts
        ["ɪ t ɪ z n oʊ t ʌ p"]
        >>> offsets = transcriber.predict(dataset, output_offsets=True)
        >>> offsets
        [
            [
                {"text": "ɪ", "start_time": 0.0, "end_time": 0.02},
                {"text": "t", "start_time": 0.26, "end_time": 0.3},
                {"text": "ɪ", "start_time": 0.34, "end_time": 0.36},
                {"text": "z", "start_time": 0.42, "end_time": 0.44},
                {"text": "n", "start_time": 0.5, "end_time": 0.54},
                {"text": "oʊ", "start_time": 0.54, "end_time": 0.58},
                {"text": "t", "start_time": 0.58, "end_time": 0.62},
                {"text": "ʌ", "start_time": 0.76, "end_time": 0.78},
                {"text": "p", "start_time": 0.92, "end_time": 0.94},
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
            },
        )

        return dataset["prediction"]
