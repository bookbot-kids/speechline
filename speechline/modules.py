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
from tqdm.auto import tqdm
from transformers import Pipeline, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from .pipelines import AudioClassificationWithPaddingPipeline


class AudioModule:
    """
    Base AudioModule. Inherit this class for other audio models.
    An AudioModule should have an inference pipeline,
    and an inference function utilizing the pipeline.

    Args:
        pipeline (Pipeline):
            HuggingFace `transformers` `Pipeline` for inference.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate


class AudioClassifier(AudioModule):
    """
    Generic AudioClassifier Module. Performs padded audio classification.

    Args:
        model_checkpoint (str):
            HuggingFace Hub model checkpoint.
    """

    def __init__(self, model_checkpoint: str, **kwargs) -> None:
        classifier = pipeline(
            "audio-classification",
            model=model_checkpoint,
            device=0 if torch.cuda.is_available() else -1,
            pipeline_class=AudioClassificationWithPaddingPipeline,
            **kwargs,
        )
        super().__init__(pipeline=classifier)

    def inference(self, batch: Dataset, batch_size: int = 1) -> List[str]:
        """
        Inference function for batched audio classification.

        Args:
            batch (Dataset):
                Dataset to be inferred.
            batch_size (int, optional):
                Batch size during inference. Defaults to `1`.

        Returns:
            List[str]:
                List of predicted labels.
        """
        prediction = [
            o["label"]
            for out in tqdm(
                self.pipeline(
                    KeyDataset(batch["audio"], key="array"),
                    batch_size=batch_size,
                    top_k=1,
                ),
                total=len(batch),
                desc="Classifying Audios",
            )
            for o in out
        ]

        return prediction


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
        chunk_length_s: int = 30,
        output_offsets: bool = False,
        offset_key: str = "text",
        return_timestamps: Union[str, bool] = True,
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
                [o["text"] for o in timestamps["chunks"] if o["text"] != " "]
            )

        prediction = self.pipeline(
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
