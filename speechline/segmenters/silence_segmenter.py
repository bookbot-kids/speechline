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

from .segmenter import Segmenter


class SilenceSegmenter(Segmenter):
    def chunk_offsets(
        self,
        offsets: List[Dict[str, Union[str, float]]],
        silence_duration: float,
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Chunk transcript offsets based on in-between silence duration.

        ### Example
        ```pycon title="example_silence_segmenter.py"
        >>> from speechline.segmenters import SilenceSegmenter
        >>> segmenter = SilenceSegmenter()
        >>> offsets = [
        ...     {"start_time": 0.0, "end_time": 0.4},
        ...     {"start_time": 0.4, "end_time": 0.5},
        ...     {"start_time": 0.6, "end_time": 0.8},
        ...     {"start_time": 1.1, "end_time": 1.3},
        ...     {"start_time": 1.5, "end_time": 1.7},
        ...     {"start_time": 1.7, "end_time": 2.0},
        ... ]
        >>> segmenter.chunk_offsets(offsets, silence_duration=0.2)
        [
            [
                {"start_time": 0.0, "end_time": 0.4},
                {"start_time": 0.4, "end_time": 0.5},
                {"start_time": 0.6, "end_time": 0.8},
            ],
            [
                {"start_time": 1.1, "end_time": 1.3}
            ],
            [
                {"start_time": 1.5, "end_time": 1.7},
                {"start_time": 1.7, "end_time": 2.0}
            ],
        ]
        >>> segmenter.chunk_offsets(offsets, silence_duration=0.1)
        [
            [
                {"start_time": 0.0, "end_time": 0.4},
                {"start_time": 0.4, "end_time": 0.5}
            ],
            [
                {"start_time": 0.6, "end_time": 0.8}
            ],
            [
                {"start_time": 1.1, "end_time": 1.3}
            ],
            [
                {"start_time": 1.5, "end_time": 1.7},
                {"start_time": 1.7, "end_time": 2.0}
            ],
        ]
        ```

        Args:
            offsets (List[Dict[str, Union[str, float]]]):
                Offsets to chunk.
            silence_duration (float):
                Minimum in-between silence duration (in seconds) to consider as gaps.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of chunked/segmented offsets.
        """
        # calculate gaps in between offsets
        gaps = [
            round(next["start_time"] - curr["end_time"], 3)
            for curr, next in zip(offsets, offsets[1:])
        ]
        # generate segment slices (start and end indices) based on silence
        slices = (
            [0]
            + [idx + 1 for idx, gap in enumerate(gaps) if gap >= silence_duration]
            + [len(offsets)]
        )
        # group consecutive offsets (segments) based on slices
        segments = [offsets[i:j] for i, j in zip(slices, slices[1:])]
        return segments
