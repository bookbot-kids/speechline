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

from difflib import SequenceMatcher
from typing import Dict, List, Union

from .segmenter import Segmenter


class WordOverlapSegmenter(Segmenter):
    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        return text

    def chunk_offsets(
        self,
        offsets: List[Dict[str, Union[str, float]]],
        ground_truth: List[str],
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Chunk transcript offsets based on overlaps with ground truth.

        ### Example
        ```pycon title="example_word_overlap_segmenter.py"
        >>> from speechline.segmenters import WordOverlapSegmenter
        >>> segmenter = WordOverlapSegmenter()
        >>> offsets = [
        ...     {'end_time': 0.28, 'start_time': 0.18, 'text': 'HER'},
        ...     {'end_time': 0.52, 'start_time': 0.34, 'text': 'RED'},
        ...     {'end_time': 1.12, 'start_time': 0.68, 'text': 'UMBRELLA'},
        ...     {'end_time': 1.46, 'start_time': 1.4, 'text': 'IS'},
        ...     {'end_time': 1.78, 'start_time': 1.56, 'text': 'JUST'},
        ...     {'end_time': 1.94, 'start_time': 1.86, 'text': 'THE'},
        ...     {'end_time': 2.3, 'start_time': 1.98, 'text': 'BEST'}
        ... ]
        >>> ground_truth = ["red", "umbrella", "just", "the", "best"]
        >>> segmenter.chunk_offsets(offsets, ground_truth)
        [
            [
                {'end_time': 0.52, 'start_time': 0.34, 'text': 'RED'},
                {'end_time': 1.12, 'start_time': 0.68, 'text': 'UMBRELLA'}
            ],
            [
                {'end_time': 1.78, 'start_time': 1.56, 'text': 'JUST'},
                {'end_time': 1.94, 'start_time': 1.86, 'text': 'THE'},
                {'end_time': 2.3, 'start_time': 1.98, 'text': 'BEST'}
            ]
        ]
        ```

        Args:
            offsets (List[Dict[str, Union[str, float]]]):
                Offsets to chunk.
            ground_truth (List[str]):
                List of ground truth words to compare with offsets.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of chunked/segmented offsets.
        """
        ground_truth = [self.normalize(g) for g in ground_truth]
        transcripts = [self.normalize(o["text"]) for o in offsets]

        matcher = SequenceMatcher(None, transcripts, ground_truth)
        idxs = [(i1, i2) for tag, i1, i2, *_ in matcher.get_opcodes() if tag == "equal"]
        segments = [offsets[i:j] for (i, j) in idxs]
        return segments
