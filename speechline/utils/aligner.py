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

import re
from difflib import SequenceMatcher
from itertools import combinations
from statistics import stdev
from typing import Callable, Dict, List, Optional, Tuple, Union


class PunctuationForcedAligner:
    """
    Force-align predicted phoneme offsets with ground truth text with punctuation.

    Args:
        g2p (Callable[[str], List[str]]):
            Callable grapheme-to-phoneme function.
        punctuations (Optional[List[str]], optional):
            List of punctuations to include. Defaults to `None`.
    """

    def __init__(
        self, g2p: Callable[[str], List[str]], punctuations: Optional[List[str]] = None
    ):
        self.punctuations = (
            ["?", ",", ".", "!", ";"] if not punctuations else punctuations
        )
        self.g2p = g2p

    def __call__(self, offsets: List[Dict[str, Union[str, float]]], text: str):
        updated_offsets = offsets[:]
        predicted_phonemes = [offset["phoneme"] for offset in updated_offsets]
        ground_truth_phonemes = self.g2p(text)

        # segment phonemes based on `self.punctuations`
        segments, cleaned_segments = self.segment_phonemes_punctuations(
            ground_truth_phonemes
        )

        # generate all possible segments from predicted phonemes
        potential_segments = self.generate_partitions(
            predicted_phonemes, n=len(cleaned_segments)
        )

        # if there are multiple possible partitions
        if len(cleaned_segments) > 1:
            # filter for highly probable candidates
            potential_segments = self._filter_candidates_stdev(
                cleaned_segments, potential_segments
            )

        # find most similar predicted segment to actual segments
        max_similarity, aligned_segments = -1, None
        for potential in potential_segments:
            similarity = sum(
                self.similarity(" ".join(hyp), seg)
                for hyp, seg in zip(potential, segments)
            ) / len(cleaned_segments)
            if similarity > max_similarity:
                max_similarity = similarity
                aligned_segments = potential

        # insert punctuations from real segment to predicted segments
        for idx, token in enumerate(segments):
            if token in self.punctuations:
                aligned_segments.insert(idx, [token])

        # add punctuations to offsets
        idx = 0
        for segment in aligned_segments:
            token = segment[0]
            # skip non-punctuation segments
            if token not in self.punctuations:
                idx += len(segment)
                continue

            # start of punctuation is end time of previous token
            start = updated_offsets[idx - 1]["end_time"]

            # end of punctuation is start time of next token
            if idx < len(updated_offsets):
                end = updated_offsets[idx]["start_time"]
            else:
                end = start  # if it's last, end = start

            offset = {"end_time": end, "phoneme": token, "start_time": start}
            updated_offsets.insert(idx, offset)
            idx += 1

        return updated_offsets

    def _filter_candidates_stdev(
        self,
        ground_truth_segments: List[List[str]],
        potential_segments: List[List[List[str]]],
        k: int = 1,
    ) -> List[List[List[str]]]:
        """
        Filters potential segment candidates based on range of
        standard deviation of segment lengths.

        Args:
            ground_truth_segments (List[List[str]]):
                Ground truth segments.
            potential_segments (List[List[List[str]]]):
                List of potential segment candidates to filter.
            k (int, optional):
                Acceptable upper/lower bounds of standard deviation.
                Defaults to `1`.

        Returns:
            List[List[List[str]]]:
                List of filtered segment candidates.
        """
        target_stdev = stdev([len(x.split()) for x in ground_truth_segments])
        stdev_lengths = [
            stdev([len(x) for x in segment]) for segment in potential_segments
        ]

        candidate_idxs = [
            i
            for i, x in enumerate(stdev_lengths)
            if target_stdev - k <= x <= target_stdev + k
        ]
        candidates = [potential_segments[i] for i in candidate_idxs]

        return candidates

    def segment_phonemes_punctuations(
        self, phonemes: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Segment/group list of phonemes consecutively, up to a punctuation.

        ### Example
        ```pycon title="example_segment_phonemes_punctuations.py"
        >>> phonemes = ['h', 'e', '.', 'l', '!', 'o', '.']
        >>> segment, cleaned_segment = segment_phonemes_punctuations(phonemes)
        >>> segment
        ['h e', '.', 'l', '!', 'o', '.']
        >>> cleaned_segment
        ['h e', 'l', 'o']
        ```

        Args:
            phonemes (List[str]):
                List of phonemes.

        Returns:
            Tuple[List[str], List[str]]:
                Pair of equivalently segmented phonemes.
                Second index returns segments without punctuations.
        """
        phoneme_string = " ".join(phonemes)
        backslash_char = "\\"
        segments = re.split(
            f"({'|'.join(f'{backslash_char}{p}' for p in self.punctuations)})",
            phoneme_string,
        )
        segments = [s.strip() for s in segments if s.strip() != ""]
        cleaned_segments = [s for s in segments if s not in self.punctuations]
        return segments, cleaned_segments

    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def generate_partitions(self, lst: List, n: int) -> List[List[List]]:
        """
        Generate all possible `n` consecutive partitions.
        Source: [StackOverflow](https://stackoverflow.com/a/73356868).

        ### Example
        ```pycon title="example_generate_partitions.py"
        >>> lst = [1, 2, 3]
        >>> generate_partitions(lst, 2)
        [[[1], [2, 3]], [[1, 2], [3]]]
        ```

        Args:
            lst (List):
                List to be partitioned.
            n (int):
                Number of partitions to generate.

        Returns:
            List[List[List]]:
                List of all possible list of segments.
        """
        result = []
        for indices in combinations(range(1, len(lst)), n - 1):
            splits = []
            start = 0
            for stop in indices:
                splits.append(lst[start:stop])
                start = stop
            splits.append(lst[start:])
            result.append(splits)
        return result
