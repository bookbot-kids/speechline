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


class PhonemeOverlapSegmenter(Segmenter):
    def __init__(self, lexicon: Dict[str, List[str]]):
        """
        Phoneme-overlap segmenter, with phoneme variations.

        Args:
            lexicon (Dict[str, List[str]]):
                Lexicon of words and their phoneme variations.
        """
        self.lexicon = lexicon

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return text

    def normalize_phonemes(self, phonemes: str) -> str:
        """
        Remove diacritics from phonemes.
        Modified from: [Michael McAuliffe](https://memcauliffe.com/speaker-dictionaries-and-multilingual-ipa.html#multilingual-ipa-mode) # noqa: E501

        Args:
            phonemes (str):
                Phonemes to normalize.

        Returns:
            str:
                Normalized phonemes.
        """
        diacritics = ["ː", "ˑ", "̆", "̯", "͡", "‿", "͜", "̩", "ˈ", "ˌ"]
        for d in diacritics:
            phonemes = phonemes.replace(d, "")
        return phonemes.strip()

    def _merge_offsets(
        self, offsets: List[Dict[str, Union[str, float]]]
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Merge phoneme-level offsets into word-bounded phoneme offsets.

        Args:
            offsets (List[Dict[str, Union[str, float]]]):
                List of phoneme offsets.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of word-bounded phoneme offset segments.
        """
        result = []
        current_item = {"text": [], "start_time": None, "end_time": None}
        for item in offsets:
            if item["text"] != " ":
                if current_item["start_time"] is None:
                    current_item["start_time"] = item["start_time"]
                current_item["end_time"] = item["end_time"]
                current_item["text"].append(item["text"])
            else:
                if current_item["start_time"] is not None:
                    result.append(current_item)
                    current_item = {"text": [], "start_time": None, "end_time": None}

        if current_item["start_time"] is not None:
            result.append(current_item)

        for r in result:
            r["text"] = " ".join(r["text"])

        return result

    def _generate_combinations(self, ground_truth: List[str]) -> List[List[str]]:
        """
        Generate all possible phoneme combinations for a given word.

        Args:
            ground_truth (List[str]):
                List of words.

        Returns:
            List[List[str]]:
                List of phoneme combinations.
        """
        combinations = []
        for word in ground_truth:
            phonemes = self.lexicon[self.normalize_text(word)]
            phonemes = set(self.normalize_phonemes(p) for p in phonemes)
            combinations.append(phonemes)
        return combinations

    def chunk_offsets(
        self,
        offsets: List[Dict[str, Union[str, float]]],
        ground_truth: List[str],
        **kwargs
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Chunk phoneme-level offsets into word-bounded phoneme offsets.

        ### Example
        ```pycon title="example_phoneme_overlap_segmenter.py"
        >>> from speechline.segmenters import PhonemeOverlapSegmenter
        >>> ground_truth = ["Her", "red", "umbrella", "is", "just", "the", "best"]
        >>> lexicon = {
        ...     "her": ["h ˈɚ", "h ɜ ɹ", "ɜ ɹ", "h ɜː ɹ", "ə ɹ"],
        ...     "red": ["ɹ ˈɛ d", "ɹ ɛ d"],
        ...     "umbrella": ["ˈʌ m b ɹ ˌɛ l ə", "ʌ m b ɹ ɛ l ə"],
        ...     "is": ["ˈɪ z", "ɪ z"],
        ...     "just": ["d͡ʒ ˈʌ s t", "d͡ʒ ʌ s t"],
        ...     "the": ["ð ə", "ð i", "ð iː", "ð ɪ"],
        ...     "best": ["b ˈɛ s t", "b ɛ s t"]
        ... }
        >>> offsets = [
        ...     {'text': 'h', 'start_time': 0.16, 'end_time': 0.18},
        ...     {'text': 'ɝ', 'start_time': 0.26, 'end_time': 0.28},
        ...     {'text': ' ', 'start_time': 0.3, 'end_time': 0.34},
        ...     {'text': 'ɹ', 'start_time': 0.36, 'end_time': 0.38},
        ...     {'text': 'ɛ', 'start_time': 0.44, 'end_time': 0.46},
        ...     {'text': 'd', 'start_time': 0.5, 'end_time': 0.52},
        ...     {'text': ' ', 'start_time': 0.6, 'end_time': 0.64},
        ...     {'text': 'ə', 'start_time': 0.72, 'end_time': 0.74},
        ...     {'text': 'm', 'start_time': 0.76, 'end_time': 0.78},
        ...     {'text': 'b', 'start_time': 0.82, 'end_time': 0.84},
        ...     {'text': 'ɹ', 'start_time': 0.84, 'end_time': 0.88},
        ...     {'text': 'ɛ', 'start_time': 0.92, 'end_time': 0.94},
        ...     {'text': 'l', 'start_time': 0.98, 'end_time': 1.0},
        ...     {'text': 'ə', 'start_time': 1.12, 'end_time': 1.14},
        ...     {'text': ' ', 'start_time': 1.3, 'end_time': 1.34},
        ...     {'text': 'ɪ', 'start_time': 1.4, 'end_time': 1.42},
        ...     {'text': 'z', 'start_time': 1.44, 'end_time': 1.46},
        ...     {'text': ' ', 'start_time': 1.52, 'end_time': 1.56},
        ...     {'text': 'dʒ', 'start_time': 1.58, 'end_time': 1.6},
        ...     {'text': 'ʌ', 'start_time': 1.66, 'end_time': 1.68},
        ...     {'text': 's', 'start_time': 1.7, 'end_time': 1.72},
        ...     {'text': 't', 'start_time': 1.78, 'end_time': 1.8},
        ...     {'text': ' ', 'start_time': 1.84, 'end_time': 1.88},
        ...     {'text': 'θ', 'start_time': 1.88, 'end_time': 1.9},
        ...     {'text': ' ', 'start_time': 1.96, 'end_time': 2.0},
        ...     {'text': 'b', 'start_time': 2.0, 'end_time': 2.02},
        ...     {'text': 'ɛ', 'start_time': 2.12, 'end_time': 2.14},
        ...     {'text': 's', 'start_time': 2.18, 'end_time': 2.2},
        ...     {'text': 't', 'start_time': 2.32, 'end_time': 2.34}
        ... ]
        >>> segmenter = PhonemeOverlapSegmenter(lexicon)
        >>> segmenter.chunk_offsets(offsets, ground_truth)
        [
            [
                {'text': 'ɹ ɛ d', 'start_time': 0.36, 'end_time': 0.52}
            ],
            [
                {'text': 'ɪ z', 'start_time': 1.4, 'end_time': 1.46},
                {'text': 'dʒ ʌ s t', 'start_time': 1.58, 'end_time': 1.8}
            ],
            [
                {'text': 'b ɛ s t', 'start_time': 2.0, 'end_time': 2.34}
            ]
        ]
        ```

        Args:
            offsets (List[Dict[str, Union[str, float]]]):
                List of phoneme offsets.
            ground_truth (List[str]):
                List of words.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of word-bounded phoneme offset segments.
        """

        ground_truth = self._generate_combinations(ground_truth)
        merged_offsets = self._merge_offsets(offsets)
        transcripts = [self.normalize_phonemes(o["text"]) for o in merged_offsets]

        idxs, index = [], 0  # index in ground truth
        for i, word in enumerate(transcripts):
            if index >= len(ground_truth):
                break
            for var in ground_truth[index:]:
                # match
                if word in var:
                    idxs.append(i)
                    break
            index += 1

        # if no matches
        if not idxs:
            return []

        # collapse longest consecutive indices
        merged_idxs = []
        start, end = idxs[0], idxs[0] + 1
        for i in idxs[1:]:
            if i == end:
                end += 1
            else:
                merged_idxs.append((start, end))
                start, end = i, i + 1
        merged_idxs.append((start, end))

        # segment according to longest consecutive indices
        segments = [merged_offsets[i:j] for (i, j) in merged_idxs]
        return segments
