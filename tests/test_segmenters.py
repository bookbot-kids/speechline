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

from speechline.segmenters import (
    PhonemeOverlapSegmenter,
    SilenceSegmenter,
    WordOverlapSegmenter,
)


def test_phoneme_overlap_segmenter():
    offsets = [
        {"text": "h", "start_time": 0.16, "end_time": 0.18},
        {"text": "ɝ", "start_time": 0.26, "end_time": 0.28},
        {"text": " ", "start_time": 0.3, "end_time": 0.34},
        {"text": "ɹ", "start_time": 0.36, "end_time": 0.38},
        {"text": "ɛ", "start_time": 0.44, "end_time": 0.46},
        {"text": "d", "start_time": 0.5, "end_time": 0.52},
        {"text": " ", "start_time": 0.6, "end_time": 0.64},
        {"text": "ə", "start_time": 0.72, "end_time": 0.74},
        {"text": "m", "start_time": 0.76, "end_time": 0.78},
        {"text": "b", "start_time": 0.82, "end_time": 0.84},
        {"text": "ɹ", "start_time": 0.84, "end_time": 0.88},
        {"text": "ɛ", "start_time": 0.92, "end_time": 0.94},
        {"text": "l", "start_time": 0.98, "end_time": 1.0},
        {"text": "ə", "start_time": 1.12, "end_time": 1.14},
        {"text": " ", "start_time": 1.3, "end_time": 1.34},
        {"text": "ɪ", "start_time": 1.4, "end_time": 1.42},
        {"text": "z", "start_time": 1.44, "end_time": 1.46},
        {"text": " ", "start_time": 1.52, "end_time": 1.56},
        {"text": "dʒ", "start_time": 1.58, "end_time": 1.6},
        {"text": "ʌ", "start_time": 1.66, "end_time": 1.68},
        {"text": "s", "start_time": 1.7, "end_time": 1.72},
        {"text": "t", "start_time": 1.78, "end_time": 1.8},
        {"text": " ", "start_time": 1.84, "end_time": 1.88},
        {"text": "θ", "start_time": 1.88, "end_time": 1.9},
        {"text": " ", "start_time": 1.96, "end_time": 2.0},
        {"text": "b", "start_time": 2.0, "end_time": 2.02},
        {"text": "ɛ", "start_time": 2.12, "end_time": 2.14},
        {"text": "s", "start_time": 2.18, "end_time": 2.2},
        {"text": "t", "start_time": 2.32, "end_time": 2.34},
    ]
    ground_truth = ["Her", "red", "umbrella", "is", "just", "the", "best"]
    lexicon = {
        "her": ["h ˈɚ", "h ɜ ɹ", "ɜ ɹ", "h ɜː ɹ", "ə ɹ"],
        "red": ["ɹ ˈɛ d", "ɹ ɛ d"],
        "umbrella": ["ˈʌ m b ɹ ˌɛ l ə", "ʌ m b ɹ ɛ l ə"],
        "is": ["ˈɪ z", "ɪ z"],
        "just": ["d͡ʒ ˈʌ s t", "d͡ʒ ʌ s t"],
        "the": ["ð ə", "ð i", "ð iː", "ð ɪ"],
        "best": ["b ˈɛ s t", "b ɛ s t"],
    }
    segmenter = PhonemeOverlapSegmenter(lexicon)
    segments = segmenter.chunk_offsets(offsets, ground_truth)
    assert segments == [
        [{"text": "ɹ ɛ d", "start_time": 0.36, "end_time": 0.52}],
        [
            {"text": "ɪ z", "start_time": 1.4, "end_time": 1.46},
            {"text": "dʒ ʌ s t", "start_time": 1.58, "end_time": 1.8},
        ],
        [{"text": "b ɛ s t", "start_time": 2.0, "end_time": 2.34}],
    ]

    offsets = [
        {"text": "ɹ", "start_time": 0.36, "end_time": 0.38},
        {"text": "ɛ", "start_time": 0.44, "end_time": 0.46},
        {"text": "d", "start_time": 0.5, "end_time": 0.52},
    ]
    ground_truth = ["red"]
    segments = segmenter.chunk_offsets(offsets, ground_truth)
    assert segments == [
        [{"text": "ɹ ɛ d", "start_time": 0.36, "end_time": 0.52}],
    ]

    offsets = [
        {"text": "h", "start_time": 0.16, "end_time": 0.18},
        {"text": "ɝ", "start_time": 0.26, "end_time": 0.28},
    ]
    ground_truth = ["her"]
    segments = segmenter.chunk_offsets(offsets, ground_truth)
    assert segments == []

    offsets = [
        {"text": "h", "start_time": 0.16, "end_time": 0.18},
        {"text": "ɝ", "start_time": 0.26, "end_time": 0.28},
        {"text": " ", "start_time": 0.3, "end_time": 0.34},
        {"text": "ɹ", "start_time": 0.36, "end_time": 0.38},
        {"text": "ɛ", "start_time": 0.44, "end_time": 0.46},
        {"text": "d", "start_time": 0.5, "end_time": 0.52},
    ]
    ground_truth = ["her"]
    segments = segmenter.chunk_offsets(offsets, ground_truth)
    assert segments == []


def test_word_overlap_segmenter():
    offsets = [
        {"end_time": 0.28, "start_time": 0.18, "text": "HER"},
        {"end_time": 0.52, "start_time": 0.34, "text": "RED"},
        {"end_time": 1.12, "start_time": 0.68, "text": "UMBRELLA"},
        {"end_time": 1.46, "start_time": 1.4, "text": "IS"},
        {"end_time": 1.78, "start_time": 1.56, "text": "JUST"},
        {"end_time": 1.94, "start_time": 1.86, "text": "THE"},
        {"end_time": 2.3, "start_time": 1.98, "text": "BEST"},
    ]
    ground_truth = ["red", "umbrella", "just", "the", "best"]
    segmenter = WordOverlapSegmenter()
    segments = segmenter.chunk_offsets(offsets, ground_truth)
    assert segments == [
        [
            {"end_time": 0.52, "start_time": 0.34, "text": "RED"},
            {"end_time": 1.12, "start_time": 0.68, "text": "UMBRELLA"},
        ],
        [
            {"end_time": 1.78, "start_time": 1.56, "text": "JUST"},
            {"end_time": 1.94, "start_time": 1.86, "text": "THE"},
            {"end_time": 2.3, "start_time": 1.98, "text": "BEST"},
        ],
    ]


def test_silence_segmenter():
    offsets = [
        {"start_time": 0.0, "end_time": 0.4},
        {"start_time": 0.4, "end_time": 0.5},
        {"start_time": 0.6, "end_time": 0.8},
        {"start_time": 1.1, "end_time": 1.3},
        {"start_time": 1.5, "end_time": 1.7},
        {"start_time": 1.7, "end_time": 2.0},
    ]
    segmenter = SilenceSegmenter()
    segments = segmenter.chunk_offsets(offsets, silence_duration=0.2)
    assert segments == [
        [
            {"start_time": 0.0, "end_time": 0.4},
            {"start_time": 0.4, "end_time": 0.5},
            {"start_time": 0.6, "end_time": 0.8},
        ],
        [{"start_time": 1.1, "end_time": 1.3}],
        [{"start_time": 1.5, "end_time": 1.7}, {"start_time": 1.7, "end_time": 2.0}],
    ]
