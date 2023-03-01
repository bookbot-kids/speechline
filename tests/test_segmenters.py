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

from speechline.segmenters import SilenceSegmenter, WordOverlapSegmenter


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
    segmenters = WordOverlapSegmenter()
    segments = segmenters.chunk_offsets(offsets, ground_truth)
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
