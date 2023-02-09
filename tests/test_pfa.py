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

import json
from glob import glob
from pathlib import Path

from gruut import sentences

from speechline.utils.aligners import PunctuationForcedAligner


def g2p(text):
    phonemes = []
    for words in sentences(text):
        for word in words:
            if word.is_major_break or word.is_minor_break:
                phonemes += word.text
            elif word.phonemes:
                phonemes += word.phonemes
    return phonemes


def test_punctuation_forced_aligner(datadir):
    pfa = PunctuationForcedAligner(g2p)
    offset_files = sorted(glob(f"{datadir}/*.json"))

    updated_offsets = []
    for offset_file in offset_files:
        transcript_path = Path(offset_file).with_suffix(".txt")
        offset = json.load(open(offset_file))
        transcript = open(transcript_path).readline()
        updated_offset = pfa(offset, transcript)
        updated_offsets.append(updated_offset)

    assert updated_offsets == [
        [
            {"phoneme": "h", "start_time": 0.0, "end_time": 0.04},
            {"phoneme": "h", "start_time": 0.14, "end_time": 0.2},
            {"phoneme": "ɚ", "start_time": 0.24, "end_time": 0.28},
            {"phoneme": "i", "start_time": 0.42, "end_time": 0.44},
            {"phoneme": "d", "start_time": 0.5, "end_time": 0.54},
            {"phoneme": "ʌ", "start_time": 0.64, "end_time": 0.66},
            {"phoneme": "m", "start_time": 0.7, "end_time": 0.74},
            {"phoneme": "b", "start_time": 0.78, "end_time": 0.82},
            {"phoneme": "ɹ", "start_time": 0.84, "end_time": 0.9},
            {"phoneme": "ɛ", "start_time": 0.92, "end_time": 0.94},
            {"phoneme": "l", "start_time": 1.0, "end_time": 1.04},
            {"phoneme": "ə", "start_time": 1.08, "end_time": 1.12},
            {"phoneme": ",", "start_time": 1.12, "end_time": 1.36},
            {"phoneme": "ɪ", "start_time": 1.36, "end_time": 1.38},
            {"phoneme": "z", "start_time": 1.54, "end_time": 1.58},
            {"phoneme": "d͡ʒ", "start_time": 1.58, "end_time": 1.62},
            {"phoneme": "ʌ", "start_time": 1.62, "end_time": 1.66},
            {"phoneme": "s", "start_time": 1.72, "end_time": 1.76},
            {"phoneme": "t", "start_time": 1.78, "end_time": 1.82},
            {"phoneme": "ð", "start_time": 1.86, "end_time": 1.88},
            {"phoneme": "ə", "start_time": 1.92, "end_time": 1.94},
            {"phoneme": "b", "start_time": 1.98, "end_time": 2.0},
            {"phoneme": "ɛ", "start_time": 2.04, "end_time": 2.06},
            {"phoneme": "s", "start_time": 2.22, "end_time": 2.26},
            {"phoneme": "t", "start_time": 2.38, "end_time": 2.4},
            {"phoneme": ".", "start_time": 2.4, "end_time": 2.4},
        ],
        [
            {"phoneme": "ɪ", "start_time": 0.0, "end_time": 0.02},
            {"phoneme": "t", "start_time": 0.26, "end_time": 0.3},
            {"phoneme": "ɪ", "start_time": 0.34, "end_time": 0.36},
            {"phoneme": "z", "start_time": 0.42, "end_time": 0.44},
            {"phoneme": "n", "start_time": 0.5, "end_time": 0.54},
            {"phoneme": "oʊ", "start_time": 0.54, "end_time": 0.58},
            {"phoneme": "t", "start_time": 0.58, "end_time": 0.62},
            {"phoneme": "ʌ", "start_time": 0.76, "end_time": 0.78},
            {"phoneme": "p", "start_time": 0.92, "end_time": 0.94},
            {"phoneme": ".", "start_time": 0.94, "end_time": 0.94},
        ],
    ]
