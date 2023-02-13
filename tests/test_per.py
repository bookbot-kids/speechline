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


import pytest

from speechline.utils.metrics import PhonemeErrorRate


def test_phoneme_error_rate():
    lexicon = {
        "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        "guy": [["g", "a", "i"]],
        "i": [["aɪ"]],
        "am": [["æ", "m"]],
        "bahrainian": [
            ["b", "ɚ", "eɪ", "n", "i", "ə", "n"],
            ["b", "aɪ", "ɹ", "ɑ", "n", "i", "ə", "n"],
            ["b", "ɑ", "ɹ", "eɪ", "n", "i", "ə", "n"],
        ],
    }

    per = PhonemeErrorRate(lexicon)

    sequences = [["i", "am", "bahrainian"]]

    assert per(sequences, [["aɪ", "æ", "m", "b", "ɚ", "eɪ", "n", "i", "ə", "n"]]) == 0.0
    assert (
        per(sequences, [["aɪ", "æ", "m", "b", "ɑ", "ɹ", "eɪ", "n", "i", "ə", "n"]])
        == 0.0
    )

    sequences = [["hello", "guy"]]

    assert per(sequences, [["h", "a", "l", "l", "o", "g", "a", "i"]]) == 0.0
    assert per(sequences, [["h", "a", "l", "l", "X", "g", "a", "i"]]) == 0.125
    assert per(sequences, [["h", "a", "l", "l", "X", "X", "X", "g", "a", "i"]]) == 0.375
    assert per(sequences, [["h", "a", "X", "g", "a", "i"]]) == 0.375
    assert per(sequences, [["h", "a", "l", "l", "X", "o", "g", "a", "i"]]) == 0.125
    assert (
        per(
            sequences,
            [["h", "a", "l", "l", "X", "X", "X", "o", "g", "a", "i"]],
        )
        == 0.375
    )
    assert per(sequences, [["h", "a", "l", "l", "g", "a", "i"]]) == 0.125
    assert per(sequences, [["h", "a", "l", "g", "a", "i"]]) == 0.25
    assert per(sequences, [["h", "e", "l", "l", "o", "g", "a", "i"]]) == 0.0
    assert (
        per(
            sequences,
            [["h", "X", "e", "X", "X", "l", "l", "o", "g", "a", "i"]],
        )
        == 0.375
    )
    assert per(sequences, [["h", "e", "X", "X", "l", "l", "o", "g", "a", "i"]]) == 0.25
    assert per(sequences, [["h", "e", "o", "g", "a", "i"]]) == 0.25
    assert per(sequences, [["h", "e", "X", "l", "o", "g", "a", "i"]]) == 0.125

    sequences = [["hello", "guy", "hello"]]

    assert (
        per(
            sequences,
            [
                [
                    "h",
                    "a",
                    "l",
                    "l",
                    "o",
                    "g",
                    "a",
                    "i",
                    "h",
                    "e",
                    "l",
                    "l",
                    "o",
                ]
            ],
        )
        == 0.0
    )
    assert (
        per(sequences, [["h", "a", "l", "l", "o", "g", "h", "e", "o"]])
        == 0.3076923076923077
    )
    assert (
        per(
            sequences,
            [
                [
                    "h",
                    "a",
                    "l",
                    "l",
                    "X",
                    "o",
                    "g",
                    "a",
                    "i",
                    "X",
                    "h",
                    "X",
                    "e",
                    "X",
                    "l",
                    "l",
                    "o",
                ]
            ],
        )
        == 0.3076923076923077
    )
    assert (
        per(
            sequences,
            [["h", "a", "X", "o", "g", "X", "i", "h", "e", "l", "l", "X"]],
        )
        == 0.3076923076923077
    )
    assert (
        per(sequences, [["g", "a", "i", "h", "e", "l", "l", "o"]])
        == 0.38461538461538464
    )
    assert (
        per(
            sequences,
            [
                [
                    "h",
                    "a",
                    "l",
                    "X",
                    "X",
                    "X",
                    "X",
                    "a",
                    "i",
                    "h",
                    "X",
                    "l",
                    "l",
                    "o",
                ]
            ],
        )
        == 0.38461538461538464
    )


def test_invalid_per_arguments():
    lexicon = {
        "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        "guy": [["g", "a", "i"]],
    }
    per = PhonemeErrorRate(lexicon)
    sequences = [["hello"], ["guy"]]
    predictions = [["h", "e", "l", "l", "o"]]
    with pytest.raises(ValueError):
        _ = per(sequences, predictions)
