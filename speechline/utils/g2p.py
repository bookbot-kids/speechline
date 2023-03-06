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

from typing import Callable

from g2p_id import G2p
from gruut import sentences


def g2p_en(text: str) -> str:
    """
    Convert English text to string of phonemes via gruut.

    Args:
        text (str):
            English text to convert.

    Returns:
        str:
            Phoneme string.
    """
    phonemes = []
    for words in sentences(text):
        for word in words:
            if word.is_major_break or word.is_minor_break:
                phonemes += word.text
            elif word.phonemes:
                phonemes += word.phonemes
    return " ".join(phonemes)


def g2p_id(text: str) -> str:
    """
    Convert Indonesian text to string of phonemes via g2p_id.

    Args:
        text (str):
            Indonesian text to convert.

    Returns:
        str:
            Phoneme string.
    """
    g2p = G2p()
    phonemes = g2p(text)
    return " ".join(p for phoneme in phonemes for p in phoneme)


def get_g2p(language: str) -> Callable:
    """
    Gets the corresponding g2p function given `language`.

    Args:
        language (str):
            Language code. Can be in the form of `en-US` or simply `en`.

    Raises:
        NotImplementedError: Language has no g2p function implemented yet.

    Returns:
        Callable:
            G2p callable function.
    """

    LANG2G2P = {
        "en": g2p_en,
        "id": g2p_id,
    }

    if language.lower() not in LANG2G2P:
        raise NotImplementedError(f"{language} has no g2p function yet!")
    return LANG2G2P[language.lower()]
