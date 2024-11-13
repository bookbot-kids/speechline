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

from typing import Callable, List
import re

from g2p_id import G2p
from gruut import sentences


def g2p_en(text: str) -> List[str]:
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
                phonemes.append(word.text)
            elif word.phonemes:
                phonemes.append(" ".join(word.phonemes))
    return phonemes


def g2p_sw(text: str) -> List[str]:
    """
    Convert Swahili text to string of phonemes via gruut.

    Args:
        text (str):
            Swahili text to convert.

    Returns:
        str:
            Phoneme string.
    """
    phonemes = []
    for words in sentences(text, lang="sw"):
        for word in words:
            if word.is_major_break or word.is_minor_break:
                phonemes.append(word.text)
            elif word.phonemes:
                _phonemes = word.phonemes[:]

                # NOTE: gruut doesn't handle "ng'" /ŋ/
                # we need to fix e.g. ng'ombe -> /ŋombe/ instead of /ᵑgombe/
                NG_GRAPHEME = "ng'"
                NG_PRENASALIZED_PHONEME = "ᵑg"
                NG_PHONEME = "ŋ"
                if NG_GRAPHEME in word.text:
                    ng_graphemes = re.findall(f"{NG_GRAPHEME}?", word.text)
                    ng_phonemes_idx = [i for i, p in enumerate(_phonemes) if p == NG_PRENASALIZED_PHONEME]
                    assert len(ng_graphemes) == len(ng_phonemes_idx)
                    for i, g in zip(ng_phonemes_idx, ng_graphemes):
                        _phonemes[i] = NG_PHONEME if g == NG_GRAPHEME else _phonemes[i]

                phonemes.append(" ".join(_phonemes))
    return phonemes


def g2p_id(text: str) -> List[str]:
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
    return [" ".join(phoneme) for phoneme in phonemes]


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
        "sw": g2p_sw,
    }

    if language.lower() not in LANG2G2P:
        raise NotImplementedError(f"{language} has no g2p function yet!")
    return LANG2G2P[language.lower()]
